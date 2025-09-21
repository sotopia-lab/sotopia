"""Social game environment that reads its rulebook and action space from JSON."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from pydantic import BaseModel, Field, RootModel, ValidationError

from sotopia.envs.parallel import ParallelSotopiaEnv, render_text_for_agent
from sotopia.messages import AgentAction, Observation, SimpleMessage


class RoleActionConfig(BaseModel):
    """Declared abilities and messaging semantics for a specific role."""

    name: str
    team: str
    description: str = ""
    goal_prompt: str = ""
    default_actions: list[str] = Field(default_factory=lambda: ["speak", "action"])
    phase_actions: dict[str, list[str]] = Field(default_factory=dict)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    allow_team_private_speech: bool = False
    allow_role_private_speech: bool = False


class RoleActionLibrary(RootModel[dict[str, RoleActionConfig]]):
    """Pydantic wrapper for mapping roles to role metadata."""

    def team_for_role(self, role: str) -> str:
        return self.root[role].team


class PhaseResolution(BaseModel):
    operation: str = Field(
        default="noop",
        description="Name of the builtin resolution handler to invoke at phase end.",
    )
    state_key: str | None = None
    visibility: str = Field(
        default="public",
        description="Default visibility for resolution feedback.",
    )


class PhaseDefinition(BaseModel):
    name: str
    kind: str = Field(
        default="discussion",
        description="Macro describing how the phase behaves (discussion, team_target, vote, single_target, announcement).",
    )
    group: str | None = Field(
        default=None,
        description="Optional label used to cluster phases into higher-level cycles (e.g., 'night', 'day').",
    )
    turn_mode: str = Field(
        default="round-robin",
        description="round-robin => sequential actors, simultaneous => everyone at once, single => one actor only.",
    )
    acting_roles: list[str] | None = None
    acting_teams: list[str] | None = None
    max_cycles: int = Field(
        default=1,
        description="Number of complete round-robin passes required before the phase advances.",
    )
    max_turns: int | None = Field(
        default=None,
        description="Optional cap on total turns inside the phase (overrides max_cycles when smaller).",
    )
    speech_visibility: str = Field(
        default="public",
        description="Where speech is visible ('public', 'team', 'private', 'hidden').",
    )
    action_visibility: str = Field(
        default="public",
        description="Where action outcomes are visible ('public', 'team', 'private', 'hidden').",
    )
    instructions: list[str] = Field(
        default_factory=list,
        description="General prompts injected into agent observations for this phase.",
    )
    role_instructions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Optional role-specific prompts keyed by role name.",
    )
    resolution: PhaseResolution | None = None
    entry_messages: list[str] = Field(default_factory=list)
    exit_messages: list[str] = Field(default_factory=list)
    description: str = ""


class EndConditionDefinition(BaseModel):
    operation: str
    team: str | None = None
    other_team: str | None = None
    winner: str | None = None
    message: str | None = None


class RulebookConfig(BaseModel):
    initial_phase: str
    phases: list[PhaseDefinition]
    phase_transitions: dict[str, str]
    end_conditions: list[EndConditionDefinition] = Field(default_factory=list)
    max_cycles: int | None = Field(
        default=None,
        description="Optional safety bound on day/night cycles to prevent infinite games.",
    )


@dataclass
class AgentState:
    name: str
    role: str
    team: str
    alive: bool = True
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseEvents:
    public: list[str] = field(default_factory=list)
    team: dict[str, list[str]] = field(default_factory=dict)
    private: dict[str, list[str]] = field(default_factory=dict)
    system: list[str] = field(default_factory=list)

    def extend(self, other: "PhaseEvents") -> None:
        self.public.extend(other.public)
        for team, messages in other.team.items():
            self.team.setdefault(team, []).extend(messages)
        for agent, messages in other.private.items():
            self.private.setdefault(agent, []).extend(messages)
        self.system.extend(other.system)

    @classmethod
    def phase_entry(cls, phase_name: str, messages: list[str]) -> "PhaseEvents":
        events = cls()
        for msg in messages:
            events.public.append(f"[God] Phase '{phase_name}' begins: {msg}")
        if not messages:
            events.public.append(f"[God] Phase '{phase_name}' begins.")
        return events


class GameRulebook:
    """Runtime state machine that enforces the JSON described social game."""

    def __init__(self, rules: RulebookConfig, roles: RoleActionLibrary) -> None:
        self.rules = rules
        self.roles = roles
        self.phase_lookup = {phase.name: phase for phase in rules.phases}
        self.agent_states: dict[str, AgentState] = {}
        self.agent_name_lookup: dict[str, str] = {}
        self.current_phase: str = rules.initial_phase
        self.phase_cycle_progress: int = 0
        self.turns_in_phase: int = 0
        self.current_actor_index: int = 0
        self.state_flags: dict[str, Any] = {}
        self.group_cycle: dict[str, int] = {}
        self.group_stage: dict[str, int] = {}
        self.current_phase_meta: dict[str, Any] = {}
        self.pending_events: PhaseEvents = PhaseEvents()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def assign_agents(
        self,
        agents: Sequence[str],
        role_assignments: dict[str, str],
    ) -> None:
        self.agent_states = {}
        self.agent_name_lookup = {}
        for name in agents:
            role = role_assignments[name]
            role_cfg = self.roles.root.get(role)
            if role_cfg is None:
                raise ValueError(f"Unknown role '{role}' for agent '{name}'")
            attrs = dict(role_cfg.initial_state)
            state = AgentState(
                name=name,
                role=role,
                team=role_cfg.team,
                alive=True,
                attributes=attrs,
            )
            self.agent_states[name] = state
            self.agent_name_lookup[name.lower()] = name
            self.agent_name_lookup[name.split()[0].lower()] = name

        self.current_phase = self.rules.initial_phase
        self.phase_cycle_progress = 0
        self.turns_in_phase = 0
        self.current_actor_index = 0
        self.state_flags = {
            "day_execution": None,
            "night_target": None,
            "witch_saved": None,
            "witch_poisoned": None,
            "seer_result": "",
        }
        self.group_cycle.clear()
        self.group_stage.clear()
        self.current_phase_meta = {}
        self._register_phase_entry(self.current_phase)
        entry_phase = self.phase_lookup[self.current_phase]
        self.pending_events = PhaseEvents.phase_entry(
            self.current_phase, entry_phase.entry_messages
        )

    # ------------------------------------------------------------------
    # Accessors used by the environment
    # ------------------------------------------------------------------
    def alive_agents(self) -> list[str]:
        return [name for name, state in self.agent_states.items() if state.alive]

    def active_agents_for_phase(self) -> list[str]:
        phase = self.phase_lookup[self.current_phase]
        eligible = self._eligible_candidates(phase)
        if not eligible:
            return []
        if phase.turn_mode == "round-robin":
            idx = self.current_actor_index
            if idx >= len(eligible):
                idx = len(eligible) - 1
            if idx < 0:
                idx = 0
            return [eligible[idx]]
        return eligible

    def available_actions(self, agent_name: str) -> list[str]:
        agent_state = self.agent_states[agent_name]
        if not agent_state.alive:
            return ["none"]
        role_cfg = self.roles.root[agent_state.role]
        actions = role_cfg.phase_actions.get(
            self.current_phase, role_cfg.default_actions
        )
        if "none" not in actions:
            actions = list(actions) + ["none"]
        return actions

    def collect_pending_events(self) -> PhaseEvents:
        events = self.pending_events
        self.pending_events = PhaseEvents()
        return events

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------
    def process_actions(
        self, actions: dict[str, AgentAction]
    ) -> tuple[PhaseEvents, bool, Optional[dict[str, str]]]:
        phase = self.phase_lookup[self.current_phase]
        acting_agents = self.active_agents_for_phase()
        events = PhaseEvents()

        if phase.kind == "announcement":
            events.extend(self._resolve_phase(phase, {}))
            winner = self._check_end_conditions()
            self._schedule_phase_exit(phase)
            return events, True, winner

        if not acting_agents:
            events.extend(self._resolve_phase(phase, {}))
            winner = self._check_end_conditions()
            self._schedule_phase_exit(phase)
            return events, True, winner

        relevant = {
            name: actions.get(name, AgentAction(action_type="none", argument=""))
            for name in acting_agents
        }

        if phase.turn_mode == "round-robin":
            actor = acting_agents[0]
            events.extend(self._record_speech(actor, relevant[actor], phase))
            events.extend(self._resolve_phase(phase, {actor: relevant[actor]}))
            self._advance_round_robin(phase)
            advance = self._should_advance(phase)
        else:
            for actor, action in relevant.items():
                events.extend(self._record_speech(actor, action, phase))
            events.extend(self._resolve_phase(phase, relevant))
            advance = True

        winner = self._check_end_conditions()
        if winner:
            self._schedule_phase_exit(phase)
            return events, True, winner

        if advance:
            self._schedule_phase_exit(phase)
        return events, advance, winner

    def start_next_phase(self) -> PhaseEvents:
        next_phase = self.rules.phase_transitions.get(self.current_phase)
        if next_phase is None:
            raise ValueError(
                f"No transition defined after phase '{self.current_phase}'"
            )
        self.current_phase = next_phase
        self.phase_cycle_progress = 0
        self.turns_in_phase = 0
        self.current_actor_index = 0
        self._register_phase_entry(next_phase)
        phase_def = self.phase_lookup[next_phase]
        entry = PhaseEvents.phase_entry(next_phase, phase_def.entry_messages)
        return entry

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _phase_group(self, phase: PhaseDefinition) -> str:
        if phase.group:
            return phase.group
        return phase.name

    def _register_phase_entry(self, phase_name: str) -> None:
        phase = self.phase_lookup[phase_name]
        group = self._phase_group(phase)
        previous_group = (
            self.current_phase_meta.get("group") if self.current_phase_meta else None
        )
        cycle = self.group_cycle.get(group, 0)
        stage = self.group_stage.get(group, 0)
        if previous_group != group:
            cycle += 1
            stage = 1
        else:
            stage += 1
        self.group_cycle[group] = cycle
        self.group_stage[group] = stage
        self.current_phase_meta = {
            "phase": phase_name,
            "group": group,
            "group_cycle": cycle,
            "group_stage": stage,
            "display_name": phase.name.replace("_", " ").title(),
        }

    def current_phase_metadata(self) -> dict[str, Any]:
        return dict(self.current_phase_meta) if self.current_phase_meta else {}

    def _eligible_candidates(self, phase: PhaseDefinition) -> list[str]:
        names = [name for name, state in self.agent_states.items() if state.alive]
        if phase.acting_roles:
            names = [
                name
                for name in names
                if self.agent_states[name].role in phase.acting_roles
            ]
        if phase.acting_teams:
            names = [
                name
                for name in names
                if self.agent_states[name].team in phase.acting_teams
            ]
        return names

    def _record_speech(
        self, actor: str, action: AgentAction, phase: PhaseDefinition
    ) -> PhaseEvents:
        events = PhaseEvents()
        if action.action_type not in {"speak", "non-verbal communication"}:
            return events
        utterance = action.argument.strip()
        if not utterance:
            return events
        line = f'{actor} said: "{utterance}"'
        if phase.speech_visibility == "team":
            team = self.agent_states[actor].team
            events.team.setdefault(team, []).append(line)
        elif phase.speech_visibility == "private":
            events.private.setdefault(actor, []).append(line)
        elif phase.speech_visibility == "hidden":
            pass
        else:
            events.public.append(line)
        return events

    def _resolve_phase(
        self,
        phase: PhaseDefinition,
        actions: dict[str, AgentAction],
    ) -> PhaseEvents:
        if phase.resolution is None:
            return PhaseEvents()
        handler = getattr(self, f"_resolve_{phase.resolution.operation}", None)
        if handler is None:
            raise ValueError(
                f"Unsupported resolution operation '{phase.resolution.operation}'"
            )
        return handler(phase, actions, phase.resolution)

    def _resolve_noop(
        self,
        phase: PhaseDefinition,
        actions: dict[str, AgentAction],
        resolution: PhaseResolution,
    ) -> PhaseEvents:
        return PhaseEvents()

    def _resolve_store_target(
        self,
        phase: PhaseDefinition,
        actions: dict[str, AgentAction],
        resolution: PhaseResolution,
    ) -> PhaseEvents:
        events = PhaseEvents()
        target = self._extract_target(actions.values())
        if target:
            self.state_flags[resolution.state_key or "night_target"] = target
            teams = phase.acting_teams or [self.agent_states[a].team for a in actions]
            for team in teams:
                events.team.setdefault(team, []).append(
                    f"[God] Target locked: {target}."
                )
        return events

    def _resolve_seer_inspect(
        self,
        phase: PhaseDefinition,
        actions: dict[str, AgentAction],
        resolution: PhaseResolution,
    ) -> PhaseEvents:
        events = PhaseEvents()
        if not actions:
            return events
        actor, action = next(iter(actions.items()))
        target = self._extract_target([action])
        if not target:
            events.private.setdefault(actor, []).append(
                "[God] Vision failed: unable to interpret your target."
            )
            return events
        team = self.agent_states[target].team
        message = f"[God] Vision reveals {target} serves team {team}."
        events.private.setdefault(actor, []).append(message)
        self.state_flags["seer_result"] = message
        return events

    def _resolve_witch_phase(
        self,
        phase: PhaseDefinition,
        actions: dict[str, AgentAction],
        resolution: PhaseResolution,
    ) -> PhaseEvents:
        events = PhaseEvents()
        if not actions:
            return events
        actor, action = next(iter(actions.items()))
        state = self.agent_states[actor]
        text = action.argument.lower()
        if "save" in text and state.attributes.get("save_available", True):
            target = self._extract_target([action]) or self.state_flags.get(
                "night_target"
            )
            if target:
                self.state_flags["witch_saved"] = target
                state.attributes["save_available"] = False
                events.private.setdefault(actor, []).append(
                    f"[God] You secretly saved {target} tonight."
                )
        if "poison" in text and state.attributes.get("poison_available", True):
            target = self._extract_target([action])
            if target:
                self.state_flags["witch_poisoned"] = target
                state.attributes["poison_available"] = False
                events.private.setdefault(actor, []).append(
                    f"[God] You poisoned {target}."
                )
        if not text.strip() or "pass" in text:
            events.private.setdefault(actor, []).append(
                "[God] You chose to remain idle."
            )
        return events

    def _resolve_resolve_night(
        self,
        phase: PhaseDefinition,
        actions: dict[str, AgentAction],
        resolution: PhaseResolution,
    ) -> PhaseEvents:
        events = PhaseEvents()
        saved = self.state_flags.get("witch_saved")
        target = self.state_flags.get("night_target")
        poison = self.state_flags.get("witch_poisoned")
        casualties: list[str] = []
        if target and target != saved:
            casualties.append(target)
        if poison and poison not in casualties:
            casualties.append(poison)
        if not casualties:
            events.public.append("[God] Dawn breaks peacefully. No one died.")
        for victim in casualties:
            if victim in self.agent_states and self.agent_states[victim].alive:
                self.agent_states[victim].alive = False
                events.public.append(f"[God] {victim} was found dead at dawn.")
        self.state_flags["night_target"] = None
        self.state_flags["witch_saved"] = None
        self.state_flags["witch_poisoned"] = None
        self.state_flags["seer_result"] = ""
        return events

    def _resolve_vote(
        self,
        phase: PhaseDefinition,
        actions: dict[str, AgentAction],
        resolution: PhaseResolution,
    ) -> PhaseEvents:
        events = PhaseEvents()
        tally: dict[str, int] = {}
        for action in actions.values():
            target = self._extract_target([action])
            if target:
                tally[target] = tally.get(target, 0) + 1
            elif "none" in action.argument.lower():
                tally.setdefault("none", 0)
                tally["none"] += 1
        if not tally:
            events.public.append("[God] No valid votes were cast.")
            self.state_flags["day_execution"] = None
            return events
        winner, votes = max(tally.items(), key=lambda kv: kv[1])
        if winner == "none":
            events.public.append("[God] The town decided to stay their hand.")
            self.state_flags["day_execution"] = None
            return events
        if list(tally.values()).count(votes) > 1:
            events.public.append("[God] The vote is tied. No execution today.")
            self.state_flags["day_execution"] = None
            return events
        self.state_flags["day_execution"] = winner
        events.public.append(
            f"[God] Majority condemns {winner}. Execution will happen at twilight."
        )
        return events

    def _resolve_post_vote_cleanup(
        self,
        phase: PhaseDefinition,
        actions: dict[str, AgentAction],
        resolution: PhaseResolution,
    ) -> PhaseEvents:
        events = PhaseEvents()
        target = self.state_flags.get("day_execution")
        if target and target in self.agent_states and self.agent_states[target].alive:
            self.agent_states[target].alive = False
            team = self.agent_states[target].team
            events.public.append(
                f"[God] {target} was executed. They belonged to team {team}."
            )
        self.state_flags["day_execution"] = None
        return events

    def _extract_target(self, actions: Iterable[AgentAction]) -> str | None:
        for action in actions:
            corpus = f"{action.action_type} {action.argument}".lower()
            for name in self.agent_states:
                if name.lower() in corpus:
                    return name
            for name in self.agent_states:
                first = name.split()[0].lower()
                if first in corpus:
                    return name
        return None

    def _advance_round_robin(self, phase: PhaseDefinition) -> None:
        base = self._eligible_candidates(phase)
        self.turns_in_phase += 1
        if not base:
            self.current_actor_index = 0
            return
        self.current_actor_index += 1
        if self.current_actor_index >= len(base):
            self.phase_cycle_progress += 1
            self.current_actor_index = 0

    def _should_advance(self, phase: PhaseDefinition) -> bool:
        if phase.turn_mode != "round-robin":
            return True
        base = self._eligible_candidates(phase)
        if not base:
            return True
        if phase.max_turns is not None and self.turns_in_phase >= phase.max_turns:
            return True
        if self.phase_cycle_progress >= phase.max_cycles:
            return True
        return False

    def _schedule_phase_exit(self, phase: PhaseDefinition) -> None:
        exit_events = PhaseEvents()
        for msg in phase.exit_messages:
            exit_events.public.append(f"[God] {msg}")
        self.pending_events.extend(exit_events)

    def _check_end_conditions(self) -> Optional[dict[str, str]]:
        for cond in self.rules.end_conditions:
            if cond.operation == "team_eliminated" and cond.team:
                alive = sum(
                    1
                    for state in self.agent_states.values()
                    if state.alive and state.team == cond.team
                )
                if alive == 0:
                    message = (
                        cond.message or f"[God] Team {cond.team} has been eliminated."
                    )
                    return {
                        "winner": cond.winner or cond.other_team or cond.team,
                        "message": message,
                    }
            if cond.operation == "parity" and cond.team and cond.other_team:
                team_count = sum(
                    1
                    for state in self.agent_states.values()
                    if state.alive and state.team == cond.team
                )
                other_count = sum(
                    1
                    for state in self.agent_states.values()
                    if state.alive and state.team == cond.other_team
                )
                if team_count >= other_count:
                    message = cond.message or (
                        f"[God] Parity reached: {cond.team} now matches or exceeds {cond.other_team}."
                    )
                    return {
                        "winner": cond.winner or cond.team,
                        "message": message,
                    }
        return None


class SocialGameEnv(ParallelSotopiaEnv):
    """Environment subclass that enforces multi-phase social game mechanics."""

    def __init__(
        self,
        env_profile,
        *,
        rulebook_path: str,
        actions_path: str,
        role_assignments: dict[str, str],
        **kwargs: Any,
    ) -> None:
        super().__init__(env_profile=env_profile, **kwargs)
        self._rulebook_path = Path(rulebook_path)
        self._actions_path = Path(actions_path)
        self._role_assignments = role_assignments
        self.game_rulebook: GameRulebook | None = None
        self._last_events: PhaseEvents = PhaseEvents()
        self._winner_payload: dict[str, str] | None = None
        self.phase_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Config loading helpers
    # ------------------------------------------------------------------
    def _load_configs(self) -> tuple[RulebookConfig, RoleActionLibrary]:
        try:
            rules = RulebookConfig.model_validate_json(self._rulebook_path.read_text())
        except ValidationError as exc:
            raise ValueError(f"Invalid rulebook config: {exc}") from exc
        actions_raw = json.loads(self._actions_path.read_text())
        try:
            roles = RoleActionLibrary.model_validate(actions_raw["roles"])
        except (KeyError, ValidationError) as exc:
            raise ValueError(f"Invalid action-space config: {exc}") from exc
        return rules, roles

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
        agents=None,
        omniscient: bool = False,
        lite: bool = False,
    ) -> dict[str, Observation]:
        base_obs = super().reset(
            seed=seed,
            options=options,
            agents=agents,
            omniscient=omniscient,
            lite=lite,
        )
        rules, role_actions = self._load_configs()
        self.game_rulebook = GameRulebook(rules, role_actions)
        self.game_rulebook.assign_agents(self.agents, self._role_assignments)
        self.phase_log = []
        self._apply_action_mask()
        self._last_events = self.game_rulebook.collect_pending_events()
        self._winner_payload = None
        self._record_phase_history(
            phase_name=self.game_rulebook.current_phase,
            actions={},
            events=self._last_events,
        )
        return self._augment_observations(base_obs, append_to_existing=True)

    def _phase_prompt_lines(
        self,
        *,
        agent_name: str,
        phase: PhaseDefinition,
        acting: bool,
        available: list[str],
    ) -> list[str]:
        assert self.game_rulebook is not None
        meta = self.game_rulebook.current_phase_metadata()
        group = meta.get("group")
        cycle = meta.get("group_cycle")
        stage = meta.get("group_stage")
        title = phase.name.replace("_", " ").title()
        if group:
            group_label = group.replace("_", " ").title()
            if cycle and stage:
                label = f"{group_label} {cycle}.{stage} – {title}"
            elif cycle:
                label = f"{group_label} {cycle} – {title}"
            else:
                label = f"{group_label}: {title}"
        else:
            label = title
        lines = [f"[God] Phase: {label}"]
        if acting:
            lines.append("[God] It is your turn to act in this phase.")
        else:
            lines.append("[God] You are observing while others act.")
        lines.append(f"[God] Available actions right now: {', '.join(available)}")
        lines.extend(f"[God] {text}" for text in phase.instructions)
        role = self.game_rulebook.agent_states[agent_name].role
        for text in phase.role_instructions.get(role, []):
            lines.append(f"[God] {text}")
        return lines

    def _record_phase_history(
        self,
        *,
        phase_name: str,
        actions: dict[str, AgentAction],
        events: PhaseEvents,
    ) -> None:
        if self.game_rulebook is None:
            return
        if not (events.public or events.team or events.private):
            if any(a.action_type != "none" for a in actions.values()):
                pass
            else:
                return
        action_summary = {
            agent: {"action_type": action.action_type, "argument": action.argument}
            for agent, action in actions.items()
            if action.action_type != "none"
        }
        phase_def = (
            self.game_rulebook.phase_lookup.get(phase_name)
            if self.game_rulebook
            else None
        )
        snapshot = {
            "phase": phase_name,
            "turn": self.turn_number,
            "public": list(events.public),
            "team": {team: list(msgs) for team, msgs in events.team.items()},
            "private": {agent: list(msgs) for agent, msgs in events.private.items()},
            "actions": action_summary,
            "meta": self.game_rulebook.current_phase_metadata()
            if self.game_rulebook
            else {},
            "instructions": phase_def.instructions if phase_def else [],
            "role_instructions": phase_def.role_instructions if phase_def else {},
        }
        self.phase_log.append(snapshot)

    def _augment_observations(
        self,
        baseline: dict[str, Observation],
        *,
        append_to_existing: bool,
    ) -> dict[str, Observation]:
        assert self.game_rulebook is not None
        acting = set(self.game_rulebook.active_agents_for_phase())
        events = self._last_events
        phase_name = self.game_rulebook.current_phase
        phase_def = self.game_rulebook.phase_lookup[phase_name]
        new_obs: dict[str, Observation] = {}
        for idx, agent_name in enumerate(self.agents):
            current = baseline[agent_name]
            available = (
                self.game_rulebook.available_actions(agent_name)
                if agent_name in acting
                else ["none"]
            )
            phase_lines = self._phase_prompt_lines(
                agent_name=agent_name,
                phase=phase_def,
                acting=agent_name in acting,
                available=available,
            )
            messages: list[str] = []
            messages.extend(events.public)
            team = self.game_rulebook.agent_states[agent_name].team
            messages.extend(events.team.get(team, []))
            messages.extend(events.private.get(agent_name, []))
            if not messages:
                messages.append("[God] Await instructions from the host.")
            segments: list[str] = []
            if append_to_existing:
                prefix = current.last_turn.strip()
                if prefix:
                    segments.append(prefix)
            segments.extend(phase_lines)
            segments.extend(messages)
            combined = "\n".join(segment for segment in segments if segment)
            new_obs[agent_name] = Observation(
                last_turn=render_text_for_agent(combined, agent_id=idx),
                turn_number=current.turn_number,
                available_actions=available,
            )
        return new_obs

    def _create_blank_observations(self) -> dict[str, Observation]:
        assert self.game_rulebook is not None
        acting = set(self.game_rulebook.active_agents_for_phase())
        blank: dict[str, Observation] = {}
        for agent_name in self.agents:
            available = (
                self.game_rulebook.available_actions(agent_name)
                if agent_name in acting
                else ["none"]
            )
            blank[agent_name] = Observation(
                last_turn="",
                turn_number=self.turn_number,
                available_actions=available,
            )
        return blank

    def _apply_action_mask(self) -> None:
        assert self.game_rulebook is not None
        acting = set(self.game_rulebook.active_agents_for_phase())
        self.action_mask = [
            agent in acting and self.game_rulebook.agent_states[agent].alive
            for agent in self.agents
        ]

    async def astep(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        assert self.game_rulebook is not None
        self._apply_action_mask()
        self.turn_number += 1
        prepared = self._coerce_actions(actions)
        self.recv_message(
            "Environment", SimpleMessage(message=f"Turn #{self.turn_number}")
        )
        for agent, action in prepared.items():
            self.recv_message(agent, action)
        phase_name = self.game_rulebook.current_phase
        events, advance, winner = self.game_rulebook.process_actions(prepared)
        exit_events = self.game_rulebook.collect_pending_events()
        events.extend(exit_events)
        self._record_phase_history(
            phase_name=phase_name,
            actions=prepared,
            events=events,
        )
        self._last_events = events
        if advance:
            next_events = self.game_rulebook.start_next_phase()
            self._record_phase_history(
                phase_name=self.game_rulebook.current_phase,
                actions={},
                events=next_events,
            )
            self._last_events.extend(next_events)
        self._apply_action_mask()
        baseline = self._create_blank_observations()
        observations = self._augment_observations(baseline, append_to_existing=False)
        rewards = {agent_name: 0 for agent_name in self.agents}
        terminated = {agent_name: bool(winner) for agent_name in self.agents}
        truncations = {agent_name: False for agent_name in self.agents}
        info = {
            agent_name: {
                "comments": winner["message"] if winner else "",
                "complete_rating": 0,
            }
            for agent_name in self.agents
        }
        if winner:
            self._winner_payload = winner
        return observations, rewards, terminated, truncations, info

    def _coerce_actions(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> dict[str, AgentAction]:
        prepared: dict[str, AgentAction] = {}
        for agent, raw in actions.items():
            if isinstance(raw, AgentAction):
                prepared[agent] = raw
            else:
                idx = int(raw.get("action_type", 0))
                action_type = self.available_action_types[idx]
                prepared[agent] = AgentAction(
                    action_type=action_type,
                    argument=str(raw.get("argument", "")),
                )
        return prepared

    def step(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        return asyncio.run(self.astep(actions))
