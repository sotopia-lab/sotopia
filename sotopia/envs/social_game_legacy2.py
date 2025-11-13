from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sotopia.envs.parallel import ParallelSotopiaEnv, render_text_for_agent
from sotopia.agents.llm_agent import Agents
from sotopia.database import EnvironmentProfile
from sotopia.messages import AgentAction, Observation, SimpleMessage


# -----------------------------
# Data containers
# -----------------------------


@dataclass
class AgentState:
    """Runtime state for an agent: identity, role/team, alive flag and extras."""

    name: str
    role: str
    team: str
    alive: bool = True
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Events:
    """Aggregated event streams for a phase step (public/team/private)."""

    public: List[str] = field(default_factory=list)
    team: Dict[str, List[str]] = field(default_factory=dict)
    private: Dict[str, List[str]] = field(default_factory=dict)

    def extend(self, other: "Events") -> None:
        self.public.extend(other.public)
        for k, v in other.team.items():
            self.team.setdefault(k, []).extend(v)
        for k, v in other.private.items():
            self.private.setdefault(k, []).extend(v)


@dataclass
class SimpleRules:
    """Parsed rulebook config.

    example JSON shape:
    {
      "initial_phase": "night",
      "next_phase": {"night": "day", "day": "night"},
      "phases": {
        "night": {
          "acting_roles": ["Werewolf", "Seer", "Witch"],
          "speech_visibility": {"Werewolf": "team", "Seer": "private", "Witch": "private"},
          "resolvers": [ {"op": "store_target", ...}, ... ]
        },
        "day": { ... }
      },
      "end_conditions": [ ... ]
    }
    """

    initial_phase: str
    next_phase: Dict[str, str]
    phases: Dict[str, Dict[str, Any]]
    end_conditions: List[Dict[str, Any]]


@dataclass
class SimpleActions:
    """Parsed action-space config.

    example JSON shape:
    {
      "teams": {"Villager": "Villagers", "Werewolf": "Werewolves", "Seer": "Villagers",...},
      "phase_actions": {"night": {"Werewolf": ["speak","action"], "Villager": ["none"]}, "day": {"*": ["speak","action"]}},
      "initial_state": {"Witch": {"save_available": true, "poison_available": true}}
    }
    """

    role_to_team: Dict[str, str]
    phase_actions: Dict[str, Dict[str, List[str]]]
    initial_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# -----------------------------
# Environment
# -----------------------------


class SocialGameEnv(ParallelSotopiaEnv):
    """Social game environment
    - Speech routing is driven by per-phase visibility rules
    - Available actions come from phase x role mapping
    - End-of-phase effects are executed by generic, JSON-declared resolvers
    """

    def __init__(
        self,
        env_profile: EnvironmentProfile,
        *,
        rulebook_path: str,
        actions_path: str,
        role_assignments: Dict[str, str],
        **kwargs: Any,
    ) -> None:
        super().__init__(env_profile=env_profile, **kwargs)
        self._rulebook_path = Path(rulebook_path)
        self._actions_path = Path(actions_path)
        self._role_assignments = role_assignments

        self.rules: Optional[SimpleRules] = None
        self.actions: Optional[SimpleActions] = None
        self.agent_states: Dict[str, AgentState] = {}
        self.current_phase: str = ""
        self._last_events: Events = Events()
        self._winner_payload: Optional[Dict[str, str]] = None

        # Ephemeral named values for cross-resolver coordination
        self.state_flags: Dict[str, Any] = {}

    # -----------------------------
    # Config loading
    # -----------------------------
    def _load_configs(self) -> None:
        rules_raw = json.loads(self._rulebook_path.read_text())
        actions_raw = json.loads(self._actions_path.read_text())

        phases = rules_raw.get("phases") or {}
        self.rules = SimpleRules(
            initial_phase=rules_raw.get("initial_phase", ""),
            next_phase=rules_raw.get("next_phase", {}),
            phases=phases,
            end_conditions=rules_raw.get("end_conditions", []),
        )

        self.actions = SimpleActions(
            role_to_team=actions_raw.get("teams", {}),
            phase_actions=actions_raw.get("phase_actions", {}),
            initial_state=actions_raw.get("initial_state", {}),
        )

    # -----------------------------
    # Lifecycle
    # -----------------------------
    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, str] | None = None,
        agents: Agents | None = None,
        omniscient: bool = False,
        lite: bool = False,
    ) -> Dict[str, Observation]:
        """Reset the environment"""
        base_obs = super().reset(
            seed=seed, options=options, agents=agents, omniscient=omniscient, lite=lite
        )

        self._load_configs()
        assert self.rules is not None and self.actions is not None

        # Assign agents
        self.agent_states.clear()
        for name in self.agents:
            role = self._role_for_agent(name)
            team = self.actions.role_to_team.get(role, "")
            attrs = dict(self.actions.initial_state.get(role, {}))
            self.agent_states[name] = AgentState(
                name=name, role=role, team=team, attributes=attrs
            )

        self.current_phase = self.rules.initial_phase
        self._winner_payload = None
        self.state_flags = {}

        self._apply_action_mask()
        self._last_events = Events(
            public=[f"[God] Phase: {self.current_phase.title()}"]
        )
        return self._augment_observations(base_obs, append_to_existing=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _role_for_agent(self, agent_name: str) -> str:
        return self._role_assignments.get(agent_name, agent_name)

    def _phase_conf(self, phase: str) -> Dict[str, Any]:
        assert self.rules is not None
        return self.rules.phases.get(phase, {})

    def _speech_visibility_for(self, phase: str, role: str) -> str:
        conf = self._phase_conf(phase)
        vis = conf.get("speech_visibility", {})
        return vis.get(role, vis.get("*", "public"))

    def _phase_actions_for_role(self, phase: str, role: str) -> List[str]:
        assert self.actions is not None
        layer = self.actions.phase_actions.get(phase, {})
        base = layer.get(role, layer.get("*", ["none"]))
        if "none" not in base:
            return list(base) + ["none"]
        return list(base)

    def _eligible_actors(self, phase: str) -> List[str]:
        conf = self._phase_conf(phase)
        acting_roles = set(conf.get("acting_roles", []))
        acting_teams = set(conf.get("acting_teams", []))
        alive = [n for n, s in self.agent_states.items() if s.alive]
        eligible = []
        for n in alive:
            st = self.agent_states[n]
            if acting_roles and st.role not in acting_roles:
                continue
            if acting_teams and st.team not in acting_teams:
                continue
            # require that role has any action other than none
            if self._phase_actions_for_role(phase, st.role) != ["none"]:
                eligible.append(n)
        return eligible

    def active_agents_for_phase(self) -> List[str]:
        return self._eligible_actors(self.current_phase)

    def available_actions(self, agent_name: str) -> List[str]:
        if not self.agent_states[agent_name].alive:
            return ["none"]
        role = self.agent_states[agent_name].role
        return self._phase_actions_for_role(self.current_phase, role)

    def _apply_action_mask(self) -> None:
        acting = set(self.active_agents_for_phase())
        self.action_mask = [
            a in acting and self.agent_states[a].alive for a in self.agents
        ]

    def _record_speech(self, actor: str, action: AgentAction) -> Events:
        events = Events()
        if action.action_type != "speak":
            return events
        text = action.argument.strip()
        if not text:
            return events
        line = f'{actor} said: "{text}"'
        vis = self._speech_visibility_for(
            self.current_phase, self.agent_states[actor].role
        )
        if vis == "team":
            team = self.agent_states[actor].team
            events.team.setdefault(team, []).append(line)
        elif vis == "private":
            events.private.setdefault(actor, []).append(line)
        elif vis == "hidden":
            return events
        else:
            events.public.append(line)
        return events

    def _extract_target_from_text(self, text: str) -> Optional[str]:
        corpus = text.lower()
        for name in self.agent_states:
            if name.lower() in corpus:
                return name
        for name in self.agent_states:
            first = name.split()[0].lower()
            if first in corpus:
                return name
        return None

    def _first_action_target(
        self, actors: Sequence[str], actions: Dict[str, AgentAction]
    ) -> Optional[str]:
        for n in actors:
            act = actions.get(n)
            if act and act.action_type == "action":
                t = self._extract_target_from_text(act.argument)
                if t:
                    return t
        return None

    # -----------------------------
    # Core loop
    # -----------------------------
    async def astep(
        self, actions: Dict[str, AgentAction] | Dict[str, Dict[str, int | str]]
    ) -> tuple[
        Dict[str, Observation],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[Any, Any]],
    ]:
        assert self.rules is not None and self.actions is not None
        self._apply_action_mask()
        self.turn_number += 1

        # Normalize actions
        prepared: Dict[str, AgentAction] = {}
        for agent, raw in actions.items():
            if isinstance(raw, AgentAction):
                prepared[agent] = raw
            else:
                idx = int(raw.get("action_type", 0))
                action_type = self.available_action_types[idx]
                prepared[agent] = AgentAction(
                    action_type=action_type, argument=str(raw.get("argument", ""))
                )

        self.recv_message(
            "Environment", SimpleMessage(message=f"Turn #{self.turn_number}")
        )
        for agent, action in prepared.items():
            self.recv_message(agent, action)

        acting = set(self.active_agents_for_phase())
        events = Events()

        # Speech first
        for actor in acting:
            events.extend(
                self._record_speech(
                    actor,
                    prepared.get(actor, AgentAction(action_type="none", argument="")),
                )
            )

        # Phase resolvers
        resolvers = self._phase_conf(self.current_phase).get("resolvers", [])
        for spec in resolvers:
            op = spec.get("op")
            if not op:
                continue
            handler = getattr(self, f"_op_{op}", None)
            if handler is None:
                continue
            events.extend(handler(spec, acting, prepared))

        winner = self._check_end_conditions()
        if winner:
            self._winner_payload = winner

        # Phase advance
        if not winner:
            self.current_phase = self.rules.next_phase.get(
                self.current_phase, self.current_phase
            )
            events.public.append(f"[God] Phase: {self.current_phase.title()}")

        self._last_events = events
        self._apply_action_mask()
        baseline = self._create_blank_observations()
        observations = self._augment_observations(baseline, append_to_existing=False)
        rewards = {a: 0.0 for a in self.agents}
        terminated = {a: bool(winner) for a in self.agents}
        truncations = {a: False for a in self.agents}
        info = {
            a: {"comments": winner["message"] if winner else "", "complete_rating": 0}
            for a in self.agents
        }
        return observations, rewards, terminated, truncations, info

    def step(
        self, actions: Dict[str, AgentAction] | Dict[str, Dict[str, int | str]]
    ) -> tuple[
        Dict[str, Observation],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[Any, Any]],
    ]:
        return asyncio.run(self.astep(actions))

    # -----------------------------
    # Generic resolvers
    # -----------------------------
    def _op_store_target(
        self, spec: Dict[str, Any], acting: set[str], actions: Dict[str, AgentAction]
    ) -> Events:
        """Aggregate acting agents' targets and store in a flag.

        spec:
          - flag (str): flag name to set
          - restrict_team (str, optional): only count actions from this team
          - message (str, optional): format with {target}
          - announce_to_team (str, optional): team name to receive announcement
        """
        events = Events()
        flag = spec.get("flag")
        if not flag:
            return events
        restrict_team = spec.get("restrict_team")

        tally: Dict[str, int] = {}
        for name in acting:
            st = self.agent_states[name]
            if restrict_team and st.team != restrict_team:
                continue
            act = actions.get(name)
            if not act or act.action_type != "action":
                continue
            t = self._extract_target_from_text(act.argument)
            if t:
                tally[t] = tally.get(t, 0) + 1

        target: Optional[str] = None
        if tally:
            target = max(tally.items(), key=lambda kv: kv[1])[0]
            self.state_flags[flag] = target
            msg_tmpl = spec.get("message")
            if msg_tmpl:
                line = msg_tmpl.format(target=target)
                team_name = spec.get("announce_to_team")
                if team_name:
                    events.team.setdefault(team_name, []).append(line)
                else:
                    events.public.append(line)
        return events

    def _op_reveal_attribute(
        self, spec: Dict[str, Any], acting: set[str], actions: Dict[str, AgentAction]
    ) -> Events:
        """Reveal a target's attribute value to each acting agent privately.

        spec:
          - attribute (str): attribute name, e.g., "team"
          - message (str, optional): template with {target} and {value}
        """
        events = Events()
        attr = spec.get("attribute")
        if not attr:
            return events
        msg_tmpl = spec.get("message", "[God] {target}: {value}")
        for name in acting:
            act = actions.get(name)
            if not act or act.action_type != "action":
                continue
            target = self._extract_target_from_text(act.argument)
            if not target:
                continue
            value = (
                self.agent_states[target].team
                if attr == "team"
                else self.agent_states[target].attributes.get(attr, "")
            )
            events.private.setdefault(name, []).append(
                msg_tmpl.format(target=target, value=value)
            )
        return events

    def _op_keyword_target_flags(
        self, spec: Dict[str, Any], acting: set[str], actions: Dict[str, AgentAction]
    ) -> Events:
        """Set one or more flags based on keywords found in actors' action text.

        spec:
          - mapping: [{"keyword":"save","flag":"witch_saved","require_attr":"save_available","set_attr_false":true}, ...]
        """
        events = Events()
        mapping = spec.get("mapping", [])
        if not mapping:
            return events
        for name in acting:
            st = self.agent_states[name]
            act = actions.get(name)
            if not act or act.action_type != "action":
                continue
            text = act.argument.lower()
            target = self._extract_target_from_text(act.argument)
            for rule in mapping:
                kw = rule.get("keyword", "").lower()
                flag = rule.get("flag")
                require_attr = rule.get("require_attr")
                set_attr_false = bool(rule.get("set_attr_false", False))
                if not kw or not flag:
                    continue
                if require_attr and not st.attributes.get(require_attr, True):
                    continue
                if kw in text and target:
                    self.state_flags[flag] = target
                    if set_attr_false and require_attr:
                        st.attributes[require_attr] = False
        return events

    def _op_kill_flags(
        self, spec: Dict[str, Any], acting: set[str], actions: Dict[str, AgentAction]
    ) -> Events:
        """Kill agents listed by flags, excluding those present in exclude flags.

        spec:
          - flags: ["night_target", "witch_poisoned"]
          - exclude: ["witch_saved"]
          - message_dead: template with {victim}
          - message_peace: message if no one dies
        """
        events = Events()
        flags = spec.get("flags", [])
        exclude = spec.get("exclude", [])
        msg_dead = spec.get("message_dead", "[God] {victim} died.")
        msg_peace = spec.get("message_peace", "[God] No one died.")

        victims: List[str] = []
        for f in flags:
            val = self.state_flags.get(f)
            if isinstance(val, str) and val:
                victims.append(val)
        excluded: List[str] = []
        for f in exclude:
            val = self.state_flags.get(f)
            if isinstance(val, str) and val:
                excluded.append(val)
        final = [v for v in victims if v not in excluded]
        if not final:
            events.public.append(msg_peace)
            return events
        for victim in final:
            if victim in self.agent_states and self.agent_states[victim].alive:
                self.agent_states[victim].alive = False
                events.public.append(msg_dead.format(victim=victim))
        return events

    def _op_vote_majority_execute(
        self, spec: Dict[str, Any], acting: set[str], actions: Dict[str, AgentAction]
    ) -> Events:
        """Tally votes and execute a unique winner.

        spec:
          - tie_policy: "no_execute" (default) | "random" (not implemented)
          - message_execute, message_tie, message_none
        """
        events = Events()
        msg_exec = spec.get(
            "message_execute",
            "[God] {target} was executed. They belonged to team {team}.",
        )
        msg_tie = spec.get("message_tie", "[God] The vote is tied. No execution today.")
        msg_none = spec.get("message_none", "[God] No valid votes were cast.")

        tally: Dict[str, int] = {}
        for name in acting:
            act = actions.get(name)
            if not act or act.action_type != "action":
                continue
            t = self._extract_target_from_text(act.argument)
            if t:
                tally[t] = tally.get(t, 0) + 1

        if not tally:
            events.public.append(msg_none)
            return events

        winner, votes = max(tally.items(), key=lambda kv: kv[1])
        if list(tally.values()).count(votes) > 1:
            events.public.append(msg_tie)
            return events

        if winner in self.agent_states and self.agent_states[winner].alive:
            self.agent_states[winner].alive = False
            team = self.agent_states[winner].team
            events.public.append(msg_exec.format(target=winner, team=team))
        return events

    def _op_clear_flags(
        self, spec: Dict[str, Any], acting: set[str], actions: Dict[str, AgentAction]
    ) -> Events:
        events = Events()
        for f in spec.get("flags", []):
            self.state_flags.pop(f, None)
        return events

    # -----------------------------
    # Observations
    # -----------------------------
    def _create_blank_observations(self) -> Dict[str, Observation]:
        blank: Dict[str, Observation] = {}
        acting = set(self.active_agents_for_phase())
        for name in self.agents:
            available = self.available_actions(name) if name in acting else ["none"]
            blank[name] = Observation(
                last_turn="", turn_number=self.turn_number, available_actions=available
            )
        return blank

    def _augment_observations(
        self,
        baseline: Dict[str, Observation],
        *,
        append_to_existing: bool,
    ) -> Dict[str, Observation]:
        acting = set(self.active_agents_for_phase())
        events = self._last_events
        new_obs: Dict[str, Observation] = {}
        for idx, name in enumerate(self.agents):
            current = baseline[name]
            available = self.available_actions(name) if name in acting else ["none"]
            lines: List[str] = []
            if append_to_existing and current.last_turn.strip():
                lines.append(current.last_turn.strip())
            lines.append(f"[God] Phase: {self.current_phase.title()}")
            lines.append(
                "[God] It is your turn to act."
                if name in acting
                else "[God] You are observing this phase."
            )
            lines.append(f"[God] Available actions: {', '.join(available)}")

            team = self.agent_states[name].team
            msgs: List[str] = []
            msgs.extend(events.public)
            msgs.extend(events.team.get(team, []))
            msgs.extend(events.private.get(name, []))
            if not msgs:
                msgs.append("[God] Await instructions from the host.")
            lines.extend(msgs)

            combined = "\n".join(seg for seg in lines if seg)
            new_obs[name] = Observation(
                last_turn=render_text_for_agent(combined, agent_id=idx),
                turn_number=current.turn_number,
                available_actions=available,
            )
        return new_obs

    # -----------------------------
    # End conditions (generic)
    # -----------------------------
    def _check_end_conditions(self) -> Optional[Dict[str, str]]:
        assert self.rules is not None
        alive_by_team: Dict[str, int] = {}
        for s in self.agent_states.values():
            if s.alive:
                alive_by_team[s.team] = alive_by_team.get(s.team, 0) + 1
        for cond in self.rules.end_conditions:
            ctype = cond.get("type")
            if ctype == "team_eliminated":
                team = cond.get("team", "")
                if alive_by_team.get(team, 0) == 0:
                    return {
                        "winner": cond.get("winner", team),
                        "message": cond.get("message", f"[God] {team} eliminated."),
                    }
            if ctype == "parity":
                team = cond.get("team", "")
                other = cond.get("other", "")
                if alive_by_team.get(team, 0) >= alive_by_team.get(other, 0) > 0:
                    return {
                        "winner": cond.get("winner", team),
                        "message": cond.get(
                            "message",
                            f"[God] Parity reached: {team} now matches or exceeds {other}.",
                        ),
                    }
        return None
