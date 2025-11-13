"""Minimal social game environment for phase-based multi-agent games like Werewolf."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sotopia.agents.llm_agent import Agents
from sotopia.database import EnvironmentProfile
from sotopia.envs.parallel import ParallelSotopiaEnv, render_text_for_agent
from sotopia.messages import AgentAction, Observation, SimpleMessage

__all__ = [
    "GameRules",
    "SocialGame",
    "SocialGameEnv",
    "RoleConfig",
    "PhaseConfig",
    "GameState",
]


# ============================================================================
# Configuration Models
# ============================================================================


class RoleConfig(BaseModel):
    """Role definition: team, actions per phase."""

    team: str
    actions: dict[str, list[str]]  # phase -> allowed actions
    goal: str = ""


class PhaseConfig(BaseModel):
    """Phase definition: who acts, visibility, resolution."""

    actors: list[str] | None = None  # roles that act (None = all alive)
    visibility: str = "public"  # public|team|private
    resolution: str = "none"  # none|kill|vote|inspect|witch|announce_deaths
    next_phase: str = ""


class GameRules(BaseModel):
    """Complete game configuration."""

    roles: dict[str, RoleConfig]
    phases: dict[str, PhaseConfig]
    start_phase: str
    win_conditions: list[dict[str, str]]  # [{type: "eliminate", team: "Werewolves"}]


# ============================================================================
# Game State
# ============================================================================


@dataclass
class GameState:
    """Runtime game state."""

    agents: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # name -> {role, team, alive, ...}
    phase: str = ""
    state: dict[str, Any] = field(
        default_factory=dict
    )  # shared state (targets, votes, etc)
    messages: list[str] = field(default_factory=list)  # pending public messages
    team_messages: dict[str, list[str]] = field(
        default_factory=dict
    )  # team -> messages
    private_messages: dict[str, list[str]] = field(
        default_factory=dict
    )  # agent -> messages

    def alive(self) -> list[str]:
        return [name for name, info in self.agents.items() if info["alive"]]

    def by_team(self, team: str) -> list[str]:
        return [
            name
            for name, info in self.agents.items()
            if info["team"] == team and info["alive"]
        ]

    def add_msg(
        self, msg: str, visibility: str = "public", target: str | None = None
    ) -> None:
        """Add message with visibility control."""
        if visibility == "public":
            self.messages.append(msg)
        elif visibility == "team" and target:
            self.team_messages.setdefault(target, []).append(msg)
        elif visibility == "private" and target:
            self.private_messages.setdefault(target, []).append(msg)

    def flush_messages(
        self,
    ) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
        """Return and clear all messages."""
        pub, team, priv = (
            self.messages[:],
            dict(self.team_messages),
            dict(self.private_messages),
        )
        self.messages.clear()
        self.team_messages.clear()
        self.private_messages.clear()
        return pub, team, priv


# ============================================================================
# Game Engine
# ============================================================================


class SocialGame:
    """Phase-based social game engine."""

    def __init__(self, rules: GameRules):
        self.rules = rules
        self.state = GameState()

    def init(self, agents: list[str], roles: dict[str, str]) -> None:
        """Initialize game with agent-role assignments."""
        self.state.agents = {
            name: {
                "role": roles[name],
                "team": self.rules.roles[roles[name]].team,
                "alive": True,
                "attrs": {},  # role-specific state
            }
            for name in agents
        }
        self.state.phase = self.rules.start_phase
        self.state.state = {
            "votes": {},
            "night_target": None,
            "witch_save": False,
            "witch_poison": None,
        }
        # Witch tracking
        for name, info in self.state.agents.items():
            if info["role"] == "Witch":
                info["attrs"]["save_used"] = False
                info["attrs"]["poison_used"] = False

    def active_agents(self) -> list[str]:
        """Who can act in current phase?"""
        phase_cfg = self.rules.phases[self.state.phase]
        if not phase_cfg.actors:
            return self.state.alive()
        return [
            n
            for n in self.state.alive()
            if self.state.agents[n]["role"] in phase_cfg.actors
        ]

    def available_actions(self, agent: str) -> list[str]:
        """What actions can agent take?"""
        if not self.state.agents[agent]["alive"]:
            return ["none"]
        role = self.state.agents[agent]["role"]
        phase = self.state.phase
        return self.rules.roles[role].actions.get(phase, ["none"])

    def process_turn(
        self, actions: dict[str, AgentAction]
    ) -> tuple[bool, dict[str, str] | None]:
        """Process actions, run resolution, return (phase_done, winner)."""
        phase_cfg = self.rules.phases[self.state.phase]

        # Record speech
        for agent, action in actions.items():
            if (
                action.action_type in ["speak", "non-verbal communication"]
                and action.argument.strip()
            ):
                msg = f"{agent}: {action.argument}"
                vis = phase_cfg.visibility
                target = (
                    self.state.agents[agent]["team"]
                    if vis == "team"
                    else agent
                    if vis == "private"
                    else None
                )
                self.state.add_msg(msg, vis, target)

        # Resolve phase
        self._resolve(phase_cfg.resolution, actions)

        # Check win
        winner = self._check_win()

        # Advance phase
        self.state.phase = phase_cfg.next_phase

        return True, winner

    def _resolve(self, resolution: str, actions: dict[str, AgentAction]) -> None:
        """Execute phase resolution logic."""
        if resolution == "none":
            return

        if resolution == "kill":
            # Werewolves pick target
            target = self._extract_name(actions)
            if target:
                self.state.state["night_target"] = target
                team = self.state.agents[list(actions.keys())[0]]["team"]
                self.state.add_msg(f"Target: {target}", "team", team)

        elif resolution == "inspect":
            # Seer inspects
            target = self._extract_name(actions)
            if target and actions:
                agent = list(actions.keys())[0]
                team = self.state.agents[target]["team"]
                self.state.add_msg(f"{target} is on team {team}", "private", agent)

        elif resolution == "witch":
            # Witch save/poison
            if not actions:
                return
            agent = list(actions.keys())[0]
            arg = list(actions.values())[0].argument.lower()

            if "save" in arg and not self.state.agents[agent]["attrs"]["save_used"]:
                target = self.state.state.get("night_target")
                if target:
                    self.state.state["witch_save"] = True
                    self.state.agents[agent]["attrs"]["save_used"] = True
                    self.state.add_msg(f"You saved {target}", "private", agent)

            if "poison" in arg and not self.state.agents[agent]["attrs"]["poison_used"]:
                target = self._extract_name(actions)
                if target:
                    self.state.state["witch_poison"] = target
                    self.state.agents[agent]["attrs"]["poison_used"] = True
                    self.state.add_msg(f"You poisoned {target}", "private", agent)

        elif resolution == "announce_deaths":
            # Resolve night kills
            killed = []
            target = self.state.state.get("night_target")
            if target and not self.state.state.get("witch_save"):
                killed.append(target)
            poison = self.state.state.get("witch_poison")
            if poison and poison not in killed:
                killed.append(poison)

            if killed:
                for victim in killed:
                    self.state.agents[victim]["alive"] = False
                    self.state.add_msg(f"{victim} died")
            else:
                self.state.add_msg("No one died")

            # Reset night state
            self.state.state.update(
                {"night_target": None, "witch_save": False, "witch_poison": None}
            )

        elif resolution == "vote":
            # Tally votes
            votes: dict[str, int] = {}
            for action in actions.values():
                target = self._extract_name({0: action})
                if target:
                    votes[target] = votes.get(target, 0) + 1

            if votes:
                winner = max(votes, key=votes.get)  # type: ignore
                max_votes = votes[winner]
                # Check tie
                if list(votes.values()).count(max_votes) == 1:
                    self.state.agents[winner]["alive"] = False
                    team = self.state.agents[winner]["team"]
                    self.state.add_msg(f"Voted out: {winner} (team {team})")
                else:
                    self.state.add_msg("Vote tied, no execution")
            else:
                self.state.add_msg("No valid votes")

    def _extract_name(self, actions: dict[Any, AgentAction]) -> str | None:
        """Extract target name from action arguments."""
        for action in actions.values():
            text = f"{action.action_type} {action.argument}".lower()
            for name in self.state.agents:
                if name.lower() in text or name.split()[0].lower() in text:
                    return name
        return None

    def _check_win(self) -> dict[str, str] | None:
        """Check win conditions."""
        for cond in self.rules.win_conditions:
            if cond["type"] == "eliminate":
                team = cond["team"]
                if not self.state.by_team(team):
                    return {
                        "winner": cond.get("winner", "Other"),
                        "message": f"Team {team} eliminated",
                    }
            elif cond["type"] == "parity":
                team1 = self.state.by_team(cond["team"])
                team2 = self.state.by_team(cond["other"])
                if len(team1) >= len(team2):
                    return {
                        "winner": cond["team"],
                        "message": f"{cond['team']} reached parity",
                    }
        return None


# ============================================================================
# Environment Wrapper
# ============================================================================


class SocialGameEnv(ParallelSotopiaEnv):
    """Sotopia environment for social games."""

    def __init__(
        self,
        env_profile: EnvironmentProfile,
        *,
        rules_path: str,
        role_assignments: dict[str, str],
        **kwargs: Any,
    ):
        super().__init__(env_profile=env_profile, **kwargs)
        self.rules_path = Path(rules_path)
        self.role_assignments = role_assignments
        self.game: SocialGame | None = None

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
        agents: Agents | None = None,
        omniscient: bool = False,
        lite: bool = False,
    ) -> dict[str, Observation]:
        obs = super().reset(seed, options, agents, omniscient, lite)

        # Load rules and init game
        rules = GameRules.model_validate_json(self.rules_path.read_text())
        self.game = SocialGame(rules)
        self.game.init(self.agents, self.role_assignments)

        return self._build_observations(obs)

    def _build_observations(
        self, base_obs: dict[str, Observation]
    ) -> dict[str, Observation]:
        """Build observations with game state."""
        assert self.game is not None

        pub, team, priv = self.game.state.flush_messages()
        active = set(self.game.active_agents())

        new_obs = {}
        for idx, agent in enumerate(self.agents):
            actions = self.game.available_actions(agent)

            # Collect visible messages
            msgs = pub[:]
            agent_team = self.game.state.agents[agent]["team"]
            msgs.extend(team.get(agent_team, []))
            msgs.extend(priv.get(agent, []))

            # Build prompt
            role = self.game.state.agents[agent]["role"]
            phase = self.game.state.phase
            lines = [
                f"Phase: {phase}",
                f"Role: {role}",
                f"Actions: {', '.join(actions)}",
                f"Active: {'Yes' if agent in active else 'No'}",
                "",
                *msgs,
            ]

            new_obs[agent] = Observation(
                last_turn=render_text_for_agent("\n".join(lines), idx),
                turn_number=self.turn_number,
                available_actions=actions,
            )

        return new_obs

    async def astep(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        assert self.game is not None

        self.turn_number += 1

        # Convert actions
        converted = {}
        for agent, action in actions.items():
            if isinstance(action, AgentAction):
                converted[agent] = action
            else:
                act_type = self.available_action_types[
                    int(action.get("action_type", 0))
                ]
                converted[agent] = AgentAction(
                    action_type=act_type, argument=str(action.get("argument", ""))
                )

        # Log
        self.recv_message(
            "Environment", SimpleMessage(message=f"Turn {self.turn_number}")
        )
        for agent, action in converted.items():
            self.recv_message(agent, action)

        # Process
        _, winner = self.game.process_turn(converted)

        # Build observations
        base_obs = {
            agent: Observation(
                last_turn="", turn_number=self.turn_number, available_actions=["none"]
            )
            for agent in self.agents
        }
        obs = self._build_observations(base_obs)

        # Results
        rewards = {a: 0.0 for a in self.agents}
        terminated = {a: bool(winner) for a in self.agents}
        truncated = {a: False for a in self.agents}
        info = {
            a: {"comments": winner["message"] if winner else "", "complete_rating": 0}
            for a in self.agents
        }

        return obs, rewards, terminated, truncated, info

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
