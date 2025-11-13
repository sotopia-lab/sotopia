from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Literal

from sotopia.envs.parallel import ParallelSotopiaEnv, render_text_for_agent
from sotopia.agents.llm_agent import Agents
from sotopia.database import EnvironmentProfile
from sotopia.messages import AgentAction, Observation, Message, SimpleMessage


class GameMessage(Message):
    """A message object with explicit recipients.

    recipients=None means public; otherwise only listed agents can view.
    """

    sender: str
    content: str
    state: str
    recipients: list[str] | None = None
    kind: Literal["speak", "action", "none"] = "none"

    def to_natural_language(self) -> str:
        if self.kind == "speak":
            return f'{self.sender} said: "{self.content}"'
        elif self.kind == "action":
            return f"{self.sender} [action] {self.content}"
        elif self.kind == "none":
            return f"{self.sender} did nothing"
        else:
            raise ValueError(f"Invalid message kind: {self.kind}")


class SocialGameEnv(ParallelSotopiaEnv):
    """
    Core concepts:
    - Per-state acting roles and action space
    - Per-message visibility: public, team, private
    - Per-agent message history is derived from the game message log
    """

    def __init__(
        self,
        env_profile: EnvironmentProfile,
        *,
        config_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_profile=env_profile, **kwargs)
        self._config_path = Path(config_path)
        self._config: Dict[str, Any] = {}

        self.role_to_team: Dict[
            str, str
        ] = {}  # Map roles to teams (e.g. Seer --> Villagers))
        self.agent_to_role: Dict[
            str, str
        ] = {}  # Map agents to roles (e.g. Aurora --> Villager)
        self.agent_alive: Dict[
            str, bool
        ] = {}  # Map agents to their alive status (e.g. Aurora --> True (alive) or False (dead))

        self.current_state: str = ""  # Current state of the game (e.g. Day_discussion)
        self.message_log: List[GameMessage] = []  # Log of all messages sent in the game
        self.state_log: List[Dict[str, Any]] = []  # Log of all states in the game
        self._state_transition: Dict[
            str, str
        ] = {}  # Map of state transitions (e.g. Night_witch --> Day_discussion)
        self._state_props: Dict[
            str, Dict[str, Any]
        ] = {}  # Map of state properties (e.g. Night --> {acting_roles: ["Werewolf"], actions: ["speak", "action"], visibility: "team"})
        self.internal_state: Dict[
            str, Any
        ] = {}  # Internal state of the game (e.g. votes, night_target, witch_save, witch_poison)

    # -----------------------------
    # Config loading
    # -----------------------------
    def _load_config(self) -> None:
        # Read config and normalize to FSM structures
        if not self._config_path.exists():
            raise FileNotFoundError(f"config_path does not exist: {self._config_path}")
        self._config = json.loads(self._config_path.read_text())

        # Build role_to_team from agents if available
        self.role_to_team = {}
        for agent in self._config.get("agents", []):
            role = agent.get("role")
            team = agent.get("team")
            if isinstance(role, str) and isinstance(team, str):
                self.role_to_team.setdefault(role, team)

        # FSM structures
        self._state_transition = dict(self._config.get("state_transition", {}))
        self._state_props = dict(self._config.get("state_properties", {}))

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
        base_obs = super().reset(
            seed=seed, options=options, agents=agents, omniscient=omniscient, lite=lite
        )

        self._load_config()

        self.agent_to_role = {}
        for name in self.agents:
            role = next(
                (
                    a.get("role", "Villager")
                    for a in self._config.get("agents", [])
                    if a.get("name") == name
                ),
                "Villager",
            )
            self.agent_to_role[name] = role

        self.agent_alive = {name: True for name in self.agents}
        self.current_state = self._config.get("initial_state", "Night")
        self.message_log = []
        self.state_log = []
        self.internal_state = {}
        init_spec = self._state_spec(self.current_state)
        if isinstance(init_spec.get("internal_state"), dict):
            self.internal_state.update(init_spec["internal_state"])

        # Prepare first observation
        self._apply_action_mask()
        self._append_system_message(f"[Game] State: {self.current_state}")
        return self._build_observations(base_obs, append_to_existing=True)

    # -----------------------------
    # Core helpers
    # -----------------------------
    def _apply_action_mask(self) -> None:
        # Determine eligible actors by role and alive status
        eligible = [
            n
            for n in self.agents
            if self.agent_alive.get(n, True)
            and (
                not set(self._state_spec(self.current_state).get("acting_roles", []))
                or self.agent_to_role.get(n, "")
                in set(self._state_spec(self.current_state).get("acting_roles", []))
            )
        ]

        # Order policy from config: "round-robin" (default) or "simultaneous"
        order = str(self._state_spec(self.current_state).get("order", "round-robin"))
        if order == "round-robin":
            mask = [False for _ in self.agents]
            if eligible:
                idx = self.turn_number % len(eligible)
                current_actor = eligible[idx]
                for i, name in enumerate(self.agents):
                    mask[i] = name == current_actor
            self.action_mask = mask
            return

        # Default: simultaneous for all eligible actors
        eligible_set = set(eligible)
        self.action_mask = [name in eligible_set for name in self.agents]

    def _state_spec(self, state: str) -> Dict[str, Any]:
        return dict(self._state_props.get(state, {}))

    def _allowed_actions_for_role(self, state: str, role: str) -> List[str]:
        spec = self._state_spec(state)
        allowed = list(spec.get("actions", ["none"]))
        if "none" not in allowed:
            allowed.append("none")
        acting_roles = spec.get("acting_roles")
        if acting_roles and role not in set(acting_roles):
            return ["none"]
        return allowed

    def available_actions(self, agent_name: str) -> List[str]:
        if not self.agent_alive.get(agent_name, True):
            return ["none"]
        role = self.agent_to_role.get(agent_name, "Villager")
        return self._allowed_actions_for_role(self.current_state, role)

    def active_agents_for_state(self) -> List[str]:
        acting_roles = set(self._state_spec(self.current_state).get("acting_roles", []))
        return [
            n
            for n in self.agents
            if self.agent_alive.get(n, True)
            and (not acting_roles or self.agent_to_role.get(n, "") in acting_roles)
        ]

    def _append_system_message(self, text: str) -> None:
        self.message_log.append(
            GameMessage(
                sender="Environment",
                content=text,
                state=self.current_state,
                recipients=None,
            )
        )

    def _can_view(self, agent_name: str, m: GameMessage) -> bool:
        return m.recipients is None or agent_name in (m.recipients or [])

    def _visible_text(self, agent_name: str) -> str:
        return "\n".join(
            m.to_natural_language()
            for m in self.message_log
            if self._can_view(agent_name, m)
        )

    def _build_observations(
        self, baseline: Dict[str, Observation], *, append_to_existing: bool
    ) -> Dict[str, Observation]:
        acting = set(self.active_agents_for_state())
        new_obs: Dict[str, Observation] = {}
        for idx, name in enumerate(self.agents):
            current = baseline[name]
            available = self.available_actions(name) if name in acting else ["none"]

            lines: List[str] = []
            if append_to_existing and current.last_turn.strip():
                lines.append(current.last_turn.strip())
            lines.append(f"[Game] State: {self.current_state}")
            lines.append(
                "[Game] It is your turn to act."
                if name in acting
                else "[Game] You are observing this state."
            )
            lines.append(f"[Game] Available actions: {', '.join(available)}")
            visible = self._visible_text(name)
            if visible:
                lines.append(visible)
            else:
                lines.append("[Game] Await instructions from the host.")

            combined = "\n".join(seg for seg in lines if seg)
            new_obs[name] = Observation(
                last_turn=render_text_for_agent(combined, agent_id=idx),
                turn_number=current.turn_number,
                available_actions=available,
            )
        return new_obs

    # -----------------------------
    # Step
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

        acting = set(self.active_agents_for_state())
        recorded_msgs: List[GameMessage] = []

        # Record messages based on visibility rules
        for actor in acting:
            act = prepared.get(actor)
            if not act or act.action_type == "none":
                continue
            if act.action_type == "speak":
                gm = GameMessage(
                    sender=actor,
                    content=act.argument.strip(),
                    state=self.current_state,
                    recipients=None,  # default public; make explicit via config if needed
                    kind="speak",
                )
                if gm.content:
                    self.message_log.append(gm)
                    recorded_msgs.append(gm)
            elif act.action_type == "action":
                gm = GameMessage(
                    sender=actor,
                    content=act.argument.strip(),
                    state=self.current_state,
                    recipients=None,  # default public; make explicit via config if needed
                    kind="action",
                )
                if gm.content:
                    self.message_log.append(gm)
                    recorded_msgs.append(gm)

        # State advancement
        self.current_state = self._state_transition.get(
            self.current_state, self.current_state
        )
        state_internal = self._state_spec(self.current_state).get("internal_state")
        if isinstance(state_internal, dict):
            self.internal_state.update(state_internal)
        self._append_system_message(f"[Game] State: {self.current_state}")

        # Append to state_log for external summarization if needed
        self.state_log.append(
            {
                "state": self.current_state,
                "public": [
                    m.to_natural_language()
                    for m in self.message_log
                    if m.recipients is None and m.state == self.current_state
                ],
            }
        )

        self._apply_action_mask()
        baseline = {
            name: Observation(
                last_turn="", turn_number=self.turn_number, available_actions=["none"]
            )
            for name in self.agents
        }
        observations = self._build_observations(baseline, append_to_existing=False)
        rewards = {a: 0.0 for a in self.agents}
        terminated = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        info = {a: {"comments": "", "complete_rating": 0} for a in self.agents}
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
