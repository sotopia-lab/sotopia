"""Social game environment for multi-state games like Werewolves, Mafia, etc."""

from __future__ import annotations

import asyncio
import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, cast

from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.envs.evaluators import unweighted_aggregate_evaluate
from sotopia.agents.llm_agent import Agents
from sotopia.database import EnvironmentProfile
from sotopia.messages import AgentAction, Observation, SimpleMessage


class SocialGameEnv(ParallelSotopiaEnv):
    """Environment for social deduction games with states, roles, and private information.

    Adds to ParallelSotopiaEnv:
    - FSM states (Night, Day, etc.)
    - Role/team system
    - Alive/dead status
    - Private information visibility
    - State transitions
    """

    def __init__(
        self,
        env_profile: EnvironmentProfile,
        *,
        config_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_profile=env_profile, **kwargs)

        # Load game configuration
        self._config_path = Path(config_path)
        self._config: Dict[str, Any] = {}

        # Game state
        self.current_state: str = ""
        self.agent_to_role: Dict[str, str] = {}  # Aurora -> Villager
        self.role_to_team: Dict[str, str] = {}  # Villager -> Villagers
        self.agent_alive: Dict[str, bool] = {}  # Aurora -> True
        self.internal_state: Dict[str, Any] = {}  # votes, targets, etc.

    def _load_config(self) -> None:
        """Load game configuration from JSON file."""
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config not found: {self._config_path}")

        self._config = json.loads(self._config_path.read_text())

        # Build role -> team mapping
        self.role_to_team = {}
        for agent_entry in self._config.get("agents", []):
            role = agent_entry.get("role")
            team = agent_entry.get("team")
            if role and team:
                self.role_to_team.setdefault(role, team)

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, str] | None = None,
        agents: Agents | None = None,
        omniscient: bool = False,
        lite: bool = False,
    ) -> Dict[str, Observation]:
        """Reset environment and initialize game state."""
        # Call parent reset
        base_obs = super().reset(
            seed=seed, options=options, agents=agents, omniscient=omniscient, lite=lite
        )

        # Load config
        self._load_config()

        # Map agent names to roles from config
        self.agent_to_role = {}
        for name in self.agents:
            role = next(
                (
                    a.get("role", "Villager")
                    for a in self._config.get("agents", [])
                    if a.get("name") == name.strip()
                ),
                "Villager",
            )
            self.agent_to_role[name] = role

        # Initialize alive status and state
        self.agent_alive = {name: True for name in self.agents}
        self.current_state = self._config.get("initial_state", "Day_discussion")
        self.internal_state = {}

        # Send initial system message
        self.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] State: {self.current_state}. The game begins!"
            ),
        )

        # Initialize round-robin counter
        self._round_robin_idx = 0

        # Initialize action mask for first turn based on state
        self._update_action_mask()

        # Update available actions based on game state
        for agent_name in self.agents:
            base_obs[agent_name].available_actions = self._get_available_actions(
                agent_name
            )

        return base_obs

    async def astep(
        self, actions: Dict[str, AgentAction] | Dict[str, Dict[str, int | str]]
    ) -> tuple[
        Dict[str, Observation],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[Any, Any]],
    ]:
        """Process one step: record actions, update state, build observations."""
        self.turn_number += 1

        # 1. Normalize actions to AgentAction objects
        normalized_actions: Dict[str, AgentAction] = {}
        for agent_name, action in actions.items():
            if isinstance(action, AgentAction):
                normalized_actions[agent_name] = action
            else:
                # Convert dict to AgentAction
                action_type = self.available_action_types[
                    int(action.get("action_type", 0))
                ]
                normalized_actions[agent_name] = AgentAction(
                    action_type=action_type,
                    argument=str(action.get("argument", "")),
                )

        # 2. Record actions to message history
        self.recv_message(
            "Environment", SimpleMessage(message=f"Turn #{self.turn_number}")
        )
        for idx, (agent_name, action) in enumerate(normalized_actions.items()):
            # Only record actions from agents who were allowed to act
            if self.agent_alive.get(agent_name, False) and self.action_mask[idx]:
                self.recv_message(agent_name, action)

        # 3. Run evaluators to check if game should terminate (e.g., max turns)
        evaluator_response = unweighted_aggregate_evaluate(
            list(
                itertools.chain(
                    *await asyncio.gather(
                        *[
                            evaluator.__acall__(
                                turn_number=self.turn_number,
                                messages=self.inbox,
                                env=self,
                            )
                            for evaluator in self.evaluators
                        ]
                    )
                )
            )
        )

        # 4. Process game-specific logic
        self._process_actions(normalized_actions)

        # 5. Check for eliminations
        self._check_eliminations()

        # 6. Check if state should transition
        should_transition = self._should_transition_state()
        print(
            f"DEBUG Turn {self.turn_number}: state={self.current_state}, should_transition={should_transition}, state_turn_count={getattr(self, '_state_turn_count', {})}"
        )
        if should_transition:
            self._transition_state()
            print(f"DEBUG: Transitioned to {self.current_state}")

        # 7. Update action mask for next turn based on state
        state_props = self._config.get("state_properties", {}).get(
            self.current_state, {}
        )
        action_order = state_props.get("action_order", self.action_order)
        print(
            f"DEBUG: About to update mask - state={self.current_state}, action_order={action_order}"
        )
        self._update_action_mask()
        print(f"DEBUG: After update_action_mask - mask={self.action_mask}")

        # 8. Build observations with visibility filtering
        observations = self._build_observations()

        # 9. Set termination from evaluators (including game-specific win conditions)
        terminated = {agent: evaluator_response.terminated for agent in self.agents}

        # 10. If terminated and terminal_evaluators exist, run them
        if evaluator_response.terminated and self.terminal_evaluators:
            terminal_response = unweighted_aggregate_evaluate(
                list(
                    itertools.chain(
                        *await asyncio.gather(
                            *[
                                evaluator.__acall__(
                                    turn_number=self.turn_number,
                                    messages=self.inbox,
                                    env=self,
                                )
                                for evaluator in self.terminal_evaluators
                            ]
                        )
                    )
                )
            )
            # Merge terminal evaluator response
            if evaluator_response.comments and terminal_response.comments:
                evaluator_response.comments += terminal_response.comments
            elif terminal_response.comments:
                evaluator_response.comments = terminal_response.comments

        rewards = {agent: 0.0 for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        info = {
            agent: {"comments": evaluator_response.comments or "", "complete_rating": 0}
            for agent in self.agents
        }

        return observations, rewards, terminated, truncations, info

    def _process_actions(self, actions: Dict[str, AgentAction]) -> None:
        """Process actions based on current state (votes, kills, etc.)."""
        state_props = self._config.get("state_properties", {}).get(
            self.current_state, {}
        )

        # Example: collect votes in voting state
        if "vote" in state_props.get("actions", []):
            for agent_name, action in actions.items():
                if action.action_type == "action" and "vote" in action.argument.lower():
                    # Parse vote target from argument
                    # Store in internal_state
                    pass

    def _check_eliminations(self) -> None:
        """Check if anyone should be eliminated (voted out, killed, etc.)."""
        # Example: tally votes and eliminate player with most votes
        pass

    def _update_action_mask(self) -> None:
        """Update action mask for next turn based on state configuration."""
        # Get action_order for this state from config, or use environment default
        state_props = self._config.get("state_properties", {}).get(
            self.current_state, {}
        )
        action_order = state_props.get("action_order", self.action_order)
        acting_roles = state_props.get("acting_roles", [])

        # Determine which agents are eligible to act in this state
        if acting_roles:
            # Only agents with specific roles can act
            eligible_indices = [
                idx
                for idx, agent_name in enumerate(self.agents)
                if self.agent_alive.get(agent_name, False)
                and self.agent_to_role.get(agent_name, "") in acting_roles
            ]
        else:
            # All alive agents can act
            eligible_indices = [
                idx
                for idx, agent_name in enumerate(self.agents)
                if self.agent_alive.get(agent_name, False)
            ]

        # Update action mask based on action order
        self.action_mask = [False for _ in self.agents]

        if not eligible_indices:
            # No eligible agents - keep all masks False
            return

        if action_order == "round-robin":
            # Cycle through eligible agents only
            if not hasattr(self, "_round_robin_idx"):
                self._round_robin_idx = 0
            # Get next eligible agent
            acting_idx = eligible_indices[self._round_robin_idx % len(eligible_indices)]
            self.action_mask[acting_idx] = True
            self._round_robin_idx += 1
        elif action_order == "random":
            # Pick random eligible agent
            acting_idx = random.choice(eligible_indices)
            self.action_mask[acting_idx] = True
        else:
            # Simultaneous: all eligible agents can act
            for idx in eligible_indices:
                self.action_mask[idx] = True

    def _should_transition_state(self) -> bool:
        """Check if we should move to next state based on how many agents have acted."""
        state_props = self._config.get("state_properties", {}).get(
            self.current_state, {}
        )
        acting_roles = state_props.get("acting_roles", [])
        action_order = state_props.get("action_order", self.action_order)

        # Initialize turn counter for this state if needed
        if not hasattr(self, "_state_turn_count"):
            self._state_turn_count: Dict[str, int] = {}
        if self.current_state not in self._state_turn_count:
            self._state_turn_count[self.current_state] = 0

        # Increment turn count for this state
        self._state_turn_count[self.current_state] += 1
        turns_in_state = self._state_turn_count[self.current_state]

        # Determine how many agents should act in this state
        if acting_roles:
            # Only specific roles act - count them
            num_acting_agents = sum(
                1
                for agent in self.agents
                if self.agent_alive.get(agent, False)
                and self.agent_to_role.get(agent, "") in acting_roles
            )
        else:
            # All alive agents act
            num_acting_agents = sum(1 for alive in self.agent_alive.values() if alive)

        # Transition logic based on action order
        if action_order == "simultaneous":
            # All agents act at once - transition after 1 turn
            return turns_in_state >= 1
        elif action_order in ["round-robin", "random"]:
            # Each agent acts once - transition after N turns
            return turns_in_state >= num_acting_agents

        return False

    def _transition_state(self) -> None:
        """Transition to next state based on FSM."""
        state_transition = self._config.get("state_transition", {})
        next_state = state_transition.get(self.current_state)

        if next_state:
            self.current_state = next_state
            # Reset turn counter for the new state
            if hasattr(self, "_state_turn_count"):
                self._state_turn_count[self.current_state] = 0
            # Reset round-robin counter for the new state
            if hasattr(self, "_round_robin_idx"):
                self._round_robin_idx = 0
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] Transitioning to state: {self.current_state}"
                ),
            )

    def _build_observations(self) -> Dict[str, Observation]:
        """Build observations for each agent based on visibility rules."""
        observations = {}

        for i, agent_name in enumerate(self.agents):
            # Get recent messages visible to this agent
            visible_history = self._get_visible_history(agent_name)

            # Get available actions for this agent
            available_actions = self._get_available_actions(agent_name)

            observations[agent_name] = Observation(
                last_turn=visible_history,
                turn_number=self.turn_number,
                available_actions=available_actions,
            )

        return observations

    def _get_visible_history(self, agent_name: str) -> str:
        """Get message history visible to this agent based on visibility rules."""
        state_props = self._config.get("state_properties", {}).get(
            self.current_state, {}
        )
        visibility = state_props.get("visibility", "public")

        visible_messages = []

        for sender, message in self.inbox[-10:]:  # Last 10 messages
            if sender == "Environment":
                # Environment messages always visible
                visible_messages.append(message.to_natural_language())
            elif visibility == "public":
                # Public: everyone sees everything
                visible_messages.append(f"{sender}: {message.to_natural_language()}")
            elif visibility == "team":
                # Team: only see teammate messages
                sender_team = self.role_to_team.get(
                    self.agent_to_role.get(sender, ""), ""
                )
                viewer_team = self.role_to_team.get(
                    self.agent_to_role.get(agent_name, ""), ""
                )
                if sender_team == viewer_team:
                    visible_messages.append(
                        f"{sender}: {message.to_natural_language()}"
                    )
            elif visibility == "private":
                # Private: only see own messages
                if sender == agent_name:
                    visible_messages.append(f"You: {message.to_natural_language()}")

        return (
            "\n".join(visible_messages) if visible_messages else "[No recent activity]"
        )

    def _get_available_actions(
        self, agent_name: str
    ) -> List[Literal["none", "speak", "non-verbal communication", "action", "leave"]]:
        """Get available actions for this agent based on state and role, restricted to allowed literals."""
        if not self.agent_alive.get(agent_name, False):
            return ["none"]

        state_props = self._config.get("state_properties", {}).get(
            self.current_state, {}
        )
        acting_roles = state_props.get("acting_roles", [])
        actions = state_props.get("actions", ["speak"])

        # If state restricts by role, check if this agent can act
        if acting_roles:
            agent_role = self.agent_to_role.get(agent_name, "")
            if agent_role not in acting_roles:
                return ["none"]

        allowed = {
            "none",
            "speak",
            "non-verbal communication",
            "action",
            "leave",
        }
        filtered = [a for a in actions if a in allowed] or ["none"]
        return cast(
            List[
                Literal["none", "speak", "non-verbal communication", "action", "leave"]
            ],
            filtered,
        )

    def get_agent_role(self, agent_name: str) -> str:
        """Get the role of an agent."""
        return self.agent_to_role.get(agent_name, "Unknown")

    def get_agent_team(self, agent_name: str) -> str:
        """Get the team of an agent."""
        role = self.get_agent_role(agent_name)
        return self.role_to_team.get(role, "Unknown")
