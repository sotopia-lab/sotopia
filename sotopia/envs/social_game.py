from __future__ import annotations

import asyncio
import itertools
import json
import logging
import random
from collections import defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, cast

from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.envs.evaluators import Evaluator, unweighted_aggregate_evaluate
from sotopia.agents.llm_agent import Agents
from sotopia.database import EnvironmentProfile
from sotopia.messages import AgentAction, Observation, SimpleMessage, Message

logger = logging.getLogger(__name__)

SOCIAL_GAME_PROMPT_TEMPLATE = """
Imagine you are playing the game as {agent}.

Here is the description of the game: {description}

Your ({agent}'s) goal: {goal}
{secret}

Here is the context of the interaction:
{history}

Your available action type(s): [{action_list}].
{action_instructions}

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
{format_instructions}
"""


class SocialGame(ParallelSotopiaEnv, ABC):
    """Abstract base class for social games.

    Defines the interface for building state, handling transitions, and building observations.
    """

    def __init__(
        self,
        env_profile: EnvironmentProfile,
        action_handler: ActionHandler | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_profile=env_profile, **kwargs)
        self.action_handler = action_handler

    @abstractmethod
    def build_state(self, actions: Dict[str, AgentAction]) -> None:
        """Update game state based on agent actions."""
        pass

    @abstractmethod
    def state_transition(self) -> None:
        """Handle state transitions (e.g., FSM updates)."""
        pass

    @abstractmethod
    def build_observation(self) -> Dict[str, Observation]:
        """Generate observations for each agent."""
        pass

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

        # 1. Normalize actions and record to history
        normalized_actions = self._process_incoming_actions(actions)

        # 2. Build State (Update internal state, process actions, check eliminations, update masks)
        self.build_state(normalized_actions)

        # 3. State Transition (Check conditions, move FSM)
        self.state_transition()

        # 4. Run evaluators (Moved to check post-transition state)
        evaluator_response = await self._run_evaluators(self.evaluators)

        # 5. Build Observation (Generate what agents see)
        observations = self.build_observation()

        # 6. Set termination
        terminated = {agent: evaluator_response.terminated for agent in self.agents}

        # 7. Terminal evaluators
        if evaluator_response.terminated and self.terminal_evaluators:
            terminal_response = await self._run_evaluators(self.terminal_evaluators)
            if evaluator_response.comments and terminal_response.comments:
                evaluator_response.comments += terminal_response.comments
            elif terminal_response.comments:
                evaluator_response.comments = terminal_response.comments

        rewards = {agent: 0.0 for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        info = {
            agent: {
                "comments": evaluator_response.comments or "",
                "complete_rating": (
                    evaluator_response.rewards.get(f"agent_{i+1}", (0, {}))[0]
                    if evaluator_response.rewards
                    else 0
                ),
            }
            for i, agent in enumerate(self.agents)
        }

        return observations, rewards, terminated, truncations, info

    async def _run_evaluators(self, evaluators: list[Evaluator]) -> Any:
        """Run evaluators and aggregate results"""
        return unweighted_aggregate_evaluate(
            list(
                itertools.chain(
                    *await asyncio.gather(
                        *[
                            evaluator.__acall__(
                                turn_number=self.turn_number,
                                messages=self.inbox,
                                env=self,
                            )
                            for evaluator in evaluators
                        ]
                    )
                )
            )
        )


class ActionHandler(ABC):
    """Abstract base class for handling game-specific actions."""

    @abstractmethod
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        """Handle a single action from an agent based on current state.

        Args:
            env: The game environment instance.
            agent_name: The name of the agent performing the action.
            action: The action object.
        """
        pass

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        """Get specific action instructions for an agent based on current state.

        Args:
            env: The game environment instance.
            agent_name: The name of the agent.

        Returns:
            A string containing instructions, or empty string.
        """
        return ""


class SocialDeductionGame(SocialGame):
    """Environment for social deduction games with states, roles, and private information.

    Adds to SocialGame:
    - FSM states (Night, Day, etc.)
    - Role/team system
    - Alive/dead status
    - Private information visibility
    - State transitions
    - Turn management (round-robin vs simultaneous)
    - Global Environment notifications (bypassing visibility filters)
    """

    def __init__(
        self,
        env_profile: EnvironmentProfile,
        *,
        config_path: str | None = None,
        config: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_profile=env_profile, include_turn_marker=False, **kwargs)

        # Load game configuration
        self._config_path = Path(config_path) if config_path else None
        self._config: Dict[str, Any] = config if config else {}

        # Agent message buffer
        self.agent_message_buffer: Dict[str, List[str]] = defaultdict(list)

        # Game state
        self.current_state: str = ""
        self.agent_to_role: Dict[str, str] = {}  # Aurora -> Villager
        self.role_to_team: Dict[
            str, str
        ] = {}  # Seer -> Villagers, Werewolf -> Werewolves
        self.agent_alive: Dict[str, bool] = {}  # Aurora -> True
        self.internal_state: Dict[str, Any] = {}  # votes, targets, etc.

    def _load_config(self) -> None:
        """Load game configuration from JSON file if not already loaded."""
        if self._config:
            pass
        elif self._config_path:
            if not self._config_path.exists():
                raise FileNotFoundError(f"Config not found: {self._config_path}")
            self._config = json.loads(self._config_path.read_text())
        else:
            raise ValueError("Neither config nor config_path provided")

        # Build role -> team mapping
        self.role_to_team = {}
        for agent_entry in self._config.get("agents", []):
            role = agent_entry.get("role")
            team = agent_entry.get("team")
            if role and team:
                self.role_to_team.setdefault(role, team)

    async def astep(
        self, actions: Dict[str, AgentAction] | Dict[str, Dict[str, int | str]]
    ) -> tuple[
        Dict[str, Observation],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[Any, Any]],
    ]:
        """Process one step: update state counters and delegate."""
        # Update state turn counter
        if not hasattr(self, "_state_turn_count"):
            self._state_turn_count: Dict[str, int] = {}

        if self.current_state not in self._state_turn_count:
            self._state_turn_count[self.current_state] = 0

        self._state_turn_count[self.current_state] += 1

        # Call super().astep to get results
        (
            observations,
            rewards,
            terminated,
            truncations,
            info,
        ) = await super().astep(actions)

        # Log termination
        if all(terminated.values()):
            # Extract comments/reasons from info
            first_agent = list(self.agents)[0]
            reason = info.get(first_agent, {}).get("comments", "Unknown reason")

            log_msg = f"Game Ends:\n{reason}\n"
            for agent_name in self.agents:
                agent_rating = info.get(agent_name, {}).get("complete_rating", 0)
                log_msg += f"{agent_name}: {agent_rating}\n"
            logger.info(log_msg)

        return observations, rewards, terminated, truncations, info

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, str] | None = None,
        agents: Agents | None = None,
        omniscient: bool = False,
        lite: bool = False,
        include_background_observations: bool = True,
    ) -> dict[str, Observation]:
        """Reset environment and initialize game state."""
        # Call parent reset
        base_obs = super().reset(
            seed=seed,
            options=options,
            agents=agents,
            omniscient=omniscient,
            include_background_observations=False,
        )

        # Load config
        self._load_config()

        # Map agent names to roles from config
        self.agent_to_role = {}
        for name in self.agents:
            role = next(
                (
                    a.get("role", "Unknown")
                    for a in self._config.get("agents", [])
                    if a.get("name") == name.strip()
                ),
                "Unknown",
            )
            self.agent_to_role[name] = role

        # Initialize alive status and state
        self.agent_alive = {name: True for name in self.agents}
        self.current_state = self._config.get("initial_state", "Unknown")
        self.internal_state = {}
        self._state_turn_count = {self.current_state: 0}

        # Send initial system message
        initial_msg_content = f"[Game] State: {self.current_state}. The game begins!"
        logger.info(initial_msg_content)
        self.recv_message(
            "Environment",
            SimpleMessage(message=initial_msg_content),
        )

        # Initialize action mask for first turn based on state
        self._update_action_mask()

        # Update available actions based on game state
        for agent_name in self.agents:
            base_obs[agent_name].available_actions = self._get_available_actions(
                agent_name
            )
            # Inject initial action instruction if handler is present
            if self.action_handler:
                instruction = self.action_handler.get_action_instruction(
                    self, agent_name
                )
                if instruction:
                    base_obs[agent_name].action_instruction = instruction

        return base_obs

    def build_state(self, actions: Dict[str, AgentAction]) -> None:
        """Update game state based on agent actions."""
        # 1. Process game-specific logic
        self._process_actions(actions)

        # 2. Check for eliminations
        self._check_eliminations()

    def state_transition(self) -> None:
        """Handle state transitions."""
        should_transition = self._should_transition_state()
        logger.debug(
            f"Turn {self.turn_number}: state={self.current_state}, should_transition={should_transition}, state_turn_count={getattr(self, '_state_turn_count', {})}"
        )
        if should_transition:
            self._perform_transition_state()
            logger.debug(f"Transitioned to {self.current_state}")

        # Update action mask for next turn based on (potentially new) state
        state_props = self._config.get("state_properties", {}).get(
            self.current_state, {}
        )
        action_order = state_props.get("action_order", self.action_order)
        logger.debug(
            f"About to update mask - state={self.current_state}, action_order={action_order}"
        )
        self._update_action_mask()
        logger.debug(f"After update_action_mask - mask={self.action_mask}")

    def build_observation(self) -> Dict[str, Observation]:
        """Generate observations for each agent."""
        return self._build_observations()

    def _process_actions(self, actions: Dict[str, AgentAction]) -> None:
        """Process actions by delegating to action_handler."""
        if self.action_handler:
            for agent_name, action in actions.items():
                self.action_handler.handle_action(self, agent_name, action)

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

        turns_in_state = self._state_turn_count.get(self.current_state, 0)

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

    def _perform_transition_state(self) -> None:
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
                SimpleMessage(message=f"[Game] Entering state: {self.current_state}"),
            )
            logger.info(f"{'-'* 50}\nTurn to {self.current_state}\n{'-'* 50}")

    def _build_observations(self) -> Dict[str, Observation]:
        """Build observations for each agent based on visibility rules."""
        observations = {}

        for agent_name in self.agents:
            observations[agent_name] = self._get_observation(agent_name)

        return observations

    def recv_message(
        self, sender: str, message: Message, receivers: List[str] | None = None
    ) -> None:
        """Receive a message and distribute it to agents based on visibility."""
        super().recv_message(sender, message)

        # Determine visibility for each agent
        state_props = self._config.get("state_properties", {}).get(
            self.current_state, {}
        )
        visibility = state_props.get("visibility", "public")

        for agent_name in self.agents:
            should_see = False

            # Check for explicit receivers
            if receivers is not None:
                if agent_name in receivers:
                    should_see = True
            elif visibility == "public":
                should_see = True
            elif visibility == "team":
                sender_team = self.role_to_team.get(
                    self.agent_to_role.get(sender, ""), ""
                )
                viewer_team = self.role_to_team.get(
                    self.agent_to_role.get(agent_name, ""), ""
                )
                should_see = sender_team == viewer_team
            elif visibility == "private":
                should_see = sender == agent_name

            # Environment messages should be public unless explicitly targeted
            if sender == "Environment" and receivers is None:
                should_see = True

            if should_see:
                if sender == "Environment":
                    self.agent_message_buffer[agent_name].append(
                        message.to_natural_language()
                    )
                else:
                    self.agent_message_buffer[agent_name].append(
                        f"{sender}: {message.to_natural_language()}"
                    )

    def _get_observation(self, agent_name: str) -> Observation:
        """Get observation for a specific agent."""
        # Get visible history from buffer
        visible_history = "\n".join(self.agent_message_buffer[agent_name])

        # Clear buffer after reading: Observation usually only sends new content; agent's memory handles accumulation.
        self.agent_message_buffer[agent_name].clear()

        # Get available actions
        available_actions = self._get_available_actions(agent_name)

        # Add specific action instructions if handler is present
        action_instruction = ""
        if self.action_handler:
            instruction = self.action_handler.get_action_instruction(self, agent_name)
            if instruction:
                action_instruction = instruction

        return Observation(
            last_turn=visible_history if visible_history else "[No recent activity]",
            turn_number=self.turn_number,
            available_actions=available_actions,
            action_instruction=action_instruction,
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

        # Check action mask (for round-robin/random ordering)
        if self.action_mask:
            try:
                agent_idx = self.agents.index(agent_name)
                if not self.action_mask[agent_idx]:
                    return ["none"]
            except ValueError:
                pass  # Should not happen if agent_name is valid

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


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load game configuration from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return cast(Dict[str, Any], json.loads(path.read_text()))
