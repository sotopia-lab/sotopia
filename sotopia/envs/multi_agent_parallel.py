"""
Multi-agent extension of ParallelSotopiaEnv to support more than 2 agents.
"""

import asyncio
import copy
import itertools
from typing import Any, Optional

try:
    from beartype import beartype  # type: ignore[import-not-found]
except ImportError:
    # Fallback if beartype is not available
    def beartype(func):  # type: ignore[no-untyped-def]
        return func


from sotopia.envs.parallel import (
    ParallelSotopiaEnv,
    get_bio,
    render_text_for_environment,
    render_text_for_agent,
    _actions_to_natural_language,
)
from sotopia.messages.message_classes import MultiAgentBackground, ScriptBackground
from sotopia.messages import MessengerMixin, Observation, AgentAction, SimpleMessage
from sotopia.envs.evaluators import unweighted_aggregate_evaluate


class MultiAgentSotopiaEnv(ParallelSotopiaEnv):
    """Extended environment that supports more than 2 agents."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # background will be properly set in reset() method

    @beartype  # type: ignore[misc]
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        agents: Optional[dict[str, Any]] = None,
        omniscient: bool = False,
        lite: bool = False,
    ) -> dict[str, Observation]:
        """Reset environment with support for multiple agents."""
        MessengerMixin.reset_inbox(self)

        if agents is not None:
            assert agents, "agents must be provided"

            agent_names = list(agents.keys())
            agent_goals = self.profile.agent_goals
            num_agents = len(agents)

            # Check if we have more than 2 agents and sufficient goals
            if num_agents > 2:
                assert (
                    len(agent_goals) >= num_agents
                ), f"Need at least {num_agents} agent goals, got {len(agent_goals)}"

                # Use multi-agent background
                agent_backgrounds = []
                for i, agent_name in enumerate(agent_names):
                    bg = get_bio(
                        self.profile.relationship,
                        agents[agent_name].profile,
                        agent_id=i,
                    )
                    agent_backgrounds.append(bg)

                raw_background = MultiAgentBackground.create(
                    scenario=self.profile.scenario,
                    agent_names=agent_names,
                    agent_backgrounds=agent_backgrounds,
                    agent_goals=[
                        f"<root viewer='agent_{i}'>{goal}</root>"
                        for i, goal in enumerate(agent_goals[:num_agents])
                    ],
                )

                if lite:
                    raw_background.agent_backgrounds = [""] * num_agents

                self.background = MultiAgentBackground.create(
                    scenario=render_text_for_environment(raw_background.scenario),
                    agent_names=raw_background.agent_names,
                    agent_backgrounds=[
                        render_text_for_environment(bg)
                        for bg in raw_background.agent_backgrounds
                    ],
                    agent_goals=[
                        render_text_for_environment(goal)
                        for goal in raw_background.agent_goals
                    ],
                )

                self.agents = agent_names

                # Create individual agent backgrounds
                agent_backgrounds_list: list[ScriptBackground] = []
                if omniscient:
                    for i in range(num_agents):
                        agent_backgrounds_list.append(copy.deepcopy(self.background))
                else:
                    for i in range(num_agents):
                        # Create background for agent i with other agents' goals hidden
                        hidden_goals = raw_background.agent_goals.copy()
                        for j, _ in enumerate(hidden_goals):
                            if j != i:
                                hidden_goals[j] = "Unknown"

                        agent_background = MultiAgentBackground.create(
                            scenario=render_text_for_agent(raw_background.scenario, i),
                            agent_names=raw_background.agent_names,
                            agent_backgrounds=[
                                render_text_for_agent(bg, i) if j == i else "Unknown"
                                for j, bg in enumerate(raw_background.agent_backgrounds)
                            ],
                            agent_goals=[
                                render_text_for_agent(goal, i) for goal in hidden_goals
                            ],
                        )
                        agent_backgrounds_list.append(agent_background)

            else:
                # Handle 2-agent case using the same logic as parent but without the assertion
                assert (
                    len(agent_goals) >= 2
                ), "Need at least 2 agent goals for 2-agent mode"

                # Use ScriptBackground for 2-agent case
                p1_bg = get_bio(
                    self.profile.relationship,
                    agents[agent_names[0]].profile,
                    agent_id=0,
                )
                p2_bg = get_bio(
                    self.profile.relationship,
                    agents[agent_names[1]].profile,
                    agent_id=1,
                )

                if lite:
                    p1_bg = ""
                    p2_bg = ""

                script_background = ScriptBackground(
                    scenario=self.profile.scenario,
                    p1_background=p1_bg,
                    p2_background=p2_bg,
                    p1_goal=f"<root viewer='agent_0'>{agent_goals[0]}</root>",
                    p2_goal=f"<root viewer='agent_1'>{agent_goals[1]}</root>",
                    p1_name=agent_names[0],
                    p2_name=agent_names[1],
                )

                self.background = ScriptBackground(
                    scenario=render_text_for_environment(script_background.scenario),
                    p1_background=render_text_for_environment(
                        script_background.p1_background
                    ),
                    p2_background=render_text_for_environment(
                        script_background.p2_background
                    ),
                    p1_goal=render_text_for_environment(script_background.p1_goal),
                    p2_goal=render_text_for_environment(script_background.p2_goal),
                    p1_name=script_background.p1_name,
                    p2_name=script_background.p2_name,
                )

                # Safe access to background attributes
                if isinstance(self.background, ScriptBackground):
                    self.agents = [self.background.p1_name, self.background.p2_name]
                else:
                    self.agents = agent_names

                # Create individual agent backgrounds for 2-agent case
                agent_backgrounds_list_2: list[ScriptBackground] = []
                if omniscient:
                    for i in range(2):
                        agent_backgrounds_list_2.append(copy.deepcopy(self.background))
                else:
                    # Create backgrounds with hidden goals for each agent
                    # Agent 0 sees their goal, Agent 1's goal is "Unknown"
                    agent_background_0 = ScriptBackground(
                        scenario=render_text_for_agent(script_background.scenario, 0),
                        p1_background=render_text_for_agent(
                            script_background.p1_background, 0
                        ),
                        p2_background=render_text_for_agent(
                            script_background.p2_background, 0
                        ),
                        p1_goal=render_text_for_agent(script_background.p1_goal, 0),
                        p2_goal="Unknown",  # Hide agent 1's goal from agent 0
                        p1_name=script_background.p1_name,
                        p2_name=script_background.p2_name,
                    )
                    # Agent 1 sees their goal, Agent 0's goal is "Unknown"
                    agent_background_1 = ScriptBackground(
                        scenario=render_text_for_agent(script_background.scenario, 1),
                        p1_background=render_text_for_agent(
                            script_background.p1_background, 1
                        ),
                        p2_background=render_text_for_agent(
                            script_background.p2_background, 1
                        ),
                        p1_goal="Unknown",  # Hide agent 0's goal from agent 1
                        p2_goal=render_text_for_agent(script_background.p2_goal, 1),
                        p1_name=script_background.p1_name,
                        p2_name=script_background.p2_name,
                    )
                    agent_backgrounds_list_2.append(agent_background_0)
                    agent_backgrounds_list_2.append(agent_background_1)

        else:
            raise ValueError("agents must be provided")

        # Create observations for each agent
        observations = {}
        final_agent_backgrounds: list[ScriptBackground] = (
            agent_backgrounds_list if num_agents > 2 else agent_backgrounds_list_2
        )
        for i, agent_name in enumerate(agent_names):
            agent_bg: ScriptBackground = final_agent_backgrounds[i]
            observations[agent_name] = Observation(
                last_turn=agent_bg.to_natural_language(),
                turn_number=0,
                available_actions=[
                    "speak",
                    "non-verbal communication",
                    "action",
                    "leave",
                ],
            )

        # Initialize turn tracking (from parent class)
        self.turn_number = 0
        self.action_mask = [False for _ in self.agents]

        # Initialize action order logic
        if hasattr(self, "action_order"):
            if self.action_order == "round-robin":
                self.action_mask[0] = True
            elif self.action_order == "random":
                import random

                self.action_mask[random.randint(0, len(self.action_mask) - 1)] = True
            else:  # simultaneous
                self.action_mask = [True for _ in self.agents]

        return observations

    @property
    def num_agents(self) -> int:
        """Return the number of agents in the environment."""
        return len(self.agents)

    async def astep(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        """Override astep to handle multiple agents properly."""
        # If we have 2 agents, use parent method
        if len(self.agents) == 2:
            return await super().astep(actions)

        # Multi-agent logic (based on parent astep but generalized)
        self.turn_number += 1

        # Convert actions to AgentAction format
        complied_actions: dict[str, AgentAction] = {}
        for key in actions.keys():
            action = actions[key]
            if isinstance(action, AgentAction):
                complied_actions[key] = action
            else:
                action["action_type"] = self.available_action_types[
                    int(action["action_type"])
                ]
                complied_actions[key] = AgentAction.model_validate(action)

        # Apply action masking
        for idx, agent in enumerate(self.agents):
            if not self.action_mask[idx]:
                complied_actions[agent] = AgentAction(action_type="none", argument="")

        # Add messages to inbox
        self.recv_message(
            "Environment", SimpleMessage(message=f"Turn #{self.turn_number}")
        )
        for agent, action in complied_actions.items():
            self.recv_message(agent, action)

        # Run evaluations
        response = unweighted_aggregate_evaluate(
            list(
                itertools.chain(
                    *await asyncio.gather(
                        *[
                            evaluator.__acall__(
                                turn_number=self.turn_number, messages=self.inbox
                            )
                            for evaluator in self.evaluators
                        ]
                    )
                )
            )
        )

        # Run terminal evaluation if conversation ended
        if response.terminated:
            _ = unweighted_aggregate_evaluate(
                list(
                    itertools.chain(
                        *await asyncio.gather(
                            *[
                                evaluator.__acall__(
                                    turn_number=self.turn_number,
                                    messages=self.inbox,
                                )
                                for evaluator in self.terminal_evaluators
                            ]
                        )
                    )
                )
            )

        # Update action mask for next turn
        if self.action_order == "round-robin":
            self.action_mask = [False for _ in self.agents]
            self.action_mask[self.turn_number % len(self.action_mask)] = True
        elif self.action_order == "random":
            import random

            self.action_mask = [False for _ in self.agents]
            self.action_mask[random.randint(0, len(self.action_mask) - 1)] = True
        else:
            self.action_mask = [True for _ in self.agents]

        # Generate natural language representation
        obs = _actions_to_natural_language(complied_actions)

        # Create info dict for all agents
        info = {}
        for agent in self.agents:
            info[agent] = {
                "comments": response.comments or "",
                "complete_rating": 0,  # We'll handle ratings differently for multi-agent
            }

        if response.terminated:
            info["rewards_prompt"] = {
                "overall_prompt": getattr(self.terminal_evaluators[0], "prompt", "")
                if self.terminal_evaluators
                else ""
            }

        # Create observations for all agents
        observations = {}
        for idx, agent in enumerate(self.agents):
            observations[agent] = Observation(
                last_turn=render_text_for_agent(obs, agent_id=idx),
                turn_number=self.turn_number,
                available_actions=list(self.available_action_types)
                if self.action_mask[idx]
                else ["none"],
            )

        # Create rewards, terminated, truncated dicts for all agents
        rewards = {agent: 0.0 for agent in self.agents}
        terminated = {agent: response.terminated for agent in self.agents}
        truncated = {agent: False for agent in self.agents}

        return observations, rewards, terminated, truncated, info
