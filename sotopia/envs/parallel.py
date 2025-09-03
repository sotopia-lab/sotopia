import asyncio
import copy
import itertools
import random
from typing import Any, Literal, Optional, Type, TypeVar

from gin import configurable
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.text import Text
from pettingzoo.utils.env import ParallelEnv
from pydantic import validate_call
from redis_om.model.model import NotFoundError

from sotopia.agents.llm_agent import Agents
from sotopia.database import EnvironmentProfile
from sotopia.database.persistent_profile import (
    AgentProfile,
    RelationshipType,
)
from sotopia.messages import (
    ActionType,
    AgentAction,
    MessengerMixin,
    Observation,
    ScriptBackground,
    SimpleMessage,
)
from sotopia.renderers import RenderContext, XMLRenderer

from .evaluators import Evaluator, unweighted_aggregate_evaluate

TBackground = TypeVar("TBackground", bound=ScriptBackground)


def _actions_to_natural_language(actions: dict[str, AgentAction]) -> str:
    action_str = ""
    for agent, action in actions.items():
        # Only record actions that did something
        if action.action_type != "none":
            if action_str != "":
                action_str += ";"  # separate actions with semicolon
            action_str += f"{agent} {action.to_natural_language()}"
    return action_str


def _map_gender_to_adj(gender: str) -> str:
    gender_to_adj = {
        "Man": "male",
        "Woman": "female",
        "Nonbinary": "nonbinary",
    }
    if gender:
        return gender_to_adj.get(gender, "")
    else:
        return ""


def _agent_profile_to_stranger_self(profile: AgentProfile, agent_id: int) -> str:
    return f"<root><p viewer='agent_{agent_id}'>{profile.first_name} {profile.last_name} is a {profile.age}-year-old {_map_gender_to_adj(profile.gender)} {profile.occupation.lower()}. {profile.gender_pronoun} pronouns. {profile.public_info} Personality and values description: {profile.personality_and_values} {profile.first_name}'s secrets: {profile.secret}</p></root>"


def _agent_profile_to_name_self(profile: AgentProfile, agent_id: int) -> str:
    return f"{profile.first_name} {profile.last_name} <p viewer='agent_{agent_id}'>is a {profile.age}-year-old {_map_gender_to_adj(profile.gender)} {profile.occupation.lower()}. {profile.gender_pronoun} pronouns. {profile.public_info} Personality and values description: {profile.personality_and_values} {profile.first_name}'s secrets: {profile.secret}</p>"


def _agent_profile_to_aquaintance_self(profile: AgentProfile, agent_id: int) -> str:
    return f"{profile.first_name} {profile.last_name} is a {profile.age}-year-old {_map_gender_to_adj(profile.gender)} {profile.occupation.lower()}. {profile.gender_pronoun} pronouns. {profile.public_info} <p viewer='agent_{agent_id}'>Personality and values description: {profile.personality_and_values} {profile.first_name}'s secrets: {profile.secret}</p>"


def _agent_profile_to_friendabove_self(profile: AgentProfile, agent_id: int) -> str:
    return f"{profile.first_name} {profile.last_name} is a {profile.age}-year-old {_map_gender_to_adj(profile.gender)} {profile.occupation.lower()}. {profile.gender_pronoun} pronouns. {profile.public_info} Personality and values description: {profile.personality_and_values} <p viewer='agent_{agent_id}'>{profile.first_name}'s secrets: {profile.secret}</p>"


def get_bio(
    relationship: RelationshipType, profile: AgentProfile, agent_id: int
) -> str:
    match relationship:
        case RelationshipType.stranger:
            return _agent_profile_to_stranger_self(profile, agent_id=agent_id)
        case RelationshipType.know_by_name:
            return _agent_profile_to_name_self(profile, agent_id=agent_id)
        case RelationshipType.acquaintance:
            return _agent_profile_to_aquaintance_self(profile, agent_id=agent_id)
        case (
            RelationshipType.friend
            | RelationshipType.romantic_relationship
            | RelationshipType.family_member
        ):
            return _agent_profile_to_friendabove_self(profile, agent_id=agent_id)
        case _:
            raise ValueError(f"Unknown relationship {relationship}")


@configurable
def render_text_for_agent(
    raw_text: str,
    agent_id: int,
    tags_to_render: list[str] = [
        "extra_info",
        "clarification_hint",
        "strategy_hint",
    ],
) -> str:
    return XMLRenderer()(
        raw_text,
        RenderContext(viewer=f"agent_{agent_id}", tags_to_render=tags_to_render),
    )


@configurable
def render_text_for_environment(
    raw_text: str,
    tags_to_render: list[str] = [
        "extra_info",
        "clarification_hint",
        "strategy_hint",
    ],
) -> str:
    return XMLRenderer()(
        raw_text,
        RenderContext(viewer="environment", tags_to_render=tags_to_render),
    )


class ParallelSotopiaEnv(ParallelEnv[str, Observation, AgentAction], MessengerMixin):
    def __init__(
        self,
        available_action_types: set[ActionType] = set(
            ["none", "speak", "non-verbal communication", "action", "leave"]
        ),
        action_order: Literal["simultaneous", "round-robin", "random"] = "simultaneous",
        evaluators: list[Evaluator] = [],
        model_name: str = "gpt-4o-mini",
        terminal_evaluators: list[Evaluator] = [],
        uuid_str: str | None = None,
        env_profile: EnvironmentProfile | None = None,
        background_class: Optional[Type[TBackground]] = None,
    ) -> None:
        """A sotopia environment for parallel agents.

        Args:
            available_action_types (set[ActionType], optional): The action types that are available to the agents. Defaults to set(["none", "speak", "non-verbal communication", "action"]).
            action_order (Literal["simultaneous", "round-robin", "random"], optional): The order in which the agents take actions. Defaults to "simultaneous".
            model_name (str, optional): The name of the language model to use. Defaults to "gpt-3.5-turbo".
        """
        super().__init__()
        if background_class is None:
            self.background_class = ScriptBackground
        else:
            self.background_class = background_class
        self.background = self.background_class(
            scenario="",
            p1_background="",
            p2_background="",
            p1_goal="",
            p2_goal="",
            p1_name="",
            p2_name="",
        )

        self.agents = []
        self.action_spaces = {}
        self.available_action_types = list(available_action_types)
        self.action_order = action_order
        self.action_mask: list[bool] = []
        self.evaluators = evaluators
        self.terminal_evaluators = terminal_evaluators
        self.model_name = model_name
        # if an environment profile is provided, use it
        assert (
            env_profile or uuid_str
        ), "Either env_profile or uuid_str must be provided"
        if env_profile is not None:
            self.profile = env_profile
        # if a uuid is provided, try to load the environment profile from the database
        elif uuid_str is not None:
            # try retrieving profile from database
            try:
                self.profile = EnvironmentProfile.get(pk=uuid_str)
            except NotFoundError:
                raise ValueError(f"Agent with uuid {uuid_str} not found in database")

    @configurable
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
        agents: Agents | None = None,
        omniscient: bool = False,
        lite: bool = False,
    ) -> dict[str, Observation]:
        """Starting a new episode. Must be called before step().

        Args:
            seed (int, optional): Seed for the environment. Defaults to None. Not used right now.
            options (dict, optional): Options for the environment. Defaults to None.
                "partial_background_file" (str): Path to a json file which need to contain a ScriptBackground object. The backgound can be incompleted ("unknown" for missing parts), and the missing parts will be filled in by the environment.
                "full_background_file" (str): Path to a json file which need to contain a ScriptBackground object. The backgound must be completed (no "unknown" for missing parts).
            omniscient (bool, optional): Whether the agents know the other agent's goal. Defaults to False.
        """
        super().__init__()
        MessengerMixin.reset_inbox(self)
        assert (
            not options
            or "partial_background_file" not in options
            and "full_background_file" not in options
        ), "partial_background_file and full_background_file are not supported anymore"
        if agents is not None:
            assert agents, "agents must be provided"
            assert len(agents) >= 2, f"At least 2 agents required, got {len(agents)}"
            agent_names = list(agents.keys())
            agent_goals = self.profile.agent_goals
            assert (
                len(agent_goals) >= 2
            ), f"At least 2 agent goals required, got {len(agent_goals)}"

            # Ensure we have enough goals for all agents
            if len(agent_goals) < len(agents):
                # Pad with generic goals if needed
                while len(agent_goals) < len(agents):
                    agent_goals.append(
                        f"Participate effectively in this {len(agents)}-agent interaction"
                    )

            # Handle both 2-agent and multi-agent scenarios
            num_agents = len(agents)
            raw_background: ScriptBackground
            if num_agents > 2:
                # Multi-agent scenario - use ScriptBackground.create_multi_agent
                raw_agent_bios = []
                for i, agent_name in enumerate(agent_names):
                    bg = get_bio(
                        self.profile.relationship,
                        agents[agent_name].profile,
                        agent_id=i,
                    )
                    raw_agent_bios.append(bg)

                raw_background = ScriptBackground.create_multi_agent(
                    scenario=self.profile.scenario,
                    agent_names=agent_names,
                    agent_backgrounds=raw_agent_bios,
                    agent_goals=[
                        f"<root viewer='agent_{i}'>{goal}</root>"
                        for i, goal in enumerate(agent_goals[:num_agents])
                    ],
                )
            else:
                # 2-agent scenario - use existing ScriptBackground
                raw_background = self.background_class(
                    scenario=self.profile.scenario,
                    p1_background=get_bio(
                        self.profile.relationship,
                        agents[agent_names[0]].profile,
                        agent_id=0,
                    ),
                    p2_background=get_bio(
                        self.profile.relationship,
                        agents[agent_names[1]].profile,
                        agent_id=1,
                    ),
                    p1_goal=f"<root viewer='agent_0'>{agent_goals[0]}</root>",
                    p2_goal=f"<root viewer='agent_1'>{agent_goals[1]}</root>",
                    p1_name=agent_names[0],
                    p2_name=agent_names[1],
                )

            if lite:
                if num_agents > 2 and raw_background.agent_names:
                    # Multi-agent lite mode
                    raw_background.agent_backgrounds = [""] * num_agents
                elif num_agents == 2:
                    # 2-agent lite mode
                    raw_background.p1_background = ""
                    raw_background.p2_background = ""

            # Create final rendered background
            if num_agents > 2 and raw_background.agent_names:
                # Multi-agent final background
                self.background = ScriptBackground.create_multi_agent(
                    scenario=render_text_for_environment(raw_background.scenario),
                    agent_names=raw_background.agent_names,
                    agent_backgrounds=[
                        render_text_for_environment(bg)
                        for bg in (raw_background.agent_backgrounds or [])
                    ],
                    agent_goals=[
                        render_text_for_environment(goal)
                        for goal in (raw_background.agent_goals or [])
                    ],
                )
            else:
                # 2-agent final background
                self.background = self.background_class(
                    scenario=render_text_for_environment(raw_background.scenario),
                    p1_background=render_text_for_environment(
                        raw_background.p1_background
                    ),
                    p2_background=render_text_for_environment(
                        raw_background.p2_background
                    ),
                    p1_goal=render_text_for_environment(raw_background.p1_goal),
                    p2_goal=render_text_for_environment(raw_background.p2_goal),
                    p1_name=raw_background.p1_name,
                    p2_name=raw_background.p2_name,
                )
        else:
            raise ValueError("agents must be provided")

        # Set agent list based on scenario type
        if num_agents > 2:
            self.agents = agent_names
        else:
            self.agents = [self.background.p1_name, self.background.p2_name]
        # Create individual agent backgrounds
        agent_backgrounds: list[ScriptBackground] = []
        if omniscient:
            for i in range(num_agents):
                agent_backgrounds.append(copy.deepcopy(self.background))
        else:
            if num_agents > 2 and raw_background.agent_names:
                # Multi-agent non-omniscient backgrounds
                for i in range(num_agents):
                    # Each agent sees their own goal, others are hidden
                    hidden_goals = list(raw_background.agent_goals or [])
                    for j in range(len(hidden_goals)):
                        if j != i:
                            hidden_goals[j] = "Unknown"

                    agent_background = ScriptBackground.create_multi_agent(
                        scenario=render_text_for_agent(raw_background.scenario, i),
                        agent_names=raw_background.agent_names,
                        agent_backgrounds=[
                            render_text_for_agent(bg, i) if j == i else "Unknown"
                            for j, bg in enumerate(
                                raw_background.agent_backgrounds or []
                            )
                        ],
                        agent_goals=[
                            render_text_for_agent(goal, i) for goal in hidden_goals
                        ],
                    )
                    agent_backgrounds.append(agent_background)
            else:
                # 2-agent non-omniscient backgrounds
                for i in range(num_agents):
                    agent_backgrounds.append(
                        self.background_class(
                            scenario=render_text_for_agent(raw_background.scenario, i),
                            p1_background=render_text_for_agent(
                                raw_background.p1_background, i
                            ),
                            p2_background=render_text_for_agent(
                                raw_background.p2_background, i
                            ),
                            p1_goal=render_text_for_agent(raw_background.p1_goal, i),
                            p2_goal=render_text_for_agent(raw_background.p2_goal, i),
                            p1_name=raw_background.p1_name,
                            p2_name=raw_background.p2_name,
                        )
                    )
        # Handle goal hiding for 2-agent case (multi-agent already handled above)
        if num_agents == 2 and not omniscient:
            # Type check to ensure we have ScriptBackground for 2-agent scenarios
            if hasattr(agent_backgrounds[0], "p2_goal") and hasattr(
                agent_backgrounds[1], "p1_goal"
            ):
                agent_backgrounds[0].p2_goal = "Unknown"
                agent_backgrounds[1].p1_goal = "Unknown"

        self.action_spaces = {
            agent: Dict(
                dict(
                    action_type=Discrete(len(self.available_action_types)),
                    argument=Text(256),
                )
            )
            for agent in self.agents
        }
        self.turn_number = 0
        self.action_mask = [False for _ in self.agents]
        if self.action_order == "round-robin":
            self.action_mask[0] = True
        elif self.action_order == "random":
            self.action_mask[random.randint(0, len(self.action_mask) - 1)] = True
        else:
            self.action_mask = [True for _ in self.agents]

        self.recv_message("Environment", self.background)

        # Create observations for each agent
        observations = {}
        for i, agent_name in enumerate(self.agents):
            agent_bg = agent_backgrounds[i]
            observations[agent_name] = Observation(
                last_turn=agent_bg.to_natural_language(),
                turn_number=0,
                available_actions=list(self.available_action_types)
                if self.action_mask[i]
                else ["none"],
            )

        return observations

    @validate_call
    def step(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        # Time step ++
        self.turn_number += 1

        # For action sampled from action space, it needs to be converted into AgentAction
        complied_actions: dict[str, AgentAction] = {}
        for key in actions.keys():
            action = actions[key]
            if isinstance(action, AgentAction):
                complied_actions[key] = action
            else:
                action["action_type"] = self.available_action_types[
                    int(action["action_type"])
                ]
                complied_actions[key] = AgentAction.parse_obj(action)

        # Masking actions from agent that are in turn
        for idx, agent in enumerate(self.agents):
            if not self.action_mask[idx]:
                complied_actions[agent] = AgentAction(action_type="none", argument="")

        self.recv_message(
            "Environment", SimpleMessage(message=f"Turn #{self.turn_number}")
        )
        for agent, action in complied_actions.items():
            self.recv_message(agent, action)

        response = unweighted_aggregate_evaluate(
            list(
                itertools.chain(
                    *(
                        evaluator(turn_number=self.turn_number, messages=self.inbox)
                        for evaluator in self.evaluators
                    )
                )
            )
        )

        self.action_mask = [False for _ in self.agents]
        if self.action_order == "round-robin":
            self.action_mask[self.turn_number % len(self.action_mask)] = True
        elif self.action_order == "random":
            self.action_mask[random.randint(0, len(self.action_mask) - 1)] = True
        else:
            self.action_mask = [True for _ in self.agents]
        obs = _actions_to_natural_language(complied_actions)

        # Create observations for all agents dynamically
        observations = {}
        for i, agent_name in enumerate(self.agents):
            observations[agent_name] = Observation(
                last_turn=render_text_for_agent(obs, agent_id=i),
                turn_number=self.turn_number,
                available_actions=list(self.available_action_types)
                if self.action_mask[i]
                else ["none"],
            )

        return (
            observations,
            # Create reward dictionary for all agents
            {agent_name: 0 for agent_name in self.agents},
            # Create info dictionary for all agents
            {agent_name: response.terminated for agent_name in self.agents},
            # Create done dictionary for all agents
            {agent_name: False for agent_name in self.agents},
            # Create info dictionary with comments and ratings for all agents
            {
                agent_name: {
                    "comments": response.comments or "",
                    "complete_rating": 0,
                }
                for agent_name in self.agents
            },
        )

    async def astep(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        # Time step ++
        self.turn_number += 1

        # For action sampled from action space, it needs to be converted into AgentAction
        complied_actions: dict[str, AgentAction] = {}
        for key in actions.keys():
            action = actions[key]
            if isinstance(action, AgentAction):
                complied_actions[key] = action
            else:
                action["action_type"] = self.available_action_types[
                    int(action["action_type"])
                ]
                complied_actions[key] = AgentAction.parse_obj(action)

        # Masking actions from agent that are in turn
        for idx, agent in enumerate(self.agents):
            if not self.action_mask[idx]:
                complied_actions[agent] = AgentAction(action_type="none", argument="")

        self.recv_message(
            "Environment", SimpleMessage(message=f"Turn #{self.turn_number}")
        )
        for agent, action in complied_actions.items():
            self.recv_message(agent, action)

        response = unweighted_aggregate_evaluate(
            list(
                itertools.chain(
                    *await asyncio.gather(
                        *[
                            evaluator.__acall__(
                                turn_number=self.turn_number,
                                messages=self.inbox,
                            )
                            for evaluator in self.evaluators
                        ]
                    )
                )
            )
        )

        if response.terminated:
            terminal_response = unweighted_aggregate_evaluate(
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
            # incorporate terminal response into response
            response.p1_rate = response.p1_rate or terminal_response.p1_rate
            response.p2_rate = response.p2_rate or terminal_response.p2_rate
            if response.comments and terminal_response.comments:
                response.comments += terminal_response.comments
            elif terminal_response.comments:
                response.comments = terminal_response.comments

        self.action_mask = [False for _ in self.agents]
        if self.action_order == "round-robin":
            self.action_mask[self.turn_number % len(self.action_mask)] = True
        elif self.action_order == "random":
            self.action_mask[random.randint(0, len(self.action_mask) - 1)] = True
        else:
            self.action_mask = [True for _ in self.agents]
        obs = _actions_to_natural_language(complied_actions)
        # Create info dictionary for all agents
        info = {
            agent_name: {
                "comments": response.comments or "",
                "complete_rating": 0,
            }
            for agent_name in self.agents
        }
        if response.terminated:
            info["rewards_prompt"] = {
                "overall_prompt": self.terminal_evaluators[0].prompt  # type: ignore
            }

        # Create observations for all agents dynamically
        observations = {}
        for i, agent_name in enumerate(self.agents):
            observations[agent_name] = Observation(
                last_turn=render_text_for_agent(obs, agent_id=i),
                turn_number=self.turn_number,
                available_actions=list(self.available_action_types)
                if self.action_mask[i]
                else ["none"],
            )

        return (
            observations,
            # Create reward dictionary for all agents
            {agent_name: 0 for agent_name in self.agents},
            # Create terminated dictionary for all agents
            {agent_name: response.terminated for agent_name in self.agents},
            # Create done dictionary for all agents
            {agent_name: False for agent_name in self.agents},
            info,
        )

    def render(self, mode: str = "human") -> None:
        pass

    def close(self) -> None:
        pass
