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
        hide_unknown: bool = False,
        include_turn_marker: bool = True,
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
        self.hide_unknown = hide_unknown
        self.include_turn_marker = include_turn_marker
        self.background = self.background_class(
            scenario="",
            agent_names=[],
            agent_backgrounds=[],
            agent_goals=[],
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
        include_background_observations: bool = True,
    ) -> dict[str, Observation]:
        """Starting a new episode. Must be called before step().

        Args:
            seed (int, optional): Seed for the environment. Defaults to None. Not used right now.
            options (dict, optional): Options for the environment. Defaults to None.
                "partial_background_file" (str): Path to a json file which need to contain a ScriptBackground object. The backgound can be incompleted ("unknown" for missing parts), and the missing parts will be filled in by the environment.
                "full_background_file" (str): Path to a json file which need to contain a ScriptBackground object. The backgound must be completed (no "unknown" for missing parts).
            omniscient (bool, optional): Whether the agents know the other agent's goal. Defaults to False.
            include_background_observations (bool, optional): Whether to include the background (Environment's message) in the observation. Defaults to True.
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

            # Handle multi-agent scenarios
            num_agents = len(agents)
            raw_background: ScriptBackground
            raw_agent_bios = []
            for i, agent_name in enumerate(agent_names):
                bg = get_bio(
                    self.profile.relationship,
                    agents[agent_name].profile,
                    agent_id=i,
                )
                raw_agent_bios.append(bg)

            raw_background = self.background_class(
                scenario=self.profile.scenario,
                agent_names=agent_names,
                agent_backgrounds=raw_agent_bios,
                agent_goals=[
                    f"<root viewer='agent_{i}'>{goal}</root>"
                    for i, goal in enumerate(agent_goals[:num_agents])
                ],
            )

            if lite:
                # Lite mode - clear backgrounds
                raw_background.agent_backgrounds = [""] * num_agents

            # Create final rendered background
            self.background = self.background_class(
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
        else:
            raise ValueError("agents must be provided")

        # Set agent list from background
        self.agents = self.background.agent_names
        # Create individual agent backgrounds
        agent_backgrounds: list[ScriptBackground] = []
        if omniscient:
            for i in range(num_agents):
                agent_backgrounds.append(copy.deepcopy(self.background))
        else:
            # Non-omniscient backgrounds - each agent sees only their own goal
            for i in range(num_agents):
                # Each agent sees their own goal, others are hidden
                hidden_goals = list(raw_background.agent_goals)
                for j in range(len(hidden_goals)):
                    if j != i:
                        hidden_goals[j] = "Unknown"

                agent_background = self.background_class(
                    scenario=render_text_for_agent(raw_background.scenario, i),
                    agent_names=raw_background.agent_names,
                    agent_backgrounds=[
                        render_text_for_agent(bg, i) if j == i else "Unknown"
                        for j, bg in enumerate(raw_background.agent_backgrounds)
                    ],
                    agent_goals=[
                        render_text_for_agent(goal, i) for goal in hidden_goals
                    ],
                    hide_unknown=self.hide_unknown,
                )
                agent_backgrounds.append(agent_background)

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

        # Create observations for each agent
        observations = {}
        if include_background_observations:
            self.recv_message("Environment", self.background)
            for i, agent_name in enumerate(self.agents):
                agent_bg = agent_backgrounds[i]
                observations[agent_name] = Observation(
                    last_turn=agent_bg.to_natural_language(),
                    turn_number=0,
                    available_actions=list(self.available_action_types)
                    if self.action_mask[i]
                    else ["none"],
                )
        else:
            for i, agent_name in enumerate(self.agents):
                observations[agent_name] = Observation(
                    last_turn="",
                    turn_number=0,
                    available_actions=list(self.available_action_types)
                    if self.action_mask[i]
                    else ["none"],
                )

        return observations

    def _process_incoming_actions(
        self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
    ) -> dict[str, AgentAction]:
        """Normalize actions, apply mask, and record to history."""
        # Normalize actions to AgentAction objects
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

        if self.include_turn_marker:
            self.recv_message(
                "Environment", SimpleMessage(message=f"Turn #{self.turn_number}")
            )
        for agent, action in complied_actions.items():
            # Only record actions from agents that are in turn
            idx = self.agents.index(agent)
            if self.action_mask[idx]:
                self.recv_message(agent, action)

        return complied_actions

    async def _run_evaluators(self, evaluators: list[Evaluator]) -> Any:
        """Run evaluators and aggregate results."""
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

        complied_actions = self._process_incoming_actions(actions)

        # Sync evaluation (not refactored to helper as it's sync vs async)
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
                    "complete_rating": (
                        response.rewards.get(f"agent_{i+1}", (0, {}))[0]  # type: ignore[index]
                        if response.rewards
                        else (
                            (response.p1_rate if i == 0 else response.p2_rate)
                            if isinstance(response.p1_rate, (int, float))
                            and isinstance(response.p2_rate, (int, float))
                            else (
                                response.p1_rate[0]
                                if i == 0 and isinstance(response.p1_rate, tuple)
                                else (
                                    response.p2_rate[0]
                                    if i == 1 and isinstance(response.p2_rate, tuple)
                                    else 0
                                )
                            )
                        )
                    ),
                }
                for i, agent_name in enumerate(self.agents)
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

        complied_actions = self._process_incoming_actions(actions)

        response = await self._run_evaluators(self.evaluators)

        if response.terminated:
            terminal_response = await self._run_evaluators(self.terminal_evaluators)
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
        if response.terminated:
            pass

        info = {
            agent_name: {
                "comments": response.comments or "",
                "complete_rating": (
                    response.rewards.get(f"agent_{i+1}", (0, {}))[0]
                    if response.rewards
                    else (
                        (response.p1_rate if i == 0 else response.p2_rate)
                        if isinstance(response.p1_rate, (int, float))
                        and isinstance(response.p2_rate, (int, float))
                        else (
                            response.p1_rate[0]
                            if i == 0 and isinstance(response.p1_rate, tuple)
                            else (
                                response.p2_rate[0]
                                if i == 1 and isinstance(response.p2_rate, tuple)
                                else 0
                            )
                        )
                    )
                ),
            }
            for i, agent_name in enumerate(self.agents)
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
