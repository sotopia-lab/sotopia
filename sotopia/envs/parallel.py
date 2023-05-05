import asyncio
import json
import logging
import random
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from beartype import beartype
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.text import Text
from pettingzoo.utils.env import ParallelEnv
from redis_om.model.model import NotFoundError

from sotopia.database import EnvironmentProfile
from sotopia.generation_utils import (
    LLM_Name,
    fill_in_background,
    generate_scenario_background,
)
from sotopia.messages import (
    ActionType,
    AgentAction,
    Message,
    MessengerMixin,
    Observation,
    ScriptBackground,
    SimpleMessage,
)

from .evaluators import (
    Evaluator,
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    unweighted_aggregate_evaluate,
)


def _actions_to_natural_language(actions: dict[str, AgentAction]) -> str:
    action_str = ""
    for agent, action in actions.items():
        # Only record actions that did something
        if action.action_type != "none":
            if action_str != "":
                action_str += ";"  # separate actions with semicolon
            action_str += f"{agent} {action.to_natural_language()}"
    return action_str


class ParallelSotopiaEnv(ParallelEnv, MessengerMixin):
    def __init__(
        self,
        available_action_types: set[ActionType] = set(
            ["none", "speak", "non-verbal communication", "action", "leave"]
        ),
        action_order: Literal[
            "simutaneous", "round-robin", "random"
        ] = "simutaneous",
        model_name: LLM_Name = "gpt-3.5-turbo",
        evaluators: list[Evaluator] = [],
        uuid_str: str | None = None,
    ) -> None:
        """A sotopia environment for parallel agents.

        Args:
            available_action_types (set[ActionType], optional): The action types that are available to the agents. Defaults to set(["none", "speak", "non-verbal communication", "action"]).
            action_order (Literal["simutaneous", "round-robin", "random"], optional): The order in which the agents take actions. Defaults to "simutaneous".
            model_name (LLM_Name, optional): The name of the language model to use. Defaults to "gpt-3.5-turbo".
        """
        super().__init__()
        self.model_name = model_name
        self.background = ScriptBackground(
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

        # if a uuid is provided, try to load the environment profile from the database
        if uuid_str is not None:
            # try retrieving profile from database
            try:
                self.profile = EnvironmentProfile.get(pk=uuid_str)
            except NotFoundError:
                raise ValueError(
                    f"Agent with uuid {uuid_str} not found in database"
                )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> dict[str, Observation]:
        """Starting a new episode. Must be called before step().

        Args:
            seed (int, optional): Seed for the environment. Defaults to None. Not used right now.
            options (dict, optional): Options for the environment. Defaults to None.
                "partial_background_file" (str): Path to a json file which need to contain a ScriptBackground object. The backgound can be incompleted ("unknown" for missing parts), and the missing parts will be filled in by the environment.
                "full_background_file" (str): Path to a json file which need to contain a ScriptBackground object. The backgound must be completed (no "unknown" for missing parts).
        """
        super().__init__()
        MessengerMixin.reset_inbox(self)
        if options and "partial_background_file" in options:
            # load pydantic background from json file
            self.background = ScriptBackground.parse_file(
                Path(options["partial_background_file"])
            )
            self.background = fill_in_background(
                self.model_name, self.background
            )
        elif options and "full_background_file" in options:
            self.background = ScriptBackground.parse_file(
                Path(options["full_background_file"])
            )
        else:
            self.background = generate_scenario_background(
                model_name=self.model_name
            )
        background_for_a = deepcopy(self.background)
        background_for_b = deepcopy(self.background)
        background_for_a.p2_goal = "Unknown"
        background_for_b.p1_goal = "Unknown"
        self.agents = [self.background.p1_name, self.background.p2_name]
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
            self.action_mask[
                random.randint(0, len(self.action_mask) - 1)
            ] = True
        else:
            self.action_mask = [True for _ in self.agents]

        self.recv_message("Environment", self.background)

        return {
            self.background.p1_name: Observation(
                last_turn=background_for_a.to_natural_language(),
                turn_number=0,
                available_actions=list(self.available_action_types)
                if self.action_mask[0]
                else ["none"],
            ),
            self.background.p2_name: Observation(
                last_turn=background_for_b.to_natural_language(),
                turn_number=0,
                available_actions=list(self.available_action_types)
                if self.action_mask[1]
                else ["none"],
            ),
        }

    @beartype
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
                complied_actions[agent] = AgentAction(
                    action_type="none", argument=""
                )

        self.recv_message(
            "Environment", SimpleMessage(message=f"Turn #{self.turn_number}")
        )
        for agent, action in complied_actions.items():
            self.recv_message(agent, action)

        response = unweighted_aggregate_evaluate(
            [
                evaluator(turn_number=self.turn_number, messages=self.inbox)
                for evaluator in self.evaluators
            ]
        )

        self.action_mask = [False for _ in self.agents]
        if self.action_order == "round-robin":
            self.action_mask[self.turn_number % len(self.action_mask)] = True
        elif self.action_order == "random":
            self.action_mask[
                random.randint(0, len(self.action_mask) - 1)
            ] = True
        else:
            self.action_mask = [True for _ in self.agents]
        obs = _actions_to_natural_language(complied_actions)
        return (
            {
                self.background.p1_name: Observation(
                    last_turn=obs,
                    turn_number=self.turn_number,
                    available_actions=list(self.available_action_types)
                    if self.action_mask[0]
                    else ["none"],
                ),
                self.background.p2_name: Observation(
                    last_turn=obs,
                    turn_number=self.turn_number,
                    available_actions=list(self.available_action_types)
                    if self.action_mask[1]
                    else ["none"],
                ),
            },
            {
                self.background.p1_name: response.p1_rate or 0,
                self.background.p2_name: response.p2_rate or 0,
            },
            {
                self.background.p1_name: response.terminated,
                self.background.p2_name: response.terminated,
            },
            {
                self.background.p1_name: False,
                self.background.p2_name: False,
            },
            {
                self.background.p1_name: {},
                self.background.p2_name: {},
            },
        )

    @beartype
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
                complied_actions[agent] = AgentAction(
                    action_type="none", argument=""
                )

        self.recv_message(
            "Environment", SimpleMessage(message=f"Turn #{self.turn_number}")
        )
        for agent, action in complied_actions.items():
            self.recv_message(agent, action)

        response = unweighted_aggregate_evaluate(
            await asyncio.gather(
                *[
                    evaluator.__acall__(
                        turn_number=self.turn_number, messages=self.inbox
                    )
                    for evaluator in self.evaluators
                ]
            )
        )

        self.action_mask = [False for _ in self.agents]
        if self.action_order == "round-robin":
            self.action_mask[self.turn_number % len(self.action_mask)] = True
        elif self.action_order == "random":
            self.action_mask[
                random.randint(0, len(self.action_mask) - 1)
            ] = True
        else:
            self.action_mask = [True for _ in self.agents]
        obs = _actions_to_natural_language(complied_actions)
        return (
            {
                self.background.p1_name: Observation(
                    last_turn=obs,
                    turn_number=self.turn_number,
                    available_actions=list(self.available_action_types)
                    if self.action_mask[0]
                    else ["none"],
                ),
                self.background.p2_name: Observation(
                    last_turn=obs,
                    turn_number=self.turn_number,
                    available_actions=list(self.available_action_types)
                    if self.action_mask[1]
                    else ["none"],
                ),
            },
            {
                self.background.p1_name: response.p1_rate or 0,
                self.background.p2_name: response.p2_rate or 0,
            },
            {
                self.background.p1_name: response.terminated,
                self.background.p2_name: response.terminated,
            },
            {
                self.background.p1_name: False,
                self.background.p2_name: False,
            },
            {
                self.background.p1_name: {},
                self.background.p2_name: {},
            },
        )

    def render(self, mode: str = "human") -> None:
        pass

    def close(self) -> None:
        pass
