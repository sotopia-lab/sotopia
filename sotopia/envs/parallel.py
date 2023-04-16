import logging
import random
from copy import deepcopy
from typing import Any, Literal, TypedDict

from beartype import beartype
from beartype.door import is_bearable
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.text import Text
from pettingzoo.utils.env import ParallelEnv

from sotopia.generation_utils.generate import (
    LLM_Name,
    generate_background,
    generate_environment_response,
    process_history,
)
from sotopia.messages import (
    ActionType,
    AgentAction,
    Message,
    Observation,
    ScriptBackground,
    SimpleMessage,
)

log = logging.getLogger("env")


def _actions_to_natural_language(actions: dict[str, AgentAction]) -> str:
    action_str = ""
    for agent, action in actions.items():
        action_str += f"{agent} {action.to_natural_language()};"
    return action_str


class ParallelSotopiaEnv(ParallelEnv):
    def __init__(
        self,
        available_action_types: set[ActionType] = set(
            ["none", "speak", "non-verbal communication", "action"]
        ),
        action_order: Literal[
            "simutaneous", "round-robin", "random"
        ] = "simutaneous",
        model_name: LLM_Name = "gpt-3.5-turbo",
    ) -> None:
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

        self.inbox: list[tuple[str, Message]] = []
        self.agents = []
        self.action_spaces = {}
        self.available_action_types = list(available_action_types)
        self.action_order = action_order
        self.action_mask: list[bool] = []

    def recv_message(self, source: str, message: Message) -> None:
        self.inbox.append((source, message))

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> dict[str, Observation]:
        self.background = generate_background(model_name=self.model_name)
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

        log.info(f"Turn {self.turn_number} begins")
        log.info(f"Background:\n{background_for_a.to_natural_language()}")
        log.info(f"Background:\n{background_for_b.to_natural_language()}")

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

        response = generate_environment_response(
            self.model_name,
            "\n".join(
                [
                    f"{x}: {y.to_natural_language()}"
                    if x != "Environment"
                    else y.to_natural_language()
                    for x, y in self.inbox
                ]
            ),
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
        log.info(f"Turn #{self.turn_number}:\n{obs}")
        log.info(
            f"Turn #{self.turn_number}:\n{response.to_natural_language()}"
        )
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
