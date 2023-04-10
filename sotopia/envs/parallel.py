from copy import deepcopy
from typing import Any, TypedDict

from beartype import beartype
from beartype.door import is_bearable
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.text import Text
from pettingzoo.utils.env import ParallelEnv

from sotopia.generation_utils.generate import (
    ActionType,
    AgentAction,
    LLM_Name,
    ScriptBackground,
    generate_background,
    generate_environment_response,
    process_history,
)


class Observation(TypedDict):
    history: str
    turn_number: int
    available_actions: list[str]


class ParallelSotopiaEnv(ParallelEnv):
    def __init__(
        self,
        available_action_types: set[ActionType] = set(
            ["none", "speak", "non-verbal communication", "action"]
        ),
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

        self.history: list[dict[str, str]] = []
        self.agents = []
        self.action_spaces = {}
        self.available_action_types = list(available_action_types)

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
        return {
            self.background.p1_name: Observation(
                history=process_history(background_for_a),
                turn_number=0,
                available_actions=list(self.available_action_types),
            ),
            self.background.p2_name: Observation(
                history=process_history(background_for_b),
                turn_number=0,
                available_actions=list(self.available_action_types),
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
        self.turn_number += 1
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
        response = generate_environment_response(
            self.model_name,
            str(self.background)
            + "\n"
            + "\n".join([str(x) for x in self.history]),
            complied_actions,
        )
        obs = process_history(complied_actions)
        return (
            {
                self.background.p1_name: Observation(
                    history=obs,
                    turn_number=self.turn_number,
                    available_actions=list(self.available_action_types),
                ),
                self.background.p2_name: Observation(
                    history=obs,
                    turn_number=self.turn_number,
                    available_actions=list(self.available_action_types),
                ),
            },
            {
                self.background.p1_name: response.p1_rate or 0,
                self.background.p2_name: response.p2_rate or 0,
            },
            {
                self.background.p1_name: response.terminate,
                self.background.p2_name: response.terminate,
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
