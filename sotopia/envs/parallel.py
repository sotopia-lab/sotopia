from copy import deepcopy
from typing import Any

from gymnasium.spaces.text import Text
from pettingzoo.utils.env import ParallelEnv

from sotopia.generation_utils.generate import (
    LLM_Name,
    ScriptBackground,
    generate_background,
    generate_environment_response,
)


class ParallelSotopiaEnv(ParallelEnv):
    def __init__(self, model_name: LLM_Name = "gpt-3.5-turbo") -> None:
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

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> dict[str, str]:
        self.background = generate_background(model_name=self.model_name)
        background_for_a = deepcopy(self.background)
        background_for_b = deepcopy(self.background)
        background_for_a.p2_goal = "Unknown"
        background_for_b.p1_goal = "Unknown"
        self.agents = [self.background.p1_name, self.background.p2_name]
        self.action_spaces = {agent: Text(256) for agent in self.agents}
        return {
            self.background.p1_name: str(background_for_a),
            self.background.p2_name: str(background_for_b),
        }

    def step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, str],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        response = generate_environment_response(
            self.model_name,
            str(self.background)
            + "\n"
            + "\n".join([str(x) for x in self.history]),
            action,
        )
        obs = str(action)
        return (
            {
                self.background.p1_name: obs,
                self.background.p2_name: obs,
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
