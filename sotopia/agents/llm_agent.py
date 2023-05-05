import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from sotopia.agents import BaseAgent
from sotopia.generation_utils.generate import (
    LLM_Name,
    agenerate_action,
    generate_action,
    generate_action_speak,
    generate_goal,
)
from sotopia.messages import AgentAction, Message, Observation


async def ainput(prompt: str = "") -> str:
    with ThreadPoolExecutor(1, "ainput") as executor:
        return (
            await asyncio.get_event_loop().run_in_executor(
                executor, input, prompt
            )
        ).rstrip()


class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        model_name: LLM_Name = "gpt-3.5-turbo",
    ) -> None:
        super().__init__(agent_name=agent_name, uuid_str=uuid_str)
        self.model_name = model_name

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        goal = generate_goal(
            self.model_name,
            background=self.inbox[0][
                1
            ].scenario,  # Only consider the first message for now
        )
        return goal

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(
        self,
        obs: Observation,
        gen_func: Callable[..., AgentAction] = generate_action,
    ) -> AgentAction:
        self.recv_message("Environment", obs)

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="")
        else:
            action = gen_func(
                self.model_name,
                history="\n".join(
                    f"{y.to_natural_language()}" for x, y in self.inbox
                ),
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=self.agent_name,
                goal=self.goal,
            )
            return action

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="")
        else:
            action = await agenerate_action(
                self.model_name,
                history="\n".join(
                    f"{y.to_natural_language()}" for x, y in self.inbox
                ),
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=self.agent_name,
                goal=self.goal,
            )
            return action


class SpeakAgent(LLMAgent):
    def act(
        self,
        obs: Observation,
        gen_func: Callable[..., AgentAction] = generate_action_speak,
    ) -> AgentAction:
        return super().act(obs, gen_func=gen_func)


class HumanAgent(BaseAgent[Observation, AgentAction]):
    """
    A human agent that takes input from the command line.
    """

    def __init__(self, agent_name: str) -> None:
        super().__init__(agent_name=agent_name)

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        goal = input("Goal: ")
        return goal

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        print("Available actions:")
        for i, action in enumerate(obs.available_actions):
            print(f"{i}: {action}")

        action_type = obs.available_actions[int(input("Action type: "))]
        argument = input("Argument: ")

        return AgentAction(action_type=action_type, argument=argument)

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        print("Available actions:")
        for i, action in enumerate(obs.available_actions):
            print(f"{i}: {action}")

        if obs.available_actions != ["none"]:
            action_type = obs.available_actions[
                int(await ainput("Action type: "))
            ]
        else:
            action_type = "none"
        if action_type in ["speak", "non-verbal communication"]:
            argument = await ainput("Argument: ")
        else:
            argument = ""

        return AgentAction(action_type=action_type, argument=argument)


class Agents(dict[str, LLMAgent | HumanAgent | SpeakAgent]):
    def reset(self) -> None:
        for agent in self.values():
            agent.reset()

    def act(self, obs: dict[str, Observation]) -> dict[str, AgentAction]:
        return {
            agent_name: agent.act(obs[agent_name])
            for agent_name, agent in self.items()
        }
