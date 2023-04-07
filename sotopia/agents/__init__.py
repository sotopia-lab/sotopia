from typing import Generic, TypeVar

from .llm_agent import Agents, LLMAgent

__all__ = ["BaseAgent", "LLMAgent", "Agents"]

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseAgent(Generic[ObsType, ActType]):
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        pass

    def act(self, obs: ObsType) -> ActType:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
