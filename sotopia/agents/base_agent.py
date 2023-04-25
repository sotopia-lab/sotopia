from typing import Generic, TypeVar

from sotopia.messages import Message

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseAgent(Generic[ObsType, ActType]):
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.inbox: list[tuple[str, Message]] = []
        pass

    def act(self, obs: ObsType) -> ActType:
        raise NotImplementedError

    def reset(self) -> None:
        self.inbox = []

    def recv_message(self, source: str, message: Message) -> None:
        self.inbox.append((source, message))
