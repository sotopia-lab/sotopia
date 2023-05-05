import uuid
from typing import Generic, TypeVar, cast

from redis_om.model.model import NotFoundError

from sotopia.database import AgentProfile
from sotopia.messages import Message, MessengerMixin

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseAgent(Generic[ObsType, ActType], MessengerMixin):
    def __init__(
        self, agent_name: str | None = None, uuid_str: str | None = None
    ) -> None:
        MessengerMixin.__init__(self)
        if uuid_str is not None:
            # try retrieving profile from database
            try:
                self.profile = AgentProfile.get(pk=uuid_str)
            except NotFoundError:
                raise ValueError(
                    f"Agent with uuid {uuid_str} not found in database"
                )
            self.agent_name = (
                self.profile.first_name + " " + self.profile.last_name
            )
        else:
            assert (
                agent_name is not None
            ), "Either agent_name or uuid_str must be provided"
            self.agent_name = agent_name

    def act(self, obs: ObsType) -> ActType:
        raise NotImplementedError

    async def aact(self, obs: ObsType) -> ActType:
        raise NotImplementedError

    def reset(self) -> None:
        self.reset_inbox()
