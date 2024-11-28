import asyncio
import sys


if sys.version_info < (3, 11):
    pass
else:
    pass


from aact import Message, NodeFactory
from aact.messages import DataModel, DataModelFactory

from typing import Literal
from pydantic import Field


from .base_agent import BaseAgent
from .datamodels import AgentAction, Observation
from sotopia.messages import ActionType


@DataModelFactory.register("observations")
class Observations(DataModel):
    observations_map: dict[str, Observation] = Field(
        description="the observations of the agents"
    )


@NodeFactory.register("moderator")
class Moderator(BaseAgent[AgentAction, Observations]):
    def __init__(
        self,
        input_channels: list[str],
        output_channels: list[str],
        agent_mapping: dict[str, str],
        redis_url: str = "redis://localhost:6379/0",
        action_order: Literal["simultaneous", "round-robin", "random"] = "round-robin",
        available_actions: list[ActionType] = [
            "none",
            "speak",
            "non-verbal communication",
            "action",
            "leave",
        ],
    ):
        super().__init__(
            input_channel_types=[
                (input_channel, AgentAction) for input_channel in input_channels
            ],
            output_channel_types=[
                (output_channel, Observations) for output_channel in output_channels
            ],
            redis_url=redis_url,
        )
        self.observation_queue: asyncio.Queue[AgentAction] = asyncio.Queue()
        self.task_scheduler: asyncio.Task[None] | None = None
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.agent_mapping: dict[str, str] = agent_mapping
        self.action_order: Literal["simultaneous", "round-robin", "random"] = (
            action_order
        )
        self.available_actions: list[ActionType] = available_actions
        self.turn_number: int = 0
        self.current_agent_index: int = 0
        self.agents: list[str] = list(agent_mapping.values())

    async def send(self, action: Observations) -> None:
        for output_channel, output_channel_type in self.output_channel_types.items():
            await self.r.publish(
                output_channel,
                Message[output_channel_type](
                    data=action.observations_map[output_channel]
                ).model_dump_json(),  # type:ignore[valid-type]
            )

    async def aact(self, agent_action: AgentAction) -> Observations | None:
        self.turn_number += 1
        observations_map: dict[str, Observation] = {}
        for output_channel, output_channel_type in self.output_channel_types.items():
            agent_name = self.agent_mapping[output_channel]
            available_actions: list[ActionType] = ["none"]
            if self.action_order == "round-robin":
                if agent_name == self.agents[self.current_agent_index]:
                    available_actions = self.available_actions
                    self.current_agent_index = (self.current_agent_index + 1) % len(
                        self.agents
                    )

            observation = Observation(
                agent_name=agent_name,
                last_turn=agent_action.to_natural_language(),
                turn_number=self.turn_number,
                available_actions=available_actions,
            )
            observations_map[output_channel] = observation

        return Observations(observations_map=observations_map)
