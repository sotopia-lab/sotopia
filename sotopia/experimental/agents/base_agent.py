import asyncio
import sys


if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from typing import Any, AsyncIterator, TypeVar

from aact import Message, Node
from aact.messages import DataModel
import logging
from rich.logging import RichHandler


T_agent_observation = TypeVar("T_agent_observation", bound=DataModel)
T_agent_action = TypeVar("T_agent_action", bound=DataModel)

log = logging.getLogger("sotopia.redis_agent")
log.setLevel(logging.INFO)
# Prevent propagation to root logger
log.propagate = False
log.addHandler(RichHandler(rich_tracebacks=True, show_time=True))


class BaseAgent(Node[T_agent_observation, T_agent_action]):
    def __init__(
        self,
        input_channel_types: list[tuple[str, type[T_agent_observation]]],
        output_channel_types: list[tuple[str, type[T_agent_action]]],
        redis_url: str = "redis://localhost:6379/0",
        node_name: str = "base_agent",
    ):
        super().__init__(
            input_channel_types=input_channel_types,
            output_channel_types=output_channel_types,
            redis_url=redis_url,
            node_name=node_name,
        )

        self.observation_queue: asyncio.Queue[T_agent_observation] = asyncio.Queue()
        self.task_scheduler: asyncio.Task[None] | None = None
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.a = node_name

    async def __aenter__(self) -> Self:
        self.task_scheduler = asyncio.create_task(self._task_scheduler())
        return await super().__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.shutdown_event.set()
        if self.task_scheduler is not None:
            self.task_scheduler.cancel()
        return await super().__aexit__(exc_type, exc_value, traceback)

    async def aact(self, observation: T_agent_observation) -> T_agent_action | None:
        raise NotImplementedError

    async def event_handler(
        self, channel: str, message: Message[T_agent_observation]
    ) -> AsyncIterator[tuple[str, Message[T_agent_action]]]:
        if channel in self.input_channel_types:
            log.info(f"Inside event_handler: {self.a}")
            await self.observation_queue.put(message.data)
            log.info(f"Put in queue: {self.a}")
        else:
            raise ValueError(f"Invalid channel: {channel}")
            yield "", self.output_type()

    async def send(self, action: T_agent_action) -> None:
        for output_channel, output_channel_type in self.output_channel_types.items():
            await self.r.publish(
                output_channel,
                Message[output_channel_type](data=action).model_dump_json(),  # type:ignore[valid-type]
            )

    async def _task_scheduler(self) -> None:
        while not self.shutdown_event.is_set():
            log.info(f"Inside task_scheduler: {self.a}")
            observation = await self.observation_queue.get()
            log.info(f"Observation in task_scheduler in {self.a}: {observation}")
            action_or_none = await self.aact(observation)
            log.info(f"Completed aact: {self.a}")
            if action_or_none is not None:
                await self.send(action_or_none)
            log.info(f"Completed send: {self.a}")
            self.observation_queue.task_done()
