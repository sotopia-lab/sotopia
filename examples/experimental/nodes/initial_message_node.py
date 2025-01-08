import sys

from typing import AsyncIterator

from aact import Message, NodeFactory, Node
from aact.messages import Text, Tick, DataModel

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@NodeFactory.register("initial_message")
class InitialMessageNode(Node[DataModel, Text]):
    def __init__(
        self,
        input_tick_channel: str,
        output_channels: list[str],
        env_scenario: str,
        node_name: str,
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(
            input_channel_types=[(input_tick_channel, Tick)],
            output_channel_types=[
                (output_channel, Text) for output_channel in output_channels
            ],
            redis_url=redis_url,
            node_name=node_name,
        )
        self.env_scenario = env_scenario
        self.output_channels = output_channels

    async def send_env_scenario(self) -> None:
        for output_channel in self.output_channels:
            await self.r.publish(
                output_channel,
                Message[Text](data=Text(text=self.env_scenario)).model_dump_json(),
            )

    async def event_loop(self) -> None:
        await self.send_env_scenario()

    async def __aenter__(self) -> Self:
        return await super().__aenter__()

    async def event_handler(
        self, _: str, __: Message[DataModel]
    ) -> AsyncIterator[tuple[str, Message[Text]]]:
        raise NotImplementedError("ScenarioContext does not have an event handler.")
        yield "", Message[Text](data=Text(text=self.env_scenario))
