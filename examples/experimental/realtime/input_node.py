import sys
from typing import AsyncIterator

from sotopia.agents.llm_agent import ainput

if sys.version_info < (3, 11):
    pass
else:
    pass
from aact import Message, Node, NodeFactory
from aact.messages import Text, Zero


@NodeFactory.register("input")
class InputNode(Node[Zero, Text]):
    def __init__(self, output_channel: str, redis_url: str) -> None:
        super().__init__(
            input_channel_types=[],
            output_channel_types=[(output_channel, Text)],
            redis_url=redis_url,
        )
        self.output_channel = output_channel

    async def event_loop(self) -> None:
        while True:
            text = await ainput("Enter text: ")
            await self.r.publish(
                self.output_channel,
                Message[Text](data=Text(text=text)).model_dump_json(),
            )

    async def event_handler(
        self, _: str, __: Message[Zero]
    ) -> AsyncIterator[tuple[str, Message[Text]]]:
        yield self.output_channel, Message[Text](data=Text(text=""))
