from aact import NodeFactory
from aact.messages import Text, Tick
from sotopia.experimental.agents.base_agent import BaseAgent


@NodeFactory.register("simple_echo_agent")
class SimpleEchoAgent(BaseAgent[Text, Text]):
    def __init__(self, input_channel: str, output_channel: str, redis_url: str) -> None:
        super().__init__(
            input_channel_types=[(input_channel, Text)],
            output_channel_types=[(output_channel, Text)],
        )

    async def aact(self, observation: Text) -> Text:
        return Text(text=f"Hello, {observation.text}!")


@NodeFactory.register("simple_tick_agent")
class SimpleTickAgent(BaseAgent[Tick, Text]):
    def __init__(self, input_channel: str, output_channel: str, redis_url: str) -> None:
        super().__init__(
            input_channel_types=[(input_channel, Tick)],
            output_channel_types=[(output_channel, Text)],
        )
        self.output_channel = output_channel

    async def aact(self, observation: Tick) -> Text:
        return Text(text=f"Tick {observation.tick}")
