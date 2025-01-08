from aact import NodeFactory, Node
from .logs import EpisodeLog
from .datamodels import AgentAction, Observation


@NodeFactory.register("evaluator")
class Evaluator(Node[AgentAction, Observation]):
    def __init__(
        self,
        node_name: str,
        input_channels: list[str],
        output_channels: list[str],
        redis_url: str,
    ):
        super().__init__(
            input_channel_types=[
                (input_channel, AgentAction) for input_channel in input_channels
            ],
            output_channel_types=[
                (output_channel, Observation) for output_channel in output_channels
            ],
            node_name=node_name,
            redis_url=redis_url,
        )

    async def aevaluate(self, episode: EpisodeLog) -> AgentAction | None:
        raise NotImplementedError
