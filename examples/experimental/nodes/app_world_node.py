import logging

from rich.logging import RichHandler

from aact import NodeFactory
from aact.nodes import RestAPINode


FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.WARNING,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)


# Maybe you need to make an AppWorldAction that inherits from a DataModelFactory


@NodeFactory.register(name="app_world")
class AppWorldController(RestAPINode):
    def __init___(
        self,
        appworld_url: str,
        input_channel: str,
        output_channel: str,
        input_type_str: str,
        output_type_str: str,
        node_name: str,
        redis_url: str,
    ):
        super().__init__(
            input_channel=input_channel,
            output_channel=output_channel,
            input_type_str=input_type_str,
            output_type_str=output_type_str,
            node_name=node_name,
            redis_url=redis_url,
        )

        self.appworld_url = appworld_url
