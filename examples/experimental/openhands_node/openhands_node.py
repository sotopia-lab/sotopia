import asyncio
from openhands.core.config import (
    AgentConfig,
    AppConfig,
    SandboxConfig,
    get_llm_config_arg,
    get_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction, BrowseURLAction

from openhands.utils.async_utils import call_async_from_sync
from openhands.runtime.base import Runtime
from typing import Any, AsyncIterator, TypeVar
from aact import Message, NodeFactory, Node
from aact.messages import Text, Tick, DataModel, DataModelFactory, Zero
from sotopia.agents.llm_agent import ainput
from sotopia.experimental.agents import BaseAgent
from aact.nodes import PrintNode

from sotopia.generation_utils import agenerate
from sotopia.generation_utils.generate import StrOutputParser
from sotopia.messages import ActionType

from pydantic import Field
import logging
from rich.logging import RichHandler
import json
import os
import sys
from typing import Optional

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.WARNING,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

base_container_image = "docker.all-hands.dev/all-hands-ai/runtime:0.11-nikolaik"


@NodeFactory.register("openhands_node")
class OpenHandsNode(Node[Text, Text]):
    def __init__(
        self,
        input_channels: list[str],
        output_channels: list[str],
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(
            input_channel_types=[(input_channel, Text) for input_channel in input_channels],
            output_channel_types=[(output_channel, Text) for output_channel in output_channels],
            redis_url=redis_url,
        )
        self.observation_queue: asyncio.Queue[Text] = asyncio.Queue()
        self.runtime: Optional[Runtime] = None
        self.task_scheduler: asyncio.Task[None] | None = None
        self.shutdown_event: asyncio.Event = asyncio.Event()

    
    async def init_runtime(self) -> None:
        config = AppConfig(
            default_agent="CodeActAgent",
            run_as_openhands=False,
            max_iterations=3,
            runtime='modal',
            modal_api_token_id = os.environ.get('MODAL_API_TOKEN_ID', ""),
            modal_api_token_secret = os.environ.get('MODAL_API_TOKEN_SECRET', ""),
            sandbox=SandboxConfig(
                base_container_image=base_container_image,
                enable_auto_lint=True,
                use_host_network=False,
                # large enough timeout, since some testcases take very long to run
                timeout=300,
                # Add platform to the sandbox config to solve issue 4401
                platform='linux/amd64',
                api_key=os.environ.get('ALLHANDS_API_KEY', None),
                remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL', ""),
                keep_remote_runtime_alive=False,
            ),
            # do not mount workspace
            workspace_base=None,
            workspace_mount_path=None,
        )        
        agent_config = AgentConfig(
            codeact_enable_jupyter=True,
            codeact_enable_browsing_delegate=True,
            codeact_enable_llm_editor=True,
        )
        config.set_agent_config(agent_config)
        self.runtime = create_runtime(config)
        if self.runtime:
            call_async_from_sync(self.runtime.connect)

            logger.info('-' * 30)
            logger.info('BEGIN Runtime Initialization Fn')
            logger.info('-' * 30)

    async def __aenter__(self) -> Self:
        self.task = asyncio.create_task(self.init_runtime())
        self.task_scheduler = asyncio.create_task(self._task_scheduler())
        logger.info("Started runtime")
        return await super().__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.runtime:
            self.runtime.close()
        self.shutdown_event.set()
        if self.task_scheduler is not None:
            self.task_scheduler.cancel()
        logger.info("Closed runtime")
        return await super().__aexit__(exc_type, exc_value, traceback)
    
    async def aact(self, observation: Text) -> Text | None:
        if self.runtime:
            logger.info("Running aact")
            action = BrowseURLAction(url=observation.text)
            action.timeout = 600
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = self.runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            return Text(text="testing")
        return None

    async def event_handler(
        self, channel: str, message: Message[Text]
    ) -> AsyncIterator[tuple[str, Message[Text]]]:
        logger.info(channel)
        logger.info(message)
        if channel in self.input_channel_types:
            await self.observation_queue.put(message.data)
        else:
            raise ValueError(f"Invalid channel: {channel}")
            yield "", self.output_type()
        print("self.observation_queue: ", self.observation_queue) 
        
    async def send(self, action: Text) -> None:
        print("Sending action: ", action)
        for output_channel, output_channel_type in self.output_channel_types.items():
            await self.r.publish(
                output_channel,
                Message[output_channel_type](data=action).model_dump_json(),  # type: ignore[valid-type]
            )

    async def _task_scheduler(self) -> None:
        logger.info("_task_scheduler")
        
        while not self.shutdown_event.is_set():
            observation = await self.observation_queue.get()
            logger.info("observation: ", observation)
            action_or_none = await self.aact(observation)
            logger.info("action_or_none: ", action_or_none)
            if action_or_none is not None:
                await self.send(action_or_none)
            self.observation_queue.task_done()









# base_container_image = "nikolaik/python-nodejs:python3.12-nodejs22"




    
# while True:
#     user_command = input("Enter a command to run (or 'exit' to quit): ")
#     if user_command.lower() == 'exit':
#         break

#     action = CmdRunAction(command=user_command)
#     action.timeout = 600
#     logger.info(action, extra={'msg_type': 'ACTION'})
#     obs = runtime.run_action(action)
#     logger.info(obs, extra={'msg_type': 'OBSERVATION'})
