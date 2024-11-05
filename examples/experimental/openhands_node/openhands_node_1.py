import asyncio
from openhands.core.config import (
    AgentConfig,
    AppConfig,
    SandboxConfig,
)
from datetime import datetime
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime
from openhands.events.action import BrowseURLAction, FileReadAction, FileWriteAction, CmdRunAction
from openhands.events.observation import BrowserOutputObservation

from openhands.utils.async_utils import call_async_from_sync
from openhands.runtime.base import Runtime
from typing import Any, AsyncIterator
from aact import Message, NodeFactory, Node
from aact.messages import Text, DataModel, Zero, Message
from aact.messages.registry import DataModelFactory

from aact.messages.commons import DataEntry
from aiofiles.threadpool.text import AsyncTextIOWrapper
from aiofiles.base import AiofilesContextManager
from aiofiles import open

import logging
from rich.logging import RichHandler
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
from sotopia.messages.message_classes import ActionType

from pydantic import Field

@DataModelFactory.register("agent_action")
class AgentAction(DataModel):
    agent_name: str = Field(description="the name of the agent")
    action_type: ActionType = Field(
        description="whether to speak at this turn or choose to not do anything"
    )
    argument: str = Field(
        description="the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action"
    )
    path: str | None = Field(
        description="path of file"
    )

    def to_natural_language(self) -> str:
        match self.action_type:
            case "none":
                return "did nothing"
            case "speak":
                return f'said: "{self.argument}"'
            case "thought":
                return f'thought: "{self.argument}"'
            case "browse":
                return f'browsed: "{self.argument}"'
            case "non-verbal communication":
                return f"[{self.action_type}] {self.argument}"
            case "action":
                return f"[{self.action_type}] {self.argument}"
            case "leave":
                return "left the conversation"
            
@NodeFactory.register("openhands")
class OpenHands(Node[DataModel, Text]):
    def __init__(
        self,
        record_channel_types: dict[str, str],
        output_channels: list[str],
        jsonl_file_path: str,
        redis_url: str,
        add_datetime: bool = False,
    ):
        input_channel_types: list[tuple[str, type[DataModel]]] = []
        
        for channel, channel_type_string in record_channel_types.items():
            input_channel_types.append(
                (channel, DataModelFactory.registry[channel_type_string])
            )
        if add_datetime:
            # add a datetime to jsonl_file_path before the extension. The file can have any extension.
            jsonl_file_path = (
                jsonl_file_path[: jsonl_file_path.rfind(".")]
                + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
                + jsonl_file_path[jsonl_file_path.rfind(".") :]
            )

        super().__init__(
            input_channel_types=input_channel_types,
            output_channel_types=[
                (output_channel, Text) for output_channel in output_channels
            ],
            redis_url=redis_url,
        )
        self.jsonl_file_path = jsonl_file_path
        self.aioContextManager: AiofilesContextManager[AsyncTextIOWrapper] | None = None
        self.json_file: AsyncTextIOWrapper | None = None
        self.write_queue: asyncio.Queue[DataEntry[DataModel]] = asyncio.Queue()
        self.write_task: asyncio.Task[None] | None = None
        self.runtime: Optional[Runtime] = None


    async def init_runtime(self) -> None:
        config = AppConfig(
            default_agent="CodeActAgent",
            run_as_openhands=False,
            max_iterations=3,
            runtime="modal",
            modal_api_token_id=os.environ.get("MODAL_API_TOKEN_ID", ""),
            modal_api_token_secret=os.environ.get("MODAL_API_TOKEN_SECRET", ""),
            sandbox=SandboxConfig(
                base_container_image=base_container_image,
                enable_auto_lint=True,
                use_host_network=False,
                # large enough timeout, since some testcases take very long to run
                timeout=300,
                # Add platform to the sandbox config to solve issue 4401
                platform="linux/amd64",
                api_key=os.environ.get("ALLHANDS_API_KEY", None),
                remote_runtime_api_url=os.environ.get(
                    "SANDBOX_REMOTE_RUNTIME_API_URL", ""
                ),
                keep_remote_runtime_alive=False,
            ),
            # do not mount workspace
            workspace_base=None,
            workspace_mount_path=None,
        )
        agent_config = AgentConfig(
            codeact_enable_jupyter=True,
            codeact_enable_browsing=True,
            codeact_enable_llm_editor=True,
        )
        config.set_agent_config(agent_config)
        self.runtime = create_runtime(config)
        if self.runtime:
            call_async_from_sync(self.runtime.connect)

            logger.info("-" * 30)
            logger.info("BEGIN Runtime Initialization Fn")
            logger.info("-" * 30)
        return
    
    
    
    async def __aenter__(self) -> Self:
        self.task = asyncio.create_task(self.init_runtime())
        self.aioContextManager = open(self.jsonl_file_path, "w")
        self.json_file = await self.aioContextManager.__aenter__()
        self.write_task = asyncio.create_task(self.write_to_file())
        return await super().__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.runtime:
            self.runtime.close()
        if self.aioContextManager:
            await self.aioContextManager.__aexit__(exc_type, exc_value, traceback)
        del self.json_file
        return await super().__aexit__(exc_type, exc_value, traceback)

    async def aact(self, observation: AgentAction) -> Text | None:
        logger.info("Entering aact")
        if self.runtime:
            logger.info("Running aact")
            print("observation: ", observation)
            print("observation.data: ", observation.data)
            if observation.data.action_type == "browse":
                action = BrowseURLAction(url=observation.data.argument)
            elif observation.data.action_type == "run":
                action = CmdRunAction(command=observation.data.argument)
            elif observation.data.action_type == "write":
                action = FileWriteAction(path=observation.data.path, content=observation.data.argument)
            elif observation.data.action_type == "read":
                action = FileWriteAction(path=observation.data.path)
            action.timeout = 600
            logger.info(action, extra={"msg_type": "ACTION"})
            obs = self.runtime.run_action(action)
            logger.info(obs, extra={"msg_type": "OBSERVATION"})
            # if isinstance(obs, BrowserOutputObservation):
            #     obs = obs.get_agent_obs_text()
            #     print("obs: ", obs)   
            return Text(text=str(obs))
        return None
    
    async def send(self, action: Text) -> None:
        for output_channel, output_channel_type in self.output_channel_types.items():
            await self.r.publish(
                output_channel,
                Message[output_channel_type](data=action).model_dump_json(),  # type: ignore[valid-type]
            )
            
    async def write_to_file(self) -> None:
        while self.json_file:
            logger.info("Entering loop")
            data_entry = await self.write_queue.get()
            action_or_none = await self.aact(data_entry)
            if action_or_none is not None:
                print("Calling send")
                await self.send(action_or_none)
            await self.json_file.write(data_entry.model_dump_json() + "\n")
            await self.json_file.flush()
            self.write_queue.task_done()

    async def event_handler(
        self, input_channel: str, input_message: Message[DataModel]
    ) -> AsyncIterator[tuple[str, Message[Zero]]]:
        if input_channel in self.input_channel_types:
            await self.write_queue.put(
                DataEntry[self.input_channel_types[input_channel]](  # type: ignore[name-defined]
                    channel=input_channel, data=input_message.data
                )
            )
        else:
            yield input_channel, Message[Zero](data=Zero())
