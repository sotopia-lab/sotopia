import asyncio
import logging
import time
import os
import sys
from enum import Enum
from typing import Any, AsyncIterator, Optional, Literal

from rich.logging import RichHandler

from pydantic import Field

from aact import Message, NodeFactory, Node
from aact.messages import Text, DataModel, Zero
from aact.messages.commons import DataEntry
from aact.messages.registry import DataModelFactory

from openhands.core.config import AgentConfig, AppConfig, SandboxConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime
from openhands.events.action import (
    BrowseURLAction,
    CmdRunAction,
    FileWriteAction,
    BrowseInteractiveAction,
)
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

BASE_CONTAINER_IMAGE = "docker.all-hands.dev/all-hands-ai/runtime:0.11-nikolaik"

# Configuration for logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.WARNING,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

class ActionType(Enum):
    NONE = "none"
    SPEAK = "speak"
    NON_VERBAL_COMMUNICATION = "non-verbal communication"
    LEAVE = "leave"
    THOUGHT = "thought"
    BROWSE = "browse"
    BROWSE_ACTION = "browse_action"
    READ = "read"
    WRITE = "write"
    RUN = "run"


@DataModelFactory.register("agent_action")
class AgentAction(DataModel):
    """
    Represents an action performed by an agent, including its type, content, and optional file path.
    """

    agent_name: str = Field(description="The name of the agent.")
    action_type: ActionType = Field(description="The type of action to perform.")
    argument: str = Field(description="The content of the action.")
    path: Optional[str] = Field(
        default=None, description="Path of the file, if applicable."
    )

    def to_natural_language(self) -> str:
        """
        Converts the action to a natural language description.

        Returns:
            str: A natural language description of the action.
        """
        action_descriptions = {
            ActionType.NONE: "did nothing",
            ActionType.SPEAK: f'said: "{self.argument}"',
            ActionType.THOUGHT: f'thought: "{self.argument}"',
            ActionType.BROWSE: f'browsed: "{self.argument}"',
            ActionType.NON_VERBAL_COMMUNICATION: f"[{self.action_type}] {self.argument}",
            ActionType.LEAVE: "left the conversation",
        }

        description = action_descriptions.get(self.action_type)
        if description is None:
            logger.warning(f"Unknown action type: {self.action_type}")
            return "performed an unknown action"

        return description


@NodeFactory.register("openhands")
class OpenHands(Node[DataModel, Text]):
    def __init__(
        self,
        input_channels: list[str],
        output_channels: list[str],
        redis_url: str,
    ):
        super().__init__(
            input_channel_types=[
                (input_channel, AgentAction) for input_channel in input_channels
            ],
            output_channel_types=[
                (output_channel, Text) for output_channel in output_channels
            ],
            redis_url=redis_url,
        )
        self.queue: asyncio.Queue[DataEntry[DataModel]] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None
        self.runtime: Optional[Runtime] = None

    async def init_runtime(self) -> None:
        """
        Initializes the runtime environment with the specified configuration.
        """
        start_time = time.time()
        modal_api_token_id = os.environ.get("MODAL_API_TOKEN_ID", "")
        modal_api_token_secret = os.environ.get("MODAL_API_TOKEN_SECRET", "")
        allhands_api_key = os.environ.get("ALLHANDS_API_KEY", None)
        sandbox_remote_runtime_api_url = os.environ.get(
            "SANDBOX_REMOTE_RUNTIME_API_URL", ""
        )

        if not modal_api_token_id or not modal_api_token_secret:
            logger.warning("Modal API tokens are not set. Check environment variables.")

        config = AppConfig(
            default_agent="CodeActAgent",
            run_as_openhands=False,
            max_iterations=3,
            runtime="modal",
            modal_api_token_id=modal_api_token_id,
            modal_api_token_secret=modal_api_token_secret,
            sandbox=SandboxConfig(
                base_container_image=BASE_CONTAINER_IMAGE,
                enable_auto_lint=True,
                use_host_network=False,
                timeout=50,
                platform="linux/amd64",
                api_key=allhands_api_key,
                remote_runtime_api_url=sandbox_remote_runtime_api_url,
                keep_remote_runtime_alive=False,
            ),
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
            logger.info("-" * 20)
            logger.info("RUNTIME CONNECTED")
            logger.info("-" * 20)
        else:
            logger.error("Failed to initialize runtime.")
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        logger.info(f"Runtime initialization took {elapsed_time:.2f} seconds.")

    async def __aenter__(self) -> Self:
        self.runtime_init_task = asyncio.create_task(self.init_runtime())
        self.task = asyncio.create_task(self.run_action())
        return await super().__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.runtime:
            self.runtime.close()
        return await super().__aexit__(exc_type, exc_value, traceback)

    async def aact(self, action: AgentAction) -> Optional[Text]:
        """
        Executes an action based on the observation and returns the result as Text.

        Args:
            observation (AgentAction): The action to be executed.

        Returns:
            Optional[Text]: The result of the action, or None if the runtime is not available.
        """
        if not self.runtime:
            logger.warning("Runtime is not initialized.")
            return None

        try:
            action = self._create_action(action)
            action.timeout = 45
            logger.info(f"Executing action: {action}", extra={"msg_type": "ACTION"})
            obs = self.runtime.run_action(action)
            logger.info(
                f"Received observation: {str(obs).splitlines()[:2]}",
                extra={"msg_type": "OBSERVATION"},
            )
            return Text(text=str(obs))
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return None

    def _create_action(self, observation: AgentAction) -> Any:
        """
        Creates an action based on the observation's action type.

        Args:
            observation (AgentAction): The observation containing the action type and arguments.

        Returns:
            Any: The created action.
        """
        action_type = observation.data.action_type
        argument = observation.data.argument
        path = observation.data.path

        if action_type == ActionType.BROWSE:
            return BrowseURLAction(url=argument)
        elif action_type == ActionType.BROWSE_ACTION:
            return BrowseInteractiveAction(browser_actions=argument)
        elif action_type == ActionType.RUN:
            return CmdRunAction(command=argument)
        elif action_type == ActionType.WRITE:
            return FileWriteAction(path=path, content=argument)
        elif action_type == ActionType.READ:
            return FileWriteAction(path=path)
        else:
            raise ValueError(f"Unsupported action type: {action_type}")

    async def send(self, action: Text) -> None:
        """
        Sends the action to all output channels.

        Args:
            action (Text): The action to be sent.
        """
        try:
            for (
                output_channel,
                output_channel_type,
            ) in self.output_channel_types.items():
                message = Message[output_channel_type](data=action).model_dump_json()
                await self.r.publish(output_channel, message)  # type: ignore[valid-type]
        except Exception as e:
            logger.error(f"Error sending action: {e}")

    async def run_action(self) -> None:
        """
        Continuously processes actions from the queue.
        """
        while self.task:
            try:
                action = await self.queue.get()
                obs = await self.aact(action)
                if obs is not None:
                    await self.send(obs)
                self.queue.task_done()
            except Exception as e:
                logger.error(f"Error processing action: {e}")

    async def event_handler(
        self, input_channel: str, input_message: Message[DataModel]
    ) -> AsyncIterator[tuple[str, Message[Zero]]]:
        """
        Handles incoming events and adds them to the processing queue.

        Args:
            input_channel (str): The channel from which the message was received.
            input_message (Message[DataModel]): The incoming message.

        Yields:
            Tuple[str, Message[Zero]]: A tuple containing the channel and a zero message if the channel is not recognized.
        """
        try:
            if input_channel in self.input_channel_types:
                data_entry = DataEntry[self.input_channel_types[input_channel]](
                    channel=input_channel, data=input_message.data
                )
                await self.queue.put(data_entry)
            else:
                logger.warning(f"Unrecognized input channel: {input_channel}")
                yield input_channel, Message[Zero](data=Zero())
        except Exception as e:
            logger.error(f"Error handling event: {e}")
