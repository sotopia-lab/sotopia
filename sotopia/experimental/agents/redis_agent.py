import logging
import json
import asyncio
import sys
from rich.logging import RichHandler
import aiohttp
from aiohttp import ClientSession, ClientWebSocketResponse

from aact import NodeFactory

from sotopia.experimental.agents.base_agent import BaseAgent
from sotopia.experimental.agents.datamodels import Observation, AgentAction
from typing import Any


# Check Python version
if sys.version_info >= (3, 11):
    pass
else:
    pass

# Configure logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.WARNING,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

logger = logging.getLogger(__name__)


@NodeFactory.register("redis_agent")
class RedisAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        input_channels: list[str],
        output_channel: str,
        node_name: str,
        other_agent_status: dict[str, bool],
        background: dict[str, Any] | None = None,
        agent_pk: str = "",
        redis_url: str = "redis://localhost:6379/0",
        websocket_url: str = "",
        pubsub_channel: str = "uuid_of_the_websocket_connection",
        websocket_wait_time: float = 5.0,
        loop_interval: float = 0.1,
    ):
        super().__init__(
            [(input_channel, Observation) for input_channel in input_channels],
            [(output_channel, AgentAction)],
            redis_url,
            node_name,
        )
        self.output_channel = output_channel
        self.message_history: list[Observation] = []
        self.agent_profile_pk: str | None = agent_pk
        self.background: dict[str, Any] | None = background
        self.awake: bool = False
        self.websocket_url = websocket_url
        self.websocket_session: ClientSession | None = None
        self.websocket: ClientWebSocketResponse | None = None
        self.pubsub_channel = pubsub_channel
        self.websocket_task: asyncio.Task[None] | None = None
        self.last_websocket_message = None
        self.websocket_wait_time = websocket_wait_time
        self.loop_interval = loop_interval
        self.shutdown_event = asyncio.Event()
        self.other_agent_status = other_agent_status
        # We'll set up the websocket connection in setup_websocket
        # which will be called during the first aact call

    async def setup_websocket(self) -> None:
        """Set up the websocket connection"""
        if self.websocket_url and not self.websocket:
            try:
                self.websocket_session = aiohttp.ClientSession()
                if self.websocket_session is not None:
                    self.websocket = await self.websocket_session.ws_connect(
                        self.websocket_url
                    )
                    self.websocket_task = asyncio.create_task(self.listen_websocket())
                    logger.info(f"Connected to websocket at {self.websocket_url}")
            except Exception as e:
                logger.error(f"Failed to connect to websocket: {e}")

    async def listen_websocket(self) -> None:
        """Listen for messages from websocket (NOTE: This is mock implementation, to be further developed)"""
        while not self.shutdown_event.is_set():
            try:
                if self.websocket:
                    async for msg in self.websocket:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            message = msg.data
                            logger.info(f"Received message from websocket: {message}")
                            self.last_websocket_message = message
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.warning("Websocket connection closed")
                            break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"Websocket error: {msg.data}")
                            break
            except Exception as e:
                logger.error(f"Error in websocket listener: {e}")

            # Try to reconnect if connection was lost
            if not self.shutdown_event.is_set():
                logger.warning("Websocket connection lost, trying to reconnect...")
                try:
                    if self.websocket_session and self.websocket_session.closed:
                        self.websocket_session = aiohttp.ClientSession()
                    if self.websocket_session is not None:
                        self.websocket = await self.websocket_session.ws_connect(
                            self.websocket_url
                        )
                    logger.info("Reconnected to websocket")
                except Exception as e:
                    logger.error(f"Failed to reconnect to websocket: {e}")
                    self.websocket = None
                    await asyncio.sleep(1)  # Wait before retrying

    async def publish_observation(self, obs: Observation) -> None:
        """Publish observation to Redis"""
        obs_json = json.dumps(obs.model_dump())
        await self.r.publish(self.pubsub_channel, obs_json)

    async def aact(self, obs: Observation) -> AgentAction | None:
        # Set up websocket on first call if needed
        if self.websocket_url and not self.websocket_task:
            await self.setup_websocket()

        await self.publish_observation(obs)
        # Handle initialization message
        if obs.turn_number == -1:
            print(f"self.awake: {self.awake}")
            if self.awake:
                return AgentAction(
                    agent_name=self.node_name,
                    output_channel=self.output_channel,
                    action_type="none",
                    argument="",
                )
            self.awake = True
            return AgentAction(
                agent_name=self.node_name,
                output_channel=self.output_channel,
                action_type="none",
                argument=json.dumps({"pk": "redis", "model_name": "redis"}),
            )
        for agent_name in self.other_agent_status.keys():
            if f"{agent_name} left." in obs.last_turn:
                self.other_agent_status[agent_name] = False
        if True not in self.other_agent_status.values():
            self.shutdown_event.set()

        # Append to message history
        self.message_history.append(obs)

        if self.websocket_url:
            """
            TODO: Implement websocket message handling
            """
            # Default action if no websocket message is available
            return AgentAction(
                agent_name=self.node_name,
                output_channel=self.output_channel,
                action_type="none",
                argument="",
            )
        return None
