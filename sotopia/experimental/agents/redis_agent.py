import logging
import json
import asyncio
import sys
from rich.logging import RichHandler
import aiohttp
from aiohttp import ClientSession, ClientWebSocketResponse, ClientWSTimeout
from typing import Any, Dict, List, Optional, Set, Union, TypeVar, cast

from aact import NodeFactory

from sotopia.experimental.agents.base_agent import BaseAgent
from sotopia.experimental.agents.datamodels import Observation, AgentAction

# Configure logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

logger = logging.getLogger(__name__)


@NodeFactory.register("redis_agent")
class RedisAgent(BaseAgent[Observation, AgentAction]):
    """
    RedisAgent serves as a bridge between WebSocket connections and the Moderator.
    
    It listens for messages from the WebSocket, forwards them to the Moderator,
    and sends responses from the Moderator back through the WebSocket.
    
    This agent is designed to handle NPC groups and targeted messaging.
    """
    
    def __init__(
        self,
        input_channels: list[str],
        output_channel: str,
        node_name: str,
        other_agent_status: Dict[str, bool],
        background: Dict[str, Any] | None = None,
        agent_pk: str = "",
        redis_url: str = "redis://localhost:6379/0",
        websocket_url: str = "",
        pubsub_channel: str = "uuid_of_the_websocket_connection",
        websocket_wait_time: float = 5.0,
        loop_interval: float = 0.1,
    ):
        """
        Initialize the RedisAgent.
        
        Parameters:
        - input_channels: Channels to listen for messages from the Moderator
        - output_channel: Channel to send messages to the Moderator
        - node_name: Name of this agent node
        - other_agent_status: Status of other agents in the simulation
        - background: Background information for the agent
        - agent_pk: Agent primary key
        - redis_url: URL of the Redis server
        - websocket_url: URL of the WebSocket server
        - pubsub_channel: Redis PubSub channel for messages
        - websocket_wait_time: Timeout for WebSocket message waiting
        - loop_interval: Interval for the main message loop
        """
        super().__init__(
            [(input_channel, Observation) for input_channel in input_channels],
            [(output_channel, AgentAction)],
            redis_url,
            node_name,
        )
        self.output_channel = output_channel
        self.message_history: list[Observation] = []
        self.agent_profile_pk: str | None = agent_pk
        self.background: Dict[str, Any] | None = background
        self.awake: bool = False
        self.websocket_url = websocket_url
        self.websocket_session: ClientSession | None = None
        self.websocket: ClientWebSocketResponse | None = None
        self.pubsub_channel = pubsub_channel
        self.websocket_task: asyncio.Task[None] | None = None
        self.websocket_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.response_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.websocket_wait_time = websocket_wait_time
        self.loop_interval = loop_interval
        self.shutdown_event = asyncio.Event()
        self.other_agent_status = other_agent_status
        
        # Fields for group-based routing
        self.npc_groups: Dict[str, List[str]] = {}  # Map from group ID to list of NPC IDs
        self.active_npcs: Set[str] = set()          # Set of active NPC IDs
        self.pending_ws_messages: List[Dict[str, Any]] = []   # Messages waiting to be processed

    async def setup_websocket(self) -> None:
        """Set up the WebSocket connection with improved error handling"""
        if self.websocket_url and not self.websocket:
            retry_count = 0
            max_retries = 3
            retry_delay = 1.0
            
            while retry_count < max_retries:
                try:
                    self.websocket_session = aiohttp.ClientSession()
                    if self.websocket_session is not None:
                        # Create a ClientWSTimeout object for the timeout
                        ws_timeout = ClientWSTimeout(timeout=10.0)
                        self.websocket = await self.websocket_session.ws_connect(
                            self.websocket_url,
                            timeout=ws_timeout,  # Now using proper type
                            heartbeat=30.0  # Keep connection alive
                        )
                        self.websocket_task = asyncio.create_task(self.listen_websocket())
                        logger.info(f"Connected to WebSocket at {self.websocket_url}")
                        return  # Success, exit the function
                        
                except aiohttp.ClientConnectorError as e:
                    logger.error(f"Connection error (attempt {retry_count+1}/{max_retries}): {e}")
                except aiohttp.WSServerHandshakeError as e:
                    logger.error(f"WebSocket handshake error (attempt {retry_count+1}/{max_retries}): {e}")
                except asyncio.TimeoutError:
                    logger.error(f"Connection timeout (attempt {retry_count+1}/{max_retries})")
                except Exception as e:
                    logger.error(f"Unexpected error connecting to WebSocket (attempt {retry_count+1}/{max_retries}): {e}")
                
                # Clean up failed session
                if self.websocket_session and not self.websocket_session.closed:
                    await self.websocket_session.close()
                
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(retry_delay * retry_count)  # Exponential backoff
            
            # All retries failed
            logger.critical(f"Failed to connect to WebSocket after {max_retries} attempts")
            self.websocket = None
            self.websocket_session = None

    async def listen_websocket(self) -> None:
        """Listen for messages from the WebSocket with improved error handling"""
        error_count = 0
        max_errors = 5
        
        while not self.shutdown_event.is_set():
            try:
                if not self.websocket:
                    logger.warning("WebSocket not connected, attempting to reconnect...")
                    await self.setup_websocket()
                    if not self.websocket:
                        await asyncio.sleep(5)  # Wait before retry
                        continue
                        
                async for msg in self.websocket:
                    # Reset error counter on successful message
                    error_count = 0
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        message_text = msg.data
                        logger.debug(f"Received message from WebSocket: {message_text[:100]}...")
                        
                        # Process the message
                        try:
                            message = json.loads(message_text)
                            
                            # Handle each message type in a try-except block
                            try:
                                if message.get("type") == "START_SIM":
                                    await self.process_start_message(message)
                                elif message.get("type") == "CLIENT_MSG":
                                    await self.process_client_message(message)
                                elif message.get("type") == "FINISH_SIM":
                                    logger.info("Received finish simulation message")
                                    # Send acknowledgment to client
                                    await self.send_to_websocket({
                                        "type": "SERVER_MSG",
                                        "data": {"type": "simulation_ending", "status": "acknowledged"}
                                    })
                                    # Trigger cleanup
                                    await self.cleanup_simulation()
                                else:
                                    logger.warning(f"Unknown message type: {message.get('type')}")
                                    
                            except Exception as e:
                                logger.error(f"Error processing message: {e}")
                                # Send error back to client
                                await self.send_to_websocket({
                                    "type": "ERROR",
                                    "data": {
                                        "type": "PROCESSING_ERROR",
                                        "details": f"Error processing message: {str(e)}"
                                    }
                                })
                                
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse WebSocket message: {message_text[:100]}...")
                            await self.send_to_websocket({
                                "type": "ERROR",
                                "data": {
                                    "type": "INVALID_JSON",
                                    "details": "Message is not valid JSON"
                                }
                            })
                            
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("WebSocket connection closed")
                        break
                        
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {msg.data}")
                        break
                        
            except asyncio.CancelledError:
                logger.info("WebSocket listener task cancelled")
                break
                
            except Exception as e:
                logger.error(f"Error in WebSocket listener: {e}")
                error_count += 1
                
                if error_count >= max_errors:
                    logger.critical(f"Too many errors ({error_count}), stopping WebSocket listener")
                    break
                    
                # Wait before trying again
                await asyncio.sleep(min(1.0 * error_count, 5.0))

            # Try to reconnect if connection was lost
            if not self.shutdown_event.is_set():
                logger.warning("WebSocket connection lost, trying to reconnect...")
                await self.setup_websocket()

        # Cleanup when exiting the loop
        await self.cleanup_websocket()

    async def process_start_message(self, message: Dict[str, Any]) -> None:
        """
        Process the start message from the client
        
        The start message contains information about the NPCs and groups
        that will be used in the simulation.
        """
        data = message.get("data", {})
        
        # Extract NPC information and group IDs
        self.active_npcs = set(data.get("npcs", []))
        self.npc_groups = data.get("groups", {})
        
        # Create and send initialization to moderator
        start_action = AgentAction(
            agent_name=self.node_name,
            output_channel=self.output_channel,
            action_type="start",
            argument=json.dumps({
                "npcs": list(self.active_npcs),
                "groups": self.npc_groups,
                "pk": "redis", 
                "model_name": "redis"
            })
        )
        
        await self.send(start_action)
        logger.info(f"Sent start message to moderator with {len(self.active_npcs)} NPCs and {len(self.npc_groups)} groups")

    async def process_client_message(self, message: Dict[str, Any]) -> None:
        """
        Process a client message and route it to the appropriate NPCs
        
        The client message contains information about the content to be sent,
        and which NPCs or groups to send it to.
        """
        data = message.get("data", {})
        content = data.get("content", "")
        target_npcs = data.get("target_npcs", [])
        target_group = data.get("target_group", None)
        
        # Create a client message action for the moderator to route
        client_action = AgentAction(
            agent_name=self.node_name,
            output_channel=self.output_channel,
            action_type="speak",
            argument=json.dumps({
                "content": content,
                "target_npcs": target_npcs,
                "target_group": target_group
            })
        )
        
        await self.send(client_action)
        logger.info(f"Routed client message to moderator for distribution")

    async def send_to_websocket(self, message: Dict[str, Any]) -> bool:
        """Send a message to the client through the WebSocket with error handling"""
        if not self.websocket:
            logger.error("Cannot send message - WebSocket not connected")
            return False
            
        try:
            await self.websocket.send_json(message)
            logger.debug(f"Sent message to WebSocket client: {message.get('type')}")
            return True
        except ConnectionResetError:
            logger.error("Connection reset while sending to WebSocket")
        except aiohttp.ClientConnectionError:
            logger.error("Connection error while sending to WebSocket")
        except Exception as e:
            logger.error(f"Failed to send message to WebSocket: {e}")
        
        # Try to reconnect
        if not self.shutdown_event.is_set():
            logger.info("Attempting to reconnect WebSocket...")
            await self.setup_websocket()
        
        return False

    async def publish_observation(self, obs: Observation) -> None:
        """Publish observation to Redis"""
        obs_json = json.dumps(obs.model_dump())
        await self.r.publish(self.pubsub_channel, obs_json)

    async def aact(self, obs: Observation) -> AgentAction | None:
        """
        Process an observation from the Moderator and take appropriate action
        
        This method handles different types of observations:
        - Initialization messages
        - Epilog messages (end of simulation)
        - NPC responses
        - Regular turn progress
        
        Returns an AgentAction or None based on the observation.
        """
        # Set up WebSocket on first call if needed
        if self.websocket_url and not self.websocket_task:
            await self.setup_websocket()

        # Publish observation to Redis for debugging/logging
        await self.publish_observation(obs)
        
        # Handle initialization message
        if obs.turn_number == -1:
            if self.awake:
                return AgentAction(
                    agent_name=self.node_name,
                    output_channel=self.output_channel,
                    action_type="none",
                    argument=""
                )
            self.awake = True
            return AgentAction(
                agent_name=self.node_name,
                output_channel=self.output_channel,
                action_type="none",
                argument=json.dumps({"pk": "redis", "model_name": "redis"})
            )
        
        # For epilog messages, forward to WebSocket
        if obs.agent_name == "epilog":
            try:
                # Parse the epilog data
                epilog_data = json.loads(obs.last_turn)
                # Create a server message to send back to the client
                server_message = {
                    "type": "SERVER_MSG",
                    "data": {
                        "type": "messages",
                        "messages": epilog_data
                    }
                }
                # Send to WebSocket
                await self.send_to_websocket(server_message)
                logger.info("Sent epilog to WebSocket client")
            except Exception as e:
                logger.error(f"Error processing epilog: {e}")
        
        # For responses from NPCs, forward to WebSocket
        elif obs.agent_name in self.active_npcs:
            try:
                # Create a message to send back to the client
                npc_response = {
                    "type": "SERVER_MSG",
                    "data": {
                        "type": "npc_response",
                        "npc_id": obs.agent_name,
                        "content": obs.last_turn
                    }
                }
                # Send to WebSocket
                await self.send_to_websocket(npc_response)
                logger.info(f"Sent NPC response from {obs.agent_name} to WebSocket client")
            except Exception as e:
                logger.error(f"Error processing NPC response: {e}")
        
        # Track agent status
        for agent_name in self.other_agent_status.keys():
            if f"{agent_name} left." in obs.last_turn:
                self.other_agent_status[agent_name] = False
        if True not in self.other_agent_status.values():
            self.shutdown_event.set()

        # Append to message history
        self.message_history.append(obs)
        
        # We don't need to return an action for most observations from moderator
        return None
    
    async def cleanup_websocket(self) -> None:
        """Clean up WebSocket resources"""
        try:
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            
            if self.websocket_session and not self.websocket_session.closed:
                await self.websocket_session.close()
                
            logger.info("WebSocket resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up WebSocket resources: {e}")
        
        self.websocket = None
        self.websocket_session = None
    
    async def cleanup_simulation(self) -> None:
        """Clean up resources when simulation ends"""
        logger.info("Cleaning up simulation resources...")
        
        # Signal other components that we're shutting down
        self.shutdown_event.set()
        
        # Clean up WebSocket
        await self.cleanup_websocket()
        
        # Cancel any pending tasks
        if self.websocket_task and not self.websocket_task.done():
            self.websocket_task.cancel()
            try:
                await self.websocket_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Simulation resources cleaned up")