import logging
import json
import asyncio
import sys
import hashlib
import copy
from rich.logging import RichHandler


from aact import NodeFactory

from sotopia.experimental.agents.base_agent import BaseAgent
from sotopia.experimental.agents.datamodels import Observation, AgentAction
from typing import Any, Dict, Set, List, Optional

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
        self.loop_interval = loop_interval
        self.shutdown_event = asyncio.Event()
        self.other_agent_status = other_agent_status

        # Redis channel patterns
        self.epilog_channel_prefix = "epilog:"
        self.command_channel_prefix = "command:"
        self.message_channel_prefix = "message:"

        # Track active connections by their IDs
        self.active_connections: Set[str] = set()
        # Maps connection IDs to agent names - populated from simulation parameters
        self.connection_to_agent: Dict[str, str] = {}

        # Attributes for group and mode support
        self.groups: Dict[
            str, List[str]
        ] = {}  # Dictionary mapping group names to lists of agent names
        self.mode = "full"  # Communication mode: "full" or "group"
        self.external_user_id = "websocket_user"  # ID for external WebSocket users
        self.pending_actions = []  # Queue for actions from WebSocket messages

        # Simplified message context tracking - just track sender and receivers
        self.message_senders: Dict[str, str] = {}  # agent -> original sender
        self.message_receivers: Dict[str, List[str]] = {}  # agent -> list of recipients

        # Track the last epilog sent to each connection to avoid duplicates
        self.last_epilog_hash: Dict[str, str] = {}

        # Task for monitoring command channels
        self.command_listener_task = None

    async def start_command_listener(self) -> None:
        """Start listening for commands on Redis channels"""
        if self.command_listener_task is None or self.command_listener_task.done():
            self.command_listener_task = asyncio.create_task(self._command_listener())
            logger.info("Started Redis command listener task")

    async def _command_listener(self) -> None:
        """Listen for commands from WebSocket clients via Redis"""
        pubsub = self.r.pubsub()

        try:
            # Subscribe to all command channels (using pattern subscription)
            pattern = f"{self.command_channel_prefix}*"
            await pubsub.psubscribe(pattern)
            logger.info(f"Subscribed to Redis command pattern: {pattern}")

            while not self.shutdown_event.is_set():
                try:
                    # Get the next message with a timeout
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=self.loop_interval,
                    )

                    if message and message["type"] == "pmessage":
                        # Extract connection ID from channel name
                        channel = message["channel"].decode()
                        connection_id = channel.replace(self.command_channel_prefix, "")

                        # Process the command
                        try:
                            command_data = json.loads(message["data"].decode())
                            logger.info(
                                f"Received command from {connection_id}: {command_data.get('type', 'unknown')}"
                            )

                            # Process simulation parameters for agent mapping
                            if command_data.get("type") == "register":
                                # Register the connection
                                await self.register_connection(connection_id)

                                # Map connection to agent if agent_ids are provided
                                if "agent_ids" in command_data:
                                    agent_ids = command_data["agent_ids"]
                                    if agent_ids and len(agent_ids) > 0:
                                        # Assume the first agent ID is the one this connection represents
                                        self.connection_to_agent[connection_id] = (
                                            agent_ids[0]
                                        )
                                        logger.info(
                                            f"Mapped connection {connection_id} to agent {agent_ids[0]}"
                                        )

                            # Process the command
                            action = await self.process_command(
                                command_data, connection_id
                            )
                            if action:
                                # Add to queue to be processed by aact loop
                                self.pending_actions.append(action)

                        except json.JSONDecodeError:
                            logger.error(
                                f"Failed to parse command: {message['data'][:200]}..."
                            )
                        except Exception as e:
                            logger.error(f"Error processing command: {e}")

                except asyncio.TimeoutError:
                    # No message available, continue
                    pass
                except Exception as e:
                    logger.error(f"Error in command listener: {e}")
                    await asyncio.sleep(1)  # Avoid tight loop on error

        except Exception as e:
            logger.error(f"Fatal error in command listener: {e}")
        finally:
            # Unsubscribe when done
            await pubsub.punsubscribe(pattern)
            logger.info("Command listener stopped")

    async def register_connection(self, connection_id: str) -> None:
        """Register a new connection"""
        self.active_connections.add(connection_id)
        logger.info(f"Registered new connection: {connection_id}")

    async def unregister_connection(self, connection_id: str) -> None:
        """Unregister a connection"""
        if connection_id in self.active_connections:
            self.active_connections.remove(connection_id)
            if connection_id in self.connection_to_agent:
                del self.connection_to_agent[connection_id]
            if connection_id in self.last_epilog_hash:
                del self.last_epilog_hash[connection_id]
            logger.info(f"Unregistered connection: {connection_id}")

    def filter_epilog_for_recipient(self, epilog_data: Dict, recipient: str) -> Dict:
        """
        Filter epilog data to only include messages relevant to a specific recipient.

        Args:
            epilog_data: The complete epilog data
            recipient: The name of the recipient agent or user

        Returns:
            Dict: Filtered epilog data
        """
        # Create a deep copy of the epilog to avoid modifying the original
        filtered_epilog = copy.deepcopy(epilog_data)

        if "messages" in filtered_epilog:
            # New list to hold filtered messages
            filtered_messages = []

            # Process each turn's messages
            for turn in filtered_epilog["messages"]:
                filtered_turn = []

                for message in turn:
                    # Extract message components based on the message format
                    # Message format can be either a tuple (sender, receiver, content) or a dict
                    try:
                        if isinstance(message, list):
                            sender = message[0]
                            receiver = message[1]
                            content = message[2]
                        else:
                            sender = message.get("sender", "")
                            receiver = message.get("receiver", "")
                            content = message.get("content", "")

                        # Parse the content as JSON if it's a string and might be JSON format
                        message_data = {}
                        try:
                            if isinstance(content, str):
                                message_data = json.loads(content)
                        except (json.JSONDecodeError, TypeError):
                            # Not JSON or invalid, use empty dict
                            message_data = {}

                        # Get targeting information
                        context = message_data.get("context", "regular")
                        target_agents = message_data.get("target_agents", [])
                        target_groups = message_data.get("target_groups", [])

                        # Logic to determine if this message should be visible to this recipient
                        should_include = False

                        # Messages from the recipient are always visible to them
                        if sender == recipient:
                            should_include = True
                        # Messages targeting the recipient directly
                        elif recipient in target_agents:
                            should_include = True
                        # Messages mentioning the recipient in the receiver field
                        elif receiver.endswith(recipient) or receiver == recipient:
                            should_include = True
                        # Messages to groups the recipient is in
                        elif any(
                            recipient in self.groups.get(group, [])
                            for group in target_groups
                        ):
                            should_include = True
                        # Broadcast messages are visible to everyone
                        elif context == "broadcast" or receiver == "Broadcast":
                            should_include = True
                        # Environment messages are visible to everyone
                        elif sender == "Environment" or receiver == "Environment":
                            should_include = True
                        # System messages are visible to everyone
                        elif context == "system":
                            should_include = True

                        if should_include:
                            filtered_turn.append(message)
                    except Exception as e:
                        logger.error(f"Error filtering message: {e}")
                        # Include the message if we can't properly filter it
                        filtered_turn.append(message)

                # Only add the turn if it has messages
                if filtered_turn:
                    filtered_messages.append(filtered_turn)

            # Replace the original messages with filtered ones
            filtered_epilog["messages"] = filtered_messages

        return filtered_epilog

    async def process_command(
        self, command_data: dict, connection_id: str
    ) -> AgentAction | None:
        """
        Process a command from a WebSocket client via Redis

        Args:
            command_data: The command data
            connection_id: The ID of the connection that sent the command

        Returns:
            AgentAction or None: Action to take based on the command
        """
        try:
            # Handle registration/unregistration
            if command_data.get("type") == "register":
                # Registration is now handled in the command listener
                return None

            if command_data.get("type") == "unregister":
                await self.unregister_connection(connection_id)
                return None

            # Handle mode setting
            if "mode" in command_data:
                self.mode = command_data["mode"]
                logger.info(f"Setting communication mode to: {self.mode}")
                return AgentAction(
                    agent_name=self.node_name,
                    output_channel=self.output_channel,
                    action_type="none",
                    argument=json.dumps({"mode": self.mode}),
                )

            # Handle group configuration
            if "groups" in command_data:
                self.groups = command_data["groups"]
                logger.info(f"Setting up groups: {self.groups}")
                return AgentAction(
                    agent_name=self.node_name,
                    output_channel=self.output_channel,
                    action_type="none",
                    argument=json.dumps({"groups": self.groups}),
                )

            # Handle messages based on mode
            if self.mode == "full":
                # In full mode, messages are just forwarded directly
                if "message" in command_data and "content" in command_data["message"]:
                    message_content = command_data["message"].get("content", "")
                    sender = command_data["message"].get(
                        "sender", self.external_user_id
                    )

                    return AgentAction(
                        agent_name=sender,
                        output_channel=self.output_channel,
                        action_type="speak",
                        argument=message_content,
                    )
            else:  # Group mode
                # In group mode, we need target information
                if "content" in command_data or (
                    "message" in command_data and "content" in command_data["message"]
                ):
                    # Extract content from either direct or nested message
                    content = command_data.get("content", "") or command_data.get(
                        "message", {}
                    ).get("content", "")

                    # Get sender - use the agent associated with this connection if available
                    sender = self.connection_to_agent.get(
                        connection_id, self.external_user_id
                    )
                    if "sender" in command_data:
                        sender = command_data["sender"]
                    elif (
                        "message" in command_data
                        and "sender" in command_data["message"]
                    ):
                        sender = command_data["message"]["sender"]

                    # Get target information
                    target_agents = command_data.get(
                        "target_agents", []
                    ) or command_data.get("message", {}).get("target_agents", [])
                    target_groups = command_data.get(
                        "target_groups", []
                    ) or command_data.get("message", {}).get("target_groups", [])

                    if not content:
                        logger.error("Message missing content")
                        return None

                    if not target_agents and not target_groups:
                        logger.error(
                            "Message must specify target_agents or target_groups in group mode"
                        )
                        return None

                    # Determine message context (for response handling)
                    context = (
                        "individual" if target_agents and not target_groups else "group"
                    )

                    # Expand target groups to include all member agents
                    expanded_agents = set(target_agents)
                    for group_name in target_groups:
                        if group_name in self.groups:
                            expanded_agents.update(self.groups[group_name])
                        else:
                            logger.warning(f"Group '{group_name}' not found")

                    # Record who sent messages to which agents for response tracking
                    for agent in expanded_agents:
                        self.message_senders[agent] = sender
                        if agent in self.message_receivers:
                            if sender not in self.message_receivers[agent]:
                                self.message_receivers[agent].append(sender)
                        else:
                            self.message_receivers[agent] = [sender]

                    # Create unified message action
                    return AgentAction(
                        agent_name=sender,
                        output_channel=self.output_channel,
                        action_type="speak",  # Using a valid action type
                        argument=json.dumps({
                            "action_metadata": "unified_message",  # Add a flag inside the argument instead
                            "content": content,
                            "target_agents": list(expanded_agents),
                            "original_target_agents": target_agents,
                            "original_target_groups": target_groups,
                            "context": context
                        }),
                    )

        except KeyError as e:
            logger.error(f"Missing key in command: {e}")
        except Exception as e:
            logger.error(f"Error processing command: {e}")

        return None

    async def process_agent_response(self, obs: Observation) -> Optional[AgentAction]:
        """
        Process a response from an agent and route it appropriately

        Args:
            obs: The observation from the agent

        Returns:
            AgentAction or None: Action for routing the response
        """
        if obs.agent_name in self.message_senders and self.mode == "group":
            # This agent has received targeted messages, route the response appropriately
            original_sender = self.message_senders.get(obs.agent_name)

            if original_sender:
                # Create unified message action for the response, using a valid action type
                return AgentAction(
                    agent_name=obs.agent_name,
                    output_channel=self.output_channel,
                    action_type="speak",  # Valid action type
                    argument=json.dumps({
                        "action_metadata": "unified_message",  # Flag for special handling
                        "content": obs.last_turn,
                        "target_agents": [original_sender],  # Send back to original sender
                        "original_target_agents": [original_sender],
                        "original_target_groups": [],
                        "context": "response"
                    }),
                )

        return None

    async def publish_observation(self, obs: Observation) -> None:
        """
        Publish observation to Redis channels.
        Simply forwards epilog updates to external connections.
        """
        # Handle epilog observations - these contain the complete conversation state
        if obs.agent_name == "epilog":
            try:
                # Parse the epilog data
                epilog_data = json.loads(obs.last_turn)

                # Generate a hash of the epilog to avoid sending duplicates
                epilog_hash = hashlib.md5(obs.last_turn.encode()).hexdigest()

                # Send the complete epilog to external connections - no filtering
                formatted_message = json.dumps({
                    "type": "SERVER_MSG",
                    "data": {"type": "episode_log", "log": epilog_data},
                })

                # Publish to each active connection's epilog channel
                for connection_id in self.active_connections:
                    last_hash = self.last_epilog_hash.get(connection_id)

                    # Only publish if it's a new epilog for this connection
                    if last_hash != epilog_hash:
                        channel = f"{self.epilog_channel_prefix}{connection_id}"
                        await self.r.publish(channel, formatted_message)
                        logger.info(f"Published epilog update to {channel}")
                        self.last_epilog_hash[connection_id] = epilog_hash

            except json.JSONDecodeError:
                logger.error(f"Failed to parse epilog data: {obs.last_turn}")
            except Exception as e:
                logger.error(f"Error publishing epilog: {e}")

    async def aact(self, obs: Observation) -> AgentAction | None:
        """Process an observation and return an action"""
        # Start the command listener if not already running
        if not self.command_listener_task or self.command_listener_task.done():
            await self.start_command_listener()

        # Publish observation to Redis channels
        await self.publish_observation(obs)

        # Handle initialization message
        if obs.turn_number == -1:
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

        # Add to message history
        self.message_history.append(obs)

        # Handle agent status updates (to detect when agents leave)
        for agent_name in self.other_agent_status.keys():
            if agent_name in obs.last_turn and "left" in obs.last_turn:
                self.other_agent_status[agent_name] = False
                # Clean up agent context data
                if agent_name in self.message_senders:
                    del self.message_senders[agent_name]
                if agent_name in self.message_receivers:
                    del self.message_receivers[agent_name]

        # Check if all agents have left
        if True not in self.other_agent_status.values():
            self.shutdown_event.set()
            # Cancel the command listener
            if self.command_listener_task and not self.command_listener_task.done():
                self.command_listener_task.cancel()

        # Process agent responses in group mode
        if self.mode == "group" and obs.agent_name != "epilog" and obs.last_turn:
            response_action = await self.process_agent_response(obs)
            if response_action:
                return response_action

        # Process any pending actions from WebSocket messages
        if self.pending_actions:
            action = self.pending_actions.pop(0)
            return action

        # Default action if no pending actions
        return AgentAction(
            agent_name=self.node_name,
            output_channel=self.output_channel,
            action_type="none",
            argument="",
        )
