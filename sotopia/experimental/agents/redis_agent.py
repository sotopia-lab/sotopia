import json
import asyncio

from aact import NodeFactory

from sotopia.experimental.agents.base_agent import BaseAgent
from sotopia.experimental.agents.datamodels import Observation, AgentAction
from typing import Any, Dict, List

# Configure logging
# FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
# logging.basicConfig(
#     level=logging.WARNING,
#     format=FORMAT,
#     datefmt="[%X]",
#     handlers=[RichHandler()],
# )

# logger = logging.getLogger(__name__)


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
        messaging_mode: str = "full",
        groups: Dict[str, List[str]] = {},
        pubsub_channel: str = "",
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
        self.connection_id = (
            pubsub_channel  # Fallback to pubsub_channel for compatibility
        )
        self.loop_interval = loop_interval
        self.shutdown_event = asyncio.Event()
        self.other_agent_status = other_agent_status
        # Group messaging support
        self.mode = messaging_mode  # Communication mode: "full" or "group"
        self.groups = groups  # Dictionary mapping group names to lists of agent names

        # Track active connections and their channels
        self.websocket_prefix = (
            "websocket:"  # Prefix for listening to websocket commands
        )

        # Track message relationships for directing responses
        self.message_senders: Dict[str, str] = {}  # agent -> original sender
        self.message_receivers: Dict[str, List[str]] = {}  # agent -> list of recipients

        # Track the last epilog hash to avoid duplicates
        self.last_epilog_hash: Dict[str, str] = {}

        # Pending actions from websocket messages
        self.pending_actions: asyncio.Queue[AgentAction] = asyncio.Queue()

        # Command listener task
        # Start the command listener if not already running
        self.command_listener_task = None
        self.start_command_listener()
        # if not self.command_listener_task or self.command_listener_task.done():
        #     await self.start_command_listener()

        print(f"RedisAgent initialized with connection_id: {self.connection_id}")
        print(f"RedisAgent mode: {self.mode}")
        print(f"RedisAgent groups: {self.groups}")

    def start_command_listener(self) -> None:
        """Start listening for commands on Redis channels"""
        if self.command_listener_task is None or self.command_listener_task.done():
            self.command_listener_task = asyncio.create_task(self._command_listener())
            print("Started Redis command listener task")

    async def _command_listener(self) -> None:
        """Listen for commands from WebSocket clients via Redis"""
        if not self.connection_id:
            print("No connection_id specified, command listener not started")
            return

        pubsub = self.r.pubsub()
        channel = f"{self.websocket_prefix}{self.connection_id}"

        try:
            # Subscribe to the websocket channel for this connection
            await pubsub.subscribe(channel)
            print(f"Subscribed to websocket channel: {channel}")

            while not self.shutdown_event.is_set():
                try:
                    # Get the next message with a timeout
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=self.loop_interval,
                    )

                    if message and message["type"] == "message":
                        # Process the command
                        try:
                            command_data = json.loads(message["data"].decode())
                            print(
                                f"Received command from websocket: {command_data.get('type', 'unknown')}"
                            )

                            # Process the command
                            action = await self.process_command(
                                command_data, self.connection_id
                            )
                            if action:
                                # Add to pending actions queue
                                await self.pending_actions.put(action)

                        except json.JSONDecodeError:
                            print(
                                f"Failed to parse websocket command: {message['data'][:200]}..."
                            )
                        except Exception as e:
                            print(f"Error processing websocket command: {e}")

                except asyncio.TimeoutError:
                    # No message available, continue
                    pass
                except Exception as e:
                    print(f"Error in command listener: {e}")
                    await asyncio.sleep(1)  # Avoid tight loop on error

        except Exception as e:
            print(f"Fatal error in command listener: {e}")
        finally:
            # Unsubscribe when done
            await pubsub.unsubscribe(channel)
            print("Command listener stopped")

    async def process_command(
        self, command_data: dict, connection_id: str
    ) -> AgentAction | None:
        """
        Process a command from a WebSocket client

        Args:
            command_data: The command data
            connection_id: The ID of the connection that sent the command

        Returns:
            AgentAction or None: Action to handle the command
        """
        try:
            # Handle mode setting
            if "mode" in command_data:
                new_mode = command_data.get("mode")
                if new_mode in ["full", "group"]:
                    self.mode = new_mode
                    print(f"Set communication mode to: {self.mode}")

                    if "groups" in command_data:
                        groups = command_data.get("groups", {})
                        self.groups = groups
                        print(f"Updated groups configuration: {self.groups}")

                        return AgentAction(
                            agent_name=self.node_name,
                            output_channel=self.output_channel,
                            action_type="setup_groups",
                            argument=json.dumps(
                                {"mode": self.mode, "groups": self.groups}
                            ),
                        )
                    return AgentAction(
                        agent_name=self.node_name,
                        output_channel=self.output_channel,
                        action_type="set_mode",
                        argument=json.dumps({"mode": self.mode}),
                    )

            # Handle message from WebSocket client
            if "message" in command_data:
                msg = command_data["message"]

                if not isinstance(msg, dict) or "content" not in msg:
                    print("Invalid message format")
                    return None

                sender = msg.get("sender", "redis_agent")
                content = msg.get("content")

                # Extract target information
                target_agents = msg.get("target_agents", [])
                target_groups = msg.get("target_groups", [])

                # If we have a single target agent (DM case)
                if len(target_agents) == 1 and not target_groups:
                    target_agent = target_agents[0]
                    print(f"Processing DM from {sender} to {target_agent}")

                    # Record message relationship for response tracking
                    self.message_senders[target_agent] = sender
                    if target_agent in self.message_receivers:
                        if sender not in self.message_receivers[target_agent]:
                            self.message_receivers[target_agent].append(sender)
                    else:
                        self.message_receivers[target_agent] = [sender]

                    return AgentAction(
                        agent_name=sender,
                        output_channel=self.output_channel,
                        action_type="speak",
                        argument=json.dumps(
                            {
                                "content": content,
                                "target_agents": [
                                    target_agent
                                ],  # Just the single target
                                "original_target_agents": [target_agent],
                                "original_target_groups": [],
                                "context": "individual",  # Mark as a direct message
                            }
                        ),
                    )

                # If we have target groups but no target agents
                elif target_groups and not target_agents:
                    # Expand target groups to include all member agents
                    expanded_agents = []
                    for group_name in target_groups:
                        if group_name in self.groups:
                            expanded_agents.extend(self.groups[group_name])

                    # Remove duplicates
                    expanded_agents = list(set(expanded_agents))

                    if not expanded_agents:
                        print(f"No agents found in groups: {target_groups}")
                        return None

                    # Record message relationships for responses
                    for agent in expanded_agents:
                        self.message_senders[agent] = sender
                        if agent in self.message_receivers:
                            if sender not in self.message_receivers[agent]:
                                self.message_receivers[agent].append(sender)
                        else:
                            self.message_receivers[agent] = [sender]

                    return AgentAction(
                        agent_name=sender,
                        output_channel=self.output_channel,
                        action_type="speak",
                        argument=json.dumps(
                            {
                                "content": content,
                                "target_agents": expanded_agents,
                                "original_target_agents": [],
                                "original_target_groups": target_groups,
                                "context": "group",  # Mark as a group message
                            }
                        ),
                    )

                # If we have both target agents and target groups
                elif target_agents and target_groups:
                    # Expand target groups to include all member agents
                    expanded_agents = list(target_agents)  # Start with direct targets
                    for group_name in target_groups:
                        if group_name in self.groups:
                            expanded_agents.extend(self.groups[group_name])

                    # Remove duplicates
                    expanded_agents = list(set(expanded_agents))

                    # Record message relationships for responses
                    for agent in expanded_agents:
                        self.message_senders[agent] = sender
                        if agent in self.message_receivers:
                            if sender not in self.message_receivers[agent]:
                                self.message_receivers[agent].append(sender)
                        else:
                            self.message_receivers[agent] = [sender]

                    return AgentAction(
                        agent_name=sender,
                        output_channel=self.output_channel,
                        action_type="speak",
                        argument=json.dumps(
                            {
                                "content": content,
                                "target_agents": expanded_agents,
                                "original_target_agents": target_agents,
                                "original_target_groups": target_groups,
                                "context": "mixed",  # Mark as mixed targeting
                            }
                        ),
                    )

                # No targets specified - broadcast to everyone
                else:
                    print(f"Broadcasting message from {sender}")
                    return AgentAction(
                        agent_name=sender,
                        output_channel=self.output_channel,
                        action_type="speak",
                        argument=json.dumps(
                            {
                                "content": content,
                                "target_agents": [],  # Empty means broadcast
                                "original_target_agents": [],
                                "original_target_groups": [],
                                "context": "broadcast",  # Mark as broadcast
                            }
                        ),
                    )

        except KeyError as e:
            print(f"Missing key in command: {e}")
        except Exception as e:
            print(f"Error processing command: {e}")

        return None

    # async def process_agent_response(self, obs: Observation) -> Optional[AgentAction]:
    #     """
    #     Process a response from an agent and route it appropriately

    #     Args:
    #         obs: The observation from the agent

    #     Returns:
    #         AgentAction or None: Action for routing the response
    #     """
    #     # Only process responses when the agent has a valid sender
    #     if obs.agent_name in self.message_senders and obs.last_turn:
    #         # Get the original sender who sent a message to this agent
    #         original_sender = self.message_senders.get(obs.agent_name)

    #         if original_sender:
    #             print(
    #                 f"Routing response from {obs.agent_name} back to {original_sender}"
    #             )

    #             # Create a unified message action to send the response back only to the original sender
    #             return AgentAction(
    #                 agent_name=obs.agent_name,
    #                 output_channel=self.output_channel,
    #                 action_type="speak",
    #                 argument=json.dumps(
    #                     {
    #                         "content": obs.last_turn,
    #                         "target_agents": [
    #                             original_sender
    #                         ],  # Only the original sender
    #                         "original_target_agents": [original_sender],
    #                         "original_target_groups": [],
    #                         "context": "response",
    #                         "responding_to": {"sender": original_sender},
    #                     }
    #                 ),
    #             )

    #     return None

    async def publish_observation(self, obs: Observation) -> None:
        """Publish observation to Redis for WebSocket clients"""
        # Only publish if we have a connection ID
        if not self.connection_id:
            print("No connection ID")
            return

        # Check if this is an epilog observation (complete conversation state)
        if obs.agent_name == "epilog":
            print("Message is an epilog")
            try:
                # Parse the epilog data
                epilog_data = json.loads(obs.last_turn)

                # Generate a hash of the epilog to avoid sending duplicates
                # epilog_hash = hashlib.md5(obs.last_turn.encode()).hexdigest()

                # Only send if different from the last epilog for this connection
                # if self.last_epilog_hash.get(self.connection_id) != epilog_hash:
                # Format as a message for the client
                formatted_message = json.dumps(
                    {
                        "type": "SERVER_MSG",
                        "data": {"type": "episode_log", "log": epilog_data},
                    }
                )

                # Publish to the connection ID channel
                await self.r.publish(self.connection_id, formatted_message)
                print(f"Published epilog update to {self.connection_id}")

            # Update hash
            # self.last_epilog_hash[self.connection_id] = epilog_hash

            except json.JSONDecodeError:
                print(f"Failed to parse epilog data: {obs.last_turn}")
            except Exception as e:
                print(f"Error publishing epilog: {e}")
        else:
            print("Unexpected message routed here ")

        # elif obs.agent_name != self.node_name and obs.last_turn:
        #     # This is a message from an agent (not system or self)
        #     try:
        #         # For agent responses, we format them as regular messages
        #         formatted_message = json.dumps(
        #             {
        #                 "type": "SERVER_MSG",
        #                 "data": {
        #                     "type": "agent_message",
        #                     "agent": obs.agent_name,
        #                     "content": obs.last_turn,
        #                     "turn": obs.turn_number,
        #                 },
        #             }
        #         )

        #         # Publish to the connection's channel
        #         await self.r.publish(self.connection_id, formatted_message)
        #         logger.info(
        #             f"Published message from {obs.agent_name} to {self.connection_id}"
        #         )
        #     except Exception as e:
        #         logger.error(f"Error publishing agent message: {e}")

    async def aact(self, obs: Observation) -> AgentAction | None:
        """
        Process an observation and return an action

        Args:
            obs: The observation to process

        Returns:
            AgentAction or None: The action to perform
        """

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
                argument=json.dumps(
                    {
                        "pk": "redis",
                        "model_name": "redis",
                        "connection_id": self.connection_id,
                    }
                ),
            )
        print("Redis is awake")

        # Update agent status
        for agent_name in self.other_agent_status.keys():
            if f"{agent_name} left." in obs.last_turn:
                self.other_agent_status[agent_name] = False
                # Clean up message tracking for this agent
                if agent_name in self.message_senders:
                    del self.message_senders[agent_name]
                if agent_name in self.message_receivers:
                    del self.message_receivers[agent_name]

        # Append to message history
        self.message_history.append(obs)
        await self.publish_observation(obs)

        # Process agent responses to route them back to the original sender
        if (
            obs.agent_name != self.node_name
            and obs.agent_name != "epilog"
            and obs.last_turn
        ):
            print("Unexpected message from agent received")
            # response_action = await self.process_agent_response(obs)
            # if response_action:
            #     print(f"Routing response from {obs.agent_name}")
            #     return response_action

        # Process any pending actions from websocket messages
        if not self.pending_actions.empty():
            action = await self.pending_actions.get()
            return action

        # Default action
        return AgentAction(
            agent_name=self.node_name,
            output_channel=self.output_channel,
            action_type="none",
            argument="",
        )
