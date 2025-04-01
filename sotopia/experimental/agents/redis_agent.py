import logging
import json
import asyncio
import sys
from rich.logging import RichHandler

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


log = logging.getLogger("sotopia.redis_agent")
log.setLevel(logging.INFO)
# Prevent propagation to root logger
log.propagate = False
log.addHandler(RichHandler(rich_tracebacks=True, show_time=True))


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
        # self.websocket_url = websocket_url
        # self.websocket_session: ClientSession | None = None
        # self.websocket: ClientWebSocketResponse | None = None
        self.pubsub_channel = pubsub_channel
        # self.websocket_task: asyncio.Task[None] | None = None
        # self.last_websocket_message = None
        # self.websocket_wait_time = websocket_wait_time
        self.loop_interval = loop_interval
        self.shutdown_event = asyncio.Event()
        self.other_agent_status = other_agent_status
        self.pending_actions: asyncio.Queue[Observation] = asyncio.Queue()
        self.command_listener_task = None
        self.websocket_prefix = (
            "websocket:"  # Prefix for listening to websocket commands
        )
        # self.start_command_listener()
        # We'll set up the websocket connection in setup_websocket
        # which will be called during the first aact call

    async def start_command_listener(self) -> None:
        """Start listening for commands on Redis channels"""
        if self.command_listener_task is None or self.command_listener_task.done():
            self.command_listener_task = asyncio.create_task(self._command_listener())
            print("Started Redis command listener task")

    async def _command_listener(self) -> None:
        """Listen for commands from WebSocket clients via Redis"""
        if not self.pubsub_channel:
            print("No connection_id specified, command listener not started")
            return

        pubsub = self.r.pubsub()
        channel = f"{self.websocket_prefix}{self.pubsub_channel}"

        try:
            # Subscribe to the websocket channel for this connection
            await pubsub.subscribe(channel)
            print(f"Subscribed to Redis channel from websocket: {channel}")

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
                            try:
                                if "message" in command_data:
                                    msg = command_data["message"]

                                    if (
                                        not isinstance(msg, dict)
                                        or "content" not in msg
                                    ):
                                        print("Invalid message format")
                                        return None

                                    sender = msg.get("sender", "redis_agent")
                                    content = msg.get("content")
                                    receiver = msg.get("receiver", "all")

                                    if receiver != "all":
                                        print(
                                            f"Processing DM from {sender} to {receiver}"
                                        )
                                    else:
                                        print(f"Broadcasting message from {sender}")
                                    action = AgentAction(
                                        agent_name=sender,
                                        output_channel=self.output_channel,
                                        action_type="speak",
                                        argument=json.dumps(
                                            {"action": content, "to": receiver}
                                        ),
                                    )
                            except KeyError as e:
                                print(f"Missing key in command: {e}")
                                action = None
                            except Exception as e:
                                print(f"Error processing command: {e}")
                                action = None

                            if action:
                                await self.send(action)
                                # Add to pending actions queue
                                # await self.pending_actions.put(action)

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

    async def publish_observation(self, obs: Observation) -> None:
        """Publish observation to Redis"""
        if not self.pubsub_channel:
            print("No connection ID")
            return
        if obs.agent_name == "epilog":
            print("Message is an epilog")
            obs_json = json.dumps(obs.model_dump())
            print(f"The epilog object looks like: {self.obs_json}")
            await self.r.publish(self.pubsub_channel, obs_json)
            print(f"Published epilog update to {self.connection_id}")
        else:
            print(f"Non-epilog message received from {obs.agent_name}")
            return

    async def aact(self, obs: Observation) -> AgentAction | None:
        if not self.command_listener_task:
            print("Redis connection not initialized from redis_agent")
            await self.start_command_listener()
        # Handle initialization message
        if obs.turn_number == -1:
            print(f"self.awake: {self.awake}")
            if self.awake:
                return AgentAction(
                    agent_name=self.node_name,
                    output_channel=self.output_channel,
                    action_type="none",
                    argument={"pk": "redis", "model_name": "redis"},
                )
            self.awake = True
            return AgentAction(
                agent_name=self.node_name,
                output_channel=self.output_channel,
                action_type="none",
                argument=json.dumps({"pk": "redis", "model_name": "redis"}),
            )
        await self.publish_observation(obs)
        for agent_name in self.other_agent_status.keys():
            if f"{agent_name} left." in obs.last_turn:
                self.other_agent_status[agent_name] = False
        if True not in self.other_agent_status.values():
            self.shutdown_event.set()

        # Append to message history
        self.message_history.append(obs)

        if not self.pending_actions.empty():
            action = await self.pending_actions.get()
            return action
        return AgentAction(
            agent_name=self.node_name,
            output_channel=self.output_channel,
            action_type="none",
            argument={"pk": "redis", "model_name": "redis"},
        )
