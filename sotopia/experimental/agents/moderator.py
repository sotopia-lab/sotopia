import asyncio
import sys
import json
import hashlib

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from aact import Message, NodeFactory, Node
from aact.messages import DataModel, DataModelFactory

from typing import Literal, Any, AsyncIterator, Dict, List
from pydantic import Field

from .datamodels import AgentAction, Observation
from sotopia.messages import ActionType
from .logs import EpisodeLog
import logging

logger = logging.getLogger(__name__)


@DataModelFactory.register("observations")
class Observations(DataModel):
    observations_map: dict[str, Observation] = Field(
        description="the observations of the agents"
    )


@NodeFactory.register("moderator")
class Moderator(Node[AgentAction, Observation]):
    def __init__(
        self,
        node_name: str,
        input_channels: list[str],
        output_channels: list[str],
        scenario: str,
        agent_mapping: dict[str, str],
        evaluator_channels: list[list[str]] = [],
        tag: str = "",
        redis_url: str = "redis://localhost:6379/0",
        action_order: Literal["simultaneous", "round-robin", "random"] = "round-robin",
        available_actions: list[ActionType] = [
            "none",
            "speak",
            "non-verbal communication",
            "action",
            "leave",
        ],
        max_turns: int = 20,
        push_to_db: bool = False,
        use_pk_value: bool = False,
        evaluate_episode: bool = False,
        redis_agent_as_actor: bool = False,
        messaging_mode: str = "full",
        groups: Dict[str, List[str]] = {},
    ) -> None:
        super().__init__(
            input_channel_types=[
                (input_channel, AgentAction) for input_channel in input_channels
            ]
            + [(channel[0], AgentAction) for channel in evaluator_channels],
            output_channel_types=[
                (output_channel, Observation) for output_channel in output_channels
            ],
            redis_url=redis_url,
            node_name=node_name,
        )
        self.observation_queue: asyncio.Queue[AgentAction] = asyncio.Queue()
        self.task_scheduler: asyncio.Task[None] | None = None
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.agent_mapping: dict[str, str] = agent_mapping
        self.tag: str = tag
        self.action_order: Literal["simultaneous", "round-robin", "random"] = (
            action_order
        )
        self.available_actions: list[ActionType] = available_actions
        self.turn_number: int = 0
        self.max_turns: int = max_turns
        self.current_agent_index: int = 0
        self.scenario: str = scenario
        self.agents: list[str] = list(agent_mapping.values())
        self.agents_awake: dict[str, bool] = {name: False for name in self.agents}
        self.all_agents_awake: asyncio.Event = asyncio.Event()
        self.evaluator_channels: list[list[str]] = evaluator_channels
        self.push_to_db: bool = push_to_db
        self.use_pk_value: bool = use_pk_value
        self.agents_pk: dict[str, str] = {}
        self.agent_models: dict[str, str] = {}
        self.redis_agent_as_actor: bool = redis_agent_as_actor
        self.evaluate_episode: bool = evaluate_episode

        # Group messaging support
        self.groups: Dict[str, List[str]] = groups  # Group name -> list of agent names
        self.mode = messaging_mode  # Communication mode: "full" or "group"

        # Message context tracking
        self.message_senders: Dict[str, str] = {}  # agent -> original sender
        self.message_receivers: Dict[str, List[str]] = {}  # agent -> list of recipients

        # Track the last epilog hash to avoid duplicates
        self.last_epilog_hash: str | None = None

        assert (not self.evaluate_episode) or len(
            evaluator_channels
        ) > 0, "if evaluate_episode is True, evaluator_channels should not be empty"

        self.epilog: EpisodeLog  # will be initialized in booting process

        print(f"Moderator initialized with messaging_mode: {self.mode}")
        print(f"Moderator groups: {self.groups}")

    async def __aenter__(self) -> Self:
        print(f"Starting moderator with scenario: {self.scenario}")
        # await self.booting()
        # assert False
        asyncio.create_task(self.booting())
        self.task_scheduler = asyncio.create_task(self._task_scheduler())
        print("Moderator booted successfully")
        return await super().__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.shutdown_event.set()
        if self.task_scheduler is not None:
            self.task_scheduler.cancel()
        return await super().__aexit__(exc_type, exc_value, traceback)

    async def send(self, output_channel: str, data: str) -> None:
        """Send data to a specific output channel."""
        await self.r.publish(output_channel, data)

    async def send_observations(self, observations: Observations) -> None:
        """Send observations to all relevant output channels."""
        for output_channel, output_channel_type in self.output_channel_types.items():
            if output_channel in observations.observations_map:
                message_json = Message[output_channel_type](  # type:ignore[valid-type]
                    data=observations.observations_map[output_channel]
                ).model_dump_json()
                await self.send(output_channel, message_json)

    async def send_epilog(self, epilog: EpisodeLog, output_channel: str) -> None:
        """Send the epilog to other agents with deduplication"""
        # Generate hash of epilog to avoid sending duplicates
        epilog_json = epilog.model_dump_json()
        # current_hash = hashlib.md5(epilog_json.encode()).hexdigest()

        # Only send if it's different from the last epilog we sent
        # if current_hash != self.last_epilog_hash:
        message_json = Message[Observation](
            data=Observation(
                agent_name="epilog",
                last_turn=epilog_json,
                turn_number=self.turn_number,
                available_actions=self.available_actions,
            )
        ).model_dump_json()
        await self.send(output_channel, message_json)
            # self.last_epilog_hash = current_hash

    async def event_handler(
        self, channel: str, message: Message[AgentAction]
    ) -> AsyncIterator[tuple[str, Message[Observation]]]:
        if channel in self.input_channel_types:
            await self.observation_queue.put(message.data)
        else:
            raise ValueError(f"Invalid channel: {channel}")
            yield "", self.output_type()

    async def _task_scheduler(self) -> None:
        await self.all_agents_awake.wait()
        while not self.shutdown_event.is_set():
            agent_action = await self.observation_queue.get()
            action_or_none = await self.astep(agent_action)
            if action_or_none is not None:
                await self.send_observations(action_or_none)
            self.observation_queue.task_done()

    async def booting(self) -> None:
        logger.info("Booting moderator and waiting for agents...")
        while not self.all_agents_awake.is_set():
            await self.send_observations(
                Observations(
                    observations_map={
                        output_channel: Observation(
                            agent_name="moderator",
                            last_turn=json.dumps(
                                {
                                    "use_pk_value": self.use_pk_value,
                                }
                            ),
                            turn_number=-1,
                            available_actions=["none"],
                        )
                        for output_channel, agent_name in self.agent_mapping.items()
                    }
                )
            )
            print("sent checking message to agents")
            await asyncio.sleep(0.2)
            while not self.observation_queue.empty():
                agent_action = await self.observation_queue.get()
                print(agent_action)
                if not self.agents_awake[agent_action.agent_name]:
                    self.agents_awake[agent_action.agent_name] = True
                    args: dict[str, Any] = json.loads(agent_action.argument)
                    self.agents_pk[agent_action.agent_name] = args["pk"]
                    self.agent_models[agent_action.agent_name] = args["model_name"]
            print(self.agents_awake)
            if False not in self.agents_awake.values():
                self.all_agents_awake.set()
                print("All agents are now awake and ready")

        # TODO: remove this once we have a better way to handle the redis_agent
        # if not self.redis_agent_as_actor:
        #     self.remove_redis_as_actor()

        # Initialize episode log with support for groups
        self.epilog = EpisodeLog(
            environment=self.scenario,
            agents=list(self.agents_pk.values()),
            tag=self.tag,
            models=list(self.agent_models.values()),
            messages=[[("Environment", "Environment", self.scenario)]],
            rewards=[0.0] * len(self.agents),
            rewards_prompt="",
            # Add group configuration
            groups=self.groups,
        )

        # Initial message to agents
        if self.action_order == "round-robin":
            await self.send_observations(
                Observations(
                    observations_map={
                        output_channel: Observation(
                            agent_name="moderator",
                            last_turn=self.scenario,
                            turn_number=0,
                            available_actions=self.available_actions
                            if agent_name == self.agents[0]
                            else ["none"],
                        )
                        for output_channel, agent_name in self.agent_mapping.items()
                    }
                )
            )
            self.current_agent_index += 1

        # Send initial epilog to RedisAgent
        if "moderator:redis_agent" in self.output_channel_types:
            await self.send_epilog(self.epilog, "moderator:redis_agent")

    def remove_redis_as_actor(self) -> None:
        return
        # Remove from output_channel_types
        if "moderator:redis_agent" in self.output_channel_types:
            self.output_channel_types.pop("moderator:redis_agent")

        # Remove from input_channel_types - need to use the correct key
        if "redis_agent:moderator" in self.input_channel_types:
            self.input_channel_types.pop("redis_agent:moderator")

        # Remove from agents list - check if it exists first
        if "redis_agent" in self.agents:
            self.agents.remove("redis_agent")

        # Remove from agent_mapping
        if "moderator:redis_agent" in self.agent_mapping:
            self.agent_mapping.pop("moderator:redis_agent")

        if "redis_agent" in self.agents_pk:
            self.agents_pk.pop("redis_agent")

        if "redis_agent" in self.agent_models:
            self.agent_models.pop("redis_agent")

        if "redis_agent" in self.agents_awake:
            self.agents_awake.pop("redis_agent")

    async def wrap_up_and_stop(self) -> None:
        """Finalize the episode and clean up resources"""
        self.shutdown_event.set()
        try:
            await asyncio.sleep(0.1)
            print("all agents have left, wrap up and stop")

            # Make sure the final epilog has been sent to RedisAgent
            # if "moderator:redis_agent" in self.output_channel_types:
            await self.send_epilog(self.epilog, "moderator:redis_agent")

            # Save to database if requested
            if self.push_to_db:
                self.epilog.save()
        except Exception as e:
            print(f"error in wrap_up_and_stop: {e}")
        await asyncio.sleep(0.5)
        print("result of this episode:\n", self.epilog.model_dump_json())
        await self.r.publish(
            "shutdown:moderator",
            "shutdown",
        )

    async def setup_groups(self, groups_data: dict) -> None:
        """
        Configure agent groups

        Args:
            groups_data: Dictionary mapping group names to lists of agent names
        """
        self.groups = groups_data

        # Update epilog with groups configuration
        self.epilog.groups = groups_data

        print(f"Updated groups configuration: {self.groups}")

        # Send updated epilog
        # if "moderator:redis_agent" in self.output_channel_types:
        #     await self.send_epilog(self.epilog, "moderator:redis_agent")

    async def set_mode(self, mode: str) -> None:
        """
        Set the communication mode

        Args:
            mode: Either "full" for normal operation or "group" for group messaging
        """
        if mode not in ["full", "group"]:
            print(f"Invalid mode: {mode}. Must be 'full' or 'group'")
            return

        self.mode = mode
        print(f"Communication mode set to: {self.mode}")

        # Send updated epilog
        # if "moderator:redis_agent" in self.output_channel_types:
        #     await self.send_epilog(self.epilog, "moderator:redis_agent")

    async def handle_unified_message(self, agent_action: AgentAction) -> Observations:
        """
        Process a unified message and route to appropriate agents
        with strict isolation for direct messages.

        Args:
            agent_action: The action containing the message

        Returns:
            Observations: Observations to send to agents
        """
        try:
            # Parse the message data
            arg_data = json.loads(agent_action.argument)

            # Extract message data
            sender = agent_action.agent_name
            content = arg_data.get("content", "")
            target_agents = arg_data.get("target_agents", [])
            original_target_agents = arg_data.get("original_target_agents", [])
            original_target_groups = arg_data.get("original_target_groups", [])
            context = arg_data.get("context", "individual")
            is_response = context == "response"
            is_dm = len(original_target_agents) == 1 and context == "individual"
            is_broadcast = (
                not original_target_agents and not original_target_groups
            ) or context == "broadcast"

            # Record message relationship for response tracking
            for agent in target_agents:
                if agent in self.agents:
                    self.message_senders[agent] = sender
                    if agent in self.message_receivers:
                        if sender not in self.message_receivers[agent]:
                            self.message_receivers[agent].append(sender)
                    else:
                        self.message_receivers[agent] = [sender]

            # Add message to epilog with appropriate format
            if original_target_groups:
                # Group message
                for group_name in original_target_groups:
                    if group_name in self.groups:
                        # Create complete message data with all metadata
                        message_data = {
                            "content": content,
                            "context": context,
                            "target_agents": original_target_agents,
                            "target_groups": original_target_groups,
                            "is_response": is_response,
                        }

                        # Add to epilog
                        receiver = f"Group:{group_name}"
                        self.epilog.messages.append(
                            [(sender, receiver, json.dumps(message_data))]
                        )

            elif is_dm:
                # Direct message to a single agent
                target_agent = original_target_agents[0]
                # Create complete message data with all metadata
                message_data = {
                    "content": content,
                    "context": "individual",
                    "target_agents": [target_agent],
                    "target_groups": [],
                    "is_response": is_response,
                }

                # Add to epilog
                receiver = f"Agent:{target_agent}"
                self.epilog.messages.append(
                    [(sender, receiver, json.dumps(message_data))]
                )

            elif is_broadcast:
                # Broadcast message
                message_data = {
                    "content": content,
                    "context": "broadcast",
                    "target_agents": self.agents,
                    "target_groups": [],
                    "is_response": is_response,
                }

                # Add to epilog
                self.epilog.messages.append(
                    [(sender, "Broadcast", json.dumps(message_data))]
                )

            elif is_response:
                # Response message - handle specially
                responding_to = arg_data.get("responding_to", {})
                original_sender = responding_to.get("sender", "unknown")

                # Create complete message data with all metadata
                message_data = {
                    "content": content,
                    "context": "response",
                    "target_agents": [original_sender],
                    "target_groups": [],
                    "responding_to": responding_to,
                    "is_response": True,
                }

                # Add to epilog
                receiver = f"Response:{original_sender}"
                self.epilog.messages.append(
                    [(sender, receiver, json.dumps(message_data))]
                )

            # Increment turn counter for new messages
            if not is_response:
                self.turn_number += 1

            # Send updated epilog to Redis agent
            # if "moderator:redis_agent" in self.output_channel_types:
            print("Epilog updated, sending to redis_agent")
            await self.send_epilog(self.epilog, "moderator:redis_agent")

            # Create observations for all agents
            observations_map = {}

            for output_channel, agent_name in self.agent_mapping.items():
                if agent_name == 'redis_agent':
                    continue
                # By default, use empty message and no actions
                message_content = ""
                available_actions = ["none"]
                redis_involved_actions = ["none", "leave", "speak"]

                # Determine if this agent should receive this message
                should_receive_message = False

                if is_dm:
                    # For DMs, only the target and sender see the message
                    target_agent = original_target_agents[0]
                    if agent_name == target_agent or agent_name == sender:
                        should_receive_message = True
                        if sender == "redis_agent":
                            available_actions = redis_involved_actions
                        else:
                            available_actions = self.available_actions

                elif is_response:
                    # For responses, only the original sender gets it
                    original_sender = arg_data.get("responding_to", {}).get(
                        "sender", None
                    )
                    if agent_name == original_sender or agent_name == sender:
                        should_receive_message = True
                        if sender == "redis_agent":
                            available_actions = redis_involved_actions
                        else:
                            available_actions = self.available_actions

                elif is_broadcast:
                    # For broadcasts, everyone gets it
                    should_receive_message = True

                    # In round-robin mode, only the current agent gets actions
                    if self.action_order == "round-robin":
                        if sender == "redis_agent":
                            available_actions = (
                                redis_involved_actions
                                if agent_name
                                == self.agents[self.current_agent_index % len(self.agents)]
                                else ["none"]
                            )
                        else:
                            available_actions = (
                                self.available_actions
                                if agent_name
                                == self.agents[self.current_agent_index % len(self.agents)]
                                else ["none"]
                            )
                    else:
                        available_actions = self.available_actions

                elif original_target_groups:
                    # For group messages, check if agent is in any of the target groups
                    in_target_group = False
                    for group_name in original_target_groups:
                        if (
                            group_name in self.groups
                            and agent_name in self.groups[group_name]
                        ):
                            in_target_group = True
                            break

                    # Agent receives message if they're in a target group or they're the sender
                    if in_target_group or agent_name == sender:
                        should_receive_message = True
                        available_actions = self.available_actions

                # Set the message content if this agent should receive it
                if should_receive_message:
                    message_content = content
                else:
                    continue

                # Create the observation
                observation = Observation(
                    agent_name=sender,
                    last_turn=message_content,
                    turn_number=self.turn_number,
                    available_actions=available_actions,
                )

                observations_map[output_channel] = observation

            return Observations(observations_map=observations_map)

        except json.JSONDecodeError:
            print(
                f"Error: Failed to parse unified message data: {agent_action.argument}"
            )
            return Observations(observations_map={})
        except Exception as e:
            print(f"Error handling unified message: {e}")
            return Observations(observations_map={})

    async def astep(self, action: AgentAction) -> Observations | None:
        """Process an agent action and return observations"""
        # Handle mode and group setup commands
        if action.action_type == "set_mode":
            try:
                mode_data = json.loads(action.argument)
                await self.set_mode(mode_data.get("mode", "full"))
            except Exception as e:
                print(f"Error setting mode: {e}")
            return None

        elif action.action_type == "setup_groups":
            try:
                groups_data = json.loads(action.argument)
                await self.set_mode(groups_data.get("mode", "group"))
                await self.setup_groups(groups_data.get("groups", {}))
            except Exception as e:
                print(f"Error setting groups: {e}")
            return None

        # Handle unified messages (group mode)
        elif action.agent_name == "redis_agent":
            return await self.handle_unified_message(action)

        # Handle regular speak actions - try to detect if it's a DM response
        elif action.action_type == "speak":
            # Check if this agent has a known sender (it's responding to someone)
            agent_name = action.agent_name
            original_sender = self.message_senders.get(agent_name)

            if original_sender and self.mode == "group":
                # This is likely a response to a DM or targeted message
                # Convert to unified message format targeting only the original sender
                unified_action = AgentAction(
                    agent_name=action.agent_name,
                    output_channel=action.output_channel,
                    action_type="speak",
                    argument=json.dumps(
                        {
                            "content": action.argument,
                            "target_agents": [
                                original_sender
                            ],  # Only send to original sender
                            "original_target_agents": [original_sender],
                            "original_target_groups": [],
                            "context": "response",
                            "responding_to": {"sender": original_sender},
                        }
                    ),
                )
                return await self.handle_unified_message(unified_action)

            # Regular broadcast message
            unified_action = AgentAction(
                agent_name=action.agent_name,
                output_channel=action.output_channel,
                action_type="speak",
                argument=json.dumps(
                    {
                        "content": action.argument,
                        "target_agents": self.agents,  # All agents for regular speak
                        "original_target_agents": [],
                        "original_target_groups": [],
                        "context": "broadcast",
                    }
                ),
            )
            return await self.handle_unified_message(unified_action)

        # Handle non-verbal communication
        elif action.action_type == "non-verbal communication":
            # Format non-verbal as *action*
            unified_action = AgentAction(
                agent_name=action.agent_name,
                output_channel=action.output_channel,
                action_type="non-verbal communication",
                argument=json.dumps(
                    {
                        "content": f"*{action.argument}*",  # Format as non-verbal
                        "target_agents": self.agents,
                        "original_target_agents": [],
                        "original_target_groups": [],
                        "context": "broadcast",
                    }
                ),
            )
            return await self.handle_unified_message(unified_action)

        # Handle physical actions
        elif action.action_type == "action":
            unified_action = AgentAction(
                agent_name=action.agent_name,
                output_channel=action.output_channel,
                action_type="action",
                argument=json.dumps(
                    {
                        "content": f"[{action.argument}]",  # Format as action
                        "target_agents": self.agents,
                        "original_target_agents": [],
                        "original_target_groups": [],
                        "context": "broadcast",
                    }
                ),
            )
            return await self.handle_unified_message(unified_action)

        # Handle leave action
        elif action.action_type == "leave":
            # Mark this agent as having left
            agent_name = action.agent_name
            if agent_name in self.agents_awake:
                self.agents_awake[agent_name] = False

            # Send a message that the agent has left
            unified_action = AgentAction(
                agent_name=action.agent_name,
                output_channel=action.output_channel,
                action_type="leave",
                argument=json.dumps(
                    {
                        "content": f"{agent_name} has left the conversation.",
                        "target_agents": self.agents,
                        "original_target_agents": [],
                        "original_target_groups": [],
                        "context": "broadcast",
                    }
                ),
            )

            result = await self.handle_unified_message(unified_action)

            # Check if all agents have left and we should wrap up
            if all(not awake for awake in self.agents_awake.values()):
                await self.wrap_up_and_stop()

            return result

        # Handle no action
        elif action.action_type == "none":
            # No action needed, just return None
            return None

        # Handle unknown action type
        else:
            print(f"Unknown action type: {action.action_type}")
            return None

    async def aeval(self, epilog: EpisodeLog) -> EpisodeLog:
        """
        evaluate the episode
        will send the epilog to evaluators, and wait for the evaluation to be finished
        """
        assert len(self.evaluator_channels) == 1, "currently only support one evaluator"

        for evaluator_channel in self.evaluator_channels:
            await self.send_epilog(epilog, evaluator_channel[1])

        print("episode eval started")

        for _ in range(
            len(self.evaluator_channels)
        ):  # the queue will take in input and output from this channel
            raw_res = await self.observation_queue.get()
            res = json.loads(raw_res.argument)
            epilog.rewards = res["reward"]
            epilog.rewards_prompt = res["reward_prompt"]

        print("episode eval finished")
        return epilog