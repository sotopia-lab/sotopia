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

from typing import Literal, Any, AsyncIterator, Dict, List, Optional
from pydantic import Field

from .datamodels import AgentAction, Observation
from sotopia.messages import ActionType
from .logs import EpisodeLog


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
    ) -> None:
        print([(channel[0], AgentAction) for channel in evaluator_channels])
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

        # Attributes for group messaging support
        self.groups: Dict[str, List[str]] = {}  # Dictionary mapping group names to lists of agent names
        self.mode: str = "full"  # Communication mode: "full" or "group"
        self.external_users: set[str] = set()  # Set of external users connected via WebSocket
        
        # Message context tracking - simplified approach matching RedisAgent
        self.message_senders: Dict[str, str] = {}  # agent -> original sender
        self.message_receivers: Dict[str, List[str]] = {}  # agent -> list of recipients
        self.message_id_counter: int = 0  # Counter for generating unique message IDs
        
        # Track the last epilog hash to avoid duplicates
        self.last_epilog_hash: str | None = None

        assert (not self.evaluate_episode) or len(
            evaluator_channels
        ) > 0, "if evaluate_episode is True, evaluator_channels should not be empty"

        self.epilog: EpisodeLog  # will be initialized in booting process

        if self.action_order == "round-robin":
            pass
        else:
            raise NotImplementedError(
                "the selected action order is currently not implemented"
            )

    def remove_redis_as_actor(self) -> None:
        """Remove RedisAgent from actors list when it's only used as a message broker"""
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

    async def __aenter__(self) -> Self:
        print(f"Starting moderator with scenario: {self.scenario}")
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
        current_hash = hashlib.md5(epilog_json.encode()).hexdigest()
        
        # Only send if it's different from the last epilog we sent
        if current_hash != self.last_epilog_hash:
            message_json = Message[Observation](
                data=Observation(
                    agent_name="epilog",
                    last_turn=epilog_json,
                    turn_number=self.turn_number,
                    available_actions=self.available_actions,
                )
            ).model_dump_json()
            await self.send(output_channel, message_json)
            self.last_epilog_hash = current_hash

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
        """Boot the moderator and initialize communication with agents"""
        print("Booting moderator and waiting for agents...")
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
                if not self.agents_awake[agent_action.agent_name]:
                    self.agents_awake[agent_action.agent_name] = True
                    args: dict[str, Any] = json.loads(agent_action.argument)
                    self.agents_pk[agent_action.agent_name] = args["pk"]
                    self.agent_models[agent_action.agent_name] = args["model_name"]
            if False not in self.agents_awake.values():
                self.all_agents_awake.set()
                print("All agents are now awake and ready")

        # TODO: remove this once we have a better way to handle the redis_agent
        if not self.redis_agent_as_actor:
            self.remove_redis_as_actor()

        # Initialize the episode log
        self.epilog = EpisodeLog(
            environment=self.scenario,
            agents=list(self.agents_pk.values()),
            tag=self.tag,
            models=list(self.agent_models.values()),
            messages=[[("Environment", "Environment", self.scenario)]],
            rewards=[0.0] * len(self.agents),
            rewards_prompt="",
        )
        
        # Add groups to epilog if it supports them
        try:
            self.epilog.groups = self.groups  # Add group configuration if supported
        except (AttributeError, TypeError):
            # EpisodeLog doesn't support groups attribute
            pass
        
        # Send initial scenario to agents
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
        await self.send_epilog(self.epilog, "moderator:redis_agent")

    async def wrap_up_and_stop(self) -> None:
        """Finalize the episode and clean up resources"""
        self.shutdown_event.set()
        try:
            await asyncio.sleep(0.1)
            print("all agents have left, wrap up and stop")
            
            # Make sure the final epilog has been sent to RedisAgent
            await self.send_epilog(self.epilog, "moderator:redis_agent")
            
            # Save to database if requested
            if self.push_to_db:
                self.epilog.save()
        except Exception as e:
            print(f"error in wrap_up_and_stop: {e}")
        
        await asyncio.sleep(0.5)
        print("result of this episode:\n", self.epilog.model_dump_json())
        
        # Publish shutdown signal
        await self.r.publish(
            "shutdown:moderator",
            "shutdown",
        )

    async def episode_log_to_messages(
        self, epilog: EpisodeLog
    ) -> list[tuple[str, str, str]]:
        """Convert epilog to a list of message tuples"""
        messages = []
        for turn_number, turn in enumerate(epilog.messages):
            for message in turn:
                messages.append((message[0], message[1], message[2]))
        return messages

    async def aeval(self, epilog: EpisodeLog) -> EpisodeLog:
        """
        Evaluate the episode by sending to evaluators and processing results
        
        Args:
            epilog: The episode log to evaluate
            
        Returns:
            EpisodeLog: The evaluated episode log with rewards
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
        
    async def setup_groups(self, groups_data: dict) -> None:
        """
        Configure agent groups
        
        Args:
            groups_data: Dictionary mapping group names to lists of agent names
        """
        self.groups = groups_data
        
        # Update epilog with groups configuration if supported
        try:
            if hasattr(self, 'epilog'):
                self.epilog.groups = groups_data
        except (AttributeError, TypeError):
            # EpisodeLog doesn't support groups attribute
            pass
            
        print(f"Updated groups configuration: {self.groups}")

    async def set_mode(self, mode: str) -> None:
        """
        Set the communication mode
        
        Args:
            mode: Either "full" for normal operation or "group" for group messaging
        """
        self.mode = mode
        print(f"Communication mode set to: {self.mode}")
        
    def generate_message_id(self) -> str:
        """Generate a unique message ID"""
        self.message_id_counter += 1
        return f"msg_{self.message_id_counter}"
        
    async def handle_unified_message(self, agent_action: AgentAction) -> Observations:
        """
        Process a unified message and route to appropriate agents
        
        Args:
            agent_action: The action containing the message
            
        Returns:
            Observations: Observations to send to agents
        """
        try:
            # Parse the message data
            message_data = json.loads(agent_action.argument)
            sender = agent_action.agent_name
            content = message_data.get("content", "")
            target_agents = message_data.get("target_agents", [])
            original_target_agents = message_data.get("original_target_agents", [])
            original_target_groups = message_data.get("original_target_groups", [])
            context = message_data.get("context", "individual")
            is_response = context == "response"
            
            # Track external users
            if sender not in self.agents and sender != "Environment":
                self.external_users.add(sender)
                
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
                        # Create message content with context information
                        message_content = json.dumps({
                            "content": content,
                            "context": context,
                            "target_agents": original_target_agents,
                            "target_groups": original_target_groups,
                            "is_response": is_response
                        })
                        
                        # Add to epilog
                        receiver = f"Group:{group_name}"
                        self.epilog.messages.append([
                            (sender, receiver, message_content)
                        ])
            
            elif original_target_agents:
                # Individual message
                for target_agent in original_target_agents:
                    # Create message content with context information
                    message_content = json.dumps({
                        "content": content,
                        "context": "individual",
                        "target_agents": [target_agent],
                        "is_response": is_response
                    })
                    
                    # Add to epilog
                    receiver = f"Agent:{target_agent}"
                    self.epilog.messages.append([
                        (sender, receiver, message_content)
                    ])
            
            elif context == "broadcast":
                # Broadcast message
                message_content = json.dumps({
                    "content": content,
                    "context": "broadcast",
                    "target_agents": self.agents,
                    "is_response": is_response
                })
                
                # Add to epilog
                self.epilog.messages.append([
                    (sender, "Broadcast", message_content)
                ])
            
            elif context == "response":
                # Response message - handle specially
                responding_to = message_data.get("responding_to", {})
                original_sender = responding_to.get("sender", "unknown")
                
                # Create message content with context information
                message_content = json.dumps({
                    "content": content,
                    "context": "response",
                    "responding_to": responding_to,
                    "is_response": True
                })
                
                # Add to epilog
                receiver = f"Response:{original_sender}"
                self.epilog.messages.append([
                    (sender, receiver, message_content)
                ])
            
            # Increment turn counter for new messages
            if not is_response:
                self.turn_number += 1
            
            # Create observations for all agents
            observations_map = {}
            for output_channel, agent_name in self.agent_mapping.items():
                # Determine if this agent should receive the message
                is_recipient = agent_name in target_agents
                
                # Set available actions based on whether this agent should respond
                available_actions = self.available_actions if is_recipient else ["none"]
                
                # Create the observation with basic info
                observation = Observation(
                    agent_name=sender,  # Set sender as the agent_name in observation
                    last_turn=content if is_recipient else "",  # Only include content for recipients
                    turn_number=self.turn_number,
                    available_actions=available_actions,
                )
                
                observations_map[output_channel] = observation
            
            # Send updated epilog to Redis agent
            await self.send_epilog(self.epilog, "moderator:redis_agent")
            
            return Observations(observations_map=observations_map)
            
        except json.JSONDecodeError:
            print(f"Error: Failed to parse unified message data: {agent_action.argument}")
            return Observations(observations_map={})
        except Exception as e:
            print(f"Error handling unified message: {e}")
            return Observations(observations_map={})

    async def astep(self, agent_action: AgentAction) -> Observations | None:
        """Process an agent action step"""
        # Handle setup_groups action
        if agent_action.action_type == "setup_groups":
            try:
                groups_data = json.loads(agent_action.argument).get("groups", {})
                await self.setup_groups(groups_data)
                return None
            except Exception as e:
                print(f"Error setting up groups: {e}")
                return None
        
        # Handle set_mode action
        if agent_action.action_type == "set_mode":
            try:
                mode_data = json.loads(agent_action.argument)
                await self.set_mode(mode_data.get("mode", "full"))
                return None
            except Exception as e:
                print(f"Error setting mode: {e}")
                return None
                
        # Handle unified_message action
        if agent_action.action_type == "unified_message":
            return await self.handle_unified_message(agent_action)
        
        # Handle responses based on message sender info in group mode
        if self.mode == "group" and agent_action.action_type == "speak":
            agent_name = agent_action.agent_name
            content = agent_action.argument
            
            # Check if this agent has a known sender (it's responding to a message)
            original_sender = self.message_senders.get(agent_name)
            if original_sender:
                # Create a unified message targeting the original sender
                try:
                    return await self.handle_unified_message(AgentAction(
                        agent_name=agent_name,
                        output_channel=agent_action.output_channel,
                        action_type="unified_message",
                        argument=json.dumps({
                            "content": content,
                            "target_agents": [original_sender],  # Response goes to original sender
                            "original_target_agents": [original_sender],
                            "original_target_groups": [],
                            "context": "response",
                            "responding_to": {
                                "sender": original_sender
                            }
                        })
                    ))
                except Exception as e:
                    print(f"Error converting speak to response message: {e}")
            
            # Special handling for regular "speak" actions in group mode when no known sender
            # In group mode, regular speak actions are converted to unified messages
            # targeting all agents if no target is specified
            try:
                # Create a unified message targeting all agents
                return await self.handle_unified_message(AgentAction(
                    agent_name=agent_action.agent_name,
                    output_channel=agent_action.output_channel,
                    action_type="unified_message",
                    argument=json.dumps({
                        "content": agent_action.argument,
                        "target_agents": self.agents,  # Target all agents
                        "original_target_agents": [],
                        "original_target_groups": [],
                        "context": "broadcast"  # Indicate this is a broadcast
                    })
                ))
            except Exception as e:
                print(f"Error converting speak to unified message: {e}")
                # Fall through to normal processing if conversion fails
        
        # Original astep functionality for standard messages
        # Add message to epilog with proper format depending on mode
        if self.mode == "full" or agent_action.action_type != "speak":
            # Standard message handling for non-speak actions or full mode
            self.epilog.messages.append([
                (
                    agent_action.agent_name,
                    "Environment",
                    agent_action.to_natural_language(),
                )
            ])
        
        # Handle leave actions
        if agent_action.action_type == "leave":
            self.agents_awake[agent_action.agent_name] = False
            # Skip redis_agent when checking if all agents have left
            if True not in self.agents_awake.values():
                if self.evaluate_episode:
                    self.epilog = await self.aeval(self.epilog)
                await self.send_epilog(self.epilog, "moderator:redis_agent")
                await self.wrap_up_and_stop()
                return None
                
        # Handle none actions
        if agent_action.action_type == "none":
            return None

        # Send updated epilog to Redis agent after any action that modifies it
        await self.send_epilog(self.epilog, "moderator:redis_agent")

        # Handle turn limits
        if self.turn_number < self.max_turns:
            self.turn_number += 1
        else:
            # Max turns reached, force agents to leave
            return Observations(
                observations_map={
                    output_channel: Observation(
                        agent_name=agent_name,
                        last_turn=agent_action.to_natural_language(),
                        turn_number=self.turn_number,
                        available_actions=["leave"],
                    )
                    for output_channel, agent_name in self.agent_mapping.items()
                }
            )

        # Create observations for next round
        observations_map: dict[str, Observation] = {}
        for output_channel, _ in self.output_channel_types.items():
            agent_name = self.agent_mapping[output_channel]
            available_actions = ["none"]
            if self.action_order == "round-robin":
                if agent_name == self.agents[self.current_agent_index]:
                    available_actions = list(self.available_actions)
                    print(f"available_actions: {available_actions}")
            observation = Observation(
                agent_name=agent_name,
                last_turn=agent_action.to_natural_language(),
                turn_number=self.turn_number,
                available_actions=available_actions,
            )
            observations_map[output_channel] = observation

        # Update the agent index for round-robin
        self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)
        return Observations(observations_map=observations_map)