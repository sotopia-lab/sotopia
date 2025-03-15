import asyncio
import sys
import json
import logging
from typing import Literal, Any, AsyncIterator, Dict, List, Set, Optional

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from aact import Message, NodeFactory, Node
from aact.messages import DataModel, DataModelFactory

from pydantic import Field

from .datamodels import AgentAction, Observation
from sotopia.messages import ActionType
from .logs import EpisodeLog


# Configure logging
logger = logging.getLogger(__name__)


@DataModelFactory.register("observations")
class Observations(DataModel):
    observations_map: Dict[str, Observation] = Field(
        description="the observations of the agents"
    )


@NodeFactory.register("moderator")
class Moderator(Node[AgentAction, Observation]):
    """
    Moderator for managing communication between agents.

    The Moderator:
    1. Receives messages from agents (including RedisAgent)
    2. Routes messages to appropriate agents based on targeting
    3. Manages turn-taking and conversation flow
    4. Handles episode evaluation
    5. Provides group-based message distribution
    """

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
        redis_agent_as_actor: bool = True,  # Changed to True by default
    ) -> None:
        super().__init__(
            node_name=node_name,  # Fixed: Added missing node_name parameter
            input_channel_types=[
                (input_channel, AgentAction) for input_channel in input_channels
            ]
            + [(channel[0], AgentAction) for channel in evaluator_channels],
            output_channel_types=[
                (output_channel, Observation) for output_channel in output_channels
            ],
            redis_url=redis_url,
        )
        self.observation_queue: asyncio.Queue[AgentAction] = asyncio.Queue()
        self.task_scheduler: asyncio.Task[None] | None = None
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.agent_mapping: Dict[str, str] = (
            agent_mapping  # Fixed: Using Dict instead of dict
        )
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
        self.agents_awake: Dict[str, bool] = {name: False for name in self.agents}
        self.all_agents_awake: asyncio.Event = asyncio.Event()
        self.evaluator_channels: list[list[str]] = evaluator_channels
        self.push_to_db: bool = push_to_db
        self.use_pk_value: bool = use_pk_value
        self.agents_pk: Dict[str, str] = {}
        self.agent_models: Dict[str, str] = {}
        self.redis_agent_as_actor: bool = redis_agent_as_actor
        self.evaluate_episode: bool = evaluate_episode

        # New fields for group-based routing
        self.npc_groups: Dict[
            str, List[str]
        ] = {}  # Map from group ID to list of NPC IDs
        self.active_npcs: Set[str] = set()  # Set of active NPC IDs
        self.current_npc_responses: Dict[str, str] = {}  # Collect responses from NPCs
        self.pending_client_messages: List[dict] = []  # Queue of messages to process
        self.group_mode: bool = False  # Flag for group-based routing

        assert (not self.evaluate_episode) or len(
            evaluator_channels
        ) > 0, "if evaluate_episode is True, evaluator_channels should not be empty"

        self.epilog: EpisodeLog  # will be initialized in booting process

    async def __aenter__(self) -> Self:
        """Set up the Moderator when entering async context"""
        logger.info(f"Starting moderator with scenario: {self.scenario}")
        asyncio.create_task(self.booting())
        self.task_scheduler = asyncio.create_task(self._task_scheduler())
        logger.info("Moderator booted successfully")
        return await super().__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Clean up when exiting async context"""
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
        """Send the epilog to other agents"""
        message_json = Message[Observation](
            data=Observation(
                agent_name="epilog",
                last_turn=epilog.model_dump_json(),
                turn_number=self.turn_number,
                available_actions=self.available_actions,
            )
        ).model_dump_json()
        await self.send(output_channel, message_json)

    async def send_to_npc(self, npc_id: str, content: str) -> None:
        """Send a message to a specific NPC"""
        output_channel = f"moderator:{npc_id}"
        if output_channel in self.output_channel_types:
            message_json = Message[Observation](
                data=Observation(
                    agent_name=npc_id,
                    last_turn=content,
                    turn_number=self.turn_number,
                    available_actions=self.available_actions,
                )
            ).model_dump_json()
            await self.send(output_channel, message_json)
            logger.info(f"Sent message to NPC {npc_id}")

    async def event_handler(
        self, channel: str, message: Message[AgentAction]
    ) -> AsyncIterator[tuple[str, Message[Observation]]]:
        """Handle incoming messages from channels"""
        if channel in self.input_channel_types:
            await self.observation_queue.put(message.data)
        else:
            raise ValueError(f"Invalid channel: {channel}")
            yield "", self.output_type()

    async def _task_scheduler(self) -> None:
        """Process messages from the observation queue"""
        await self.all_agents_awake.wait()
        while not self.shutdown_event.is_set():
            agent_action = await self.observation_queue.get()
            action_or_none = await self.astep(agent_action)
            if action_or_none is not None:
                await self.send_observations(action_or_none)
            self.observation_queue.task_done()

    async def booting(self) -> None:
        """Initialize the Moderator and wait for all agents to connect"""
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
            logger.info("Sent checking message to agents")
            await asyncio.sleep(0.2)
            while not self.observation_queue.empty():
                agent_action = await self.observation_queue.get()

                # Handle special case for start message from RedisAgent
                # Fixed: Added type checking to avoid comparison-overlap error
                if (
                    agent_action.action_type == "start"
                    and agent_action.agent_name == "redis_agent"
                ):
                    try:
                        start_data = json.loads(agent_action.argument)
                        if "npcs" in start_data:
                            self.active_npcs = set(start_data["npcs"])
                        if "groups" in start_data:
                            self.npc_groups = start_data["groups"]
                        self.group_mode = True
                        logger.info(
                            f"Received start message with {len(self.active_npcs)} NPCs and {len(self.npc_groups)} groups"
                        )
                    except Exception as e:
                        logger.error(f"Error parsing start message: {e}")

                # Handle normal agent initialization
                if not self.agents_awake.get(agent_action.agent_name, False):
                    self.agents_awake[agent_action.agent_name] = True
                    try:
                        args: Dict[str, Any] = json.loads(agent_action.argument)
                        self.agents_pk[agent_action.agent_name] = args.get("pk", "")
                        self.agent_models[agent_action.agent_name] = args.get(
                            "model_name", ""
                        )
                    except Exception as e:
                        # Handle case where argument isn't JSON
                        logger.warning(f"Failed to parse agent arguments: {e}")
                        self.agents_pk[agent_action.agent_name] = ""
                        self.agent_models[agent_action.agent_name] = ""

            if False not in self.agents_awake.values():
                self.all_agents_awake.set()
                logger.info("All agents are now awake and ready")

        self.epilog = EpisodeLog(
            environment=self.scenario,
            agents=list(self.agents_pk.values()),
            tag=self.tag,
            models=list(self.agent_models.values()),
            messages=[[("Environment", "Environment", self.scenario)]],
            rewards=[0.0] * len(self.agents),
            rewards_prompt="",
        )

        # Initialize with initial observations to all agents
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

    async def wrap_up_and_stop(self) -> None:
        """Clean up and terminate the simulation"""
        self.shutdown_event.set()
        try:
            await asyncio.sleep(0.1)
            logger.info("All agents have left, wrapping up and stopping")
            if self.push_to_db:
                self.epilog.save()
        except Exception as e:
            logger.error(f"Error in wrap_up_and_stop: {e}")
        await asyncio.sleep(0.5)
        logger.info(f"Result of this episode:\n{self.epilog.model_dump_json()}")
        await self.r.publish(
            "shutdown:moderator",
            "shutdown",
        )

    async def route_message_to_npcs(
        self,
        content: str,
        target_npcs: Optional[List[str]] = None,  # Fixed: Made parameter optional
        target_group: Optional[str] = None,  # Fixed: Made parameter optional
    ) -> Observations:
        """Route a message to specific NPCs or all NPCs in a group"""
        npcs_to_message = set()

        # Add specifically targeted NPCs
        if target_npcs:
            npcs_to_message.update(target_npcs)

        # Add all NPCs in the target group
        if target_group and target_group in self.npc_groups:
            npcs_to_message.update(self.npc_groups[target_group])

        # If no targeting specified and we're in group mode, message all active NPCs
        if not target_npcs and not target_group and self.group_mode:
            npcs_to_message.update(self.active_npcs)

        # Only message NPCs that exist in our mapping
        valid_npcs = set()
        for npc_id in npcs_to_message:
            output_channel = f"moderator:{npc_id}"
            if output_channel in self.output_channel_types:
                valid_npcs.add(npc_id)
            else:
                logger.warning(f"NPC {npc_id} not found in output channels")

        # Create observations for each target NPC
        observations_map = {}
        for npc_id in valid_npcs:
            output_channel = f"moderator:{npc_id}"
            observations_map[output_channel] = Observation(
                agent_name=npc_id,
                last_turn=content,
                turn_number=self.turn_number,
                available_actions=self.available_actions,
            )

        # Return observations to be sent
        return Observations(observations_map=observations_map)

    async def aeval(self, epilog: EpisodeLog) -> EpisodeLog:
        """
        Evaluate the episode
        Will send the epilog to evaluators and wait for the evaluation to be finished
        """
        assert len(self.evaluator_channels) == 1, "currently only support one evaluator"

        for evaluator_channel in self.evaluator_channels:
            await self.send_epilog(epilog, evaluator_channel[1])

        logger.info("Episode eval started")

        for _ in range(
            len(self.evaluator_channels)
        ):  # the queue will take in input and output from this channel
            raw_res = await self.observation_queue.get()
            res = json.loads(raw_res.argument)
            epilog.rewards = res["reward"]
            epilog.rewards_prompt = res["reward_prompt"]

        logger.info("Episode eval finished")
        return epilog

    async def astep(self, agent_action: AgentAction) -> Optional[Observations]:
        """
        Process an agent action and determine the next step in the simulation

        This method handles:
        - Regular agent messages
        - Client messages from RedisAgent
        - Agent leaving
        - Evaluation
        """
        # Special handling for client messages from RedisAgent
        if (
            agent_action.agent_name == "redis_agent"
            and agent_action.action_type == "speak"
        ):
            try:
                # Parse the message data
                message_data = json.loads(agent_action.argument)
                content = message_data.get("content", "")
                target_npcs = message_data.get("target_npcs", [])
                target_group = message_data.get("target_group", None)

                # Add to the episode log
                self.epilog.messages.append(
                    [
                        (
                            "Client",
                            "NPCs",
                            content,
                        )
                    ]
                )

                # Increment turn number for client messages
                self.turn_number += 1

                # Route the message to appropriate NPCs
                return await self.route_message_to_npcs(
                    content=content, target_npcs=target_npcs, target_group=target_group
                )
            except Exception as e:
                logger.error(f"Error routing client message: {e}")
                return None

        # Regular message handling for agent actions
        self.epilog.messages.append(
            [
                (
                    agent_action.agent_name,
                    "Environment",
                    agent_action.to_natural_language(),
                )
            ]
        )

        # Handle agent leaving
        if agent_action.action_type == "leave":
            self.agents_awake[agent_action.agent_name] = False
            # Skip redis_agent when checking if all agents have left
            remaining_agents = {
                k: v for k, v in self.agents_awake.items() if k != "redis_agent"
            }
            if True not in remaining_agents.values():
                if self.evaluate_episode:
                    self.epilog = await self.aeval(self.epilog)
                await self.send_epilog(self.epilog, "moderator:redis_agent")
                await self.wrap_up_and_stop()
                return None

        # Skip empty actions
        if agent_action.action_type == "none":
            return None

        # Always send the current state to RedisAgent
        await self.send_epilog(self.epilog, "moderator:redis_agent")

        # Check if we've reached max turns
        if self.turn_number < self.max_turns:
            if not self.group_mode:  # Only increment for regular turn-based mode
                self.turn_number += 1
        else:
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

        # For NPC responses in group mode, send to RedisAgent individually
        if self.group_mode and agent_action.agent_name in self.active_npcs:
            # Send this NPC's response to the RedisAgent
            redis_observations_map = {  # Fixed: Renamed variable to avoid redefinition
                "moderator:redis_agent": Observation(
                    agent_name=agent_action.agent_name,
                    last_turn=agent_action.argument,
                    turn_number=self.turn_number,
                    available_actions=["none"],
                )
            }
            return Observations(observations_map=redis_observations_map)

        # Normal turn-based progression for regular agents
        observations_map: Dict[str, Observation] = {}
        for output_channel, _ in self.output_channel_types.items():
            agent_name = self.agent_mapping.get(output_channel, "")
            if not agent_name:
                continue

            available_actions = ["none"]
            if self.action_order == "round-robin" and not self.group_mode:
                if agent_name == self.agents[self.current_agent_index]:
                    available_actions = list(self.available_actions)

            observation = Observation(
                agent_name=agent_name,
                last_turn=agent_action.to_natural_language(),
                turn_number=self.turn_number,
                available_actions=available_actions,
            )
            observations_map[output_channel] = observation

        if not self.group_mode:
            self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)

        return Observations(observations_map=observations_map)
