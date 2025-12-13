import asyncio
import sys
import json

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from aact import Message, NodeFactory, Node
from aact.messages import DataModel, DataModelFactory

from typing import Literal, Any, AsyncIterator
from pydantic import Field

from .datamodels import AgentAction, Observation
from sotopia.messages import ActionType
from .logs import EpisodeLog

import logging
from rich.logging import RichHandler

# Configure logger with rich formatting
log = logging.getLogger("sotopia.moderator")
log.setLevel(logging.INFO)
# Prevent propagation to root logger
log.propagate = False
log.addHandler(RichHandler(rich_tracebacks=True, show_time=True))


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
        log.info("Booting moderator and waiting for agents...")
        asyncio.create_task(self.booting())
        self.task_scheduler = asyncio.create_task(self._task_scheduler())
        log.info("Moderator booted successfully")
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
                    log.info(f"Agent {agent_action.argument} is awake")
                    args: dict[str, Any] = json.loads(agent_action.argument)
                    self.agents_pk[agent_action.agent_name] = args["pk"]
                    self.agent_models[agent_action.agent_name] = args["model_name"]
            if False not in self.agents_awake.values():
                self.all_agents_awake.set()
                print("All agents are now awake and ready")

        # TODO: remove this once we have a better way to handle the redis_agent
        if not self.redis_agent_as_actor:
            self.remove_redis_as_actor()

        self.epilog = EpisodeLog(
            environment=self.scenario,
            agents=list(self.agents_pk.values()),
            tag=self.tag,
            models=list(self.agent_models.values()),
            messages=[[("Environment", "Environment", self.scenario)]],
            rewards=[0.0] * len(self.agents),
            rewards_prompt="",
        )
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

    async def wrap_up_and_stop(self) -> None:
        self.shutdown_event.set()
        try:
            await asyncio.sleep(0.1)
            print("all agents have left, wrap up and stop")
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

    async def episode_log_to_messages(
        self, epilog: EpisodeLog
    ) -> list[tuple[str, str, str]]:
        messages = []
        for turn_number, turn in enumerate(epilog.messages):
            for message in turn:
                messages.append((message[0], message[1], message[2]))
        return messages

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

    async def astep(self, agent_action: AgentAction) -> Observations | None:
        # message (sender, receivers (seperated by comma), message content)
        self.epilog.messages.append(
            [
                (
                    agent_action.agent_name,
                    "Environment",
                    agent_action.to_natural_language(),
                )
            ]
        )
        if agent_action.action_type == "leave":
            self.agents_awake[agent_action.agent_name] = False
            # Skip redis_agent when checking if all agents have left
            if True not in self.agents_awake.values():
                if self.evaluate_episode:
                    self.epilog = await self.aeval(self.epilog)
                await self.send_epilog(self.epilog, "moderator:redis_agent")
                await self.wrap_up_and_stop()
                return None
        if agent_action.action_type == "none":
            return None

        await self.send_epilog(self.epilog, "moderator:redis_agent")

        if self.turn_number < self.max_turns:
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

        self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)
        return Observations(observations_map=observations_map)
