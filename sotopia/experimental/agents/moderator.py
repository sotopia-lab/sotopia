import asyncio
import sys


if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


from aact import Message, NodeFactory, Node
from aact.messages import DataModel, DataModelFactory

from typing import Literal, Any, AsyncIterator
from pydantic import Field

from sotopia.database import EpisodeLog
from .datamodels import AgentAction, Observation
from sotopia.messages import ActionType


@DataModelFactory.register("observations")
class Observations(DataModel):
    observations_map: dict[str, Observation] = Field(
        description="the observations of the agents"
    )


@NodeFactory.register("moderator")
class Moderator(Node[AgentAction, Observation]):
    def __init__(
        self,
        input_channels: list[str],
        output_channels: list[str],
        scenario: str,
        agent_mapping: dict[str, str],
        node_name: str,
        agent_backgrounds: dict[str, str],
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
    ):
        super().__init__(
            input_channel_types=[
                (input_channel, AgentAction) for input_channel in input_channels
            ],
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
        self.action_order: Literal["simultaneous", "round-robin", "random"] = (
            action_order
        )
        self.available_actions: list[ActionType] = available_actions
        self.turn_number: int = 0
        self.max_turns: int = max_turns
        self.current_agent_index: int = 0
        self.scenario: str = scenario
        self.agents: list[str] = list(agent_mapping.values())
        self.agent_models: dict[str, str] = {}
        self.agents_awake: dict[str, bool] = {name: False for name in self.agents}
        self.all_agents_awake: asyncio.Event = asyncio.Event()
        self.message_history: list[list[tuple[str, str, str]]] = [
            [("Environment", "Environment", self.scenario)]
        ]
        self.push_to_db = push_to_db
        self.agent_backgrounds = agent_backgrounds

        if self.action_order == "round-robin":
            pass
        else:
            raise NotImplementedError(
                "the selected action order is currently not implemented"
            )

    async def __aenter__(self) -> Self:
        print(self.scenario)
        asyncio.create_task(self.booting())
        self.task_scheduler = asyncio.create_task(self._task_scheduler())
        return await super().__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.shutdown_event.set()
        if self.task_scheduler is not None:
            self.task_scheduler.cancel()
        return await super().__aexit__(exc_type, exc_value, traceback)

    async def send(self, observations: Observations) -> None:
        for output_channel, output_channel_type in self.output_channel_types.items():
            if output_channel in observations.observations_map:
                await self.r.publish(
                    output_channel,
                    Message[output_channel_type](  # type:ignore[valid-type]
                        data=observations.observations_map[output_channel]
                    ).model_dump_json(),
                )

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
            observation = await self.observation_queue.get()
            action_or_none = await self.aact(observation)
            if action_or_none is not None:
                await self.send(action_or_none)
            self.observation_queue.task_done()

    async def booting(self) -> None:
        """
        1. send checking message to agents for every 0.1 seconds, until all agents are awake
        - this message has turn_number of -1 for identification, agents should not record this into actual message_history
        - if the agent booted succesfully, he is expected to return its model name for record.
        2. (under round-robin action order)after all agents are awake, send agent[0] a message to allow the agent to start speaking
        """
        while not self.all_agents_awake.is_set():
            await self.send(
                Observations(
                    observations_map={
                        output_channel: Observation(
                            agent_name="moderator",
                            last_turn=self.scenario,
                            turn_number=-1,
                            available_actions=["none"],
                        )
                        for output_channel, agent_name in self.agent_mapping.items()
                    }
                )
            )
            await asyncio.sleep(0.1)
            while not self.observation_queue.empty():
                agent_action = await self.observation_queue.get()
                self.agents_awake[agent_action.agent_name] = True
                self.agent_models[agent_action.agent_name] = agent_action.argument
            if False not in self.agents_awake.values():
                self.all_agents_awake.set()

        if self.action_order == "round-robin":
            await self.send(
                Observations(
                    observations_map={
                        output_channel: Observation(
                            agent_name="moderator",
                            last_turn=self.agent_backgrounds[agent_name],
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
        if self.push_to_db:
            await self.save()
        await asyncio.sleep(0.5)
        print("stopping all agents")
        await self.r.publish(
            f"shutdown:{self.node_name}",
            "shutdown",
        )

    async def save(self) -> EpisodeLog:
        """
        save the EpisodeLog to redis, without evaluating
        TODO: specify what to be added inside tag
        TODO: update the code so that EpisodeLog.render_for_humans() can work
            -currently it cannot work because no AgentProfile has been uploaded to redis
            -such a process should be done back in the agents' end
            -also the current agentslist is consist of names, but not uuid's of agents
        """
        epilog = EpisodeLog(
            environment=self.scenario,
            agents=self.agents,
            tag=None,
            models=list(self.agent_models.values()),
            messages=self.message_history,
            reasoning="",
            rewards=[0] * len(self.agents),
            rewards_prompt="",
        )
        epilog.save()
        # print(epilog.render_for_humans())
        return epilog

    async def aact(self, agent_action: AgentAction) -> Observations | None:
        if agent_action.action_type == "leave":
            self.agents_awake[agent_action.agent_name] = False
            if True not in self.agents_awake.values():
                await self.wrap_up_and_stop()
                return None
        if agent_action.action_type == "none":
            return None

        if len(self.message_history) == 1:
            self.message_history[0].append(
                (
                    agent_action.agent_name,
                    "Environment",
                    agent_action.to_natural_language(),
                )
            )
        else:
            self.message_history.append(
                [
                    (
                        agent_action.agent_name,
                        "Environment",
                        agent_action.to_natural_language(),
                    )
                ]
            )

        if self.turn_number < self.max_turns:
            self.turn_number += 1
        else:
            return Observations(
                observations_map={
                    output_channel: Observation(
                        agent_name="moderator",
                        last_turn=self.scenario,
                        turn_number=self.turn_number + 1,
                        available_actions=["leave"],
                    )
                    for output_channel, agent_name in self.agent_mapping.items()
                }
            )

        observations_map: dict[str, Observation] = {}
        for output_channel, output_channel_type in self.output_channel_types.items():
            agent_name = self.agent_mapping[output_channel]
            available_actions: list[ActionType] = ["none"]
            if self.action_order == "round-robin":
                if agent_name == self.agents[self.current_agent_index]:
                    available_actions = self.available_actions

            observation = Observation(
                agent_name=agent_name,
                last_turn=agent_action.to_natural_language(),
                turn_number=self.turn_number,
                available_actions=available_actions,
            )
            observations_map[output_channel] = observation
        self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)

        return Observations(observations_map=observations_map)
