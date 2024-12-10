import asyncio
import sys


if sys.version_info < (3, 11):
    pass
else:
    pass


from aact import Message, NodeFactory
from aact.messages import DataModel, DataModelFactory

from typing import Literal, Self
from pydantic import Field

from sotopia.database import EpisodeLog
from .base_agent import BaseAgent
from .datamodels import AgentAction, Observation
from sotopia.messages import ActionType


@DataModelFactory.register("observations")
class Observations(DataModel):
    observations_map: dict[str, Observation] = Field(
        description="the observations of the agents"
    )


@NodeFactory.register("moderator")
class Moderator(BaseAgent[AgentAction, Observation]):
    def __init__(
        self,
        input_channels: list[str],
        output_channels: list[str],
        scenario: str,
        agent_mapping: dict[str, str],
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

    async def send(self, action: Observations) -> None:
        for output_channel, output_channel_type in self.output_channel_types.items():
            if output_channel in action.observations_map:
                await self.r.publish(
                    output_channel,
                    Message[output_channel_type](  # type:ignore[valid-type]
                        data=action.observations_map[output_channel]
                    ).model_dump_json(),
                )

    async def __aenter__(self) -> Self:
        print(self.scenario)
        asyncio.create_task(self.booting())
        self.task_scheduler = asyncio.create_task(self._task_scheduler())
        return await super().__aenter__()

    async def _task_scheduler(self) -> None:
        await self.all_agents_awake.wait()
        return await super()._task_scheduler()

    async def booting(self) -> None:
        """
        1. send checking message to agents for every 5 seconds, until all agents are awake
        - the agent should use this waking message to send their information to moderator for record
        - if further information of the agents are needed, should be communicated through the booting process
        2. after all agents are awake, send agent[0] a message to allow the agent to start speaking
        """
        while not self.all_agents_awake.is_set():
            await self.send(
                Observations(
                    observations_map={
                        output_channel: Observation(
                            agent_name="moderator",
                            last_turn=self.scenario,
                            turn_number=0,
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

        for output_channel, agent_name in self.agent_mapping.items():
            if agent_name == self.agents[0]:
                await self.send(
                    Observations(
                        observations_map={
                            output_channel: Observation(
                                agent_name="moderator",
                                last_turn=self.scenario,
                                turn_number=0,
                                available_actions=self.available_actions,
                            )
                        }
                    )
                )
                break
        self.current_agent_index += 1

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

        if (
            self.turn_number < self.max_turns
        ):  # minor changes: from 20 to self.max_turns
            self.turn_number += 1
        else:
            await self.save()
            self.shutdown_event.set()
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
