from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
)
from sotopia.agents import Agents, LLMAgent
from sotopia.messages import Observation, AgentAction
from sotopia.envs import ParallelSotopiaEnv
from sotopia.database import EnvironmentProfile, AgentProfile

from enum import Enum
from typing import TypedDict, Any
from pydantic import BaseModel
import asyncio
from typing import AsyncGenerator


class WSMessageType(str, Enum):
    SERVER_MSG = "SERVER_MSG"
    CLIENT_MSG = "CLIENT_MSG"
    ERROR = "ERROR"
    START_SIM = "START_SIM"
    END_SIM = "END_SIM"


class ErrorType(str, Enum):
    NOT_AUTHORIZED = "NOT_AUTHORIZED"
    SIMULATION_ALREADY_STARTED = "SIMULATION_ALREADY_STARTED"
    SIMULATION_NOT_STARTED = "SIMULATION_NOT_STARTED"
    SIMULATION_ISSUE = "SIMULATION_ISSUE"
    OTHER = "OTHER"


class MessageForRendering(TypedDict):
    role: str
    type: str
    content: str


class WSMessage(BaseModel):
    type: WSMessageType
    data: dict[str, Any]

    model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.type.value,  # TODO check whether we want to use the enum value or the enum itself
            "data": self.data,
        }


def get_env_agents(
    env_id: str, agent_ids: list[str]
) -> tuple[ParallelSotopiaEnv, Agents, dict[str, Observation]]:
    # environment_profile = EnvironmentProfile.find().all()[0]
    # agent_profiles = AgentProfile.find().all()[:2]
    environment_profile: EnvironmentProfile = EnvironmentProfile.get(env_id)
    agent_profiles: list[AgentProfile] = [
        AgentProfile.get(agent_id) for agent_id in agent_ids
    ]

    agent_list = [
        LLMAgent(
            agent_profile=agent_profile,
            model_name="gpt-4o-mini",
        )
        for agent_idx, agent_profile in enumerate(agent_profiles)
    ]
    for idx, goal in enumerate(environment_profile.agent_goals):
        agent_list[idx].goal = goal

    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env = ParallelSotopiaEnv(
        action_order="round-robin",
        model_name="gpt-4o-mini",
        evaluators=[
            RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
        ],
        terminal_evaluators=[
            ReachGoalLLMEvaluator(
                "gpt-4o",
                EvaluationForTwoAgents[SotopiaDimensions],
            ),
        ],
        env_profile=environment_profile,
    )

    environment_messages = env.reset(agents=agents, omniscient=False)
    agents.reset()

    return env, agents, environment_messages


class WebSocketSotopiaSimulator:
    def __init__(self, env_id: str, agent_ids: list[str]) -> None:
        self.env, self.agents, self.environment_messages = get_env_agents(
            env_id, agent_ids
        )
        self.messages: list[list[tuple[str, str, str]]] = []
        self.messages.append(
            [
                (
                    "Environment",
                    agent_name,
                    self.environment_messages[agent_name].to_natural_language(),
                )
                for agent_name in self.env.agents
            ]
        )
        for index, agent_name in enumerate(self.env.agents):
            self.agents[agent_name].goal = self.env.profile.agent_goals[index]

    async def run_one_step(self) -> AsyncGenerator[dict[str, Any], None]:
        done = False

        turn = self.messages[-1]
        messages_for_rendering = [
            {"role": "Background Info", "type": "info", "content": turn[0][2]},
            {"role": "Background Info", "type": "info", "content": turn[1][2]},
            {"role": "System", "type": "divider", "content": "Start Simulation"},
        ]
        for msg in messages_for_rendering:
            yield msg

        while not done:
            # gather agent messages
            agent_messages: dict[str, AgentAction] = dict()
            actions = await asyncio.gather(
                *[
                    self.agents[agent_name].aact(self.environment_messages[agent_name])
                    for agent_name in self.env.agents
                ]
            )

            for idx, agent_name in enumerate(self.env.agents):
                agent_messages[agent_name] = actions[idx]

                self.messages[-1].append(
                    (
                        agent_name,
                        "Environment",
                        agent_messages[agent_name].to_natural_language(),
                    )
                )

            # send agent messages to environment
            (
                self.environment_messages,
                rewards_in_turn,
                terminated,
                ___,
                info,
            ) = await self.env.astep(agent_messages)

            self.messages.append(
                [
                    (
                        "Environment",
                        agent_name,
                        self.environment_messages[agent_name].to_natural_language(),
                    )
                    for agent_name in self.env.agents
                ]
            )

            messages_in_this_turn = []
            for sender, receiver, message in self.messages[-2]:
                if receiver == "Environment":
                    if sender != "Environment":
                        if "did nothing" in message:
                            continue
                        else:
                            if "said:" in message:
                                messages_in_this_turn.append(f"{sender} {message}")
                            else:
                                messages_in_this_turn.append(f"{sender}: {message}")
                    else:
                        messages_in_this_turn.append(message)
            print("\n".join(messages_in_this_turn))
            yield {
                "role": "agent",  # TODO separate agent 1 and 2
                "type": "action",
                "content": messages_in_this_turn[0],
            }

            done = all(terminated.values())

        reasoning = info[self.env.agents[0]]["comments"]
        yield {
            "role": "agent",  # TODO separate agent 1 and 2
            "type": "comment",
            "content": reasoning,
        }
        rewards = [
            info[agent_name]["complete_rating"] for agent_name in self.env.agents
        ]
        yield {
            "role": "agent",  # TODO separate agent 1 and 2
            "type": "comment",
            "content": rewards,
        }
