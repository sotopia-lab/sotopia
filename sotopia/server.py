import asyncio
from typing import Literal, cast

from beartype import beartype

from sotopia.agents import Agents, HumanAgent, LLMAgent
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
)
from sotopia.generation_utils.generate import LLM_Name
from sotopia.messages import AgentAction, Message


@beartype
def run_sync_server(
    model_dict: dict[str, LLM_Name],
    action_order: Literal["simutaneous", "round-robin", "random"],
    partial_background_file: str | None = None,
) -> list[tuple[str, str, Message]]:

    # Create Environment and agents
    # This step will be moved to outside this function

    env = ParallelSotopiaEnv(
        model_name=model_dict["env"],
        action_order=action_order,
        evaluators=[
            ReachGoalLLMEvaluator(model_dict["env"]),
            RuleBasedTerminatedEvaluator(),
        ],
    )
    if partial_background_file:
        environment_messages = env.reset(
            options={"partial_background_file": partial_background_file}
        )
    else:
        environment_messages = env.reset()
    agents = Agents()
    agents_model_names = [model_dict["agent1"], model_dict["agent2"]]
    for agent_name, agent_model in zip(env.agents, agents_model_names):
        if agent_model == "human":
            agents[agent_name] = HumanAgent(agent_name)
        else:
            agents[agent_name] = LLMAgent(agent_name, model_name=agent_model)
    agents.reset()

    messages: list[tuple[str, str, Message]] = []

    # Main Event Loop
    done = False
    for agent_name in env.agents:
        messages.append(
            ("Environment", agent_name, environment_messages[agent_name])
        )
    while not done:
        # gather agent messages
        agent_messages: dict[str, AgentAction] = dict()
        for agent_name in env.agents:
            agent_messages[agent_name] = agents[agent_name].act(
                environment_messages[agent_name]
            )
            messages.append(
                (agent_name, "Environment", agent_messages[agent_name])
            )

        # send agent messages to environment
        environment_messages, _, terminated, ___, ____ = env.step(
            agent_messages
        )
        for agent_name in env.agents:
            messages.append(
                ("Environment", agent_name, environment_messages[agent_name])
            )
        done = all(terminated.values())

    return messages


@beartype
async def run_async_server(
    model_dict: dict[str, LLM_Name],
    action_order: Literal["simutaneous", "round-robin", "random"],
    partial_background_file: str | None = None,
) -> list[tuple[str, str, Message]]:

    # Create Environment and agents
    # This step will be moved to outside this function

    env = ParallelSotopiaEnv(
        model_name=model_dict["env"],
        action_order=action_order,
        evaluators=[
            ReachGoalLLMEvaluator(model_dict["env"]),
            RuleBasedTerminatedEvaluator(),
        ],
    )
    if partial_background_file:
        environment_messages = env.reset(
            options={"partial_background_file": partial_background_file}
        )
    else:
        environment_messages = env.reset()
    agents = Agents()
    agents_model_names = [model_dict["agent1"], model_dict["agent2"]]
    for agent_name, agent_model in zip(env.agents, agents_model_names):
        if agent_model == "human":
            agents[agent_name] = HumanAgent(agent_name)
        else:
            agents[agent_name] = LLMAgent(agent_name, model_name=agent_model)
    agents.reset()

    messages: list[tuple[str, str, Message]] = []

    # Main Event Loop
    done = False
    for agent_name in env.agents:
        messages.append(
            ("Environment", agent_name, environment_messages[agent_name])
        )
    while not done:
        # gather agent messages
        agent_messages: dict[str, AgentAction] = dict()
        actions = await asyncio.gather(
            *[
                agents[agent_name].aact(environment_messages[agent_name])
                for agent_name in env.agents
            ]
        )
        actions = cast(list[AgentAction], actions)
        for idx, agent_name in enumerate(env.agents):
            agent_messages[agent_name] = actions[idx]

            messages.append(
                (agent_name, "Environment", agent_messages[agent_name])
            )

        # send agent messages to environment
        environment_messages, _, terminated, ___, ____ = await env.astep(
            agent_messages
        )
        for agent_name in env.agents:
            messages.append(
                ("Environment", agent_name, environment_messages[agent_name])
            )
        done = all(terminated.values())

    return messages
