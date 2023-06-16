import asyncio
import functools
import itertools
from typing import Literal, cast

from beartype import beartype

from sotopia.agents import Agents, HumanAgent, LLMAgent, SpeakAgent
from sotopia.database import EpisodeLog
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
)
from sotopia.generation_utils.generate import LLM_Name
from sotopia.messages import AgentAction, Message, Observation
from sotopia.samplers import UniformSampler


@beartype
def run_sync_server(
    model_name_dict: dict[str, LLM_Name],
    action_order: Literal["simutaneous", "round-robin", "random"],
    agents_info: dict[str, dict[str, str]] | None = None,
    partial_background_file: str | None = None,
    full_background_file: str | None = None,
    mode: str | None = None,
) -> list[tuple[str, str, Message]]:

    # Create Environment and agents
    # This step will be moved to outside this function

    env = ParallelSotopiaEnv(
        model_name=model_name_dict["env"],
        action_order=action_order,
        evaluators=[
            RuleBasedTerminatedEvaluator(),
        ],
    )
    if partial_background_file:
        environment_messages = env.reset(
            options={"partial_background_file": partial_background_file}
        )
    elif full_background_file:
        environment_messages = env.reset(
            options={"full_background_file": full_background_file}
        )
    else:
        environment_messages = env.reset()
    agents = Agents()
    agents_model_names = [model_name_dict["agent1"], model_name_dict["agent2"]]
    for agent_name, agent_model in zip(env.agents, agents_model_names):
        if agent_model == "human":
            agents[agent_name] = HumanAgent(agent_name)
        elif mode == "speak":
            agents[agent_name] = SpeakAgent(agent_name, model_name=agent_model)
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
            if agents_info is not None:
                agents[agent_name].goal = agents_info[agent_name]["goal"]
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
) -> list[tuple[str, str, Message]]:

    # Create Environment and agents
    # This step will be moved to outside this function

    env_params = {
        "model_name": model_dict["env"],
        "action_order": action_order,
        "evaluators": [
            ReachGoalLLMEvaluator(model_dict["env"]),
            RuleBasedTerminatedEvaluator(max_turn_number=2),
        ],
    }
    agents_model_dict = {
        "agent1": model_dict["agent1"],
        "agent2": model_dict["agent2"],
    }
    sampler = UniformSampler[Observation, AgentAction]()
    env, agent_list = sampler.sample(
        agent_classes=[
            LLMAgent if model_name != "human" else HumanAgent
            for model_name in agents_model_dict.values()
        ],
        n_agent=len(agents_model_dict),
        env_params=env_params,
        agents_params=[
            {"model_name": model_name} if model_name != "human" else {}
            for model_name in agents_model_dict.values()
        ],
    )

    agents = Agents({agent.agent_name: agent for agent in agent_list})
    environment_messages = env.reset(agents=agents)
    agents_model_names = [model_dict["agent1"], model_dict["agent2"]]
    for agent_name, agent_model in zip(env.agents, agents_model_names):
        if agent_model == "human":
            agents[agent_name] = HumanAgent(agent_name)
        else:
            agents[agent_name] = LLMAgent(agent_name, model_name=agent_model)
    agents.reset()

    messages: list[list[tuple[str, str, Message]]] = []

    # Main Event Loop
    done = False
    messages.append(
        [
            ("Environment", agent_name, environment_messages[agent_name])
            for agent_name in env.agents
        ]
    )
    # set goal for agents
    for index, agent_name in enumerate(env.agents):
        agents[agent_name].goal = env.profile.agent_goals[index]
    rewards: list[list[float]] = []
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

            messages[-1].append(
                (agent_name, "Environment", agent_messages[agent_name])
            )

        # send agent messages to environment
        (
            environment_messages,
            rewards_in_turn,
            terminated,
            ___,
            ____,
        ) = await env.astep(agent_messages)
        messages.append(
            [
                ("Environment", agent_name, environment_messages[agent_name])
                for agent_name in env.agents
            ]
        )
        rewards.append(
            [rewards_in_turn[agent_name] for agent_name in env.agents]
        )
        done = all(terminated.values())

    EpisodeLog(
        environment=env.profile.pk,
        agents=[agent.profile.pk for agent in agent_list],
        messages=[
            [
                (m[0], m[1], m[2].to_natural_language())
                for m in messages_in_turn
            ]
            for messages_in_turn in messages
        ],
        rewards=rewards,
    ).save()
    # flatten nested list messages
    return list(itertools.chain(*messages))
