from typing import Literal

from sotopia.agents import Agents, LLMAgent
from sotopia.envs import ParallelSotopiaEnv
from sotopia.generation_utils.generate import LLM_Name
from sotopia.messages import AgentAction, Message


def run_sync_server(
    model_name: LLM_Name,
    action_order: Literal["simutaneous", "round-robin", "random"],
    partial_background_file: str | None = None,
) -> list[tuple[str, str, Message]]:

    # Create Environment and agents
    # This step will be moved to outside this function

    env = ParallelSotopiaEnv(model_name=model_name, action_order=action_order)
    if partial_background_file:
        environment_messages = env.reset(
            options={"partial_background_file": partial_background_file}
        )
    else:
        environment_messages = env.reset()
    agents = Agents()
    for agent_name in env.agents:
        agents[agent_name] = LLMAgent(agent_name)
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
