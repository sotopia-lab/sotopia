import pytest

from sotopia.agents import Agents, LLMAgent
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
)
from sotopia.envs import ParallelSotopiaEnv
from sotopia.messages import AgentAction


@pytest.mark.asyncio
async def test_parallel_sotopia_env() -> None:
    env_profile = EnvironmentProfile(
        pk="test_pk",
        code_name="test",
        scenario="test",
        agent_goals=["test 1", "test 2"],
    )
    env = ParallelSotopiaEnv(env_profile=env_profile)

    agents = Agents(
        {
            "agent1": LLMAgent(
                "agent1",
                model_name="gpt-4o-mini",
                agent_profile=AgentProfile(
                    **{
                        "first_name": "John",
                        "last_name": "Doe",
                        "pk": "test_pk_agent_1",
                    },
                ),
            ),
            "agent2": LLMAgent(
                "agent2",
                model_name="gpt-4o-mini",
                agent_profile=AgentProfile(
                    **{
                        "first_name": "Jane",
                        "last_name": "Doe",
                        "pk": "test_pk_agent_2",
                    }
                ),
            ),
        }
    )
    env.reset(agents=agents)
    max_steps = 5
    while env.agents:
        max_steps -= 1
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        await env.astep(actions)
        if not max_steps:
            break


@pytest.mark.asyncio
async def test_parallel_sotopia_env_script_writing_single_step() -> None:
    env_profile = EnvironmentProfile(
        pk="test_pk",
        code_name="test",
        scenario="test",
        agent_goals=["test 1", "test 2"],
    )
    env = ParallelSotopiaEnv(env_profile=env_profile)

    agents = Agents(
        {
            "agent1": LLMAgent(
                "agent1",
                model_name="gpt-4o-mini",
                agent_profile=AgentProfile(
                    **{
                        "first_name": "John",
                        "last_name": "Doe",
                        "pk": "test_pk_agent_1",
                    }
                ),
                script_like=True,
            ),
            "agent2": LLMAgent(
                "agent2",
                model_name="gpt-4o-mini",
                agent_profile=AgentProfile(
                    **{
                        "first_name": "Jane",
                        "last_name": "Doe",
                        "pk": "test_pk_agent_2",
                    }
                ),
                script_like=True,
            ),
        }
    )
    env.reset(agents=agents)

    max_steps = 5
    while env.agents:
        max_steps -= 1
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        await env.astep(actions)
        if not max_steps:
            break


@pytest.mark.asyncio
async def test_parallel_sotopia_env_multi_agents_private_messages() -> None:
    """
    Test if agent messages with `to` field (private messages) are correctly
    routed to corresponding agents (visible to sender and recipients only).
    """
    env_profile = EnvironmentProfile(
        pk="test_pk",
        code_name="test",
        scenario="test",
        agent_goals=["test 1", "test 2", "test 3"],
    )
    env = ParallelSotopiaEnv(env_profile=env_profile)

    agents = Agents(
        {
            "agent1": LLMAgent(
                "agent1",
                model_name="gpt-4o-mini",
                agent_profile=AgentProfile(
                    **{
                        "first_name": "John",
                        "last_name": "Doe",
                        "pk": "test_pk_agent_1",
                    },
                ),
            ),
            "agent2": LLMAgent(
                "agent2",
                model_name="gpt-4o-mini",
                agent_profile=AgentProfile(
                    **{
                        "first_name": "Jane",
                        "last_name": "Doe",
                        "pk": "test_pk_agent_2",
                    }
                ),
            ),
            "agent3": LLMAgent(
                "agent3",
                model_name="gpt-4o-mini",
                agent_profile=AgentProfile(
                    **{
                        "first_name": "Marry",
                        "last_name": "Doe",
                        "pk": "test_pk_agent_3",
                    }
                ),
            ),
        }
    )
    env.reset(agents=agents)

    actions = {
        "agent1": AgentAction(
            action_type="speak",
            argument="psst only for agent2",
            to=["agent2"],
        ),
        "agent2": AgentAction(action_type="speak", argument="hi all"),
        "agent3": AgentAction(
            action_type="speak",
            argument="psst only for agent1",
            to=["agent1"],
        ),
    }

    observations, _, _, _, _ = await env.astep(actions)

    # Private content is visible to sender and recipient
    # Note: private messages include the [private to ...] prefix
    assert (
        "agent1 [private to ['agent2']]  said: \"psst only for agent2\""
        in observations["agent1"].last_turn
    )
    assert (
        "agent1 [private to ['agent2']]  said: \"psst only for agent2\""
        in observations["agent2"].last_turn
    )
    assert (
        "agent3 [private to ['agent1']]  said: \"psst only for agent1\""
        in observations["agent1"].last_turn
    )
    assert (
        "agent3 [private to ['agent1']]  said: \"psst only for agent1\""
        in observations["agent3"].last_turn
    )

    # Private messages should NOT be visible to non-recipients
    assert (
        "agent1 [private to ['agent2']]  said: \"psst only for agent2\""
        not in observations["agent3"].last_turn
    )
    assert (
        "agent3 [private to ['agent1']]  said: \"psst only for agent1\""
        not in observations["agent2"].last_turn
    )

    # Public content is visible to everyone
    assert 'agent2  said: "hi all"' in observations["agent1"].last_turn
    assert 'agent2  said: "hi all"' in observations["agent2"].last_turn
    assert 'agent2  said: "hi all"' in observations["agent3"].last_turn


@pytest.mark.asyncio
async def test_parallel_sotopia_env_invalid_receipients() -> None:
    """
    Test if env throws error when message recipients are invalid
    """
    env_profile = EnvironmentProfile(
        pk="test_pk",
        code_name="test",
        scenario="test",
        agent_goals=["test 1", "test 2"],
    )
    env = ParallelSotopiaEnv(env_profile=env_profile)

    agents = Agents(
        {
            "agent1": LLMAgent(
                "agent1",
                model_name="gpt-4o-mini",
                agent_profile=AgentProfile(
                    **{
                        "first_name": "John",
                        "last_name": "Doe",
                        "pk": "test_pk_agent_1",
                    },
                ),
            ),
            "agent2": LLMAgent(
                "agent2",
                model_name="gpt-4o-mini",
                agent_profile=AgentProfile(
                    **{
                        "first_name": "Jane",
                        "last_name": "Doe",
                        "pk": "test_pk_agent_2",
                    }
                ),
            ),
        }
    )
    env.reset(agents=agents)

    actions = {
        "agent1": AgentAction(
            action_type="speak",
            argument="psst only for agent2",
            to=["invalid_agent"],
        ),
        "agent2": AgentAction(action_type="speak", argument="hi all"),
    }

    # Agent1's action to 'invalid_agent' should raise an error, see
    # `ParallelSotopiaEnv.astep()`
    with pytest.raises(ValueError, match=r"Invalid recipient.*in 'to'"):
        await env.astep(actions)
