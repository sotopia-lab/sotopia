from sotopia.agents import LLMAgent
from sotopia.database import AgentProfile, EnvironmentProfile
from sotopia.envs.parallel import ParallelSotopiaEnv


def test_create_env_profile() -> None:
    env_profile = EnvironmentProfile(
        scenario="The conversation between two friends in a cafe",
        agent_goals=[
            "trying to figure out the gift preference of the other agent, but not let them know you are buying gift for them",
            "to have a good time",
        ],
    )

    env_profile.save()
    pk = env_profile.pk
    env = ParallelSotopiaEnv(uuid_str=pk)
    assert env.profile == env_profile
    env.close()
    EnvironmentProfile.delete(pk)


def test_create_agent_profile() -> None:
    agent_profile = AgentProfile(
        first_name="John",
        last_name="Doe",
    )
    agent_profile.save()
    pk = agent_profile.pk
    agent = LLMAgent(uuid_str=pk)
    assert agent.profile == agent_profile
    AgentProfile.delete(pk)
