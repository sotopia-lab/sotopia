from sotopia.agents.base_agent import BaseAgent
from sotopia.agents.llm_agent import Agents
from sotopia.database import AgentProfile, EnvironmentProfile
from sotopia.envs import ParallelSotopiaEnv
from sotopia.renderers import RenderContext, XMLRenderer


def test_nested_visibility() -> None:
    assert (
        XMLRenderer()(
            "<root><a viewer='agent_1'>test<b viewer='agent_2'>test</b></a></root>",
            RenderContext(viewer="agent_2", tags_to_render=["a", "b"]),
        )
        == ""
    )
    assert (
        XMLRenderer()(
            "<root><a viewer='agent_2'>test<b viewer='agent_2'>test</b></a></root>",
            RenderContext(viewer="agent_2", tags_to_render=["a", "b"]),
        )
        == "testtest"
    )
    assert (
        XMLRenderer()(
            "<root><a viewer='agent_2'>test<b viewer='agent_2'>test</b></a></root>",
            RenderContext(viewer="agent_2", tags_to_render=["a"]),
        )
        == "test"
    )
    assert (
        XMLRenderer()(
            "<root><a viewer='agent_2'>test<b viewer='agent_2'>test</b></a></root>",
            RenderContext(viewer="agent_2", tags_to_render=["b"]),
        )
        == ""
    )
    assert (
        XMLRenderer()(
            "test<b viewer='agent_2'>test</b>test",
            RenderContext(viewer="agent_2", tags_to_render=["a", "b"]),
        )
        == "testtesttest"
    )


def test_renderer_in_env() -> None:
    env = ParallelSotopiaEnv(
        env_profile=EnvironmentProfile(
            scenario="test",
            agent_goals=["agent_1's goal", "agent_2's goal"],
        )
    )

    agents = Agents(
        {
            "John": BaseAgent(
                agent_profile=AgentProfile(
                    first_name="John",
                    last_name="Doe",
                )
            ),
            "Jane": BaseAgent(
                agent_profile=AgentProfile(
                    first_name="Jane",
                    last_name="Doe",
                )
            ),
        }
    )

    obs = env.reset(agents=agents)
    assert (
        obs["John"].last_turn == "Here is the context of this interaction:\n"
        "Scenario: test\n"
        "Participants: John and Jane\n"
        "John's background: John Doe is a 0-year-old  .  pronouns.  Personality and values description:  John's secrets: \n"
        "Jane's background: Jane Doe is a 0-year-old  .  pronouns.  Personality and values description:  Jane's secrets: \n"
        "John's goal: agent_1's goal\n"
        "Jane's goal: Unknown"
    )
