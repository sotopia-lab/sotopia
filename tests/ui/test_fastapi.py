from fastapi.testclient import TestClient
from sotopia.database import EnvironmentProfile, AgentProfile, EpisodeLog
from sotopia.messages import SimpleMessage
from sotopia.ui.fastapi_server import app
import pytest
from typing import Generator, Callable

client = TestClient(app)


def create_dummy_episode_log() -> None:
    episode = EpisodeLog(
        environment="tmppk_env_profile",
        agents=["tmppk_agent1", "tmppk_agent2"],
        messages=[
            [
                (
                    "tmppk_agent1",
                    "tmppk_agent2",
                    SimpleMessage(message="Hello").to_natural_language(),
                ),
                (
                    "tmppk_agent2",
                    "tmppk_agent1",
                    SimpleMessage(message="Hi").to_natural_language(),
                ),
            ],
            [
                (
                    "Environment",
                    "tmppk_agent2",
                    SimpleMessage(message="Hello").to_natural_language(),
                ),
                (
                    "tmppk_agent2",
                    "tmppk_agent1",
                    SimpleMessage(message="Hi").to_natural_language(),
                ),
            ],
        ],
        rewards=[
            (0, {"believability": 9.0}),
            (
                0,
                {
                    "believability": 9.0,
                    "relationship": 2.0,
                    "knowledge": 1.0,
                    "secret": 0.0,
                    "social_rules": 0.0,
                    "financial_and_material_benefits": 0.0,
                    "goal": 10.0,
                    "overall_score": 0,
                },
            ),
        ],
        reasoning="",
        pk="tmppk_episode_log",
        rewards_prompt="",
        tag="test_tag",
    )
    episode.save()


@pytest.fixture
def create_mock_data() -> Generator[None, None, None]:
    def _create_mock_agent_profile() -> None:
        AgentProfile(first_name="John", last_name="Doe", pk="tmppk_agent1").save()
        AgentProfile(first_name="Jane", last_name="Doe", pk="tmppk_agent2").save()

    def _create_mock_env_profile() -> None:
        env_profile = EnvironmentProfile(
            codename="test_codename",
            scenario="A",
            agent_goals=[
                "B",
                "C",
            ],
            pk="tmppk_env_profile",
        )
        env_profile.save()

    _create_mock_agent_profile()
    _create_mock_env_profile()

    yield

    AgentProfile.delete("tmppk_agent1")
    AgentProfile.delete("tmppk_agent2")
    EnvironmentProfile.delete("tmppk_env_profile")
    EpisodeLog.delete("tmppk_episode_log")


def test_get_scenarios_all(create_mock_data: Callable[[], None]) -> None:
    response = client.get("/scenarios")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_scenarios_by_id(create_mock_data: Callable[[], None]) -> None:
    response = client.get("/scenarios/id/tmppk_env_profile")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["scenario"] == "A"


def test_get_scenarios_by_codename(create_mock_data: Callable[[], None]) -> None:
    codename = "test_codename"
    response = client.get(f"/scenarios/codename/{codename}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["codename"] == codename


def test_get_agents_all(create_mock_data: Callable[[], None]) -> None:
    response = client.get("/agents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_agents_by_id(create_mock_data: Callable[[], None]) -> None:
    response = client.get("/agents/id/tmppk_agent1")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["first_name"] == "John"


def test_get_agents_by_gender(create_mock_data: Callable[[], None]) -> None:
    gender = "test_gender"
    response = client.get(f"/agents/gender/{gender}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["gender"] == gender


def test_get_agents_by_occupation(create_mock_data: Callable[[], None]) -> None:
    occupation = "test_occupation"
    response = client.get(f"/agents/occupation/{occupation}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["occupation"] == occupation


def test_get_episodes_by_id(create_mock_data: Callable[[], None]) -> None:
    create_dummy_episode_log()
    response = client.get("/episodes/id/tmppk_episode_log")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_episodes_by_tag(create_mock_data: Callable[[], None]) -> None:
    create_dummy_episode_log()
    tag = "test_tag"
    response = client.get(f"/episodes/tag/{tag}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["tag"] == tag


def test_create_agent(create_mock_data: Callable[[], None]) -> None:
    agent_data = {
        "first_name": "test_first_name",
        "last_name": "test_last_name",
    }
    response = client.post("/agents/", json=agent_data)
    assert response.status_code == 200
    assert isinstance(response.json(), str)


def test_create_scenario(create_mock_data: Callable[[], None]) -> None:
    scenario_data = {
        "codename": "test_codename",
        "scenario": "test_scenario",
        "tag": "test",
    }
    response = client.post("/scenarios/", json=scenario_data)
    assert response.status_code == 200
    assert isinstance(response.json(), str)


def test_delete_agent(create_mock_data: Callable[[], None]) -> None:
    response = client.delete("/agents/tmppk_agent1")
    assert response.status_code == 200
    assert isinstance(response.json(), str)


def test_delete_scenario(create_mock_data: Callable[[], None]) -> None:
    response = client.delete("/scenarios/tmppk_env_profile")
    assert response.status_code == 200
    assert isinstance(response.json(), str)
