from fastapi.testclient import TestClient
from sotopia.database import EnvironmentProfile, AgentProfile, EpisodeLog
from sotopia.messages import SimpleMessage
from sotopia.ui.fastapi_server import app

client = TestClient(app)


def create_mock_agent_profile() -> str:
    agent_profile = AgentProfile(
        first_name="John",
        last_name="Doe",
        age=30,
        occupation="test_occupation",
        gender="test_gender",
        gender_pronoun="He/Him",
        public_info="Public Info",
        big_five="OCEAN",
        moral_values=["Honesty", "Integrity"],
        schwartz_personal_values=["Self-Direction", "Stimulation"],
        personality_and_values="Personality and Values",
        decision_making_style="Analytical",
        secret="Secret",
        model_id="model_123",
        mbti="INTJ",
    )
    agent_profile.save()
    pk = agent_profile.pk
    assert pk is not None
    return pk


def delete_mock_agent_profile(pk: str) -> None:
    AgentProfile.delete(pk)


def create_mock_env_profile() -> str:
    env_profile = EnvironmentProfile(
        codename="test_codename",
        scenario="A",
        agent_goals=[
            "B",
            "C",
        ],
    )
    env_profile.save()
    pk = env_profile.pk
    assert pk is not None
    return pk


def delete_mock_env_profile(pk: str) -> None:
    EnvironmentProfile.delete(pk)


def create_mock_episode_log() -> str:
    agent1 = "agent1"
    agent2 = "agent2"
    episode_log = EpisodeLog(
        environment="test_environment",
        agents=[agent1, agent2],
        messages=[
            [
                (
                    agent1,
                    agent2,
                    SimpleMessage(message="Hello").to_natural_language(),
                ),
                (
                    agent2,
                    agent1,
                    SimpleMessage(message="Hi").to_natural_language(),
                ),
            ],
            [
                (
                    "Environment",
                    agent2,
                    SimpleMessage(message="Hello").to_natural_language(),
                ),
                (
                    agent2,
                    agent1,
                    SimpleMessage(message="Hi again").to_natural_language(),
                ),
            ],
        ],
        reasoning="Reasoning",
        rewards=[1.0, 2.0],
        rewards_prompt="Rewards Prompt",
        tag="test_tag",
    )
    episode_log.save()
    pk = episode_log.pk
    assert pk is not None
    return pk


def delete_mock_episode_log(pk: str) -> None:
    EpisodeLog.delete(pk)


def test_get_scenarios_all() -> None:
    pk = create_mock_env_profile()
    response = client.get("/scenarios")
    delete_mock_env_profile(pk)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_scenarios_by_id() -> None:
    pk = create_mock_env_profile()
    response = client.get(f"/scenarios/id/{pk}")
    delete_mock_env_profile(pk)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["scenario"] == "A"


def test_get_scenarios_by_codename() -> None:
    codename = "test_codename"
    pk = create_mock_env_profile()
    response = client.get(f"/scenarios/codename/{codename}")
    delete_mock_env_profile(pk)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["codename"] == codename


def test_get_agents_all() -> None:
    pk = create_mock_agent_profile()
    response = client.get("/agents")
    delete_mock_agent_profile(pk)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_agents_by_id() -> None:
    pk = create_mock_agent_profile()
    response = client.get(f"/agents/id/{pk}")
    delete_mock_agent_profile(pk)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["first_name"] == "John"


def test_get_agents_by_gender() -> None:
    gender = "test_gender"
    pk = create_mock_agent_profile()
    response = client.get(f"/agents/gender/{gender}")
    delete_mock_agent_profile(pk)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["gender"] == gender


def test_get_agents_by_occupation() -> None:
    occupation = "test_occupation"
    pk = create_mock_agent_profile()
    response = client.get(f"/agents/occupation/{occupation}")
    delete_mock_agent_profile(pk)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["occupation"] == occupation


def test_get_episodes_by_id() -> None:
    pk = create_mock_episode_log()
    response = client.get(f"/episodes/id/{pk}")
    delete_mock_episode_log(pk)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_episodes_by_tag() -> None:
    tag = "test_tag"
    pk = create_mock_episode_log()
    response = client.get(f"/episodes/tag/{tag}")
    delete_mock_episode_log(pk)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["tag"] == tag


def test_create_agent() -> None:
    agent_data = {
        "first_name": "test_first_name",
        "last_name": "test_last_name",
    }
    response = client.post("/agents/", json=agent_data)
    assert response.status_code == 200
    assert isinstance(response.json(), str)
    pk = response.json()
    delete_mock_agent_profile(pk)


def test_create_scenario() -> None:
    scenario_data = {
        "codename": "test_codename",
        "scenario": "test_scenario",
        "tag": "test",
    }
    response = client.post("/scenarios/", json=scenario_data)
    assert response.status_code == 200
    assert isinstance(response.json(), str)
    pk = response.json()
    delete_mock_env_profile(pk)


def test_delete_agent() -> None:
    pk = create_mock_agent_profile()
    response = client.delete(f"/agents/{pk}")
    assert response.status_code == 200
    assert isinstance(response.json(), str)


def test_delete_scenario() -> None:
    pk = create_mock_env_profile()
    response = client.delete(f"/scenarios/{pk}")
    assert response.status_code == 200
    assert isinstance(response.json(), str)
