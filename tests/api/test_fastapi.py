from fastapi.testclient import TestClient
from sotopia.database import (
    EnvironmentProfile,
    AgentProfile,
    EpisodeLog,
    RelationshipProfile,
    CustomEvaluationDimension,
    CustomEvaluationDimensionList,
)
from sotopia.messages import SimpleMessage
from sotopia.api.fastapi_server import app
import pytest
from typing import Generator, Callable, Any

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
def create_mock_data(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    for_posting = request.param if hasattr(request, "param") else False

    def _create_mock_agent_profile() -> None:
        AgentProfile(
            first_name="John",
            last_name="Doe",
            occupation="test_occupation",
            gender="test_gender",
            pk="tmppk_agent1",
            tag="test_tag",
        ).save()
        AgentProfile(
            first_name="Jane",
            last_name="Doe",
            occupation="test_occupation",
            gender="test_gender",
            pk="tmppk_agent2",
            tag="test_tag",
        ).save()

    def _create_mock_env_profile() -> None:
        env_profile = EnvironmentProfile(
            codename="test_codename",
            scenario="A",
            agent_goals=[
                "B",
                "C",
            ],
            pk="tmppk_env_profile",
            tag="test_tag",
        )
        env_profile.save()

    def _create_mock_relationship() -> None:
        RelationshipProfile(
            pk="tmppk_relationship",
            agent_1_id="tmppk_agent1",
            agent_2_id="tmppk_agent2",
            relationship=1.0,
        ).save()

    def _create_mock_evaluation_dimension() -> None:
        CustomEvaluationDimension(
            pk="tmppk_evaluation_dimension",
            name="test_dimension",
            description="test_description",
            range_high=10,
            range_low=-10,
        ).save()
        CustomEvaluationDimensionList(
            pk="tmppk_evaluation_dimension_list",
            name="test_dimension_list",
            dimension_pks=["tmppk_evaluation_dimension"],
        ).save()

    if not for_posting:
        _create_mock_agent_profile()
        _create_mock_env_profile()
        _create_mock_relationship()
        _create_mock_evaluation_dimension()
        print("created mock data")
    yield

    try:
        AgentProfile.delete("tmppk_agent1")
    except Exception as e:
        print(e)
    try:
        AgentProfile.delete("tmppk_agent2")
    except Exception as e:
        print(e)
    try:
        EnvironmentProfile.delete("tmppk_env_profile")
    except Exception as e:
        print(e)
    try:
        RelationshipProfile.delete("tmppk_relationship")
    except Exception as e:
        print(e)
    try:
        EpisodeLog.delete("tmppk_episode_log")
    except Exception as e:
        print(e)

    try:
        EpisodeLog.delete("tmppk_episode_log")
    except Exception as e:
        print(e)

    try:
        episodes = EpisodeLog.find(EpisodeLog.tag == "test_tag").all()
        for episode in episodes:
            EpisodeLog.delete(episode.pk)
    except Exception as e:
        print(e)

    try:
        CustomEvaluationDimension.delete("tmppk_evaluation_dimension")
    except Exception as e:
        print(e)
    try:
        CustomEvaluationDimensionList.delete("tmppk_evaluation_dimension_list")
    except Exception as e:
        print(e)


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


def test_get_relationship(create_mock_data: Callable[[], None]) -> None:
    response = client.get("/relationship/tmppk_agent1/tmppk_agent2")
    assert response.status_code == 200
    assert isinstance(response.json(), str)
    assert response.json() == "1: know_by_name"


def test_get_evaluation_dimensions(create_mock_data: Callable[[], None]) -> None:
    response = client.get("/evaluation_dimensions")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert response.json()["test_dimension_list"][0]["name"] == "test_dimension"


@pytest.mark.parametrize("create_mock_data", [True], indirect=True)
def test_create_agent(create_mock_data: Callable[[], None]) -> None:
    agent_data = {
        "pk": "tmppk_agent1",
        "first_name": "test_first_name",
        "last_name": "test_last_name",
    }
    response = client.post("/agents", json=agent_data)
    assert response.status_code == 200
    assert isinstance(response.json(), str)


@pytest.mark.parametrize("create_mock_data", [True], indirect=True)
def test_create_scenario(create_mock_data: Callable[[], None]) -> None:
    scenario_data = {
        "pk": "tmppk_env_profile",
        "codename": "test_codename",
        "scenario": "test_scenario",
        "tag": "test",
    }
    response = client.post("/scenarios", json=scenario_data)
    assert response.status_code == 200
    assert isinstance(response.json(), str)


@pytest.mark.parametrize("create_mock_data", [True], indirect=True)
def test_create_relationship(create_mock_data: Callable[[], None]) -> None:
    relationship_data = {
        "pk": "tmppk_relationship",
        "agent_1_id": "tmppk_agent1",
        "agent_2_id": "tmppk_agent2",
        "relationship": 1.0,
        "tag": "test_tag",
    }
    response = client.post("/relationship", json=relationship_data)
    assert response.status_code == 200
    assert isinstance(response.json(), str)


@pytest.mark.parametrize("create_mock_data", [True], indirect=True)
def test_create_evaluation_dimensions(create_mock_data: Callable[[], None]) -> None:
    evaluation_dimension_data = {
        "pk": "tmppk_evaluation_dimension_list",
        "name": "test_dimension_list",
        "dimensions": [
            {
                "pk": "tmppk_evaluation_dimension",
                "name": "test_dimension",
                "description": "test_description",
                "range_high": 10,
                "range_low": -10,
            }
        ],
    }
    response = client.post("/evaluation_dimensions", json=evaluation_dimension_data)
    print(response.json())
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


def test_delete_relationship(create_mock_data: Callable[[], None]) -> None:
    response = client.delete("/relationship/tmppk_relationship")
    assert response.status_code == 200
    assert isinstance(response.json(), str)


def test_delete_evaluation_dimension(create_mock_data: Callable[[], None]) -> None:
    response = client.delete("/evaluation_dimensions/tmppk_evaluation_dimension_list")
    assert response.status_code == 200
    assert isinstance(response.json(), str)


def test_websocket_simulate(create_mock_data: Callable[[], None]) -> None:
    LOCAL_MODEL = "custom/llama3.2:1b@http://localhost:8000/v1"
    with client.websocket_connect("/ws/simulation?token=test") as websocket:
        start_msg = {
            "type": "START_SIM",
            "data": {
                "env_id": "tmppk_env_profile",
                "agent_ids": ["tmppk_agent1", "tmppk_agent2"],
                "agent_models": [LOCAL_MODEL, LOCAL_MODEL],
                "evaluator_model": LOCAL_MODEL,
                "evaluation_dimension_list_name": "test_dimension_list",
            },
        }
        websocket.send_json(start_msg)

        # check the streaming response, stop when we received 2 messages
        messages: list[dict[str, Any]] = []
        while len(messages) < 2:
            message = websocket.receive_json()
            assert (
                message["type"] == "SERVER_MSG"
            ), f"Expected SERVER_MSG, got {message['type']}, full msg: {message}"
            messages.append(message)

        # send the end message
        end_msg = {
            "type": "FINISH_SIM",
        }
        websocket.send_json(end_msg)


# def test_simulate(create_mock_data: Callable[[], None]) -> None:
#     response = client.post(
#         "/simulate",
#         json={
#             "env_id": "tmppk_env_profile",
#             "agent_ids": ["tmppk_agent1", "tmppk_agent2"],
#             "models": [
#                 # "custom/llama3.2:1b@http://localhost:8000/v1",
#                 # "custom/llama3.2:1b@http://localhost:8000/v1",
#                 # "custom/llama3.2:1b@http://localhost:8000/v1"
#                 "gpt-4o-mini",
#                 "gpt-4o-mini",
#                 "gpt-4o-mini",
#             ],
#             "max_turns": 2,
#             "tag": "test_tag",
#         },
#     )
#     assert response.status_code == 200
#     assert isinstance(response.json(), str)
#     max_retries = 20
#     retry_count = 0
#     while retry_count < max_retries:
#         try:
#             status = NonStreamingSimulationStatus.find(
#                 NonStreamingSimulationStatus.episode_pk == response.json()
#             ).all()[0]
#             assert isinstance(status, NonStreamingSimulationStatus)
#             print(status)
#             if status.status == "Error":
#                 raise Exception("Error running simulation")
#             elif status.status == "Completed":
#                 # EpisodeLog.get(response.json())
#                 break
#             # Status is "Started", keep polling
#             time.sleep(1)
#             retry_count += 1
#         except Exception as e:
#             print(f"Error checking simulation status: {e}")
#             time.sleep(1)
#             retry_count += 1
#     else:
#         raise TimeoutError("Simulation timed out after 10 retries")
