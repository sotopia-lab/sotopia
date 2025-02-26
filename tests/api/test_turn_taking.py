from fastapi.testclient import TestClient
import pytest
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
from typing import Callable, Generator, Any
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
                "tmppk_agent1",
                SimpleMessage(message="Welcome to the simulation").to_natural_language(),
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
    """
    Fixture to create dummy data:
    - Two AgentProfiles (tmppk_agent1 and tmppk_agent2)
    - One EnvironmentProfile (tmppk_env_profile) with agent goals
    - One RelationshipProfile between the two agents
    - A custom evaluation dimension and list
    Then yields control to the tests and performs cleanup afterward.
    """
    # Create dummy agent profiles
    def _create_mock_agent_profiles() -> None:
        AgentProfile(
            first_name="John",
            last_name="Doe",
            occupation="tester",
            gender="male",
            pk="tmppk_agent1",
            tag="test_tag",
        ).save()
        AgentProfile(
            first_name="Jane",
            last_name="Doe",
            occupation="tester",
            gender="female",
            pk="tmppk_agent2",
            tag="test_tag",
        ).save()

    # Create a dummy environment profile
    def _create_mock_env_profile() -> None:
        env_profile = EnvironmentProfile(
            codename="test_codename",
            scenario="A",
            agent_goals=["Goal1", "Goal2"],
            pk="tmppk_env_profile",
            tag="test_tag",
        )
        env_profile.save()

    # Create a dummy relationship profile
    def _create_mock_relationship() -> None:
        RelationshipProfile(
            pk="tmppk_relationship",
            agent_1_id="tmppk_agent1",
            agent_2_id="tmppk_agent2",
            relationship=1.0,
        ).save()

    # Create a dummy evaluation dimension and list
    def _create_mock_evaluation_dimension() -> None:
        CustomEvaluationDimension(
            pk="tmppk_evaluation_dimension",
            name="test_dimension",
            description="test description",
            range_high=10,
            range_low=-10,
        ).save()
        CustomEvaluationDimensionList(
            pk="tmppk_evaluation_dimension_list",
            name="test_dimension_list",
            dimension_pks=["tmppk_evaluation_dimension"],
        ).save()

    _create_mock_agent_profiles()
    _create_mock_env_profile()
    _create_mock_relationship()
    _create_mock_evaluation_dimension()

    yield

    # Cleanup code: Delete dummy data from the database.
    try:
        AgentProfile.delete("tmppk_agent1")
    except Exception as e:
        print(f"Error deleting tmppk_agent1: {e}")
    try:
        AgentProfile.delete("tmppk_agent2")
    except Exception as e:
        print(f"Error deleting tmppk_agent2: {e}")
    try:
        EnvironmentProfile.delete("tmppk_env_profile")
    except Exception as e:
        print(f"Error deleting tmppk_env_profile: {e}")
    try:
        RelationshipProfile.delete("tmppk_relationship")
    except Exception as e:
        print(f"Error deleting tmppk_relationship: {e}")
    try:
        EpisodeLog.delete("tmppk_episode_log")
    except Exception as e:
        print(f"Error deleting tmppk_episode_log: {e}")
    try:
        CustomEvaluationDimension.delete("tmppk_evaluation_dimension")
    except Exception as e:
        print(f"Error deleting tmppk_evaluation_dimension: {e}")
    try:
        CustomEvaluationDimensionList.delete("tmppk_evaluation_dimension_list")
    except Exception as e:
        print(f"Error deleting tmppk_evaluation_dimension_list: {e}")


def test_basic_turn_taking(create_mock_data: Callable[[], None], caplog) -> None:
    """
    Test sending a single TURN_REQUEST and verifying that a TURN_RESPONSE is returned.
    """
    caplog.set_level(logging.DEBUG)
    logging.info("Starting test_basic_turn_taking")
    with client.websocket_connect("/ws/simulation?token=test") as websocket:
        # Send the START_SIM message to initialize simulation.
        start_msg = {
            "type": "START_SIM",
            "data": {
                "env_id": "tmppk_env_profile",
                "agent_ids": ["tmppk_agent1", "tmppk_agent2"],
                "agent_models": ["gpt-4o-mini", "gpt-4o-mini"],
                "evaluator_model": "gpt-4o",
                "evaluation_dimension_list_name": "test_dimension_list",
                "mode": "turn"
            },
        }
        websocket.send_json(start_msg)
        # Optionally, wait for an initial SERVER_MSG indicating simulation start.
        # server_init = websocket.receive_json()
        # logging.info(f"Received initial message: {server_init}")
        # assert server_init["type"] in ("SERVER_MSG", "messages")
        
        # Send a TURN_REQUEST.
        turn_request = {
            "type": "TURN_REQUEST",
            "data": {
                "agent_id": "John Doe",
                "content": "Hello, how are you?"
            }
        }
        websocket.send_json(turn_request)
        logging.info("Sent TURN_REQUEST.")
        response = websocket.receive_json()
        logging.info(f"Received TURN_RESPONSE: {response}")
        assert response["type"] == "TURN_RESPONSE"
        data = response["data"]
        assert "agent_response" in data
        assert "action_type" in data
        # End the simulation.
        websocket.send_json({"type": "FINISH_SIM"})
        end_response = websocket.receive_json()
        assert end_response["type"] == "END_SIM"
        error_logs = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
        if error_logs:
            for rec in error_logs:
                logging.debug(f"Logged error: {rec.getMessage()}")


def test_multi_turn_conversation(create_mock_data: Callable[[], None]) -> None:
    """
    Test multiple back-and-forth TURN_REQUEST messages to ensure the conversation
    continues to evolve correctly.
    """
    with client.websocket_connect("/ws/simulation?token=test") as websocket:
        # Initialize simulation.
        start_msg = {
            "type": "START_SIM",
            "data": {
                "env_id": "tmppk_env_profile",
                "agent_ids": ["tmppk_agent1", "tmppk_agent2"],
                "agent_models": ["gpt-4o-mini", "gpt-4o-mini"],
                "evaluator_model": "gpt-4o",
                "evaluation_dimension_list_name": "test_dimension_list",
                "mode": "turn"
            },
        }
        websocket.send_json(start_msg)
        # Send several TURN_REQUEST messages.
        for i in range(3):
            turn_request = {
                "type": "TURN_REQUEST",
                "data": {
                    "agent_id": "Jane Doe",
                    "content": f"Turn {i+1}: message content"
                }
            }
            websocket.send_json(turn_request)
            response = websocket.receive_json()
            assert response["type"] == "TURN_RESPONSE"
            data = response["data"]
            assert "action_type" in data
            if data["action_type"] == "leave":
                assert data["agent_response"] == ""
                break
            else:
                assert data["agent_response"]  # non-empty response expected
        
        websocket.send_json({"type": "FINISH_SIM"})


def test_alternating_agents(create_mock_data: Callable[[], None]) -> None:
    """
    Test a conversation that alternates between two different agents.
    """
    with client.websocket_connect("/ws/simulation?token=test") as websocket:
        start_msg = {
            "type": "START_SIM",
            "data": {
                "env_id": "tmppk_env_profile",
                "agent_ids": ["tmppk_agent1", "tmppk_agent2"],
                "agent_models": ["gpt-4o-mini", "gpt-4o-mini"],
                "evaluator_model": "gpt-4o",
                "evaluation_dimension_list_name": "test_dimension_list",
                "mode": "turn"
            },
        }
        websocket.send_json(start_msg)
        agents = ["Jane Doe", "John Doe"]
        for i in range(4):
            turn_request = {
                "type": "TURN_REQUEST",
                "data": {
                    "agent_id": agents[i % 2],
                    "content": f"Message from turn {i+1}"
                }
            }
            websocket.send_json(turn_request)
            response = websocket.receive_json()
            assert response["type"] == "TURN_RESPONSE"
            data = response["data"]
            assert data["agent_id"] == agents[i % 2]
        
        websocket.send_json({"type": "FINISH_SIM"})


def test_invalid_agent_id(create_mock_data: Callable[[], None]) -> None:
    """
    Test that providing an invalid agent_id in the TURN_REQUEST yields an ERROR message.
    """
    with client.websocket_connect("/ws/simulation?token=test") as websocket:
        start_msg = {
            "type": "START_SIM",
            "data": {
                "env_id": "tmppk_env_profile",
                "agent_ids": ["tmppk_agent1", "tmppk_agent2"],
                "agent_models": ["gpt-4o-mini", "gpt-4o-mini"],
                "evaluator_model": "gpt-4o",
                "evaluation_dimension_list_name": "test_dimension_list",
                "mode": "turn"
            },
        }
        websocket.send_json(start_msg)
        turn_request = {
        "type": "TURN_REQUEST",
        "data": {
            "agent_id": "nonexistent_agent",
            "content": "This should trigger an error"
        }
        }
        websocket.send_json(turn_request)
        error_response = websocket.receive_json()
        assert error_response["type"] == "ERROR"
        assert "not found" in error_response["data"]["details"]
        
        websocket.send_json({"type": "FINISH_SIM"})


def test_full_simulation_streaming(create_mock_data: Callable[[], None]) -> None:
    """
    Test the full simulation streaming mode (run_simulation) by collecting several streaming
    SERVER_MSG messages before ending the simulation.
    """
    with client.websocket_connect("/ws/simulation?token=test") as websocket:
        start_msg = {
            "type": "START_SIM",
            "data": {
                "env_id": "tmppk_env_profile",
                "agent_ids": ["tmppk_agent1", "tmppk_agent2"],
                "agent_models": ["gpt-4o-mini", "gpt-4o-mini"],
                "evaluator_model": "gpt-4o",
                "evaluation_dimension_list_name": "test_dimension_list",
                "mode": "full"
            },
        }
        websocket.send_json(start_msg)
        messages = []
        # Collect several streaming messages.
        while len(messages) < 2:
            msg = websocket.receive_json()
            assert msg["type"] in ["SERVER_MSG", "messages"]
            messages.append(msg)
        
        websocket.send_json({"type": "FINISH_SIM"})



