import pytest
from typing import List, Dict, Any
import logging
from pydantic import BaseModel
from sotopia.database.persistent_profile import RelationshipType
<<<<<<< HEAD:tests/api/test_turntaking_atomic.py
from sotopia.api.websocket_utils import build_observation
=======

>>>>>>> 96b64ded57b634854ad22b2f5488adfdd66f8634:tests/api/more_tests.py
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

from sotopia.api.websocket_utils import (
    build_observation,
    get_env_agents,
)
from sotopia.agents import LLMAgent, Agents
from sotopia.messages import Observation, AgentAction
from sotopia.database import (
    EnvironmentProfile,
    AgentProfile,
    EvaluationDimensionBuilder,
)

# =============================================================================
# Test for build_observation
# =============================================================================


def test_build_observation():
    """
    Test that build_observation returns an Observation with the correct
    last_turn, turn_number and available_actions.
    """
    conversation_history = [
        {"role": "client", "content": "Hello"},
        {"role": "agent", "content": "Hi, how may I help you?"},
    ]
    turn_number = len(conversation_history)
    obs: Observation = build_observation(turn_number, conversation_history)
    assert obs.last_turn == "Hi, how may I help you?"
    assert obs.turn_number == turn_number
    expected_actions: List[str] = [
        "speak",
        "non-verbal communication",
        "action",
        "leave",
    ]
    assert obs.available_actions == expected_actions


# =============================================================================
# Test for get_env_agents
# =============================================================================


# Define dummy profile classes to simulate database objects.
class DummyAgentProfile:
    def __init__(
        self,
        pk: str,
        first_name: str = "John",
        last_name: str = "Doe",
        age: int = 30,
        occupation: str = "tester",
        gender: str = "male",
        gender_pronoun: str = "he",
        public_info: str = "",
        big_five: str = "",
        moral_values: list[str] = None,
        schwartz_personal_values: list[str] = None,
        personality_and_values: str = "",
        decision_making_style: str = "",
        secret: str = "",
        model_id: str = "",
        mbti: str = "",
        tag: str = "test_tag",
    ):
        self.pk = pk
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.occupation = occupation
        self.gender = gender
        self.gender_pronoun = gender_pronoun
        self.public_info = public_info
        self.big_five = big_five
        self.moral_values = moral_values or []
        self.schwartz_personal_values = schwartz_personal_values or []
        self.personality_and_values = personality_and_values
        self.decision_making_style = decision_making_style
        self.secret = secret
        self.model_id = model_id
        self.mbti = mbti
        self.tag = tag


class DummyEnvProfile:
    def __init__(self, pk: str):
        self.pk = pk
        self.codename = "test_codename"
        self.source = ""
        self.scenario = "A concrete scenario description that meets the guidelines."
        self.agent_goals = ["Goal1", "Goal2"]
<<<<<<< HEAD:tests/api/test_turntaking_atomic.py
        self.relationship = RelationshipType.stranger
=======
        # Provide a valid RelationshipType value:
        self.relationship = (
            RelationshipType.stranger
        )  # or another valid member of the enum
>>>>>>> 96b64ded57b634854ad22b2f5488adfdd66f8634:tests/api/more_tests.py
        self.age_constraint = None
        self.occupation_constraint = None
        self.agent_constraint = None
        self.tag = "test_tag"


def fake_agent_get(agent_id: str) -> DummyAgentProfile:
    dummy = {
        "agent1": DummyAgentProfile(
            "agent1", first_name="John", last_name="Doe", age=30
        ),
        "agent2": DummyAgentProfile(
            "agent2", first_name="Jane", last_name="Doe", age=25
        ),
    }
    if agent_id in dummy:
        return dummy[agent_id]
    raise Exception(f"AgentProfile with id {agent_id} not found")


def fake_env_get(env_id: str) -> DummyEnvProfile:
    return DummyEnvProfile(env_id)


@pytest.fixture
def mp(monkeypatch):
    monkeypatch.setattr(EnvironmentProfile, "get", fake_env_get)
    monkeypatch.setattr(AgentProfile, "get", fake_agent_get)
    monkeypatch.setattr(
        EvaluationDimensionBuilder,
        "select_existing_dimension_model_by_list_name",
        lambda list_name: BaseModel,
    )
    original_init = LLMAgent.__init__

    def patched_init(
        self,
        agent_name=None,
        uuid_str=None,
        agent_profile=None,
        model_name="gpt-4o-mini",
        script_like=False,
    ):
        if agent_name is None and agent_profile is not None:
            agent_name = agent_profile.pk
        original_init(
            self, agent_name, uuid_str, agent_profile, model_name, script_like
        )

    monkeypatch.setattr(LLMAgent, "__init__", patched_init)
    return monkeypatch


def test_get_env_agents(mp):
    """
    Test that get_env_agents returns an environment, an agents dictionary keyed by
    agent names (which should be the pks) and non-empty environment messages.
    """
    env, agents, env_msgs = get_env_agents("env1", ["agent1", "agent2"], ["model1", "model2"], "eval_model", "dummy_list")
    assert set(agents.keys()) == {"John Doe", "Jane Doe"}

    # Each agent should have been assigned one of the dummy environment goals.
    for agent in agents.values():
        assert agent.goal in ["Goal1", "Goal2"]

    # Check that environment messages is a dict.
    assert isinstance(env_msgs, dict)


# =============================================================================
# Atomic test for process_turn
# =============================================================================


# Create a dummy agent by subclassing LLMAgent that returns a fixed AgentAction.
class DummyAgent(LLMAgent):
    async def aact(self, obs: Observation) -> AgentAction:
        # Always return a known action, for example, a "speak" action.
        return AgentAction(action_type="speak", argument="dummy response")


@pytest.fixture
def dummy_simulator(mp) -> Any:
    """
    Create a dummy simulator that mimics WebSocketSotopiaSimulator. It sets up:
      - A dummy Agents dictionary with one DummyAgent.
      - A dummy environment with one goal.
      - A conversation_history seeded with the initial environment message.
    """
    # Create dummy agents dictionary.
    agents = Agents(
        {
            "agent1": DummyAgent(
                agent_name="agent1", agent_profile=None, model_name="dummy"
            )
        }
    )

    # Create a dummy environment profile for simulation.
    class DummyEnv:
        def __init__(self, goals: List[str]):
            self.agents = list(agents.keys())
            self.profile = type(
                "DummyProfile", (), {"agent_goals": goals, "pk": "env1"}
            )

    dummy_env = DummyEnv(["goal1"])
    # Create dummy environment messages (simulate initial reset).
    dummy_msgs = {
        "agent1": type(
            "DummyObs", (), {"to_natural_language": lambda self: "initial message"}
        )()
    }

    # Define a minimal dummy simulator class.
    class DummySimulator:
        def __init__(self):
            self.env = dummy_env
            self.agents = agents
            self.environment_messages = dummy_msgs
            # Initialize conversation history with the initial environment message.
<<<<<<< HEAD:tests/api/test_turntaking_atomic.py
            self.conversation_history: List[Dict[str, str]] = [{
                "role": "environment",
                "agent": "agent1",
                "content": dummy_msgs["agent1"].to_natural_language()
            }]
        async def process_turn(self, client_data: dict) -> dict:
=======
            self.conversation_history: List[Dict[str, str]] = [
                {
                    "role": "environment",
                    "agent": "agent1",
                    "content": dummy_msgs["agent1"].to_natural_language(),
                }
            ]

        async def process_turn(self, client_data: dict) -> dict:
            # Reuse the actual process_turn logic from WebSocketSotopiaSimulator.
            # Import build_observation from the proper module.
            from sotopia.api.websocket_utils import build_observation

>>>>>>> 96b64ded57b634854ad22b2f5488adfdd66f8634:tests/api/more_tests.py
            # Append client's input.
            self.conversation_history.append(
                {"role": "client", "content": client_data.get("content", "")}
            )
            agent_id = client_data.get("agent_id")
            if agent_id not in self.agents:
                raise ValueError(f"Agent with id {agent_id} not found")
            obs = build_observation(
                len(self.conversation_history), self.conversation_history
            )
            agent = self.agents[agent_id]
            agent_action = await agent.aact(obs)
            self.conversation_history.append(
                {"role": "agent", "content": agent_action.argument}
            )
            return {
                "turn": len(self.conversation_history),
                "agent_id": agent_id,
                "agent_response": agent_action.argument,
                "action_type": agent_action.action_type,
            }

    return DummySimulator()


@pytest.mark.asyncio
async def test_process_turn_success(dummy_simulator):
    """
    Test that process_turn returns correct TURN_RESPONSE data when valid input is provided.
    """
    simulator = dummy_simulator
    client_data = {"agent_id": "agent1", "content": "Hello!"}
    result = await simulator.process_turn(client_data)
    # Initially: environment message (1 msg). After processing turn → add client and agent messages, so total is 3.
    assert result["turn"] == 3
    assert result["agent_id"] == "agent1"
    # DummyAgent always returns the same response.
    assert result["agent_response"] == "dummy response"
    assert result["action_type"] == "speak"


@pytest.mark.asyncio
async def test_process_turn_invalid_agent(dummy_simulator):
    """
    Test that process_turn raises a ValueError for an invalid agent_id.
    """
    simulator = dummy_simulator
    client_data = {"agent_id": "nonexistent", "content": "Hello!"}
    with pytest.raises(ValueError) as excinfo:
        await simulator.process_turn(client_data)
    assert "Agent with id nonexistent not found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_multiple_turns_accumulate_history(dummy_simulator):
    """
    Test that multiple calls to process_turn properly accumulate conversation history.
    """
    simulator = dummy_simulator
    # Initial history already has one environment message.
    initial_length = len(simulator.conversation_history)
    # Process first turn.
    await simulator.process_turn({"agent_id": "agent1", "content": "Turn one"})
    # Process second turn.
    await simulator.process_turn({"agent_id": "agent1", "content": "Turn two"})
    # There should be: 1 (env) + 2*2 messages = 5 messages total.
    assert len(simulator.conversation_history) == initial_length + 4

<<<<<<< HEAD:tests/api/test_turntaking_atomic.py
=======

# You can add additional atomic tests if needed.
>>>>>>> 96b64ded57b634854ad22b2f5488adfdd66f8634:tests/api/more_tests.py
