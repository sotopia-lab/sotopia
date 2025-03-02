from typing import Any, Dict, List, Optional, Union
import logging
import pytest
from pydantic import BaseModel
from sotopia.database.persistent_profile import RelationshipType
from sotopia.api.websocket_utils import build_observation, get_env_agents
from sotopia.agents import LLMAgent, Agents
from sotopia.messages import Observation, AgentAction
from sotopia.database import EnvironmentProfile, AgentProfile, EvaluationDimensionBuilder

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

# =============================================================================
# Test for build_observation
# =============================================================================

def test_build_observation() -> None:
    """
    Test that build_observation returns an Observation with the correct
    last_turn, turn_number and available_actions.
    """
    conversation_history: List[Dict[str, str]] = [
        {"role": "client", "content": "Hello"},
        {"role": "agent", "content": "Hi, how may I help you?"},
    ]
    turn_number: int = len(conversation_history)
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
# Dummy classes to simulate database objects
# =============================================================================

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
        moral_values: Optional[List[str]] = None,
        schwartz_personal_values: Optional[List[str]] = None,
        personality_and_values: str = "",
        decision_making_style: str = "",
        secret: str = "",
        model_id: str = "",
        mbti: str = "",
        tag: str = "test_tag",
    ) -> None:
        self.pk = pk
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.occupation = occupation
        self.gender = gender
        self.gender_pronoun = gender_pronoun
        self.public_info = public_info
        self.big_five = big_five
        # Use "Optional[List[str]]" for defaults and set empty list if None.
        self.moral_values: List[str] = moral_values or []
        self.schwartz_personal_values: List[str] = schwartz_personal_values or []
        self.personality_and_values = personality_and_values
        self.decision_making_style = decision_making_style
        self.secret = secret
        self.model_id = model_id
        self.mbti = mbti
        self.tag = tag

class DummyEnvProfile:
    def __init__(self, pk: str) -> None:
        self.pk = pk
        self.codename: str = "test_codename"
        self.source: str = ""
        self.scenario: str = "A concrete scenario description that meets the guidelines."
        self.agent_goals: List[str] = ["Goal1", "Goal2"]
        self.relationship: RelationshipType = RelationshipType.stranger  
        self.age_constraint = None
        self.occupation_constraint = None
        self.agent_constraint = None
        self.tag = "test_tag"

def fake_agent_get(agent_id: str) -> DummyAgentProfile:
    dummy: Dict[str, DummyAgentProfile] = {
        "agent1": DummyAgentProfile("agent1", first_name="John", last_name="Doe", age=30),
        "agent2": DummyAgentProfile("agent2", first_name="Jane", last_name="Doe", age=25),
    }
    if agent_id in dummy:
        return dummy[agent_id]
    raise Exception(f"AgentProfile with id {agent_id} not found")

def fake_env_get(env_id: str) -> DummyEnvProfile:
    return DummyEnvProfile(env_id)

# =============================================================================
# Fixture for monkeypatching
# =============================================================================

@pytest.fixture
def mp(monkeypatch: Any) -> Any:
    monkeypatch.setattr(EnvironmentProfile, "get", fake_env_get)
    monkeypatch.setattr(AgentProfile, "get", fake_agent_get)
    monkeypatch.setattr(
        EvaluationDimensionBuilder,
        "select_existing_dimension_model_by_list_name",
        lambda list_name: BaseModel,
    )
    original_init = LLMAgent.__init__

    def patched_init(
        self: LLMAgent,
        agent_name: Optional[str] = None,
        uuid_str: Optional[str] = None,
        agent_profile: Optional[DummyAgentProfile] = None,
        model_name: str = "gpt-4o-mini",
        script_like: bool = False,
    ) -> None:
        if agent_name is None and agent_profile is not None:
            agent_name = agent_profile.pk
        original_init(self, agent_name, uuid_str, agent_profile, model_name, script_like)

    monkeypatch.setattr(LLMAgent, "__init__", patched_init)
    return monkeypatch

def test_get_env_agents(mp: Any) -> None:
    """
    Test that get_env_agents returns an environment, an agents dictionary keyed by
    agent names (which should be the pks) and non-empty environment messages.
    """
    env, agents, env_msgs = get_env_agents("env1", ["agent1", "agent2"], ["model1", "model2"], "eval_model", "dummy_list")
    assert set(agents.keys()) == {"John Doe", "Jane Doe"}

    for agent in agents.values():
        assert agent.goal in ["Goal1", "Goal2"]

    assert isinstance(env_msgs, dict)

# =============================================================================
# DummyAgent and simulator for process_turn tests
# =============================================================================

class DummyAgent(LLMAgent):
    async def aact(self, obs: Observation) -> AgentAction:
        # Always return a known action.
        return AgentAction(action_type="speak", argument="dummy response")

@pytest.fixture
def dummy_simulator(mp: Any) -> Any:
    agents_instance: Agents = Agents({
        "agent1": DummyAgent(agent_name="agent1", agent_profile=None, model_name="dummy")
    })

    class DummyEnv:
        def __init__(self, goals: List[str]) -> None:
            self.agents: List[str] = list(agents_instance.keys())
            self.profile = type("DummyProfile", (), {"agent_goals": goals, "pk": "env1"})

    dummy_env = DummyEnv(["goal1"])
    dummy_msgs: Dict[str, Any] = {
        "agent1": type("DummyObs", (), {"to_natural_language": lambda self: "initial message"})()
    }

    class DummySimulator:
        def __init__(self) -> None:
            self.env: DummyEnv = dummy_env
            self.agents: Agents = agents_instance
            self.environment_messages: Dict[str, Any] = dummy_msgs
            self.conversation_history: List[Dict[str, str]] = [{
                "role": "environment",
                "agent": "agent1",
                "content": dummy_msgs["agent1"].to_natural_language()
            }]

        async def process_turn(self, client_data: Dict[str, str]) -> Dict[str, Union[int, str]]:
            self.conversation_history.append({"role": "client", "content": client_data.get("content", "")})
            agent_id: str = client_data.get("agent_id")  # Assumed to be str
            if agent_id not in self.agents:
                raise ValueError(f"Agent with id {agent_id} not found")
            obs: Observation = build_observation(len(self.conversation_history), self.conversation_history)
            agent = self.agents[agent_id]  # type: ignore
            agent_action: AgentAction = await agent.aact(obs)
            self.conversation_history.append({"role": "agent", "content": agent_action.argument})
            return {
                "turn": len(self.conversation_history),
                "agent_id": agent_id,
                "agent_response": agent_action.argument,
                "action_type": agent_action.action_type,
            }

    return DummySimulator()

@pytest.mark.asyncio
async def test_process_turn_success(dummy_simulator: Any) -> None:
    """
    Test that process_turn returns correct TURN_RESPONSE data when valid input is provided.
    """
    simulator = dummy_simulator
    client_data: Dict[str, str] = {"agent_id": "agent1", "content": "Hello!"}
    result: Dict[str, Union[int, str]] = await simulator.process_turn(client_data)
    # Initially: environment message (1 msg). After processing turn → add client and agent messages, so total is 3.
    assert result["turn"] == 3
    assert result["agent_id"] == "agent1"
    assert result["agent_response"] == "dummy response"
    assert result["action_type"] == "speak"

@pytest.mark.asyncio
async def test_process_turn_invalid_agent(dummy_simulator: Any) -> None:
    """
    Test that process_turn raises a ValueError for an invalid agent_id.
    """
    simulator = dummy_simulator
    client_data: Dict[str, str] = {"agent_id": "nonexistent", "content": "Hello!"}
    with pytest.raises(ValueError) as excinfo:
        await simulator.process_turn(client_data)
    assert "Agent with id nonexistent not found" in str(excinfo.value)

@pytest.mark.asyncio
async def test_multiple_turns_accumulate_history(dummy_simulator: Any) -> None:
    """
    Test that multiple calls to process_turn properly accumulate conversation history.
    """
    simulator = dummy_simulator
    initial_length: int = len(simulator.conversation_history)
    await simulator.process_turn({"agent_id": "agent1", "content": "Turn one"})
    await simulator.process_turn({"agent_id": "agent1", "content": "Turn two"})
    # There should be: 1 (initial env message) + 2 turns × 2 messages each = initial_length + 4 messages.
    assert len(simulator.conversation_history) == initial_length + 4
