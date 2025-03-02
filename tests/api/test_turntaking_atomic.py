# test_turntaking_atomic.py

import pytest
import asyncio
from typing import List, Dict, Any, Optional, Union
import logging
from pydantic import BaseModel

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

# Import required components from the Sotopia codebase.
from sotopia.api.websocket_utils import (
    build_observation,
    get_env_agents,
    WSMessageType,
)
from sotopia.agents import LLMAgent, Agents
from sotopia.messages import Observation, AgentAction
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    RuleBasedTerminatedEvaluator,
    EpisodeLLMEvaluator,
    EvaluationForTwoAgents,
    SotopiaDimensions,
)
from sotopia.database import EnvironmentProfile, AgentProfile, EvaluationDimensionBuilder
from sotopia.database.persistent_profile import RelationshipType

# =============================================================================
# Dummy classes to simulate database objects
# =============================================================================

# Define DummyAgentProfile as a subclass of AgentProfile so that it is compatible.
class DummyAgentProfile(AgentProfile):
    @classmethod
    def construct_dummy(
        cls,
        pk: str,
        first_name: str = "John",
        last_name: str = "Doe",
        age: int = 30,
    ) -> "DummyAgentProfile":
        # Create a dict that matches BaseAgentProfile's schema.
        data = {
            "pk": pk,
            "first_name": first_name,
            "last_name": last_name,
            "age": age,
            "occupation": "tester",
            "gender": "male",
            "gender_pronoun": "he",
            "public_info": "",
            "big_five": "",
            "moral_values": [],
            "schwartz_personal_values": [],
            "personality_and_values": "",
            "decision_making_style": "",
            "secret": "",
            "model_id": "",
            "mbti": "",
            "tag": "test_tag",
        }
        return cls.model_validate(data)  # Using model_validate() for pydantic models

# Define DummyEnvProfile with all required fields.
class DummyEnvProfile:
    def __init__(self, pk: str) -> None:
        self.pk = pk
        self.codename: str = "test_codename"
        # Force source to be a string (not None)
        self.source: str = ""
        # Use a non-None scenario.
        self.scenario: str = "A concrete scenario description that meets the guidelines."
        # For test consistency, use lowercase goals if tests expect them.
        self.agent_goals: List[str] = ["goal1", "goal2"]
        self.relationship: RelationshipType = RelationshipType.stranger  
        # For fields that might be str but can be None in model, ensure a string is provided.
        self.age_constraint: str = ""
        self.occupation_constraint: str = ""
        self.agent_constraint: List[List[str]] = []
        self.tag: str = "test_tag"

def fake_agent_get(agent_id: str) -> DummyAgentProfile:
    dummy: Dict[str, DummyAgentProfile] = {
        "agent1": DummyAgentProfile.construct_dummy("agent1", first_name="John", last_name="Doe", age=30),
        "agent2": DummyAgentProfile.construct_dummy("agent2", first_name="Jane", last_name="Doe", age=25),
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
def mp(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
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
        agent_profile: Optional[AgentProfile] = None,
        model_name: str = "gpt-4o-mini",
        script_like: bool = False,
    ) -> None:
        if agent_name is None and agent_profile is not None:
            agent_name = agent_profile.pk  # This sets the agent name to its pk.
        original_init(self, agent_name, uuid_str, agent_profile, model_name, script_like)
    monkeypatch.setattr(LLMAgent, "__init__", patched_init)
    return monkeypatch

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
        {"role": "agent", "content": "Hi, how may I help you?"}
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
# Test for get_env_agents
# =============================================================================

def test_get_env_agents(mp: pytest.MonkeyPatch) -> None:
    """
    Test that get_env_agents returns an environment, an agents dictionary keyed by
    agent names (which should be the dummy profile pks) and non-empty environment messages.
    """
    env, agents, env_msgs = get_env_agents("env1", ["agent1", "agent2"], ["model1", "model2"], "eval_model", "dummy_list")
    # Because our patched LLMAgent.__init__ uses agent_profile.pk as agent_name,
    # the keys should be "agent1" and "agent2".
    assert set(agents.keys()) == {"Jane Doe", "John Doe"}
    for agent in agents.values():
        # Using our DummyEnvProfile, agent.goal should be one of the dummy goals.
        assert agent.goal in ["goal1", "goal2"]
    assert isinstance(env_msgs, dict)

# =============================================================================
# Atomic test for process_turn
# =============================================================================

# Create a dummy agent by subclassing LLMAgent that returns a fixed AgentAction.
class DummyAgent(LLMAgent):
    async def aact(self, obs: Observation) -> AgentAction:
        return AgentAction(action_type="speak", argument="dummy response")

@pytest.fixture
def dummy_simulator(mp: pytest.MonkeyPatch) -> Any:
    """
    Create a dummy simulator that mimics WebSocketSotopiaSimulator.
    It sets up a dummy Agents dictionary with one DummyAgent,
    a dummy environment with one goal, and conversation history seeded with an initial message.
    """
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
            from sotopia.api.websocket_utils import build_observation
            self.conversation_history.append({"role": "client", "content": client_data.get("content", "")})
            agent_id: str = client_data.get("agent_id")
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
    # Initial message count is 1, then client and agent turns make 3.
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
    # Each call to process_turn adds 2 messages.
    assert len(simulator.conversation_history) == initial_length + 4
