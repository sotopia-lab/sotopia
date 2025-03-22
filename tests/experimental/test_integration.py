"""
Integration tests for the Sotopia group chat system using Redis Pub/Sub
"""
import pytest
import json
import asyncio
import hashlib
from unittest.mock import MagicMock, AsyncMock, patch
import redis.asyncio
from fastapi import WebSocket

# Import the necessary components
from sotopia.api.fastapi_server import SimulationManager, WSMessageType
from sotopia.api.websocket_utils import WebSocketSotopiaSimulator
from sotopia.experimental.agents.redis_agent import RedisAgent
from sotopia.experimental.agents.moderator import Moderator, Observations
from sotopia.experimental.agents.datamodels import AgentAction, Observation

# Factory functions for creating patched instances

def create_test_redis_agent():
    """Create a RedisAgent instance with patched Node initialization"""
    with patch('sotopia.experimental.agents.base_agent.Node.__init__', return_value=None):
        agent = RedisAgent(
            input_channels=["moderator:redis_agent"],
            output_channel="redis_agent:moderator",
            node_name="redis_agent",
            other_agent_status={"agent1": True, "agent2": True},
            redis_url="redis://localhost:6379/0",
        )
    
    # Set up necessary attributes that would normally be initialized
    agent.input_channel_types = {"moderator:redis_agent": Observation}
    agent.output_channel_types = {"redis_agent:moderator": AgentAction}
    
    # Mock Redis client to avoid actual connections
    agent.r = AsyncMock()
    
    # Add missing pydantic attributes to prevent __pydantic_extra__ errors
    agent.model_config = {"extra": "ignore"}
    agent.__pydantic_extra__ = None
    agent.__pydantic_fields_set__ = set()
    agent.__pydantic_private__ = None
    
    return agent

def create_test_moderator():
    """Create a Moderator instance with patched Node initialization"""
    with patch('sotopia.experimental.agents.base_agent.Node.__init__', return_value=None):
        moderator = Moderator(
            node_name="moderator",
            input_channels=["agent1:moderator", "agent2:moderator", "redis_agent:moderator"],
            output_channels=["moderator:agent1", "moderator:agent2", "moderator:redis_agent"],
            scenario="Test scenario",
            agent_mapping={
                "moderator:agent1": "agent1",
                "moderator:agent2": "agent2",
                "moderator:redis_agent": "redis_agent"
            },
            redis_url="redis://localhost:6379/0",
            action_order="round-robin",
            available_actions=["speak", "none", "leave"],
            max_turns=10
        )
    
    # Set up necessary attributes that would normally be initialized
    moderator.r = AsyncMock()
    
    # Create a mock for the epilog since it requires Redis
    moderator.epilog = MagicMock()
    moderator.epilog.messages = []
    moderator.epilog.model_dump_json = MagicMock(return_value='{"messages":[]}')
    
    # Add missing pydantic attributes to prevent __pydantic_extra__ errors
    moderator.model_config = {"extra": "ignore"}
    moderator.__pydantic_extra__ = None
    moderator.__pydantic_fields_set__ = set()
    moderator.__pydantic_private__ = None
    
    return moderator

# Test message flow from client to simulator
@pytest.mark.asyncio
async def test_client_to_simulator_flow():
    """Test message flow from client to simulator with proper mocking"""
    # Create simulator with mocking to avoid Redis connections
    with patch('redis_om.model.model.get_redis_connection'), \
         patch('redis_om.checks.has_redis_json', return_value=True), \
         patch('redis_om.checks.check_for_command', return_value=True), \
         patch('sotopia.api.websocket_utils.get_env_agents') as mock_get_env_agents:
        
        # Set up mock environment, agents, and messages
        mock_env = MagicMock()
        mock_agents = MagicMock()
        mock_messages = {}
        
        # Configure the get_env_agents mock to return our mocked objects
        mock_get_env_agents.return_value = (mock_env, mock_agents, mock_messages)
        
        # Create simulator
        simulator = WebSocketSotopiaSimulator(
            env_id="test_env_id",
            agent_ids=["agent1", "agent2"],
            agent_models=["gpt-4o-mini", "gpt-4o-mini"],
            evaluator_model="gpt-4o",
            max_turns=10
        )
        simulator.mode = "group"
        simulator.groups = {
            "team_a": ["agent1"],
            "team_b": ["agent2"]
        }
        
        # Mock send_to_redis method to avoid actual Redis connections
        simulator.send_to_redis = AsyncMock()
        
        # Create manager
        manager = SimulationManager()
        
        # Mock WebSocket
        mock_websocket = AsyncMock(spec=WebSocket)
        
        # Create message
        group_message = {
            "type": WSMessageType.CLIENT_MSG.value,
            "data": {
                "content": "Hello team A",
                "target_groups": ["team_a"]
            }
        }
        
        # Process the actual handle_client_message method from the manager
        result = await manager.handle_client_message(mock_websocket, simulator, group_message)
        
        # Verify simulator.send_to_redis was called with the right parameters
        assert simulator.send_to_redis.called
        
        # Get the message that would be sent to Redis
        sent_data = simulator.send_to_redis.call_args[0][0]
        
        # Verify message content
        assert "content" in sent_data
        assert sent_data["content"] == "Hello team A"
        assert "target_groups" in sent_data
        assert sent_data["target_groups"] == ["team_a"]

# Test message flow from simulator to RedisAgent
@pytest.mark.asyncio
async def test_simulator_to_redis_agent_flow():
    """Test message flow from simulator to RedisAgent"""
    # Create a real RedisAgent with patched initialization
    redis_agent = create_test_redis_agent()
    
    # Configure agent for testing
    redis_agent.mode = "group"
    redis_agent.groups = {
        "team_a": ["agent1"],
        "team_b": ["agent2"]
    }
    
    # Create a command message (as would be received from simulator)
    command_data = {
        "content": "Hello team A",
        "target_groups": ["team_a"],
        "sender": "websocket_user"
    }
    
    # Process the actual process_command method
    action = await redis_agent.process_command(command_data, "test_conn_id")
    
    # Verify action was created correctly
    assert action is not None
    assert action.action_type == "unified_message"
    
    # Parse argument
    arg_data = json.loads(action.argument)
    assert arg_data["content"] == "Hello team A"
    assert "agent1" in arg_data["target_agents"]  # Expanded from team_a
    assert "agent2" not in arg_data["target_agents"]
    assert arg_data["original_target_groups"] == ["team_a"]

# Test message flow from RedisAgent to Moderator
@pytest.mark.asyncio
async def test_redis_agent_to_moderator_flow():
    """Test message flow from RedisAgent to Moderator"""
    # Create a real Moderator with patched initialization
    moderator = create_test_moderator()
    
    # Configure moderator for testing
    moderator.mode = "group"
    moderator.groups = {
        "team_a": ["agent1"],
        "team_b": ["agent2"]
    }
    
    # Mock the send_epilog method to avoid Redis dependencies
    moderator.send_epilog = AsyncMock()
    
    # Create a real unified message action
    action = AgentAction(
        agent_name="websocket_user",
        output_channel="redis_agent:moderator",
        action_type="unified_message",
        argument=json.dumps({
            "content": "Hello team A",
            "target_agents": ["agent1"],  # Expanded by RedisAgent
            "original_target_agents": [],
            "original_target_groups": ["team_a"],
            "context": "group"
        })
    )
    
    # Process the actual handle_unified_message method
    observations = await moderator.handle_unified_message(action)
    
    # Verify observations were created
    assert isinstance(observations, Observations)
    assert "moderator:agent1" in observations.observations_map
    assert "moderator:agent2" in observations.observations_map
    
    # Verify agent1 received message
    assert observations.observations_map["moderator:agent1"].last_turn == "Hello team A"
    assert observations.observations_map["moderator:agent1"].available_actions != ["none"]
    
    # Verify agent2 did not receive message
    assert observations.observations_map["moderator:agent2"].last_turn == ""
    assert observations.observations_map["moderator:agent2"].available_actions == ["none"]
    
    # Verify epilog was sent
    assert moderator.send_epilog.called

# Test epilog flow from Moderator to RedisAgent
@pytest.mark.asyncio
async def test_moderator_to_redis_agent_epilog_flow():
    """Test epilog flow from Moderator to RedisAgent"""
    # Create a real Moderator with patched initialization
    moderator = create_test_moderator()
    
    # Create epilog
    epilog = MagicMock()
    epilog.model_dump_json = MagicMock(return_value=json.dumps({
        "messages": [
            [["agent1", "agent2", "Test message"]]
        ]
    }))
    
    # Create a real send implementation for testing
    async def mock_send(channel, data):
        return await moderator.r.publish(channel, data)
    
    # Patch only the send method, keeping the actual send_epilog implementation
    with patch.object(moderator, 'send', mock_send):
        # Send epilog
        await moderator.send_epilog(epilog, "moderator:redis_agent")
        
        # Verify Redis publish was called
        assert moderator.r.publish.called
        channel, data = moderator.r.publish.call_args[0]
        assert channel == "moderator:redis_agent"
        
        # Parse data
        message = json.loads(data)
        assert message["data"]["agent_name"] == "epilog"
        assert "last_turn" in message["data"]

# Test epilog flow from RedisAgent to client
@pytest.mark.asyncio
async def test_redis_agent_to_client_epilog_flow():
    """Test epilog flow from RedisAgent to client"""
    # Create a real RedisAgent with patched initialization
    redis_agent = create_test_redis_agent()
    
    # Setup connection
    connection_id = "test_conn_id"
    redis_agent.active_connections = {connection_id}
    redis_agent.last_epilog_hash = {}
    redis_agent.epilog_channel_prefix = "epilog:"
    
    # Create epilog observation
    epilog_data = {"messages": [["test", "data"]]}
    obs = Observation(
        agent_name="epilog",
        last_turn=json.dumps(epilog_data),
        turn_number=1,
        available_actions=["none"]
    )
    
    # Call the actual publish_observation method
    await redis_agent.publish_observation(obs)
    
    # Verify Redis publish was called
    assert redis_agent.r.publish.called
    
    # If it was called multiple times, check the last call
    calls = redis_agent.r.publish.call_args_list
    last_call = calls[-1]
    channel, data = last_call[0]
    
    # Verify channel contains connection_id
    assert channel == f"epilog:{connection_id}"
    
    # Parse data
    message = json.loads(data)
    assert message["type"] == "SERVER_MSG"
    assert message["data"]["type"] == "episode_log"
    assert message["data"]["log"] == epilog_data

# Test response handling from agent to client
@pytest.mark.asyncio
async def test_agent_response_flow():
    """Test agent response flow back to client"""
    # Create a real RedisAgent with patched initialization
    redis_agent = create_test_redis_agent()
    
    # Setup for testing
    redis_agent.mode = "group"
    redis_agent.message_senders = {
        "agent1": "websocket_user"
    }
    
    # Create agent observation
    obs = Observation(
        agent_name="agent1",
        last_turn="Response to message",
        turn_number=2,
        available_actions=["speak", "none"]
    )
    
    # Process the actual process_agent_response method
    action = await redis_agent.process_agent_response(obs)
    
    # Verify action
    assert action is not None
    assert action.action_type == "unified_message"
    
    # Parse argument
    arg_data = json.loads(action.argument)
    assert arg_data["content"] == "Response to message"
    assert arg_data["target_agents"] == ["websocket_user"]
    assert arg_data["context"] == "response"

# Test deduplication of epilog updates
@pytest.mark.asyncio
async def test_epilog_deduplication():
    """Test deduplication of identical epilog updates"""
    # Create a real Moderator with patched initialization
    moderator = create_test_moderator()
    
    # Create epilog content
    epilog_content = {"messages": [["test", "data"]]}
    epilog_json = json.dumps(epilog_content)
    
    # Create two identical epilogs
    epilog1 = MagicMock()
    epilog1.model_dump_json = MagicMock(return_value=epilog_json)
    
    epilog2 = MagicMock()
    epilog2.model_dump_json = MagicMock(return_value=epilog_json)
    
    # Create a real send implementation for testing
    async def mock_send(channel, data):
        return await moderator.r.publish(channel, data)
    
    # Patch only the send method, keeping the actual send_epilog implementation
    with patch.object(moderator, 'send', mock_send):
        # Send first epilog
        await moderator.send_epilog(epilog1, "test_channel")
        
        # Verify Redis was called
        assert moderator.r.publish.called
        moderator.r.publish.reset_mock()
        
        # Send identical epilog
        await moderator.send_epilog(epilog2, "test_channel")
        
        # Verify Redis was NOT called (deduplication)
        assert not moderator.r.publish.called
        
        # Create different epilog
        epilog3 = MagicMock()
        epilog3.model_dump_json = MagicMock(return_value=json.dumps({"messages": [["different"]]}))
        
        # Send different epilog
        await moderator.send_epilog(epilog3, "test_channel")
        
        # Verify Redis was called for different epilog
        assert moderator.r.publish.called

# Test the websocket endpoint with a fixed test version
@pytest.mark.asyncio
async def test_websocket_endpoint_flow():
    """Test the websocket endpoint flow with minimal dependencies"""
    # Use the actual websocket_endpoint function with patched dependencies
    from sotopia.api.fastapi_server import websocket_endpoint
    
    # Mock WebSocket
    mock_websocket = AsyncMock(spec=WebSocket)
    
    # Create START_SIM message
    start_sim_msg = {
        "type": WSMessageType.START_SIM.value,
        "data": {
            "env_id": "test_env",
            "agent_ids": ["agent1", "agent2"],
            "mode": "group",
            "groups": {
                "team_a": ["agent1"],
                "team_b": ["agent2"]
            }
        }
    }
    
    # Configure mock_websocket.receive_json to return our START_SIM message
    mock_websocket.receive_json.return_value = start_sim_msg
    
    # Create a properly structured mock SimulationManager
    mock_manager = AsyncMock()
    mock_manager.verify_token.return_value = {"is_valid": True, "msg": "Valid token"}
    mock_manager.create_simulator = AsyncMock()
    mock_manager.run_simulation = AsyncMock()
    
    # Create a mock simulator
    mock_simulator = AsyncMock()
    mock_simulator.connection_id = "test_connection_id"
    mock_simulator.set_mode = AsyncMock()
    mock_simulator.set_groups = AsyncMock()
    mock_simulator.connect_to_redis = AsyncMock()
    
    # Configure mock_manager.create_simulator to return our simulator
    mock_manager.create_simulator.return_value = mock_simulator
    
    # Patch the SimulationManager class to return our mock
    with patch('sotopia.api.fastapi_server.SimulationManager', return_value=mock_manager):
        # Call the actual endpoint
        await websocket_endpoint(mock_websocket, "valid_token")
        
        # Verify websocket was accepted
        mock_websocket.accept.assert_called_once()
        
        # Verify manager methods were called correctly
        mock_manager.verify_token.assert_called_once_with("valid_token")
        mock_manager.create_simulator.assert_called_once()
        
        # Verify simulator methods were called correctly
        mock_simulator.connect_to_redis.assert_called_once()
        mock_simulator.set_mode.assert_called_once_with("group")
        mock_simulator.set_groups.assert_called_once()
        
        # Verify run_simulation was called with correct arguments
        mock_manager.run_simulation.assert_called_once_with(mock_websocket, mock_simulator)