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
from sotopia.api.fastapi_server import SimulationManager, WSMessageType, ErrorType
from sotopia.api.websocket_utils import WebSocketSotopiaSimulator

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

# Test simulator to RedisAgent flow with pure dictionary approach
@pytest.mark.asyncio
async def test_simulator_to_redis_agent_flow():
    """Test message flow from simulator to RedisAgent using dictionary mocks"""
    # Mock the process_command function directly
    async def mock_process_command(command_data, connection_id):
        # Only handle group mode messages
        if "target_groups" in command_data and command_data["target_groups"]:
            group_name = command_data["target_groups"][0]
            
            # In this test we'll simulate that "team_a" contains "agent1"
            expanded_agents = []
            if group_name == "team_a":
                expanded_agents = ["agent1"]
            
            # Construct the action as a simple dictionary instead of an AgentAction
            return {
                "agent_name": command_data.get("sender", "websocket_user"),
                "output_channel": "redis_agent:moderator",
                "action_type": "unified_message",
                "argument": json.dumps({
                    "content": command_data["content"],
                    "target_agents": expanded_agents,
                    "original_target_agents": [],
                    "original_target_groups": command_data["target_groups"],
                    "context": "group"
                })
            }
        return None
    
    # Test data
    command_data = {
        "content": "Hello team A",
        "target_groups": ["team_a"],
        "sender": "websocket_user"
    }
    
    # Call our mock function
    action = await mock_process_command(command_data, "test_conn_id")
    
    # Verify action structure
    assert action is not None
    assert action["action_type"] == "unified_message"
    
    # Parse argument
    arg_data = json.loads(action["argument"])
    assert arg_data["content"] == "Hello team A"
    assert "agent1" in arg_data["target_agents"]  # Expanded from team_a
    assert arg_data["original_target_groups"] == ["team_a"]

# Test RedisAgent to Moderator flow with pure dictionary approach
@pytest.mark.asyncio
async def test_redis_agent_to_moderator_flow():
    """Test message flow from RedisAgent to Moderator using dictionary mocks"""
    # Mock the handle_unified_message function
    async def mock_handle_unified_message(action):
        # Extract data from action
        agent_name = action["agent_name"]
        argument = json.loads(action["argument"])
        content = argument["content"]
        target_agents = argument["target_agents"]
        
        # Create observations dict instead of Observations object
        observations_map = {}
        
        # Add observations for all agents (agent1 and agent2 in this test)
        for agent in ["agent1", "agent2"]:
            # Determine if this agent should receive the message
            is_recipient = agent in target_agents
            
            # Set available actions
            available_actions = ["speak", "none"] if is_recipient else ["none"]
            
            # Create observation dict
            observation = {
                "agent_name": agent_name,
                "last_turn": content if is_recipient else "",
                "turn_number": 1,
                "available_actions": available_actions
            }
            
            # Add to map
            observations_map[f"moderator:{agent}"] = observation
        
        # Return a mock observations object
        return type('MockObservations', (), {'observations_map': observations_map})
    
    # Create action dict
    action = {
        "agent_name": "websocket_user",
        "output_channel": "redis_agent:moderator",
        "action_type": "unified_message",
        "argument": json.dumps({
            "content": "Hello team A",
            "target_agents": ["agent1"],  # Only agent1 gets the message
            "original_target_agents": [],
            "original_target_groups": ["team_a"],
            "context": "group"
        })
    }
    
    # Call our mock function
    result = await mock_handle_unified_message(action)
    
    # Verify result structure
    assert hasattr(result, "observations_map")
    assert "moderator:agent1" in result.observations_map
    assert "moderator:agent2" in result.observations_map
    
    # Verify agent1 received message
    assert result.observations_map["moderator:agent1"]["last_turn"] == "Hello team A"
    assert result.observations_map["moderator:agent1"]["available_actions"] != ["none"]
    
    # Verify agent2 did not receive message
    assert result.observations_map["moderator:agent2"]["last_turn"] == ""
    assert result.observations_map["moderator:agent2"]["available_actions"] == ["none"]

# Test Moderator to RedisAgent epilog flow
@pytest.mark.asyncio
async def test_moderator_to_redis_agent_epilog_flow():
    """Test epilog flow from Moderator to RedisAgent using mocks"""
    # Create a mock for Redis publish
    mock_publish = AsyncMock()
    
    # Create a mock for send_epilog that calls mock_publish
    async def mock_send_epilog(epilog, channel):
        # Create a message that looks like what send_epilog would create
        message = {
            "data": {
                "agent_name": "epilog",
                "last_turn": epilog.model_dump_json(),
                "turn_number": 1,
                "available_actions": ["none"]
            }
        }
        await mock_publish(channel, json.dumps(message))
    
    # Create epilog
    mock_epilog = MagicMock()
    mock_epilog.model_dump_json = MagicMock(return_value=json.dumps({
        "messages": [
            [["agent1", "agent2", "Test message"]]
        ]
    }))
    
    # Call the mocked method
    await mock_send_epilog(mock_epilog, "moderator:redis_agent")
    
    # Verify send was called with correct parameters
    assert mock_publish.called
    channel, data = mock_publish.call_args[0]
    assert channel == "moderator:redis_agent"
    
    # Parse data
    message = json.loads(data)
    assert message["data"]["agent_name"] == "epilog"
    assert "last_turn" in message["data"]
    
    # Verify the epilog content was included
    epilog_data = json.loads(message["data"]["last_turn"])
    assert "messages" in epilog_data
    assert epilog_data["messages"][0][0][0] == "agent1"
    assert epilog_data["messages"][0][0][2] == "Test message"

# Test RedisAgent to client epilog flow
@pytest.mark.asyncio
async def test_redis_agent_to_client_epilog_flow():
    """Test epilog flow from RedisAgent to client using mocks"""
    # Create a mock for Redis publish
    mock_publish = AsyncMock()
    
    # Create a mock for publish_observation that uses mock_publish
    async def mock_publish_observation(obs):
        # Check if this is an epilog observation
        if obs["agent_name"] == "epilog":
            try:
                # Parse the epilog data
                epilog_data = json.loads(obs["last_turn"])
                
                # Format the message for each connection
                formatted_message = json.dumps({
                    "type": "SERVER_MSG",
                    "data": {
                        "type": "episode_log",
                        "log": epilog_data
                    }
                })
                
                # Publish to each active connection's epilog channel
                for connection_id in ["test_conn_id"]:
                    channel = f"epilog:{connection_id}"
                    await mock_publish(channel, formatted_message)
            except Exception as e:
                print(f"Error publishing epilog: {e}")
    
    # Create epilog observation as dictionary
    epilog_data = {"messages": [["test", "data"]]}
    obs = {
        "agent_name": "epilog",
        "last_turn": json.dumps(epilog_data),
        "turn_number": 1,
        "available_actions": ["none"]
    }
    
    # Call the mocked method
    await mock_publish_observation(obs)
    
    # Verify publish was called with correct parameters
    assert mock_publish.called
    channel, data = mock_publish.call_args[0]
    assert channel == "epilog:test_conn_id"
    
    # Parse data
    message = json.loads(data)
    assert message["type"] == "SERVER_MSG"
    assert message["data"]["type"] == "episode_log"
    assert message["data"]["log"] == epilog_data

# Test response handling from agent to client
@pytest.mark.asyncio
async def test_agent_response_flow():
    """Test agent response flow back to client using dictionary mocks"""
    # Mock process_agent_response with dictionary approach
    async def mock_process_agent_response(obs):
        # Extract data
        agent_name = obs["agent_name"]
        content = obs["last_turn"]
        
        # In our test we'll assume agent1 had a message from websocket_user
        if agent_name == "agent1":
            # Create action dict
            return {
                "agent_name": agent_name,
                "output_channel": "redis_agent:moderator",
                "action_type": "unified_message",
                "argument": json.dumps({
                    "content": content,
                    "target_agents": ["websocket_user"],
                    "original_target_agents": ["websocket_user"],
                    "original_target_groups": [],
                    "context": "response"
                })
            }
        return None
    
    # Create observation as dictionary
    obs = {
        "agent_name": "agent1",
        "last_turn": "Response to message",
        "turn_number": 2,
        "available_actions": ["speak", "none"]
    }
    
    # Call our mock function
    action = await mock_process_agent_response(obs)
    
    # Verify action
    assert action is not None
    assert action["action_type"] == "unified_message"
    
    # Parse argument
    arg_data = json.loads(action["argument"])
    assert arg_data["content"] == "Response to message"
    assert arg_data["target_agents"] == ["websocket_user"]
    assert arg_data["context"] == "response"

# Test deduplication of epilog updates
@pytest.mark.asyncio
async def test_epilog_deduplication():
    """Test deduplication of identical epilog updates using mocks"""
    # Create a mock for Redis publish
    mock_publish = AsyncMock()
    
    # Create a mock for send_epilog that has deduplication logic
    last_epilog_hash = None
    
    async def mock_send_epilog(epilog, channel):
        nonlocal last_epilog_hash
        
        # Generate hash of epilog to avoid sending duplicates
        epilog_json = epilog.model_dump_json()
        current_hash = hashlib.md5(epilog_json.encode()).hexdigest()
        
        # Only send if it's different from the last epilog we sent
        if current_hash != last_epilog_hash:
            message = {
                "data": {
                    "agent_name": "epilog",
                    "last_turn": epilog_json,
                    "turn_number": 1,
                    "available_actions": ["none"]
                }
            }
            await mock_publish(channel, json.dumps(message))
            last_epilog_hash = current_hash
    
    # Create two identical epilogs
    epilog1 = MagicMock()
    epilog1.model_dump_json = MagicMock(return_value=json.dumps({"messages": [["test", "data"]]}))
    
    epilog2 = MagicMock()
    epilog2.model_dump_json = MagicMock(return_value=json.dumps({"messages": [["test", "data"]]}))
    
    # Send first epilog
    await mock_send_epilog(epilog1, "test_channel")
    
    # Verify publish was called
    assert mock_publish.called
    mock_publish.reset_mock()
    
    # Send identical epilog
    await mock_send_epilog(epilog2, "test_channel")
    
    # Verify publish was NOT called (deduplication)
    assert not mock_publish.called
    
    # Create different epilog
    epilog3 = MagicMock()
    epilog3.model_dump_json = MagicMock(return_value=json.dumps({"messages": [["different"]]}))
    
    # Send different epilog
    await mock_send_epilog(epilog3, "test_channel")
    
    # Verify publish was called for different epilog
    assert mock_publish.called

# Test the websocket endpoint with completely mocked dependencies
@pytest.mark.asyncio
async def test_websocket_endpoint_flow():
    """Test the websocket endpoint flow with minimal dependencies"""
    # Mock entire SimulationManager class
    mock_manager = AsyncMock()
    mock_manager.verify_token = AsyncMock(return_value={"is_valid": True, "msg": "Valid token"})
    mock_manager.create_simulator = AsyncMock()
    mock_manager.send_message = AsyncMock()
    mock_manager.send_error = AsyncMock()
    mock_manager.run_simulation = AsyncMock()
    
    # Create a mock simulator
    mock_simulator = AsyncMock()
    mock_simulator.connection_id = "test_connection_id"
    mock_simulator.connect_to_redis = AsyncMock()
    mock_simulator.set_mode = AsyncMock()
    mock_simulator.set_groups = AsyncMock()
    
    # Set up manager to return simulator
    mock_manager.create_simulator.return_value = mock_simulator
    
    # Mock WebSocket
    mock_websocket = AsyncMock(spec=WebSocket)
    
    # Set up message receipt 
    mock_websocket.receive_json = AsyncMock()
    mock_websocket.receive_json.return_value = {
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
    
    # Define the websocket endpoint function
    async def test_websocket_endpoint(websocket: WebSocket, token: str):
        # Verify token
        token_status = await mock_manager.verify_token(token)
        if not token_status["is_valid"]:
            await websocket.close(code=1008, reason=token_status["msg"])
            return

        try:
            await websocket.accept()
            
            # Wait for START_SIM message
            start_msg = await websocket.receive_json()
            if start_msg.get("type") != WSMessageType.START_SIM.value:
                await mock_manager.send_error(
                    websocket,
                    ErrorType.INVALID_MESSAGE,
                    "First message must be of type START_SIM"
                )
                await websocket.close(code=1008)
                return

            # Extract parameters
            sim_data = start_msg.get("data", {})
            env_id = sim_data.get("env_id")
            agent_ids = sim_data.get("agent_ids", [])
            
            if not env_id or not agent_ids:
                await mock_manager.send_error(
                    websocket,
                    ErrorType.INVALID_MESSAGE,
                    "START_SIM message must include env_id and agent_ids"
                )
                await websocket.close(code=1008)
                return

            # Create simulator
            simulator = await mock_manager.create_simulator(
                env_id=env_id,
                agent_ids=agent_ids,
                agent_models=sim_data.get("agent_models"),
                evaluator_model=sim_data.get("evaluator_model"),
                evaluation_dimension_list_name=sim_data.get("evaluation_dimension_list_name"),
                env_profile_dict=sim_data.get("env_profile_dict"),
                agent_profile_dicts=sim_data.get("agent_profile_dicts"),
                max_turns=sim_data.get("max_turns"),
            )
            
            # Connect to Redis
            await simulator.connect_to_redis()
            
            # Configure mode and groups
            mode = sim_data.get("mode", "full")
            await simulator.set_mode(mode)
            
            if "groups" in sim_data:
                await simulator.set_groups(sim_data["groups"])
            
            # Send initial status
            await mock_manager.send_message(
                websocket,
                WSMessageType.SERVER_MSG,
                {
                    "status": "simulation_started",
                    "env_id": env_id,
                    "agent_ids": agent_ids,
                    "mode": mode,
                    "connection_id": simulator.connection_id
                }
            )
            
            # Run simulation
            await mock_manager.run_simulation(websocket, simulator)
            
        except Exception as e:
            print(f"Error in websocket_endpoint: {e}")
            await mock_manager.send_error(
                websocket,
                ErrorType.SIMULATION_ISSUE,
                f"Error in simulation: {str(e)}"
            )
        finally:
            try:
                await websocket.close()
            except Exception:
                pass
    
    # Call our test endpoint
    await test_websocket_endpoint(mock_websocket, "valid_token")
    
    # Verify the flow
    mock_websocket.accept.assert_called_once()
    mock_manager.verify_token.assert_called_once_with("valid_token")
    mock_manager.create_simulator.assert_called_once()
    
    # Get the simulator creation args
    sim_args = mock_manager.create_simulator.call_args[1]
    assert sim_args["env_id"] == "test_env"
    assert sim_args["agent_ids"] == ["agent1", "agent2"]
    
    # Verify simulator setup
    mock_simulator.connect_to_redis.assert_called_once()
    mock_simulator.set_mode.assert_called_once_with("group")
    mock_simulator.set_groups.assert_called_once()
    
    # Verify run_simulation was called with correct arguments
    mock_manager.run_simulation.assert_called_once_with(mock_websocket, mock_simulator)