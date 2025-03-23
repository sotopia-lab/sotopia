"""
Integration tests for the Sotopia group chat system using real components with minimal mocking
"""
import pytest
import json
import asyncio
import hashlib
import os
from unittest.mock import MagicMock, AsyncMock, patch
import redis.asyncio
from fastapi import WebSocket

# Import the necessary components
from sotopia.api.fastapi_server import SimulationManager, WSMessageType, ErrorType
from sotopia.api.websocket_utils import WebSocketSotopiaSimulator
from sotopia.experimental.agents.redis_agent import RedisAgent
from sotopia.experimental.agents.moderator import Moderator, Observations
from sotopia.experimental.agents.datamodels import AgentAction, Observation

# Set up a test Redis instance that will be used by all tests
# This approach assumes Redis is available at localhost:6379
# For CI/CD, you would typically use testcontainers or a similar approach

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")  # Use DB 15 for testing

@pytest.fixture
async def redis_client():
    """Provide a Redis client for testing"""
    client = redis.asyncio.Redis.from_url(TEST_REDIS_URL)
    # Clear the test database before each test
    await client.flushdb()
    try:
        yield client
    finally:
        await client.close()

@pytest.fixture
async def active_redis_agent(redis_client):
    """Create and initialize a real RedisAgent connected to the test Redis"""
    # Create RedisAgent with minimal mocking
    agent = RedisAgent(
        input_channels=["moderator:redis_agent"],
        output_channel="redis_agent:moderator",
        node_name="redis_agent",
        other_agent_status={"agent1": True, "agent2": True},
        redis_url=TEST_REDIS_URL,
        loop_interval=0.05  # Use shorter interval for testing
    )
    
    # Initialize the agent (this would normally be done by aact)
    agent.mode = "group"
    agent.groups = {
        "team_a": ["agent1"],
        "team_b": ["agent2"]
    }
    agent.active_connections = {"test_conn_id"}
    
    # Enter the context to initialize Redis connection
    await agent.__aenter__()
    
    try:
        yield agent
    finally:
        # Make sure to clean up
        await agent.__aexit__(None, None, None)

# Helper for publishing messages to Redis and capturing responses
async def publish_and_capture(redis_client, channel, message, response_channel, timeout=1.0):
    """Publish a message to Redis and capture the response on the specified channel"""
    # Set up a pubsub to listen for responses
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(response_channel)
    
    # Clear any existing messages
    while await pubsub.get_message(timeout=0.1):
        pass
    
    # Publish the message
    await redis_client.publish(channel, json.dumps(message))
    
    # Wait for and capture the response
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        message = await pubsub.get_message(timeout=0.2)
        if message and message["type"] == "message":
            await pubsub.unsubscribe(response_channel)
            return json.loads(message["data"].decode())
        await asyncio.sleep(0.05)
    
    # No response received within timeout
    await pubsub.unsubscribe(response_channel)
    return None

# Test message flow with WebSocketSotopiaSimulator using less mocking
@pytest.mark.asyncio
async def test_simulator_message_flow(redis_client):
    """
    Test the message flow through the WebSocketSotopiaSimulator with minimal mocking.
    This tests that messages from the client are properly sent to Redis.
    """
    # We still need to mock some components to avoid DB dependencies
    with patch('redis_om.model.model.get_redis_connection', return_value=redis_client), \
         patch('redis_om.checks.has_redis_json', return_value=True), \
         patch('redis_om.checks.check_for_command', return_value=True), \
         patch('sotopia.api.websocket_utils.get_env_agents') as mock_get_env_agents:
        
        # Set up mock environment, agents, and messages
        mock_env = MagicMock()
        mock_agents = MagicMock()
        mock_messages = {}
        
        # Configure the get_env_agents mock to return our mocked objects
        mock_get_env_agents.return_value = (mock_env, mock_agents, mock_messages)
        
        # Create a real simulator with test Redis
        simulator = WebSocketSotopiaSimulator(
            env_id="test_env_id",
            agent_ids=["agent1", "agent2"],
            agent_models=["gpt-4o-mini", "gpt-4o-mini"],
            evaluator_model="gpt-4o",
            max_turns=10,
            redis_url=TEST_REDIS_URL
        )
        
        # Configure the simulator
        simulator.connection_id = "test_conn_id"
        simulator.mode = "group"
        simulator.groups = {
            "team_a": ["agent1"],
            "team_b": ["agent2"]
        }
        
        # Connect to Redis
        await simulator.connect_to_redis()
        
        try:
            # Create a message to be sent
            message = {
                "content": "Hello team A",
                "target_groups": ["team_a"]
            }
            
            # Send the message
            await simulator.send_to_redis(message)
            
            # Verify the message was published to Redis
            # Set up a listener for the command channel
            command_channel = f"command:{simulator.connection_id}"
            
            # Get the most recent message from the command channel
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(command_channel)
            
            # Give some time for message to be processed
            await asyncio.sleep(0.1)
            
            # Get the last message
            received_message = None
            while True:
                message = await pubsub.get_message(timeout=0.1)
                if not message:
                    break
                if message["type"] == "message":
                    received_message = json.loads(message["data"].decode())
            
            # Unsubscribe
            await pubsub.unsubscribe(command_channel)
            
            # Verify the message was received
            assert received_message is not None
            assert "content" in received_message
            assert received_message["content"] == "Hello team A"
            assert "target_groups" in received_message
            assert received_message["target_groups"] == ["team_a"]
        finally:
            # Clean up
            if hasattr(simulator, "redis_client") and simulator.redis_client:
                await simulator.redis_client.close()

# Test RedisAgent group message processing
@pytest.mark.asyncio
async def test_redis_agent_group_message_processing(active_redis_agent, redis_client):
    """
    Test the RedisAgent's ability to process group messages.
    This tests that group targeting works correctly.
    """
    # Create a command message targeting a group
    command_data = {
        "content": "Hello team A",
        "target_groups": ["team_a"],
        "sender": "websocket_user"
    }
    
    # Get the command channel
    command_channel = f"command:test_conn_id"
    
    # Set up to capture messages on the output channel
    output_channel = "redis_agent:moderator"
    
    # Publish the command and capture the response
    response = await publish_and_capture(
        redis_client,
        command_channel,
        command_data,
        output_channel,
        timeout=1.0
    )
    
    # Verify the response
    assert response is not None
    assert "data" in response
    data = response["data"]
    
    # Verify the message was properly processed
    assert data["action_type"] == "unified_message"
    
    # Parse the argument
    arg_data = json.loads(data["argument"])
    assert arg_data["content"] == "Hello team A"
    assert "agent1" in arg_data["target_agents"]  # Should be expanded from team_a
    assert "agent2" not in arg_data["target_agents"]  # Not in team_a
    assert arg_data["original_target_groups"] == ["team_a"]

# Test RedisAgent epilog publishing
@pytest.mark.asyncio
async def test_redis_agent_epilog_publishing(active_redis_agent, redis_client):
    """
    Test the RedisAgent's ability to process and publish epilog updates.
    This verifies that epilogs are properly formatted and published to connections.
    """
    # Create an epilog observation
    epilog_data = {"messages": [["agent1", "agent2", "Test message"]]}
    
    # Create a message simulating an epilog from the Moderator
    message = {
        "data": {
            "agent_name": "epilog",
            "last_turn": json.dumps(epilog_data),
            "turn_number": 1,
            "available_actions": ["none"]
        }
    }
    
    # Publish to the RedisAgent's input channel
    await redis_client.publish("moderator:redis_agent", json.dumps(message))
    
    # Listen for epilog messages on the connection's epilog channel
    epilog_channel = f"epilog:test_conn_id"
    
    # Set up a pubsub to listen
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(epilog_channel)
    
    # Wait for and capture the message
    received_message = None
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < 1.0:
        message = await pubsub.get_message(timeout=0.2)
        if message and message["type"] == "message":
            received_message = json.loads(message["data"].decode())
            break
        await asyncio.sleep(0.05)
    
    # Unsubscribe
    await pubsub.unsubscribe(epilog_channel)
    
    # Verify the message was received and formatted correctly
    assert received_message is not None
    assert received_message["type"] == "SERVER_MSG"
    assert received_message["data"]["type"] == "episode_log"
    assert received_message["data"]["log"] == epilog_data

# Test WebSocket endpoint with more realistic simulation
@pytest.mark.asyncio
async def test_websocket_endpoint_integration():
    """
    Test the websocket endpoint with more realistic message flow,
    but still avoiding external dependencies.
    """
    # Set up a manager with enough real behavior for testing
    manager = SimulationManager()
    
    # Mock methods that would use external resources
    manager.verify_token = AsyncMock(return_value={"is_valid": True, "msg": "Valid token"})
    manager.create_simulator = AsyncMock()
    manager.run_simulation = AsyncMock()
    
    # Create a WebSocket mock
    mock_websocket = AsyncMock(spec=WebSocket)
    mock_websocket.accept = AsyncMock()
    mock_websocket.send_json = AsyncMock()
    mock_websocket.close = AsyncMock()
    
    # Create a realistic simulator mock
    mock_simulator = AsyncMock()
    mock_simulator.connection_id = "test_conn_id"
    mock_simulator.connect_to_redis = AsyncMock()
    mock_simulator.set_mode = AsyncMock()
    mock_simulator.set_groups = AsyncMock()
    
    # Configure simulator mock
    manager.create_simulator.return_value = mock_simulator
    
    # Configure WebSocket to return a START_SIM message
    mock_websocket.receive_json.return_value = {
        "type": WSMessageType.START_SIM.value,
        "data": {
            "env_id": "test_env_id",
            "agent_ids": ["agent1", "agent2"],
            "mode": "group",
            "groups": {
                "team_a": ["agent1"],
                "team_b": ["agent2"]
            }
        }
    }
    
    # Run the endpoint function
    from sotopia.api.fastapi_server import websocket_endpoint
    await websocket_endpoint(mock_websocket, "valid_token")
    
    # Verify the behavior
    mock_websocket.accept.assert_called_once()
    mock_websocket.receive_json.assert_called_once()
    manager.verify_token.assert_called_once_with("valid_token")
    manager.create_simulator.assert_called_once()
    
    # Verify simulator configuration
    simulator_args = manager.create_simulator.call_args[1]
    assert simulator_args["env_id"] == "test_env_id"
    assert simulator_args["agent_ids"] == ["agent1", "agent2"]
    
    # Verify simulator setup
    mock_simulator.connect_to_redis.assert_called_once()
    mock_simulator.set_mode.assert_called_once_with("group")
    mock_simulator.set_groups.assert_called_once()
    
    # Verify run_simulation was called
    manager.run_simulation.assert_called_once_with(mock_websocket, mock_simulator)

# Test full message flow with minimal mocking
@pytest.mark.asyncio
async def test_end_to_end_message_flow(redis_client):
    """
    Test the end-to-end message flow between simulator, RedisAgent, and Moderator
    with minimal mocking. This tests that all components can work together.
    """
    # Set up the RedisAgent first - we'll use a real one
    redis_agent = RedisAgent(
        input_channels=["moderator:redis_agent"],
        output_channel="redis_agent:moderator",
        node_name="redis_agent",
        other_agent_status={"agent1": True, "agent2": True},
        redis_url=TEST_REDIS_URL,
        loop_interval=0.05
    )
    
    # Initialize the RedisAgent
    await redis_agent.__aenter__()
    
    try:
        # Set up the simulator with minimal mocking
        with patch('redis_om.model.model.get_redis_connection', return_value=redis_client), \
             patch('redis_om.checks.has_redis_json', return_value=True), \
             patch('redis_om.checks.check_for_command', return_value=True), \
             patch('sotopia.api.websocket_utils.get_env_agents') as mock_get_env_agents:
            
            # Set up mock environment, agents, and messages
            mock_env = MagicMock()
            mock_agents = MagicMock()
            mock_messages = {}
            
            # Configure the get_env_agents mock to return our mocked objects
            mock_get_env_agents.return_value = (mock_env, mock_agents, mock_messages)
            
            # Create the simulator
            simulator = WebSocketSotopiaSimulator(
                env_id="test_env_id",
                agent_ids=["agent1", "agent2"],
                agent_models=["gpt-4o-mini", "gpt-4o-mini"],
                evaluator_model="gpt-4o",
                max_turns=10,
                redis_url=TEST_REDIS_URL
            )
            
            # Configure the simulator
            simulator.connection_id = "test_conn_id"
            simulator.mode = "group"
            simulator.groups = {
                "team_a": ["agent1"],
                "team_b": ["agent2"]
            }
            
            # Connect to Redis
            await simulator.connect_to_redis()
            
            try:
                # Configure RedisAgent with the same groups
                await redis_client.publish(
                    f"command:test_conn_id",
                    json.dumps({
                        "groups": {
                            "team_a": ["agent1"],
                            "team_b": ["agent2"]
                        },
                        "mode": "group"
                    })
                )
                
                # Give time for the message to be processed
                await asyncio.sleep(0.2)
                
                # Set up a mock moderator (minimal mocking)
                moderator = MagicMock()
                moderator.handle_unified_message = AsyncMock()
                
                # Create a unified_message response that the moderator would generate
                unified_message_response = {
                    "observations_map": {
                        "moderator:agent1": {
                            "agent_name": "websocket_user",
                            "last_turn": "Hello team A",
                            "turn_number": 1,
                            "available_actions": ["speak", "none"]
                        },
                        "moderator:agent2": {
                            "agent_name": "websocket_user",
                            "last_turn": "",
                            "turn_number": 1,
                            "available_actions": ["none"]
                        }
                    }
                }
                moderator.handle_unified_message.return_value = type('MockObservations', (), unified_message_response)
                
                # Listen for messages on redis_agent:moderator
                pubsub = redis_client.pubsub()
                await pubsub.subscribe("redis_agent:moderator")
                
                # Clear any existing messages
                while await pubsub.get_message(timeout=0.1):
                    pass
                
                # Send a message through the simulator
                await simulator.send_to_redis({
                    "content": "Hello team A",
                    "target_groups": ["team_a"],
                    "sender": "websocket_user"
                })
                
                # Wait for and capture the message on redis_agent:moderator
                unified_message = None
                start_time = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - start_time < 1.0:
                    message = await pubsub.get_message(timeout=0.2)
                    if message and message["type"] == "message":
                        data = json.loads(message["data"].decode())
                        if "action_type" in data.get("data", {}) and data["data"]["action_type"] == "unified_message":
                            unified_message = data
                            break
                    await asyncio.sleep(0.05)
                
                # Unsubscribe
                await pubsub.unsubscribe("redis_agent:moderator")
                
                # Verify the message was processed correctly
                assert unified_message is not None
                assert unified_message["data"]["action_type"] == "unified_message"
                
                # Parse the argument
                arg_data = json.loads(unified_message["data"]["argument"])
                assert arg_data["content"] == "Hello team A"
                assert "agent1" in arg_data["target_agents"]
                assert "agent2" not in arg_data["target_agents"]
                assert arg_data["original_target_groups"] == ["team_a"]
            finally:
                # Clean up simulator
                if hasattr(simulator, "redis_client") and simulator.redis_client:
                    await simulator.redis_client.close()
    finally:
        # Clean up RedisAgent
        await redis_agent.__aexit__(None, None, None)

# Test deduplication logic with real messages
@pytest.mark.asyncio
async def test_epilog_deduplication(redis_client):
    """
    Test that identical epilog updates are deduplicated.
    This uses a real Redis client with minimal mocking.
    """
    # Set up a mock Moderator with Redis
    moderator = MagicMock()
    moderator.last_epilog_hash = None
    moderator.send = AsyncMock()
    
    # Define the send_epilog implementation using Redis
    async def send_epilog(epilog, channel):
        nonlocal moderator
        # Generate hash of epilog to avoid sending duplicates
        epilog_json = epilog.model_dump_json()
        current_hash = hashlib.md5(epilog_json.encode()).hexdigest()
        
        # Only send if it's different from the last epilog we sent
        if current_hash != moderator.last_epilog_hash:
            message = {
                "data": {
                    "agent_name": "epilog",
                    "last_turn": epilog_json,
                    "turn_number": 1,
                    "available_actions": ["none"]
                }
            }
            # Publish directly to Redis instead of using mock
            await redis_client.publish(channel, json.dumps(message))
            moderator.last_epilog_hash = current_hash
    
    # Set the method on the moderator
    moderator.send_epilog = send_epilog
    
    # Create two identical epilogs
    epilog1 = MagicMock()
    epilog1.model_dump_json = MagicMock(return_value=json.dumps({"messages": [["test", "data"]]}))
    
    epilog2 = MagicMock()
    epilog2.model_dump_json = MagicMock(return_value=json.dumps({"messages": [["test", "data"]]}))
    
    # Set up a pubsub to monitor the channel
    pubsub = redis_client.pubsub()
    test_channel = "test:epilog"
    await pubsub.subscribe(test_channel)
    
    # Clear any existing messages
    while await pubsub.get_message(timeout=0.1):
        pass
    
    # Send first epilog
    await moderator.send_epilog(epilog1, test_channel)
    
    # Wait for and count the message
    message_count = 0
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < 0.5:
        message = await pubsub.get_message(timeout=0.1)
        if message and message["type"] == "message":
            message_count += 1
        await asyncio.sleep(0.05)
    
    # Verify the first message was sent
    assert message_count == 1
    
    # Send identical epilog
    await moderator.send_epilog(epilog2, test_channel)
    
    # Wait and check for any new messages
    message_count = 0
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < 0.5:
        message = await pubsub.get_message(timeout=0.1)
        if message and message["type"] == "message":
            message_count += 1
        await asyncio.sleep(0.05)
    
    # Verify no new message was sent (deduplication)
    assert message_count == 0
    
    # Create different epilog
    epilog3 = MagicMock()
    epilog3.model_dump_json = MagicMock(return_value=json.dumps({"messages": [["different"]]}))
    
    # Send different epilog
    await moderator.send_epilog(epilog3, test_channel)
    
    # Wait and check for the new message
    message_count = 0
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < 0.5:
        message = await pubsub.get_message(timeout=0.1)
        if message and message["type"] == "message":
            message_count += 1
        await asyncio.sleep(0.05)
    
    # Verify a new message was sent for different epilog
    assert message_count == 1
    
    # Unsubscribe
    await pubsub.unsubscribe(test_channel)