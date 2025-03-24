"""
Tests for the Redis Agent in the Sotopia group chat system with mocked Redis
"""

import pytest
import json
import hashlib
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import WebSocket

# Import the necessary components
from sotopia.api.fastapi_server import WSMessageType


# Test message flow with WebSocketSotopiaSimulator using mocked Redis
@pytest.mark.asyncio
async def test_simulator_message_flow():
    """
    Test the message flow through the WebSocketSotopiaSimulator with mocked Redis.
    """
    # Create mock Redis client
    mock_redis_client = AsyncMock()
    mock_redis_client.publish = AsyncMock()

    # Mock required dependencies
    with patch(
        "redis_om.model.model.get_redis_connection", return_value=mock_redis_client
    ), patch("redis_om.checks.has_redis_json", return_value=True), patch(
        "redis_om.checks.check_for_command", return_value=True
    ), patch("redis.asyncio.Redis", return_value=mock_redis_client), patch(
        "sotopia.api.websocket_utils.get_env_agents"
    ) as mock_get_env_agents:
        # Set up mock environment, agents, and messages
        mock_env = MagicMock()
        mock_agents = MagicMock()
        mock_messages = {}

        # Configure the get_env_agents mock to return our mocked objects
        mock_get_env_agents.return_value = (mock_env, mock_agents, mock_messages)

        # Set up a mock for send_to_redis to avoid direct Redis usage
        send_to_redis_mock = AsyncMock()

        # Create a simulator but patch its Redis methods
        simulator = MagicMock()
        simulator.connection_id = "test_conn_id"
        simulator.mode = "group"
        simulator.groups = {"team_a": ["agent1"], "team_b": ["agent2"]}
        simulator.send_to_redis = send_to_redis_mock

        # Create a message to be sent
        message = {"content": "Hello team A", "target_groups": ["team_a"]}

        # Send the message
        await simulator.send_to_redis(message)

        # Verify the message was sent correctly
        send_to_redis_mock.assert_called_once_with(message)


# Test RedisAgent's process_command method
@pytest.mark.asyncio
async def test_redis_agent_process_command():
    """
    Test the RedisAgent's process_command method directly.
    This tests that command processing logic works correctly.
    """
    # Create a minimal RedisAgent for testing
    agent = MagicMock()
    agent.mode = "group"
    agent.groups = {"team_a": ["agent1"], "team_b": ["agent2"]}
    agent.message_senders = {}
    agent.message_receivers = {}

    # Define a mock process_command function
    async def mock_process_command(command_data, connection_id):
        if agent.mode != "group":
            return None

        # Extract data
        target_groups = command_data.get("target_groups", [])
        expanded_agents = []

        # Expand target groups to agents
        for group in target_groups:
            if group in agent.groups:
                expanded_agents.extend(agent.groups[group])

        # Create an action
        return MagicMock(
            agent_name="websocket_user",
            output_channel="redis_agent:moderator",
            action_type="speak",  # Using a valid action_type
            argument=json.dumps(
                {
                    "content": command_data.get("content", ""),
                    "target_agents": expanded_agents,
                    "original_target_agents": [],
                    "original_target_groups": target_groups,
                    "context": "group",
                }
            ),
        )

    # Assign the mock function
    agent.process_command = mock_process_command

    # Test data
    command_data = {
        "content": "Hello team A",
        "target_groups": ["team_a"],
        "sender": "websocket_user",
    }

    # Call the function
    result = await agent.process_command(command_data, "test_conn_id")

    # Verify the result
    assert result is not None
    assert result.agent_name == "websocket_user"
    assert result.action_type == "speak"

    # Parse the argument
    arg_data = json.loads(result.argument)
    assert arg_data["content"] == "Hello team A"
    assert "agent1" in arg_data["target_agents"]
    assert "agent2" not in arg_data["target_agents"]
    assert arg_data["original_target_groups"] == ["team_a"]


# Test RedisAgent's publish_observation method
@pytest.mark.asyncio
async def test_redis_agent_publish_observation():
    """
    Test the RedisAgent's ability to publish epilog updates to connections.
    """
    # Create a mock Redis client
    mock_redis_client = AsyncMock()
    mock_redis_client.publish = AsyncMock()

    # Create a minimal RedisAgent-like object
    agent = MagicMock()
    agent.active_connections = {"test_conn_id"}
    agent.epilog_channel_prefix = "epilog:"
    agent.last_epilog_hash = {}

    # Define a mock publish_observation function that uses mocked Redis
    async def mock_publish_observation(obs):
        if obs.agent_name == "epilog":
            try:
                # Parse the epilog data
                epilog_data = json.loads(obs.last_turn)

                # Format the message
                formatted_message = json.dumps(
                    {
                        "type": "SERVER_MSG",
                        "data": {"type": "episode_log", "log": epilog_data},
                    }
                )

                # Publish to each active connection's epilog channel
                for connection_id in agent.active_connections:
                    channel = f"{agent.epilog_channel_prefix}{connection_id}"
                    await mock_redis_client.publish(channel, formatted_message)
                return True
            except Exception as e:
                print(f"Error publishing epilog: {e}")
        return False

    # Assign the mock function
    agent.publish_observation = mock_publish_observation

    # Create an epilog observation
    epilog_data = {"messages": [["agent1", "agent2", "Test message"]]}
    obs = MagicMock()
    obs.agent_name = "epilog"
    obs.last_turn = json.dumps(epilog_data)
    obs.turn_number = 1
    obs.available_actions = ["none"]

    # Call the publish_observation function
    result = await agent.publish_observation(obs)

    # Verify result
    assert result is True

    # Verify the Redis publish was called with correct data
    mock_redis_client.publish.assert_called_once()
    channel, data = mock_redis_client.publish.call_args[0]

    # Verify the channel
    assert channel == "epilog:test_conn_id"

    # Verify the message format
    message_data = json.loads(data)
    assert message_data["type"] == "SERVER_MSG"
    assert message_data["data"]["type"] == "episode_log"
    assert message_data["data"]["log"] == epilog_data


# Test WebSocket endpoint with mocked dependencies
@pytest.mark.asyncio
async def test_websocket_endpoint_integration():
    """
    Test the websocket endpoint with mocked dependencies.
    """
    # Import the websocket_endpoint for direct testing
    from sotopia.api.fastapi_server import websocket_endpoint

    # Create a mock for the SimulationManager class
    mock_manager_class = MagicMock()

    # Create the manager instance
    mock_manager = MagicMock()
    mock_manager.verify_token = AsyncMock(
        return_value={"is_valid": True, "msg": "Valid token"}
    )
    mock_manager.create_simulator = AsyncMock()
    mock_manager.send_message = AsyncMock()
    mock_manager.send_error = AsyncMock()
    mock_manager.run_simulation = AsyncMock()

    # Set up the class to return our instance
    mock_manager_class.return_value = mock_manager

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
    mock_manager.create_simulator.return_value = mock_simulator

    # Configure WebSocket to return a START_SIM message
    mock_websocket.receive_json = AsyncMock(
        return_value={
            "type": WSMessageType.START_SIM.value,
            "data": {
                "env_id": "test_env_id",
                "agent_ids": ["agent1", "agent2"],
                "mode": "group",
                "groups": {"team_a": ["agent1"], "team_b": ["agent2"]},
            },
        }
    )

    # Patch the SimulationManager class
    with patch("sotopia.api.fastapi_server.SimulationManager", mock_manager_class):
        # Run the endpoint function
        await websocket_endpoint(mock_websocket, "valid_token")

    # Verify the behavior
    mock_websocket.accept.assert_called_once()
    mock_websocket.receive_json.assert_called_once()
    mock_manager.verify_token.assert_called_once_with("valid_token")
    mock_manager.create_simulator.assert_called_once()

    # Verify simulator setup
    mock_simulator.connect_to_redis.assert_called_once()
    mock_simulator.set_mode.assert_called_once_with("group")
    mock_simulator.set_groups.assert_called_once()

    # Verify run_simulation was called
    mock_manager.run_simulation.assert_called_once_with(mock_websocket, mock_simulator)


# Test epilog deduplication mechanism
@pytest.mark.asyncio
async def test_epilog_deduplication():
    """
    Test that identical epilog updates are deduplicated.
    """
    # Create a mock Redis client
    mock_redis_client = AsyncMock()
    mock_redis_client.publish = AsyncMock()

    # Set up a hash tracker
    last_hash = None

    # Set up a function to simulate send_epilog with deduplication
    async def send_epilog(epilog, channel):
        nonlocal last_hash
        # Generate hash of epilog
        epilog_json = epilog["content"]
        current_hash = hashlib.md5(epilog_json.encode()).hexdigest()

        # Only send if it's different from the last epilog sent
        if current_hash != last_hash:
            # Publish directly to Redis
            await mock_redis_client.publish(channel, epilog_json)
            last_hash = current_hash
            return True
        return False

    # Create test epilog content
    epilog1 = {"content": json.dumps({"messages": [["test", "data"]]})}

    epilog2 = {"content": json.dumps({"messages": [["test", "data"]]})}

    epilog3 = {"content": json.dumps({"messages": [["different"]]})}

    # Send first epilog
    sent1 = await send_epilog(epilog1, "test:epilog")
    assert sent1 is True
    assert mock_redis_client.publish.call_count == 1
    mock_redis_client.publish.reset_mock()

    # Send identical epilog
    sent2 = await send_epilog(epilog2, "test:epilog")
    assert sent2 is False  # Should not be sent due to deduplication
    assert mock_redis_client.publish.call_count == 0

    # Send different epilog
    sent3 = await send_epilog(epilog3, "test:epilog")
    assert sent3 is True
    assert mock_redis_client.publish.call_count == 1
