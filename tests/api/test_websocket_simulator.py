"""
Test the WebSocketSotopiaSimulator group communication functionality
"""

import pytest
import json
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from sotopia.experimental.agents import RedisAgent
# Import the necessary components
from sotopia.api.websocket_utils import WebSocketSotopiaSimulator


class MockPubSub:
    """A properly mocked Redis PubSub client that handles async operations"""

    def __init__(self):
        self.channels = set()
        self.messages = []

    async def subscribe(self, channel):
        """Mock subscribing to a channel"""
        self.channels.add(channel)
        return True

    async def unsubscribe(self, channel):
        """Mock unsubscribing from a channel"""
        if channel in self.channels:
            self.channels.remove(channel)
        return True

    async def psubscribe(self, pattern):
        """Mock pattern subscribing"""
        self.channels.add(f"pattern:{pattern}")
        return True

    async def punsubscribe(self, pattern):
        """Mock pattern unsubscribing"""
        if f"pattern:{pattern}" in self.channels:
            self.channels.remove(f"pattern:{pattern}")
        return True

    async def get_message(self, ignore_subscribe_messages=False, timeout=None):
        """Mock getting a message"""
        if self.messages:
            return self.messages.pop(0)
        return None

    def add_message(self, message):
        """Add a test message to the queue"""
        self.messages.append(message)


# Fixture for mocking Redis
@pytest.fixture
def mock_redis():
    """Create a fully mocked Redis setup that doesn't try to connect to a real server"""
    with patch("redis_om.model.model.get_redis_connection"), patch(
        "redis_om.checks.has_redis_json", return_value=True
    ), patch("redis_om.checks.check_for_command", return_value=True):
        # Mock Redis class
        mock_redis_class = MagicMock()

        # Create mock Redis client
        mock_client = AsyncMock()
        mock_client.pubsub = AsyncMock()
        mock_client.publish = AsyncMock()
        mock_client.close = AsyncMock()

        # Create mock pubsub
        mock_pubsub = MockPubSub()

        # Setup pubsub.get_message to return nothing by default
        mock_pubsub.get_message = AsyncMock(return_value=None)

        # Set up return values
        mock_client.pubsub.return_value = mock_pubsub
        mock_redis_class.return_value = mock_client

        yield mock_redis_class, mock_client, mock_pubsub


@pytest.fixture
def redis_agent(mock_redis):
    """Create a test RedisAgent with Redis connection properly mocked"""
    redis_client, pubsub = mock_redis

    # Create the agent
    agent = RedisAgent(
        input_channels=["moderator:redis_agent"],
        output_channel="redis_agent:moderator",
        node_name="redis_agent",
        other_agent_status={"agent1": True, "agent2": True},
        redis_url="redis://localhost:6379/0",
    )

    # Replace the Redis connection
    agent.r = redis_client

    # Ensure we don't attempt to connect during the test
    agent.start_command_listener = AsyncMock()

    # Setup active connections
    agent.active_connections = {"test_conn_id"}
    agent.last_epilog_hash = {}

    # Setup test data
    agent.mode = "full"
    agent.groups = {}
    agent.pending_actions = []

    return agent, redis_client, pubsub


# Fixture for creating a test simulator with mocks
@pytest.fixture
def test_simulator(mock_redis):
    """Create a test simulator with all Redis-related components properly mocked"""
    # Unpack mocks
    mock_redis_class, mock_redis_client, mock_pubsub = mock_redis

    # Need to patch EnvironmentProfile.get to avoid Redis connection
    with patch(
        "sotopia.api.websocket_utils.EnvironmentProfile.get", side_effect=Exception
    ), patch(
        "sotopia.api.websocket_utils.AgentProfile.get", side_effect=Exception
    ), patch("sotopia.api.websocket_utils.get_env_agents") as mock_get_env_agents:
        # Set up mock environment, agents, and messages
        mock_env = MagicMock()
        mock_agents = MagicMock()
        mock_messages = {}

        # Configure the get_env_agents mock to return our mocked objects
        mock_get_env_agents.return_value = (mock_env, mock_agents, mock_messages)

        # Create the simulator with minimal configuration
        simulator = WebSocketSotopiaSimulator(
            env_id="test_env_id",
            agent_ids=["agent1", "agent2"],
            agent_models=["gpt-4o-mini", "gpt-4o-mini"],
            evaluator_model="gpt-4o",
            max_turns=10,
        )

        # Set up the Redis client manually to avoid actual connections
        simulator.redis_client = mock_redis_client
        simulator.redis_pubsub = mock_pubsub

        # Initialize with an empty latest_epilog
        simulator.latest_epilog = None
        simulator.connection_id = "test_connection_id"

        return simulator, mock_redis_client, mock_pubsub


# Test set_mode method
@pytest.mark.asyncio
async def test_set_mode(test_simulator):
    simulator, redis_client, _ = test_simulator

    # Mock send_to_redis
    simulator.send_to_redis = AsyncMock()

    # Test full mode
    await simulator.set_mode("full")

    # Verify mode was set and send_to_redis was called
    assert simulator.mode == "full"
    simulator.send_to_redis.assert_called_once_with({"mode": "full"})
    simulator.send_to_redis.reset_mock()

    # Test group mode
    await simulator.set_mode("group")

    # Verify mode was set and send_to_redis was called
    assert simulator.mode == "group"
    simulator.send_to_redis.assert_called_once_with({"mode": "group"})


# Test set_groups method
@pytest.mark.asyncio
async def test_set_groups(test_simulator):
    simulator, redis_client, _ = test_simulator

    # Mock send_to_redis
    simulator.send_to_redis = AsyncMock()

    # Test groups data
    groups_data = {"team_a": ["agent1"], "team_b": ["agent2"]}

    # Call the method
    await simulator.set_groups(groups_data)

    # Verify groups were set and send_to_redis was called
    assert simulator.groups == groups_data
    simulator.send_to_redis.assert_called_once_with({"groups": groups_data})


# Test send_message method in full mode
@pytest.mark.asyncio
async def test_send_message_full_mode(test_simulator):
    simulator, redis_client, _ = test_simulator

    # Set mode to full
    simulator.mode = "full"

    # Mock send_to_redis
    simulator.send_to_redis = AsyncMock()

    # Test message data
    message_data = {"content": "Hello from full mode", "sender": "user1"}

    # Call the method
    await simulator.send_message(message_data)

    # Verify send_to_redis was called with the right payload
    expected_payload = {
        "message": {"content": "Hello from full mode", "sender": "user1"}
    }
    simulator.send_to_redis.assert_called_once_with(expected_payload)


# Test process_group_message method
@pytest.mark.asyncio
async def test_process_group_message(test_simulator):
    simulator, redis_client, _ = test_simulator

    # Set mode to group
    simulator.mode = "group"

    # Mock send_to_redis
    simulator.send_to_redis = AsyncMock()

    # Set up groups
    simulator.groups = {"team_a": ["agent1"], "team_b": ["agent2"]}

    # Test message targeting specific agents
    agent_message = {
        "content": "Message to specific agents",
        "target_agents": ["agent1"],
        "target_groups": [],
        "sender": "user1",
    }

    # Process agent message
    await simulator.process_group_message(agent_message)

    # Verify message was sent to Redis
    simulator.send_to_redis.assert_called_once()
    payload = simulator.send_to_redis.call_args[0][0]

    assert payload["content"] == "Message to specific agents"
    assert payload["target_agents"] == ["agent1"]
    assert payload["target_groups"] == []

    # Reset mock
    simulator.send_to_redis.reset_mock()

    # Test message targeting groups
    group_message = {
        "content": "Message to team A",
        "target_agents": [],
        "target_groups": ["team_a"],
        "sender": "user1",
    }

    # Process group message
    await simulator.process_group_message(group_message)

    # Verify message was sent to Redis
    simulator.send_to_redis.assert_called_once()
    payload = simulator.send_to_redis.call_args[0][0]

    assert payload["content"] == "Message to team A"
    assert payload["target_agents"] == []
    assert payload["target_groups"] == ["team_a"]


# Test handle_client_message functionality based on mode
@pytest.mark.asyncio
async def test_handle_client_message(test_simulator):
    simulator, redis_client, _ = test_simulator

    # Mock methods
    simulator.send_message = AsyncMock()
    simulator.process_group_message = AsyncMock()

    # Test in full mode
    simulator.mode = "full"

    full_message = {"content": "Full mode message", "sender": "user1"}

    # Handle message in full mode
    await simulator.handle_client_message(full_message)

    # Verify send_message was called
    simulator.send_message.assert_called_once_with(full_message)
    simulator.process_group_message.assert_not_called()

    # Reset mocks
    simulator.send_message.reset_mock()
    simulator.process_group_message.reset_mock()

    # Test in group mode
    simulator.mode = "group"

    group_message = {
        "content": "Group mode message",
        "target_groups": ["team_a"],
        "sender": "user1",
    }

    # Handle message in group mode
    await simulator.handle_client_message(group_message)

    # Verify process_group_message was called
    simulator.process_group_message.assert_called_once_with(group_message)
    simulator.send_message.assert_not_called()


# Test _redis_subscriber method with epilog message
@pytest.mark.asyncio
async def test_redis_subscriber_epilog(test_simulator):
    simulator, redis_client, pubsub = test_simulator

    # Mock pubsub.get_message to return an epilog message, then cancel
    epilog_data = {
        "type": "SERVER_MSG",
        "data": {"type": "episode_log", "log": {"messages": [["test", "data"]]}},
    }

    # Setup get_message side effect
    pubsub.get_message.side_effect = [
        {"type": "message", "data": json.dumps(epilog_data).encode()},
        None,  # Second call returns None
        asyncio.CancelledError(),  # Force exit from loop
    ]

    # Start subscriber task
    task = asyncio.create_task(simulator._redis_subscriber())

    # Wait a short time for processing
    await asyncio.sleep(0.1)

    # Cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Verify epilog was stored
    assert simulator.latest_epilog == {"messages": [["test", "data"]]}


# Test end-to-end in group mode
@pytest.mark.asyncio
async def test_end_to_end_group_mode(test_simulator, mock_redis):
    simulator, redis_client, pubsub = test_simulator

    # Setup for group mode
    await simulator.set_mode("group")

    # Setup groups
    groups_data = {"team_a": ["agent1"], "team_b": ["agent2"]}
    await simulator.set_groups(groups_data)

    # Verify Redis was updated with mode and groups
    assert redis_client.publish.call_count >= 2

    # Mock send_to_redis to track final message
    original_send_to_redis = simulator.send_to_redis
    simulator.send_to_redis = AsyncMock(side_effect=original_send_to_redis)

    # Create group message
    group_message = {
        "content": "Hello team A from test",
        "target_groups": ["team_a"],
        "sender": "test_user",
    }

    # Process message
    await simulator.process_group_message(group_message)

    # Verify message was sent to Redis
    simulator.send_to_redis.assert_called_once()
    sent_data = simulator.send_to_redis.call_args[0][0]

    # Verify correct fields
    assert sent_data["content"] == "Hello team A from test"
    assert sent_data["target_groups"] == ["team_a"]
    assert sent_data["sender"] == "test_user"
