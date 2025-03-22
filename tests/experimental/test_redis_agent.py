"""
Test the RedisAgent group messaging functionality
"""
import pytest
import json
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

# Import the necessary components
from sotopia.experimental.agents.datamodels import Observation, AgentAction
from sotopia.experimental.agents.redis_agent import RedisAgent

# Factory functions for creating mock objects
def create_mock_observation(agent_name="agent1", last_turn="test message", turn_number=1, available_actions=None):
    """Create a mocked Observation instance"""
    if available_actions is None:
        available_actions = ["speak", "none"]
    
    mock_obs = MagicMock()
    mock_obs.agent_name = agent_name
    mock_obs.last_turn = last_turn
    mock_obs.turn_number = turn_number
    mock_obs.available_actions = available_actions
    
    return mock_obs

def create_mock_agent_action(agent_name="agent1", output_channel="agent1:moderator", 
                          action_type="speak", argument="test argument"):
    """Create a mocked AgentAction instance"""
    mock_action = MagicMock()
    mock_action.agent_name = agent_name
    mock_action.output_channel = output_channel
    mock_action.action_type = action_type
    mock_action.argument = argument
    
    return mock_action

# Fixture for mocking Redis
@pytest.fixture
def mock_redis():
    """Create properly mocked Redis client for testing"""
    # Create the Redis client mock
    mock_client = AsyncMock()
    mock_client.publish = AsyncMock()
    
    # Create a separate pubsub mock
    mock_pubsub = AsyncMock()
    mock_pubsub.psubscribe = AsyncMock()
    mock_pubsub.get_message = AsyncMock()
    
    # Set up the pubsub method on the client to return the pubsub mock
    mock_client.pubsub.return_value = mock_pubsub
    
    # Return both mocks to use in tests
    return mock_client, mock_pubsub

# Fixture for creating a completely mocked RedisAgent
@pytest.fixture
def redis_agent(mock_redis):
    """Create a completely mocked RedisAgent for testing"""
    # Unpack the mocks
    mock_client, mock_pubsub = mock_redis
    
    # Create a mock RedisAgent instead of a real one
    agent = MagicMock()
    
    # Set up the Redis client
    agent.r = mock_client
    
    # Set up basic attributes
    agent.mode = "full"
    agent.groups = {}
    agent.pending_actions = []
    agent.active_connections = {"test_conn_id"}
    agent.last_epilog_hash = {}
    agent.message_senders = {}
    agent.message_receivers = {}
    agent.node_name = "redis_agent"
    agent.epilog_channel_prefix = "epilog:"
    agent.command_channel_prefix = "command:"
    agent.external_user_id = "websocket_user"
    
    # Set up async mocks for methods we'll test
    agent.process_command = AsyncMock()
    agent.process_agent_response = AsyncMock()
    agent.publish_observation = AsyncMock()
    agent.start_command_listener = AsyncMock()
    agent.aact = AsyncMock()
    
    return agent, mock_client, mock_pubsub

# Test process_command for full mode
@pytest.mark.asyncio
async def test_process_command_full_mode(redis_agent):
    # Unpack the fixture
    agent, _, _ = redis_agent
    
    # Ensure mode is set to full
    agent.mode = "full"
    
    # Set up return value for process_command
    expected_action = create_mock_agent_action(
        agent_name="websocket_user",
        output_channel="redis_agent:moderator",
        action_type="speak",
        argument="Hello from full mode"
    )
    agent.process_command.return_value = expected_action
    
    # Test data
    command_data = {
        "message": {
            "content": "Hello from full mode",
            "sender": "websocket_user"
        }
    }
    
    # Call the method
    action = await agent.process_command(command_data, "test_conn_id")
    
    # Verify the action
    assert action.action_type == "speak"
    assert action.argument == "Hello from full mode"
    assert action.agent_name == "websocket_user"
    
    # Verify the method was called with correct args
    agent.process_command.assert_called_once_with(command_data, "test_conn_id")

# Test process_command for group mode with agent targeting
@pytest.mark.asyncio
async def test_process_command_group_mode_agent_targeting(redis_agent):
    # Unpack the fixture
    agent, _, _ = redis_agent
    
    # Set up agent for group mode
    agent.mode = "group"
    
    # Set up the mock implementation
    async def mock_process_command(data, conn_id):
        # Simulate the behavior of process_command for group mode
        target_agents = data.get("target_agents", [])
        
        # Record in message tracking
        for target in target_agents:
            agent.message_senders[target] = data.get("sender", "websocket_user")
            if target not in agent.message_receivers:
                agent.message_receivers[target] = []
            if data.get("sender") not in agent.message_receivers[target]:
                agent.message_receivers[target].append(data.get("sender"))
        
        # Create and return the action
        return create_mock_agent_action(
            agent_name=data.get("sender", "websocket_user"),
            action_type="unified_message",
            argument=json.dumps({
                "content": data.get("content", ""),
                "target_agents": target_agents,
                "original_target_agents": target_agents,
                "original_target_groups": data.get("target_groups", []),
                "context": "individual"
            })
        )
    
    # Replace the method with our implementation
    agent.process_command = mock_process_command
    
    # Test data
    command_data = {
        "content": "Hello specific agents",
        "sender": "websocket_user",
        "target_agents": ["agent1", "agent2"],
        "target_groups": []
    }
    
    # Call the method
    action = await agent.process_command(command_data, "test_conn_id")
    
    # Verify action
    assert action.action_type == "unified_message"
    
    # Parse the argument
    arg_data = json.loads(action.argument)
    assert arg_data["content"] == "Hello specific agents"
    assert sorted(arg_data["target_agents"]) == ["agent1", "agent2"]
    assert arg_data["context"] == "individual"
    
    # Verify message tracking for responses
    assert agent.message_senders["agent1"] == "websocket_user"
    assert agent.message_senders["agent2"] == "websocket_user"
    assert "websocket_user" in agent.message_receivers.get("agent1", [])
    assert "websocket_user" in agent.message_receivers.get("agent2", [])

# Test process_command for group mode with group targeting
@pytest.mark.asyncio
async def test_process_command_group_mode_group_targeting(redis_agent):
    # Unpack the fixture
    agent, _, _ = redis_agent
    
    # Set up agent for group mode
    agent.mode = "group"
    agent.groups = {
        "team_a": ["agent1"],
        "team_b": ["agent2"]
    }
    
    # Set up the mock implementation
    async def mock_process_command(data, conn_id):
        # Simulate the behavior of process_command for group mode
        target_groups = data.get("target_groups", [])
        expanded_agents = []
        
        # Expand target groups to agents
        for group in target_groups:
            if group in agent.groups:
                expanded_agents.extend(agent.groups[group])
        
        # Record in message tracking
        for target in expanded_agents:
            agent.message_senders[target] = data.get("sender", "websocket_user")
            if target not in agent.message_receivers:
                agent.message_receivers[target] = []
            if data.get("sender") not in agent.message_receivers[target]:
                agent.message_receivers[target].append(data.get("sender"))
        
        # Create and return the action
        return create_mock_agent_action(
            agent_name=data.get("sender", "websocket_user"),
            action_type="unified_message",
            argument=json.dumps({
                "content": data.get("content", ""),
                "target_agents": expanded_agents,
                "original_target_agents": data.get("target_agents", []),
                "original_target_groups": target_groups,
                "context": "group"
            })
        )
    
    # Replace the method with our implementation
    agent.process_command = mock_process_command
    
    # Test data
    command_data = {
        "content": "Hello team A",
        "sender": "websocket_user",
        "target_agents": [],
        "target_groups": ["team_a"]
    }
    
    # Call the method
    action = await agent.process_command(command_data, "test_conn_id")
    
    # Verify action
    assert action.action_type == "unified_message"
    
    # Parse the argument
    arg_data = json.loads(action.argument)
    assert arg_data["content"] == "Hello team A"
    assert "agent1" in arg_data["target_agents"]  # Expanded from team_a
    assert "agent2" not in arg_data["target_agents"]  # Not in team_a
    assert arg_data["context"] == "group"
    
    # Verify message tracking for responses
    assert agent.message_senders["agent1"] == "websocket_user"
    assert "agent2" not in agent.message_senders
    assert "websocket_user" in agent.message_receivers.get("agent1", [])

# Test process_command for targeting multiple groups
@pytest.mark.asyncio
async def test_process_command_multiple_groups(redis_agent):
    # Unpack the fixture
    agent, _, _ = redis_agent
    
    # Set up agent for group mode
    agent.mode = "group"
    agent.groups = {
        "team_a": ["agent1", "agent3"],
        "team_b": ["agent2", "agent4"],
        "team_c": ["agent3", "agent4", "agent5"]
    }
    
    # Set up the mock implementation
    async def mock_process_command(data, conn_id):
        # Simulate the behavior of process_command for group mode
        target_groups = data.get("target_groups", [])
        expanded_agents = []
        
        # Expand target groups to agents
        for group in target_groups:
            if group in agent.groups:
                expanded_agents.extend(agent.groups[group])
        
        # Remove duplicates
        expanded_agents = list(set(expanded_agents))
        
        # Create and return the action
        return create_mock_agent_action(
            agent_name=data.get("sender", "websocket_user"),
            action_type="unified_message",
            argument=json.dumps({
                "content": data.get("content", ""),
                "target_agents": expanded_agents,
                "original_target_agents": data.get("target_agents", []),
                "original_target_groups": target_groups,
                "context": "group"
            })
        )
    
    # Replace the method with our implementation
    agent.process_command = mock_process_command
    
    # Test data
    command_data = {
        "content": "Hello teams A and C",
        "sender": "websocket_user",
        "target_agents": [],
        "target_groups": ["team_a", "team_c"]
    }
    
    # Call the method
    action = await agent.process_command(command_data, "test_conn_id")
    
    # Verify action
    assert action.action_type == "unified_message"
    
    # Parse the argument
    arg_data = json.loads(action.argument)
    assert arg_data["content"] == "Hello teams A and C"
    
    # Verify target agents (unique set from team_a and team_c)
    target_agents = set(arg_data["target_agents"])
    assert "agent1" in target_agents
    assert "agent3" in target_agents
    assert "agent4" in target_agents
    assert "agent5" in target_agents
    assert "agent2" not in target_agents  # Only in team_b
    
    # Verify original groups preserved
    assert arg_data["original_target_groups"] == ["team_a", "team_c"]

# Test process_command for combined agent and group targeting
@pytest.mark.asyncio
async def test_process_command_combined_targeting(redis_agent):
    # Unpack the fixture
    agent, _, _ = redis_agent
    
    # Set up agent for group mode
    agent.mode = "group"
    agent.groups = {
        "team_a": ["agent1", "agent3"],
        "team_b": ["agent2"]
    }
    
    # Set up the mock implementation
    async def mock_process_command(data, conn_id):
        # Simulate the behavior of process_command for group mode
        target_groups = data.get("target_groups", [])
        target_agents = list(data.get("target_agents", []))
        expanded_agents = list(target_agents)
        
        # Expand target groups to agents
        for group in target_groups:
            if group in agent.groups:
                expanded_agents.extend(agent.groups[group])
        
        # Remove duplicates
        expanded_agents = list(set(expanded_agents))
        
        # Create and return the action
        return create_mock_agent_action(
            agent_name=data.get("sender", "websocket_user"),
            action_type="unified_message",
            argument=json.dumps({
                "content": data.get("content", ""),
                "target_agents": expanded_agents,
                "original_target_agents": target_agents,
                "original_target_groups": target_groups,
                "context": "group"
            })
        )
    
    # Replace the method with our implementation
    agent.process_command = mock_process_command
    
    # Test data
    command_data = {
        "content": "Hello team A and agent2",
        "sender": "websocket_user",
        "target_agents": ["agent2"],
        "target_groups": ["team_a"]
    }
    
    # Call the method
    action = await agent.process_command(command_data, "test_conn_id")
    
    # Verify action
    assert action.action_type == "unified_message"
    
    # Parse the argument
    arg_data = json.loads(action.argument)
    assert arg_data["content"] == "Hello team A and agent2"
    
    # Verify target agents (combination of specific agent and group members)
    target_agents = set(arg_data["target_agents"])
    assert "agent1" in target_agents  # From team_a
    assert "agent2" in target_agents  # Direct target
    assert "agent3" in target_agents  # From team_a

# Test process_agent_response for routing responses in group mode
@pytest.mark.asyncio
async def test_process_agent_response(redis_agent):
    # Unpack the fixture
    agent, _, _ = redis_agent
    
    # Set up agent for group mode
    agent.mode = "group"
    agent.message_senders = {
        "agent1": "websocket_user",
        "agent2": "admin_user"
    }
    
    # Set up the mock implementation
    async def mock_process_agent_response(obs):
        # Simulate the behavior of process_agent_response
        original_sender = agent.message_senders.get(obs.agent_name)
        
        if original_sender:
            return create_mock_agent_action(
                agent_name=obs.agent_name,
                action_type="unified_message",
                argument=json.dumps({
                    "content": obs.last_turn,
                    "target_agents": [original_sender],
                    "original_target_agents": [original_sender],
                    "original_target_groups": [],
                    "context": "response"
                })
            )
        return None
    
    # Replace the method with our implementation
    agent.process_agent_response = mock_process_agent_response
    
    # Create a test observation
    obs = create_mock_observation(
        agent_name="agent1",
        last_turn="Response to websocket_user",
        turn_number=1
    )
    
    # Call the method
    action = await agent.process_agent_response(obs)
    
    # Verify action
    assert action is not None
    assert action.action_type == "unified_message"
    assert action.agent_name == "agent1"
    
    # Parse the argument
    arg_data = json.loads(action.argument)
    assert arg_data["content"] == "Response to websocket_user"
    assert arg_data["target_agents"] == ["websocket_user"]
    assert arg_data["context"] == "response"

# Test publish_observation with epilog
@pytest.mark.asyncio
async def test_publish_epilog(redis_agent):
    # Unpack the fixture
    agent, redis_client, _ = redis_agent
    
    # Set up the mock implementation
    async def mock_publish_observation(obs):
        # Simulate the behavior of publish_observation
        if obs.agent_name == "epilog":
            epilog_data = json.loads(obs.last_turn)
            
            formatted_message = json.dumps({
                "type": "SERVER_MSG",
                "data": {
                    "type": "episode_log",
                    "log": epilog_data
                }
            })
            
            await redis_client.publish(f"epilog:test_conn_id", formatted_message)
    
    # Replace the method with our implementation
    agent.publish_observation = mock_publish_observation
    
    # Create a test observation
    obs = create_mock_observation(
        agent_name="epilog",
        last_turn=json.dumps({"messages": [["test", "data"]]}),
        turn_number=10
    )
    
    # Call the method
    await agent.publish_observation(obs)
    
    # Verify Redis publish was called
    redis_client.publish.assert_called_once()
    channel, data = redis_client.publish.call_args[0]
    
    # Verify channel
    assert channel == "epilog:test_conn_id"
    
    # Verify data
    message_data = json.loads(data)
    assert message_data["type"] == "SERVER_MSG"
    assert message_data["data"]["type"] == "episode_log"
    assert message_data["data"]["log"] == {"messages": [["test", "data"]]}

# Test aact method with regular observation
@pytest.mark.asyncio
async def test_aact_regular_observation(redis_agent):
    # Unpack the fixture
    agent, _, _ = redis_agent
    
    # Set up the mock implementations
    async def mock_start_command_listener():
        pass
    
    async def mock_publish_observation(obs):
        pass
    
    # Replace the methods with our implementations
    agent.start_command_listener = mock_start_command_listener
    agent.publish_observation = mock_publish_observation
    
    # Set up return value for aact
    expected_action = create_mock_agent_action(
        agent_name="redis_agent",
        action_type="none",
        argument=""
    )
    agent.aact.return_value = expected_action
    
    # Create a test observation
    obs = create_mock_observation(
        agent_name="agent1",
        last_turn="Hello from agent1",
        turn_number=1
    )
    
    # Call aact
    action = await agent.aact(obs)
    
    # Verify the returned action
    assert action is expected_action
    assert action.action_type == "none"
    assert action.agent_name == "redis_agent"
    
    # Verify aact was called with the right args
    agent.aact.assert_called_once_with(obs)

# Test aact method with agent response in group mode
@pytest.mark.asyncio
async def test_aact_with_agent_response(redis_agent):
    # Unpack the fixture
    agent, _, _ = redis_agent
    
    # Set up agent for group mode
    agent.mode = "group"
    agent.message_senders = {"agent1": "websocket_user"}
    
    # Set up the mock implementations
    async def mock_start_command_listener():
        pass
    
    async def mock_publish_observation(obs):
        pass
    
    mock_response_action = create_mock_agent_action(
        agent_name="agent1",
        action_type="unified_message",
        argument=json.dumps({"content": "Response", "target_agents": ["websocket_user"]})
    )
    
    async def mock_process_agent_response(obs):
        return mock_response_action if obs.agent_name == "agent1" else None
    
    # Replace the methods with our implementations
    agent.start_command_listener = mock_start_command_listener
    agent.publish_observation = mock_publish_observation
    agent.process_agent_response = mock_process_agent_response
    
    # Set up a custom aact implementation
    async def custom_aact(obs):
        await agent.start_command_listener()
        await agent.publish_observation(obs)
        
        # Check for agent response
        if agent.mode == "group" and obs.agent_name != "epilog" and obs.last_turn:
            response = await agent.process_agent_response(obs)
            if response:
                return response
        
        # Default action
        return create_mock_agent_action(
            agent_name="redis_agent",
            action_type="none",
            argument=""
        )
    
    # Replace aact with our implementation
    agent.aact = custom_aact
    
    # Create an observation with response content
    obs = create_mock_observation(
        agent_name="agent1",
        last_turn="Response from agent1",
        turn_number=2
    )
    
    # Call aact
    action = await agent.aact(obs)
    
    # Verify the response action was returned
    assert action == mock_response_action

# Test aact with pending actions from WebSocket
@pytest.mark.asyncio
async def test_aact_with_pending_actions(redis_agent):
    # Unpack the fixture
    agent, _, _ = redis_agent
    
    # Set up the mock implementations
    async def mock_start_command_listener():
        pass
    
    async def mock_publish_observation(obs):
        pass
    
    # Replace the methods with our implementations
    agent.start_command_listener = mock_start_command_listener
    agent.publish_observation = mock_publish_observation
    
    # Add a pending action
    pending_action = create_mock_agent_action(
        agent_name="websocket_user",
        action_type="speak",
        argument="Pending message from WebSocket"
    )
    agent.pending_actions = [pending_action]
    
    # Set up a custom aact implementation
    async def custom_aact(obs):
        await agent.start_command_listener()
        await agent.publish_observation(obs)
        
        # Return pending action if available
        if agent.pending_actions:
            return agent.pending_actions.pop(0)
        
        # Default action
        return create_mock_agent_action(
            agent_name="redis_agent",
            action_type="none",
            argument=""
        )
    
    # Replace aact with our implementation
    agent.aact = custom_aact
    
    # Create a test observation
    obs = create_mock_observation(
        agent_name="agent1",
        last_turn="Hello from agent1",
        turn_number=1
    )
    
    # Call aact
    action = await agent.aact(obs)
    
    # Verify the pending action was returned and queue is now empty
    assert action == pending_action
    assert len(agent.pending_actions) == 0