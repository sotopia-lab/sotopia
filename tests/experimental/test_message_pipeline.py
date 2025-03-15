import pytest
import asyncio
import json
import logging
from unittest.mock import MagicMock, patch, AsyncMock

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("message_pipeline_test")

# --------------------------------
# Mock Classes
# --------------------------------

class MockWebSocket:
    """Mock WebSocket class for testing"""
    def __init__(self):
        self.sent_messages = []
        self.response_queue = []
        
    async def send_json(self, data):
        """Record sent messages"""
        self.sent_messages.append(data)
        logger.info(f"Sent message: {data}")
        return True
        
    async def receive_json(self, timeout=None):
        """Return mock responses"""
        # Generate response based on the last sent message
        if not self.sent_messages:
            return {"type": "ERROR", "data": {"message": "No messages sent"}}
            
        last_message = self.sent_messages[-1]
        msg_type = last_message.get("type", "")
        
        if msg_type == "START_SIM":
            # For START_SIM messages
            mode = last_message["data"].get("mode", "full")
            npcs = last_message["data"].get("npcs", [])
            groups = last_message["data"].get("groups", {})
            
            return {
                "type": "SERVER_MSG",
                "data": {
                    "type": "initialization",
                    "status": "ready",
                    "npcs": npcs,
                    "groups": groups,
                    "mode": mode
                }
            }
            
        elif msg_type == "CLIENT_MSG":
            # For CLIENT_MSG messages
            content = last_message["data"].get("content", "")
            target_npcs = last_message["data"].get("target_npcs", [])
            target_group = last_message["data"].get("target_group", None)
            
            # Determine which NPCs to include in response
            npcs_to_respond = []
            
            # Handle targeting
            groups = {'group1': ['agent1', 'agent2'], 'group2': ['agent2', 'agent3']}
            all_npcs = ['agent1', 'agent2', 'agent3']
            
            if target_npcs:
                npcs_to_respond = target_npcs
            elif target_group and target_group in groups:
                npcs_to_respond = groups[target_group]
            else:
                npcs_to_respond = all_npcs
            
            # Generate responses
            responses = {}
            for npc in npcs_to_respond:
                responses[npc] = {
                    "content": f"Mock response from {npc} to: {content}",
                    "action_type": "speak"
                }
            
            return {
                "type": "SERVER_MSG",
                "data": {
                    "type": "npc_responses",
                    "status": "success",
                    "turn": len(self.sent_messages),
                    "responses": responses
                }
            }
            
        elif msg_type == "TURN_REQUEST":
            # For TURN_REQUEST messages
            agent_id = last_message["data"].get("agent_id", "")
            content = last_message["data"].get("content", "")
            
            return {
                "type": "TURN_RESPONSE",
                "data": {
                    "turn": len(self.sent_messages),
                    "agent_id": agent_id,
                    "agent_response": f"Mock response from {agent_id} to: {content}",
                    "action_type": "speak"
                }
            }
            
        elif msg_type == "FINISH_SIM":
            # For FINISH_SIM messages
            return {
                "type": "END_SIM",
                "data": {}
            }
            
        # Default response
        return {
            "type": "SERVER_MSG",
            "data": {"message": "Mock response"}
        }
        
    async def close(self):
        """Mock close method"""
        return True

class MockClientSession:
    """Mock ClientSession for testing"""
    def __init__(self):
        self.websocket = MockWebSocket()
        
    async def ws_connect(self, url, timeout=None):
        """Return mock websocket"""
        logger.info(f"Mock connection to {url}")
        return self.websocket
        
    async def close(self):
        """Mock close method"""
        return True

# --------------------------------
# Test utilities
# --------------------------------

async def send_start_sim(ws, mode="group"):
    """Send a START_SIM message"""
    npcs = ["agent1", "agent2", "agent3"]
    groups = {"group1": ["agent1", "agent2"], "group2": ["agent2", "agent3"]}
    
    LOCAL_MODEL = "custom/llama3.2:1b@http://localhost:8000/v1"
    start_message = {
        "type": "START_SIM",
        "data": {
            "env_id": "mock_env_id",
            "agent_ids": ["agent1", "agent2"],
            "agent_models": [LOCAL_MODEL, LOCAL_MODEL],
            "evaluator_model": LOCAL_MODEL,
            "mode": mode,
            "npcs": npcs,
            "groups": groups
        }
    }
    await ws.send_json(start_message)
    return await ws.receive_json(timeout=5.0)

async def send_client_message(ws, content, target_npcs=None, target_group=None):
    """Send a CLIENT_MSG message"""
    message = {
        "type": "CLIENT_MSG",
        "data": {
            "content": content
        }
    }
    
    if target_npcs:
        message["data"]["target_npcs"] = target_npcs
    if target_group:
        message["data"]["target_group"] = target_group
        
    await ws.send_json(message)
    return await ws.receive_json(timeout=5.0)

async def send_turn_request(ws, agent_id, content):
    """Send a TURN_REQUEST message"""
    message = {
        "type": "TURN_REQUEST",
        "data": {
            "agent_id": agent_id,
            "content": content
        }
    }
    await ws.send_json(message)
    return await ws.receive_json(timeout=5.0)

async def end_simulation(ws):
    """Send a FINISH_SIM message"""
    await ws.send_json({"type": "FINISH_SIM", "data": {}})
    return await ws.receive_json(timeout=5.0)

# --------------------------------
# Tests
# --------------------------------

@pytest.fixture
def mock_websocket():
    """Fixture for mock websocket"""
    return MockWebSocket()

@pytest.mark.asyncio
async def test_group_messaging(mock_websocket):
    """Test the group-based messaging pipeline"""
    # Initialize the simulation
    response = await send_start_sim(mock_websocket, mode="group")
    logger.info(f"Start simulation response: {response}")
    
    # Verify the initialization response
    assert response["type"] == "SERVER_MSG", "Wrong response type"
    assert response["data"]["type"] == "initialization", "Not an initialization message"
    assert "npcs" in response["data"], "No NPCs in response"
    assert "groups" in response["data"], "No groups in response"
    
    # Send a message to a group
    message = "Hello to group1"
    response = await send_client_message(mock_websocket, message, target_group="group1")
    logger.info(f"Group message response: {response}")
    
    # Verify the message response
    assert response["type"] == "SERVER_MSG", "Wrong response type"
    assert "responses" in response["data"], "No responses in data"
    
    # Check that responses came from the right group members
    responses = response["data"]["responses"]
    assert "agent1" in responses, "Missing response from agent1"
    assert "agent2" in responses, "Missing response from agent2"
    assert "agent3" not in responses, "Unexpected response from agent3"
    
    # Clean up
    response = await end_simulation(mock_websocket)
    assert response["type"] == "END_SIM", "Failed to end simulation"

@pytest.mark.asyncio
async def test_targeted_messaging(mock_websocket):
    """Test messaging to specific NPCs"""
    # Initialize the simulation
    await send_start_sim(mock_websocket, mode="group")
    
    # Send a message to specific NPCs
    message = "Hello to specific NPCs"
    response = await send_client_message(
        mock_websocket, 
        message, 
        target_npcs=["agent1", "agent3"]
    )
    
    # Verify the message response
    assert response["type"] == "SERVER_MSG", "Wrong response type"
    assert "responses" in response["data"], "No responses in data"
    
    # Check that responses came from the right NPCs
    responses = response["data"]["responses"]
    assert "agent1" in responses, "Missing response from agent1"
    assert "agent3" in responses, "Missing response from agent3"
    assert "agent2" not in responses, "Unexpected response from agent2"
    
    # Clean up
    await end_simulation(mock_websocket)

@pytest.mark.asyncio
async def test_turn_based_messaging(mock_websocket):
    """Test turn-based messaging pipeline"""
    # Initialize the simulation
    await send_start_sim(mock_websocket, mode="turn")
    
    # Send turn requests
    agents = ["John Doe", "Jane Doe"]
    for agent in agents:
        response = await send_turn_request(
            mock_websocket,
            agent,
            f"Hello from {agent}"
        )
        
        # Verify the turn response
        assert response["type"] == "TURN_RESPONSE", "Wrong response type"
        assert "agent_id" in response["data"], "No agent_id in response"
        assert "agent_response" in response["data"], "No agent_response in response"
        assert response["data"]["agent_id"] == agent, f"Wrong agent_id in response, expected {agent}"
    
    # Clean up
    await end_simulation(mock_websocket)

@pytest.mark.asyncio
async def test_group_membership_overlap(mock_websocket):
    """Test that NPCs can belong to multiple groups"""
    # Initialize the simulation
    await send_start_sim(mock_websocket, mode="group")
    
    # Send messages to different groups with overlapping membership
    groups = {"group1": ["agent1", "agent2"], "group2": ["agent2", "agent3"]}
    
    for group_name, expected_members in groups.items():
        response = await send_client_message(
            mock_websocket,
            f"Hello {group_name}",
            target_group=group_name
        )
        
        # Verify the message response
        assert response["type"] == "SERVER_MSG", "Wrong response type"
        assert "responses" in response["data"], "No responses in data"
        
        # Check that responses came from the right group members
        responses = response["data"]["responses"]
        for member in expected_members:
            assert member in responses, f"Missing response from {member} in {group_name}"
    
    # Clean up
    await end_simulation(mock_websocket)

@pytest.mark.asyncio
async def test_multi_turn_conversation(mock_websocket):
    """Test a realistic multi-turn conversation between client and NPCs"""
    # Initialize the simulation
    response = await send_start_sim(mock_websocket, mode="group")
    logger.info(f"Start simulation response: {response}")
    
    # Verify the initialization
    assert response["type"] == "SERVER_MSG", "Wrong response type"
    assert response["data"]["type"] == "initialization", "Not an initialization message"
    
    # --------- TURN 1: Client greets the group ---------
    greeting = "Hello everyone in group1! How are you all doing today?"
    response1 = await send_client_message(mock_websocket, greeting, target_group="group1")
    
    # Verify responses
    assert response1["type"] == "SERVER_MSG", "Wrong response type"
    assert "responses" in response1["data"], "No responses in data"
    
    # Check group responses
    responses1 = response1["data"]["responses"]
    assert "agent1" in responses1, "Missing response from agent1"
    assert "agent2" in responses1, "Missing response from agent2"
    assert "agent3" not in responses1, "Unexpected response from agent3"
    
    logger.info(f"Turn 1 - Client: {greeting}")
    for npc, resp in responses1.items():
        logger.info(f"Turn 1 - {npc}: {resp['content']}")
    
    # --------- TURN 2: Client asks a follow-up question ---------
    follow_up = "What kind of tasks can you help me with as a group?"
    response2 = await send_client_message(mock_websocket, follow_up, target_group="group1")
    
    # Verify responses
    responses2 = response2["data"]["responses"]
    assert len(responses2) == 2, "Expected 2 responses"
    
    logger.info(f"Turn 2 - Client: {follow_up}")
    for npc, resp in responses2.items():
        logger.info(f"Turn 2 - {npc}: {resp['content']}")
    
    # --------- TURN 3: Client addresses one specific NPC ---------
    # Pick the first NPC that responded
    specific_npc = list(responses2.keys())[0]
    direct_question = f"Hey {specific_npc}, can you tell me more about yourself?"
    
    response3 = await send_client_message(
        mock_websocket, 
        direct_question, 
        target_npcs=[specific_npc]
    )
    
    # Verify responses
    responses3 = response3["data"]["responses"]
    assert len(responses3) == 1, "Expected only 1 response"
    assert specific_npc in responses3, f"Missing response from {specific_npc}"
    
    logger.info(f"Turn 3 - Client: {direct_question}")
    logger.info(f"Turn 3 - {specific_npc}: {responses3[specific_npc]['content']}")
    
    # --------- TURN 4: Client addresses a different group ---------
    group_switch = "Now I'd like to talk to group2. Hello group2!"
    response4 = await send_client_message(mock_websocket, group_switch, target_group="group2")
    
    # Verify responses
    responses4 = response4["data"]["responses"]
    assert "agent2" in responses4, "Missing response from agent2"
    assert "agent3" in responses4, "Missing response from agent3"
    assert "agent1" not in responses4, "Unexpected response from agent1"
    
    logger.info(f"Turn 4 - Client: {group_switch}")
    for npc, resp in responses4.items():
        logger.info(f"Turn 4 - {npc}: {resp['content']}")
    
    # --------- TURN 5: Client addresses the intersection of groups ---------
    # agent2 is in both groups, so target them specifically
    intersection_msg = "Let's talk to the agent who's in both groups. That would be agent2."
    response5 = await send_client_message(mock_websocket, intersection_msg, target_npcs=["agent2"])
    
    # Verify responses
    responses5 = response5["data"]["responses"]
    assert len(responses5) == 1, "Expected only 1 response"
    assert "agent2" in responses5, "Missing response from agent2"
    
    logger.info(f"Turn 5 - Client: {intersection_msg}")
    logger.info(f"Turn 5 - agent2: {responses5['agent2']['content']}")
    
    # --------- TURN 6: Client wraps up the conversation ---------
    final_msg = "Thanks everyone for chatting with me today!"
    response6 = await send_client_message(mock_websocket, final_msg, target_npcs=["agent1", "agent2", "agent3"])
    
    # Verify responses from all NPCs
    responses6 = response6["data"]["responses"]
    assert len(responses6) == 3, "Expected 3 responses"
    assert "agent1" in responses6, "Missing response from agent1"
    assert "agent2" in responses6, "Missing response from agent2"
    assert "agent3" in responses6, "Missing response from agent3"
    
    logger.info(f"Turn 6 - Client: {final_msg}")
    for npc, resp in responses6.items():
        logger.info(f"Turn 6 - {npc}: {resp['content']}")
    
    # Clean up
    await end_simulation(mock_websocket)
    logger.info("Multi-turn conversation test completed successfully!")