import pytest
import logging
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("message_pipeline_test")

# --------------------------------
# Mock Classes
# --------------------------------


class MockWebSocket:
    """Mock WebSocket class for testing"""

    def __init__(self) -> None:
        self.sent_messages: List[Dict[str, Any]] = []
        self.response_queue: List[Dict[str, Any]] = []
        self.npc_responses: Dict[
            str, Dict[str, Any]
        ] = {}  # Store NPC responses for testing

    async def send_json(self, data: Dict[str, Any]) -> bool:
        """Record sent messages"""
        self.sent_messages.append(data)
        logger.info(f"Sent message: {data}")
        return True

    async def receive_json(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Return mock responses"""
        # If we have queued responses, return those first
        if self.response_queue:
            return self.response_queue.pop(0)

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
                    "mode": mode,
                },
            }

        elif msg_type == "CLIENT_MSG":
            # For CLIENT_MSG messages
            content = last_message["data"].get("content", "")
            target_npcs = last_message["data"].get("target_npcs", [])
            target_group = last_message["data"].get("target_group", None)

            # Determine which NPCs to include in response
            npcs_to_respond: List[str] = []

            # Handle targeting
            groups: Dict[str, List[str]] = {
                "group1": ["agent1", "agent2"],
                "group2": ["agent2", "agent3"],
            }
            all_npcs: List[str] = ["agent1", "agent2", "agent3"]

            if target_npcs:
                npcs_to_respond = target_npcs
            elif target_group and target_group in groups:
                npcs_to_respond = groups[target_group]
            else:
                npcs_to_respond = all_npcs

            # Store response data for testing
            self.npc_responses = {}
            for npc in npcs_to_respond:
                self.npc_responses[npc] = {
                    "content": f"Mock response from {npc} to: {content}",
                    "action_type": "speak",
                }

            # Return the first response immediately
            # In a real situation, redis_agent would send one at a time
            if npcs_to_respond:
                first_npc = npcs_to_respond[0]

                # Queue the rest of the responses for future receive_json calls
                for npc in npcs_to_respond[1:]:
                    self.response_queue.append(
                        {
                            "type": "SERVER_MSG",
                            "data": {
                                "type": "npc_response",
                                "npc_id": npc,
                                "content": f"Mock response from {npc} to: {content}",
                            },
                        }
                    )

                return {
                    "type": "SERVER_MSG",
                    "data": {
                        "type": "npc_response",
                        "npc_id": first_npc,
                        "content": f"Mock response from {first_npc} to: {content}",
                    },
                }
            else:
                return {
                    "type": "SERVER_MSG",
                    "data": {"type": "info", "message": "No NPCs to respond"},
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
                    "action_type": "speak",
                },
            }

        elif msg_type == "FINISH_SIM":
            # For FINISH_SIM messages
            return {"type": "END_SIM", "data": {}}

        # Default response
        return {"type": "SERVER_MSG", "data": {"message": "Mock response"}}

    async def close(self) -> bool:
        """Mock close method"""
        return True

    async def get_all_npc_responses(self) -> List[Dict[str, Any]]:
        """Helper to collect all queued NPC responses"""
        # First get any remaining responses from the queue
        responses: List[Dict[str, Any]] = []
        while self.response_queue:
            responses.append(await self.receive_json())

        return responses


class MockClientSession:
    """Mock ClientSession for testing"""

    def __init__(self) -> None:
        self.websocket = MockWebSocket()

    async def ws_connect(
        self, url: str, timeout: Optional[float] = None
    ) -> MockWebSocket:
        """Return mock websocket"""
        logger.info(f"Mock connection to {url}")
        return self.websocket

    async def close(self) -> bool:
        """Mock close method"""
        return True


# --------------------------------
# Test utilities
# --------------------------------


async def send_start_sim(ws: MockWebSocket, mode: str = "group") -> Dict[str, Any]:
    """Send a START_SIM message"""
    npcs: List[str] = ["agent1", "agent2", "agent3"]
    groups: Dict[str, List[str]] = {
        "group1": ["agent1", "agent2"],
        "group2": ["agent2", "agent3"],
    }

    LOCAL_MODEL = "custom/llama3.2:1b@http://localhost:8000/v1"
    start_message: Dict[str, Any] = {
        "type": "START_SIM",
        "data": {
            "env_id": "mock_env_id",
            "agent_ids": ["agent1", "agent2"],
            "agent_models": [LOCAL_MODEL, LOCAL_MODEL],
            "evaluator_model": LOCAL_MODEL,
            "mode": mode,
            "npcs": npcs,
            "groups": groups,
        },
    }
    await ws.send_json(start_message)
    return await ws.receive_json(timeout=5.0)


async def send_client_message(
    ws: MockWebSocket,
    content: str,
    target_npcs: Optional[List[str]] = None,
    target_group: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Send a CLIENT_MSG message and collect all NPC responses"""
    message: Dict[str, Any] = {"type": "CLIENT_MSG", "data": {"content": content}}

    if target_npcs:
        message["data"]["target_npcs"] = target_npcs
    if target_group:
        message["data"]["target_group"] = target_group

    await ws.send_json(message)

    # Get the first response
    first_response = await ws.receive_json(timeout=5.0)

    # If we have multiple NPC responses, collect them
    additional_responses = await ws.get_all_npc_responses()

    return first_response, additional_responses


async def send_turn_request(
    ws: MockWebSocket, agent_id: str, content: str
) -> Dict[str, Any]:
    """Send a TURN_REQUEST message"""
    message: Dict[str, Any] = {
        "type": "TURN_REQUEST",
        "data": {"agent_id": agent_id, "content": content},
    }
    await ws.send_json(message)
    return await ws.receive_json(timeout=5.0)


async def end_simulation(ws: MockWebSocket) -> Dict[str, Any]:
    """Send a FINISH_SIM message"""
    await ws.send_json({"type": "FINISH_SIM", "data": {}})
    return await ws.receive_json(timeout=5.0)


# --------------------------------
# Tests
# --------------------------------


@pytest.fixture
def mock_websocket() -> MockWebSocket:
    """Fixture for mock websocket"""
    return MockWebSocket()


@pytest.mark.asyncio
async def test_group_messaging(mock_websocket: MockWebSocket) -> None:
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
    first_response, additional_responses = await send_client_message(
        mock_websocket, message, target_group="group1"
    )

    # Verify the first NPC response
    assert first_response["type"] == "SERVER_MSG", "Wrong response type"
    assert first_response["data"]["type"] == "npc_response", "Not an NPC response"
    assert "npc_id" in first_response["data"], "No NPC ID in response"

    # Check first response's NPC ID belongs to group1
    first_npc_id = first_response["data"]["npc_id"]
    assert first_npc_id in ["agent1", "agent2"], f"NPC {first_npc_id} not in group1"

    # Check additional responses if any
    for resp in additional_responses:
        assert (
            resp["type"] == "SERVER_MSG"
        ), "Wrong response type in additional response"
        assert resp["data"]["type"] == "npc_response", "Not an NPC response"
        npc_id = resp["data"]["npc_id"]
        assert npc_id in ["agent1", "agent2"], f"NPC {npc_id} not in group1"

    # Verify we got responses from all expected NPCs
    all_npc_ids: List[str] = [first_response["data"]["npc_id"]] + [
        r["data"]["npc_id"] for r in additional_responses
    ]
    assert "agent1" in all_npc_ids, "Missing response from agent1"
    assert "agent2" in all_npc_ids, "Missing response from agent2"
    assert "agent3" not in all_npc_ids, "Unexpected response from agent3"

    # Clean up
    response = await end_simulation(mock_websocket)
    assert response["type"] == "END_SIM", "Failed to end simulation"


@pytest.mark.asyncio
async def test_targeted_messaging(mock_websocket: MockWebSocket) -> None:
    """Test messaging to specific NPCs"""
    # Initialize the simulation
    await send_start_sim(mock_websocket, mode="group")

    # Send a message to specific NPCs
    message = "Hello to specific NPCs"
    first_response, additional_responses = await send_client_message(
        mock_websocket, message, target_npcs=["agent1", "agent3"]
    )

    # Verify the first response
    assert first_response["type"] == "SERVER_MSG", "Wrong response type"
    assert first_response["data"]["type"] == "npc_response", "Not an NPC response"
    first_npc_id = first_response["data"]["npc_id"]
    assert first_npc_id in [
        "agent1",
        "agent3",
    ], f"NPC {first_npc_id} not in target list"

    # Check additional responses if any
    for resp in additional_responses:
        assert (
            resp["type"] == "SERVER_MSG"
        ), "Wrong response type in additional response"
        assert resp["data"]["type"] == "npc_response", "Not an NPC response"
        npc_id = resp["data"]["npc_id"]
        assert npc_id in ["agent1", "agent3"], f"NPC {npc_id} not in target list"

    # Verify we got responses from all expected NPCs
    all_npc_ids: List[str] = [first_response["data"]["npc_id"]] + [
        r["data"]["npc_id"] for r in additional_responses
    ]
    assert "agent1" in all_npc_ids, "Missing response from agent1"
    assert "agent3" in all_npc_ids, "Missing response from agent3"
    assert "agent2" not in all_npc_ids, "Unexpected response from agent2"

    # Clean up
    await end_simulation(mock_websocket)


@pytest.mark.asyncio
async def test_turn_based_messaging(mock_websocket: MockWebSocket) -> None:
    """Test turn-based messaging pipeline"""
    # Initialize the simulation
    await send_start_sim(mock_websocket, mode="turn")

    # Send turn requests
    agents: List[str] = ["John Doe", "Jane Doe"]
    for agent in agents:
        response = await send_turn_request(mock_websocket, agent, f"Hello from {agent}")

        # Verify the turn response
        assert response["type"] == "TURN_RESPONSE", "Wrong response type"
        assert "agent_id" in response["data"], "No agent_id in response"
        assert "agent_response" in response["data"], "No agent_response in response"
        assert (
            response["data"]["agent_id"] == agent
        ), f"Wrong agent_id in response, expected {agent}"

    # Clean up
    await end_simulation(mock_websocket)


@pytest.mark.asyncio
async def test_group_membership_overlap(mock_websocket: MockWebSocket) -> None:
    """Test that NPCs can belong to multiple groups"""
    # Initialize the simulation
    await send_start_sim(mock_websocket, mode="group")

    # Send messages to different groups with overlapping membership
    groups: Dict[str, List[str]] = {
        "group1": ["agent1", "agent2"],
        "group2": ["agent2", "agent3"],
    }

    for group_name, expected_members in groups.items():
        first_response, additional_responses = await send_client_message(
            mock_websocket, f"Hello {group_name}", target_group=group_name
        )

        # Verify the first response
        assert first_response["type"] == "SERVER_MSG", "Wrong response type"
        assert first_response["data"]["type"] == "npc_response", "Not an NPC response"
        first_npc_id = first_response["data"]["npc_id"]
        assert (
            first_npc_id in expected_members
        ), f"NPC {first_npc_id} not in {group_name}"

        # Check additional responses
        for resp in additional_responses:
            assert (
                resp["type"] == "SERVER_MSG"
            ), "Wrong response type in additional response"
            assert resp["data"]["type"] == "npc_response", "Not an NPC response"
            npc_id = resp["data"]["npc_id"]
            assert npc_id in expected_members, f"NPC {npc_id} not in {group_name}"

        # Verify we got responses from all expected NPCs
        all_npc_ids: List[str] = [first_response["data"]["npc_id"]] + [
            r["data"]["npc_id"] for r in additional_responses
        ]
        for member in expected_members:
            assert (
                member in all_npc_ids
            ), f"Missing response from {member} in {group_name}"

    # Clean up
    await end_simulation(mock_websocket)


@pytest.mark.asyncio
async def test_multi_turn_conversation(mock_websocket: MockWebSocket) -> None:
    """Test a realistic multi-turn conversation between client and NPCs"""
    # Initialize the simulation
    response = await send_start_sim(mock_websocket, mode="group")
    logger.info(f"Start simulation response: {response}")

    # Verify the initialization
    assert response["type"] == "SERVER_MSG", "Wrong response type"
    assert response["data"]["type"] == "initialization", "Not an initialization message"

    # --------- TURN 1: Client greets the group ---------
    greeting = "Hello everyone in group1! How are you all doing today?"
    first_response1, additional_responses1 = await send_client_message(
        mock_websocket, greeting, target_group="group1"
    )

    # Log the conversation
    logger.info(f"Turn 1 - Client: {greeting}")
    logger.info(
        f"Turn 1 - {first_response1['data']['npc_id']}: {first_response1['data']['content']}"
    )
    for resp in additional_responses1:
        logger.info(f"Turn 1 - {resp['data']['npc_id']}: {resp['data']['content']}")

    # Verify all group1 members responded
    all_npc_ids1: List[str] = [first_response1["data"]["npc_id"]] + [
        r["data"]["npc_id"] for r in additional_responses1
    ]
    assert "agent1" in all_npc_ids1, "Missing response from agent1"
    assert "agent2" in all_npc_ids1, "Missing response from agent2"
    assert "agent3" not in all_npc_ids1, "Unexpected response from agent3"

    # --------- TURN 2: Client asks a follow-up question ---------
    follow_up = "What kind of tasks can you help me with as a group?"
    first_response2, additional_responses2 = await send_client_message(
        mock_websocket, follow_up, target_group="group1"
    )

    # Log the conversation
    logger.info(f"Turn 2 - Client: {follow_up}")
    logger.info(
        f"Turn 2 - {first_response2['data']['npc_id']}: {first_response2['data']['content']}"
    )
    for resp in additional_responses2:
        logger.info(f"Turn 2 - {resp['data']['npc_id']}: {resp['data']['content']}")

    # Verify all group1 members responded
    all_npc_ids2: List[str] = [first_response2["data"]["npc_id"]] + [
        r["data"]["npc_id"] for r in additional_responses2
    ]
    assert len(all_npc_ids2) == 2, "Expected 2 responses"

    # --------- TURN 3: Client addresses one specific NPC ---------
    # Pick the first NPC that responded
    specific_npc = first_response2["data"]["npc_id"]
    direct_question = f"Hey {specific_npc}, can you tell me more about yourself?"

    first_response3, additional_responses3 = await send_client_message(
        mock_websocket, direct_question, target_npcs=[specific_npc]
    )

    # Log the conversation
    logger.info(f"Turn 3 - Client: {direct_question}")
    logger.info(
        f"Turn 3 - {first_response3['data']['npc_id']}: {first_response3['data']['content']}"
    )

    # Verify only the specific NPC responded
    assert (
        first_response3["data"]["npc_id"] == specific_npc
    ), f"Wrong NPC responded: {first_response3['data']['npc_id']}"
    assert not additional_responses3, "Unexpected additional responses"

    # --------- TURN 4: Client addresses a different group ---------
    group_switch = "Now I'd like to talk to group2. Hello group2!"
    first_response4, additional_responses4 = await send_client_message(
        mock_websocket, group_switch, target_group="group2"
    )

    # Log the conversation
    logger.info(f"Turn 4 - Client: {group_switch}")
    logger.info(
        f"Turn 4 - {first_response4['data']['npc_id']}: {first_response4['data']['content']}"
    )
    for resp in additional_responses4:
        logger.info(f"Turn 4 - {resp['data']['npc_id']}: {resp['data']['content']}")

    # Verify all group2 members responded
    all_npc_ids4: List[str] = [first_response4["data"]["npc_id"]] + [
        r["data"]["npc_id"] for r in additional_responses4
    ]
    assert "agent2" in all_npc_ids4, "Missing response from agent2"
    assert "agent3" in all_npc_ids4, "Missing response from agent3"
    assert "agent1" not in all_npc_ids4, "Unexpected response from agent1"

    # --------- TURN 5: Client addresses the intersection of groups ---------
    # agent2 is in both groups, so target them specifically
    intersection_msg = (
        "Let's talk to the agent who's in both groups. That would be agent2."
    )
    first_response5, additional_responses5 = await send_client_message(
        mock_websocket, intersection_msg, target_npcs=["agent2"]
    )

    # Log the conversation
    logger.info(f"Turn 5 - Client: {intersection_msg}")
    logger.info(
        f"Turn 5 - {first_response5['data']['npc_id']}: {first_response5['data']['content']}"
    )

    # Verify only agent2 responded
    assert first_response5["data"]["npc_id"] == "agent2", "Wrong NPC responded"
    assert not additional_responses5, "Unexpected additional responses"

    # --------- TURN 6: Client wraps up the conversation ---------
    final_msg = "Thanks everyone for chatting with me today!"
    first_response6, additional_responses6 = await send_client_message(
        mock_websocket, final_msg, target_npcs=["agent1", "agent2", "agent3"]
    )

    # Log the conversation
    logger.info(f"Turn 6 - Client: {final_msg}")
    logger.info(
        f"Turn 6 - {first_response6['data']['npc_id']}: {first_response6['data']['content']}"
    )
    for resp in additional_responses6:
        logger.info(f"Turn 6 - {resp['data']['npc_id']}: {resp['data']['content']}")

    # Verify all NPCs responded
    all_npc_ids6: List[str] = [first_response6["data"]["npc_id"]] + [
        r["data"]["npc_id"] for r in additional_responses6
    ]
    assert "agent1" in all_npc_ids6, "Missing response from agent1"
    assert "agent2" in all_npc_ids6, "Missing response from agent2"
    assert "agent3" in all_npc_ids6, "Missing response from agent3"

    # Clean up
    await end_simulation(mock_websocket)
    logger.info("Multi-turn conversation test completed successfully!")
