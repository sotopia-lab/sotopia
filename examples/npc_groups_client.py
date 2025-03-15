import asyncio
import aiohttp
import json
import logging
import argparse
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# WebSocket message types
class WSMessageType:
    SERVER_MSG = "SERVER_MSG"
    CLIENT_MSG = "CLIENT_MSG"
    ERROR = "ERROR"
    START_SIM = "START_SIM"
    END_SIM = "END_SIM"
    FINISH_SIM = "FINISH_SIM"


async def send_message_to_npcs(
    websocket, content: str, target_npcs: List[str] = None, target_group: str = None
):
    """Send a message to specific NPCs or a group"""
    message = {"type": WSMessageType.CLIENT_MSG, "data": {"content": content}}

    # Add targeting information if provided
    if target_npcs:
        message["data"]["target_npcs"] = target_npcs

    if target_group:
        message["data"]["target_group"] = target_group

    await websocket.send_json(message)
    logger.info(
        f"Sent message: {content} to {target_npcs or target_group or 'all NPCs'}"
    )


async def finish_simulation(websocket):
    """Send a message to finish the simulation"""
    await websocket.send_json({"type": WSMessageType.FINISH_SIM, "data": {}})
    logger.info("Sent finish simulation message")


async def start_npc_group_simulation(
    websocket, npcs: List[str], groups: Dict[str, List[str]]
):
    """Start a simulation with defined NPCs and groups"""
    start_message = {
        "type": WSMessageType.START_SIM,
        "data": {
            "env_id": "environment_12345",  # Can be any unique ID
            "npcs": npcs,
            "groups": groups,
            "mode": "group",
        },
    }
    await websocket.send_json(start_message)
    logger.info(f"Started simulation with {len(npcs)} NPCs and {len(groups)} groups")


async def handle_server_messages(websocket):
    """Handle messages from the server"""
    async for msg in websocket:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
                msg_type = data.get("type")

                if msg_type == WSMessageType.SERVER_MSG:
                    content_type = data.get("data", {}).get("type")

                    # Handle initialization
                    if content_type == "initialization":
                        init_data = data.get("data", {}).get("data", {})
                        logger.info(
                            f"Simulation initialized with {len(init_data.get('npcs', []))} NPCs"
                        )

                    # Handle NPC responses
                    elif content_type == "npc_responses":
                        responses = (
                            data.get("data", {}).get("data", {}).get("responses", {})
                        )
                        logger.info(f"Received responses from {len(responses)} NPCs:")
                        for npc_id, response in responses.items():
                            logger.info(
                                f"  {npc_id}: {response.get('content')} ({response.get('action_type')})"
                            )

                    # Handle other message types
                    else:
                        logger.info(f"Received server message: {data}")

                elif msg_type == WSMessageType.ERROR:
                    error_type = data.get("data", {}).get("type")
                    error_details = data.get("data", {}).get("details")
                    logger.error(f"Error from server: {error_type} - {error_details}")

                elif msg_type == WSMessageType.END_SIM:
                    logger.info("Simulation ended by server")
                    return

            except json.JSONDecodeError:
                logger.error(f"Failed to parse message: {msg.data}")

        elif msg.type == aiohttp.WSMsgType.CLOSED:
            logger.info("WebSocket connection closed")
            break

        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger.error(f"WebSocket error: {msg.data}")
            break


async def interactive_mode(websocket):
    """Run an interactive session with the NPCs"""
    # Sample NPCs and groups
    npcs = ["agent1", "agent2", "agent3", "agent4"]
    groups = {
        "group1": ["agent1", "agent2"],
        "group2": ["agent3", "agent4"],
        "all": ["agent1", "agent2", "agent3", "agent4"],
    }

    # Start the simulation
    await start_npc_group_simulation(websocket, npcs, groups)

    # Start a task to handle server messages
    server_task = asyncio.create_task(handle_server_messages(websocket))

    try:
        while True:
            print("\nOptions:")
            print("1. Send a message to a specific NPC")
            print("2. Send a message to a group")
            print("3. Send a message to all NPCs")
            print("4. Exit")

            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
                npc_id = input(f"Enter NPC ID ({', '.join(npcs)}): ").strip()
                if npc_id not in npcs:
                    print(f"Invalid NPC ID. Must be one of: {', '.join(npcs)}")
                    continue

                content = input("Enter your message: ").strip()
                await send_message_to_npcs(websocket, content, target_npcs=[npc_id])

            elif choice == "2":
                group_id = input(
                    f"Enter group ID ({', '.join(groups.keys())}): "
                ).strip()
                if group_id not in groups:
                    print(
                        f"Invalid group ID. Must be one of: {', '.join(groups.keys())}"
                    )
                    continue

                content = input("Enter your message: ").strip()
                await send_message_to_npcs(websocket, content, target_group=group_id)

            elif choice == "3":
                content = input("Enter your message: ").strip()
                await send_message_to_npcs(websocket, content)

            elif choice == "4":
                print("Exiting simulation...")
                await finish_simulation(websocket)
                break

            else:
                print("Invalid choice. Please enter a number between 1 and 4.")

    except asyncio.CancelledError:
        logger.info("Interactive session cancelled")
    finally:
        # Make sure to cancel the server task
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


async def automated_demo(websocket):
    """Run an automated demonstration of the NPC group messaging"""
    # Sample NPCs and groups
    npcs = ["agent1", "agent2", "agent3", "agent4"]
    groups = {
        "group1": ["agent1", "agent2"],
        "group2": ["agent3", "agent4"],
        "all": ["agent1", "agent2", "agent3", "agent4"],
    }

    # Start the simulation
    await start_npc_group_simulation(websocket, npcs, groups)

    # Start a task to handle server messages
    server_task = asyncio.create_task(handle_server_messages(websocket))

    try:
        # Wait for initialization
        await asyncio.sleep(2)

        # Send a message to all NPCs
        logger.info("Sending message to all NPCs")
        await send_message_to_npcs(
            websocket, "Hello everyone! Can you all introduce yourselves?"
        )
        await asyncio.sleep(5)  # Wait for responses

        # Send a message to a specific group
        logger.info("Sending message to group1")
        await send_message_to_npcs(
            websocket, "Group 1, what's your specialty?", target_group="group1"
        )
        await asyncio.sleep(5)  # Wait for responses

        # Send a message to a specific NPC
        logger.info("Sending message to agent3")
        await send_message_to_npcs(
            websocket,
            "Agent3, can you tell me more about yourself?",
            target_npcs=["agent3"],
        )
        await asyncio.sleep(5)  # Wait for responses

        # Send a message to another group
        logger.info("Sending message to group2")
        await send_message_to_npcs(
            websocket,
            "Group 2, what projects are you working on?",
            target_group="group2",
        )
        await asyncio.sleep(5)  # Wait for responses

        # Finish the simulation
        await finish_simulation(websocket)

    except asyncio.CancelledError:
        logger.info("Demo cancelled")
    finally:
        # Make sure to cancel the server task
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


async def main():
    parser = argparse.ArgumentParser(description="NPC Groups Client Example")
    parser.add_argument(
        "--server",
        default="localhost:8800",
        help="Server address (default: localhost:8800)",
    )
    parser.add_argument(
        "--token",
        default="demo_token",
        help="Authentication token (default: demo_token)",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo"],
        default="interactive",
        help="Run mode: interactive or demo (default: interactive)",
    )

    args = parser.parse_args()

    # Construct WebSocket URL
    ws_url = f"ws://{args.server}/ws/npc_groups?token={args.token}"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(ws_url) as websocket:
                logger.info(f"Connected to {ws_url}")

                if args.mode == "interactive":
                    await interactive_mode(websocket)
                else:
                    await automated_demo(websocket)

        except aiohttp.ClientError as e:
            logger.error(f"Connection error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
