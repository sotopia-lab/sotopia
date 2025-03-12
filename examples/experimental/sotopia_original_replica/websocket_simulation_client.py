import asyncio
import json
import logging
from typing import List, Optional, Any
from rich.console import Console
from rich.logging import RichHandler
import rich
from rich import print
import aiohttp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(console=Console())],
)
logger = logging.getLogger("sotopia-websocket-client")

# API endpoint configuration
BASE_URL = "http://localhost:8080"  # Updated to match your server port
WS_URL = (
    "ws://localhost:8080/ws/simulation?token=demo-token"  # Updated WebSocket endpoint
)


class WSMessageType:
    SERVER_MSG = "SERVER_MSG"
    CLIENT_MSG = "CLIENT_MSG"
    ERROR = "ERROR"
    START_SIM = "START_SIM"
    END_SIM = "END_SIM"
    FINISH_SIM = "FINISH_SIM"


async def check_api_connection() -> bool:
    """
    Ping the API to check if the connection is available.
    """
    try:
        async with aiohttp.ClientSession() as session:
            # Use the health check endpoint
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"API health check: {health_data['status']}")
                    if health_data["status"] == "ok":
                        return True
                    else:
                        logger.warning(
                            f"API health check returned degraded status: {health_data}"
                        )
                        return False
                else:
                    logger.warning(
                        f"API health check failed with status code: {response.status}"
                    )
                    return False
    except Exception as e:
        logger.error(f"Failed to connect to API: {e}")
        return False


async def run_simulation(
    env_id: str,
    agent_ids: List[str],
    agent_models: Optional[List[str]] = None,
    env_profile_dict: Optional[dict[str, Any]] = None,
    agent_profile_dicts: Optional[List[dict[str, Any]]] = None,
    evaluator_model: Optional[str] = None,
    evaluation_dimension_list_name: Optional[str] = None,
) -> None:
    """
    Connect to the WebSocket endpoint and run a simulation.

    Args:
        env_id: ID of the environment to use
        agent_ids: List of agent IDs to participate in the simulation
        agent_models: List of models to use for each agent
        evaluator_model: Model to use for evaluation
        evaluation_dimension_list_name: Name of evaluation dimension list
    """
    # Check API connection
    if not await check_api_connection():
        logger.error("Cannot proceed with simulation: API connection failed")
        return

    # Prepare simulation start message
    start_message = {
        "type": WSMessageType.START_SIM,
        "data": {
            "env_id": env_id,
            "agent_ids": agent_ids,
        },
    }

    # Add optional parameters if provided
    if agent_models:
        start_message["data"]["agent_models"] = agent_models
    if evaluator_model:
        start_message["data"]["evaluator_model"] = evaluator_model
    if evaluation_dimension_list_name:
        start_message["data"]["evaluation_dimension_list_name"] = (
            evaluation_dimension_list_name
        )
    if env_profile_dict:
        start_message["data"]["env_profile_dict"] = env_profile_dict
    if agent_profile_dicts:
        start_message["data"]["agent_profile_dicts"] = agent_profile_dicts

    # Connect to WebSocket
    session = aiohttp.ClientSession()
    try:
        async with session.ws_connect(WS_URL) as ws:
            logger.info("Connected to WebSocket")
            # Send simulation start message
            await ws.send_json(start_message)
            logger.info(f"Sent simulation start message: {start_message}")

            # Listen for messages
            while True:
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")

                    if msg_type == WSMessageType.SERVER_MSG:
                        server_data = data.get("data", {})

                        # Rich print the last messages if available
                        if (
                            server_data.get("type") == "messages"
                            and "messages" in server_data
                        ):
                            messages_data = server_data["messages"]
                            if (
                                "messages" in messages_data
                                and messages_data["messages"]
                            ):
                                last_message = messages_data["messages"][
                                    -1
                                ]  # Get the last 3 messages
                                if last_message and len(last_message) > 0:
                                    sender = last_message[0][0]
                                    recipient = last_message[0][1]
                                    content = last_message[0][2]
                                    # Format the message with rich styling
                                    message_content = (
                                        content.get("message", "")
                                        if isinstance(content, dict)
                                        else content
                                    )
                                    panel = rich.panel.Panel(
                                        message_content,
                                        title=f"[bold blue]{sender}[/bold blue] → [bold green]{recipient}[/bold green]",
                                        border_style="cyan",
                                        padding=(1, 2),
                                    )
                                    print(panel)
                    elif (
                        msg_type == WSMessageType.END_SIM
                        or msg_type == WSMessageType.FINISH_SIM
                    ):
                        logger.info("Simulation completed!")
                        logger.info(
                            f"Result: {json.dumps(data.get('data', {}), indent=2)}"
                        )
                        break
                    elif msg_type == WSMessageType.ERROR:
                        logger.error(f"Error: {data.get('data', {}).get('message')}")
                        break
                    else:
                        logger.info(
                            f"Received message of type {msg_type}: {data.get('data', {})}"
                        )
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket connection closed")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break
    except Exception as e:
        logger.error(f"Error during WebSocket communication: {e}")
    finally:
        await session.close()


async def main():
    # Example parameters
    env_id = "env_123"  # Replace with actual environment ID
    agent_ids = ["agent_1", "agent_2", "agent_3"]  # Replace with actual agent IDs
    agent_models = ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"]
    evaluator_model = "gpt-4o"
    env_profile_dict = {
        "codename": "test",
        "scenario": "Just chat (finish the conversation in 2 turns)",
        "agent_goals": ["Just chat"] * len(agent_ids),
    }
    agent_profile_dicts = [
        {
            "first_name": f"agent_{agent_id}",
            "last_name": f"agent_{agent_id}",
        }
        for agent_id in agent_ids
    ]
    await run_simulation(
        env_id=env_id,
        agent_ids=agent_ids,
        agent_models=agent_models,
        env_profile_dict=env_profile_dict,
        agent_profile_dicts=agent_profile_dicts,
        evaluator_model=evaluator_model,
        evaluation_dimension_list_name="sotopia",
    )


if __name__ == "__main__":
    asyncio.run(main())
