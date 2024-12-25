"""
A test client for the WebSocket server
"""

import json
from sotopia.database import EnvironmentProfile, AgentProfile

import asyncio
import aiohttp
import sys
from pathlib import Path


class WebSocketClient:
    def __init__(self, url: str, token: str, client_id: int):
        self.url = url
        self.token = token
        self.client_id = client_id
        self.message_file = Path(f"message_{client_id}.txt")

    async def save_message(self, message: str) -> None:
        """Save received message to a file"""
        with open(self.message_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")

    async def connect(self) -> None:
        """Establish and maintain websocket connection"""
        url_with_token = f"{self.url}?token=test_token_{self.client_id}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url_with_token) as ws:
                    print(f"Client {self.client_id}: Connected to {self.url}")

                    # Send initial message
                    # Note: You'll need to implement the logic to get agent_ids and env_id
                    # This is just an example structure
                    agent_ids = [agent.pk for agent in AgentProfile.find().all()[:2]]
                    env_id = EnvironmentProfile.find().all()[0].pk
                    start_message = {
                        "type": "START_SIM",
                        "data": {
                            "env_id": env_id,  # Replace with actual env_id
                            "agent_ids": agent_ids,  # Replace with actual agent_ids
                        },
                    }
                    await ws.send_json(start_message)
                    print(f"Client {self.client_id}: Sent START_SIM message")

                    # Receive and process messages
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            print(
                                f"\nClient {self.client_id} received message:",
                                json.dumps(json.loads(msg.data), indent=2),
                            )
                            await self.save_message(msg.data)
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            print(f"Client {self.client_id}: Connection closed")
                            break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            print(f"Client {self.client_id}: Connection error")
                            break

        except Exception as e:
            print(f"Client {self.client_id} connection error:", str(e))


async def main() -> None:
    # Create multiple WebSocket clients
    num_clients = 0
    url = "ws://localhost:8800/ws/simulation"

    # Create and store client instances
    clients = [
        WebSocketClient(url=url, token=f"test_token_{i}", client_id=i)
        for i in range(num_clients)
    ]
    clients.append(WebSocketClient(url=url, token="test_token_10", client_id=10))
    clients.append(
        WebSocketClient(url=url, token="test_token_10", client_id=10)
    )  # test duplicate token

    # Create tasks for each client
    tasks = [asyncio.create_task(client.connect()) for client in clients]

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down clients...")
        sys.exit(0)
