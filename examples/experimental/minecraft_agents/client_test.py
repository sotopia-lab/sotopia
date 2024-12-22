import asyncio
import websockets


async def test_client():
    uri = "ws://localhost:8000/ws/test_client"  # WebSocket server (default listening port for Uvicorn is 8000)
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket server")

            test_message = "Hello from client!"
            print(f"Sending: {test_message}")
            await websocket.send(test_message)

            while True:
                response = await websocket.recv()
                print(f"Received: {response}")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    asyncio.run(test_client())
