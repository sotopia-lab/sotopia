import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.DEBUG)

async def test_websocket():
    uri = "ws://localhost:8000/ws/test_session"
    try:
        async with websockets.connect(uri) as websocket:
            logging.info("Connected to WebSocket")
            
            # Send a message
            message = {
                "action_type": "speak",
                "argument": "Hello, world!"
            }
            await websocket.send(json.dumps(message))
            logging.info(f"Sent message: {message}")
            
            # Wait for a short time to allow server processing
            await asyncio.sleep(0.5)
            
            # Receive a message
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                logging.info(f"Received: {response}")
            except asyncio.TimeoutError:
                logging.error("Timeout waiting for response")
            
    except websockets.exceptions.ConnectionClosedOK:
        logging.info("WebSocket connection closed normally")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.error(f"WebSocket connection closed unexpectedly: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_websocket())
