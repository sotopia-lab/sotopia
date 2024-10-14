from fastapi import FastAPI, WebSocket, WebSocketDisconnect, WebSocketException
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
import asyncio
from minimalist_demo import run_simulation, store_message_callback
import logging
from sotopia.messages import AgentAction

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be cautious with this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add this line to enable WebSocket CORS
from fastapi.middleware.trustedhost import TrustedHostMiddleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

redis_client = redis.Redis(host='localhost', port=6379, db=0)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    logging.debug(f"Attempting to accept WebSocket connection for session: {session_id}")
    await websocket.accept()
    logging.info(f"WebSocket connection accepted for session: {session_id}")
    
    # Start the simulation in the background
    # simulation_task = asyncio.create_task(run_simulation(session_id))
    
    # Listen for new messages and send them to the client
    pubsub = redis_client.pubsub()
    pubsub.subscribe(f"chat:{session_id}")
    logging.info(f"Subscribed to chat:{session_id}")
    
    try:
        while True:
            try:
                # Increase timeout to 1 second
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                logging.info(f"Received data: {data}")
                if data:
                    message = json.loads(data)
                    action = AgentAction(action_type=message['action_type'], argument=message['argument'])
                    redis_client.publish(f"human_input:{session_id}", json.dumps(action.__dict__))
                    logging.info(f"Published action to Redis: {action}")
                    
                    # Send a confirmation message back to the client
                    await websocket.send_json({
                        "sender": "server",
                        "content": f"Received action: {action.action_type} with argument: {action.argument}",
                        "timestamp": str(redis_client.time()[0])
                    })
            except asyncio.TimeoutError:
                # Instead of continuing silently, check Redis for any messages
                message = pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    data = message['data'].decode('utf-8').split(':', 1)
                    logging.info(f"Redis message: {data}")
                    await websocket.send_json({
                        "sender": data[0],
                        "content": data[1],
                        "timestamp": str(redis_client.time()[0])
                    })
                continue
            except WebSocketDisconnect:
                logging.info(f"WebSocket disconnected for session: {session_id}")
                break
            except WebSocketException as e:
                logging.error(f"WebSocket error: {str(e)}")
                break
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON received: {data}. Error: {str(e)}")
                continue
            except Exception as e:
                logging.error(f"Error processing message: {str(e)}")
                break
    finally:
        pubsub.unsubscribe(f"chat:{session_id}")
        # simulation_task.cancel()
        logging.info(f"Unsubscribed from chat:{session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
