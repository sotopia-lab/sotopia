from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
import asyncio
from minimalist_demo import run_simulation, store_message_callback
import logging

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
    simulation_task = asyncio.create_task(run_simulation(session_id))
    
    # Listen for new messages and send them to the client
    pubsub = redis_client.pubsub()
    pubsub.subscribe(f"chat:{session_id}")
    print(f"Subscribed to chat:{session_id}")
    
    try:
        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True)
            if message:
                data = message['data'].decode('utf-8').split(':', 1)
                print(f"Message: {data}")
                await websocket.send_json({
                    "sender": data[0],
                    "content": data[1],
                    "timestamp": str(redis_client.time()[0])
                })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session: {session_id}")
    finally:
        pubsub.unsubscribe(f"chat:{session_id}")
        simulation_task.cancel()
        print(f"Cleaned up for session: {session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
