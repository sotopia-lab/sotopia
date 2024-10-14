# This demo servers as a minimal example of how to use the sotopia library.

# 1. Import the sotopia library
# 1.1. Import the `run_async_server` function: In sotopia, we use Python Async
#     API to optimize the throughput.
import asyncio
import logging
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server
from rich.logging import RichHandler
import redis
from sotopia.messages import AgentAction
# Configure Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Configure logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=20,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

# Custom callback to store messages in Redis
async def store_message_callback(session_id, sender, message):
    redis_client.publish(f"chat:{session_id}", f"{sender}:{message}")

# Run the simulation
async def run_simulation(session_id):
    print(f"Starting simulation for session {session_id}")
    messages = await run_async_server(
        model_dict={
            "env": "gpt-4o-mini",
            "agent1": "gpt-4o-mini",
            "agent2": "human",
        },
        sampler=UniformSampler(),
        session_id=session_id,
    )    
    print(f"All episodes processed for session {session_id}")

if __name__ == "__main__":
    asyncio.run(run_simulation("test_session"))
