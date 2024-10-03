# This demo servers as a minimal example of how to use the sotopia library.

# 1. Import the sotopia library
# 1.1. Import the `run_async_server` function: In sotopia, we use Python Async
#     API to optimize the throughput.
import asyncio
import logging

# 1.2. Import the `UniformSampler` class: In sotopia, we use samplers to sample
#     the social tasks.
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server
from rich.logging import RichHandler

# 2. Run the server

# 2.1. Configure the logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=20,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

# 2.2. Run the simulation
asyncio.run(
    run_async_server(
        model_dict={
            "env": "gpt-4",
            "agent1": "gpt-4o-mini",
            "agent2": "gpt-4o-mini",
        },
        sampler=UniformSampler(),
    )
)
