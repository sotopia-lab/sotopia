import asyncio
import logging
from logging import FileHandler
from typing import Literal

from rich import print
from rich.logging import RichHandler

from sotopia.generation_utils.generate import LLM_Name, process_history
from sotopia.server import run_async_server, run_sync_server

# date and message only
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(),
        FileHandler("./logs/round_robin_parallel_sotopia_env_2.log"),
    ],
)

model_names: dict[str, LLM_Name] = {
    "env": "gpt-4",
    "agent1": "gpt-4",
    "agent2": "human",
}

messages = asyncio.run(
    run_async_server(model_dict=model_names, action_order="round-robin")
)
for index, (sender, receiver, message) in enumerate(messages):
    if sender == "Environment" and index % 2 == 0:
        print(message.to_natural_language())
