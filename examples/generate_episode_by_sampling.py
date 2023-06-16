import asyncio
import logging
from logging import FileHandler
from typing import Literal

from rich import print
from rich.logging import RichHandler

from sotopia.generation_utils.generate import LLM_Name
from sotopia.server import run_async_server

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
    "agent1": "gpt-3.5-turbo",
    "agent2": "gpt-3.5-turbo",
}

messages = asyncio.run(
    run_async_server(model_dict=model_names, action_order="round-robin")
)

env_messages = []
for index, (sender, receiver, message) in enumerate(messages):
    if receiver == "Environment":
        env_messages.append((sender, message))

history = "\n".join(
    [
        f"{x}: {y.to_natural_language()}"
        if x != "Environment"
        else y.to_natural_language()
        for x, y in env_messages
    ]
)
print(history)
