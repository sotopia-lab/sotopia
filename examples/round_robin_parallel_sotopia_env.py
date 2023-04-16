import logging
from logging import FileHandler

from rich import print
from rich.logging import RichHandler

from sotopia.agents.llm_agent import Agents, LLMAgent
from sotopia.envs import ParallelSotopiaEnv
from sotopia.generation_utils.generate import process_history
from sotopia.sync_server import run_sync_server

# date and message only
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(),
        FileHandler("./logs/round_robin_parallel_sotopia_env.log"),
    ],
)

messages = run_sync_server(
    model_name="gpt-3.5-turbo", action_order="round-robin"
)
for sender, receiver, message in messages:
    print(sender, receiver, message.to_natural_language())
