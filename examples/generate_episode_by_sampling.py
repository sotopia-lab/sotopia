import asyncio
import logging
import sys
from logging import FileHandler
from typing import Literal

from rich import print
from rich.logging import RichHandler

from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
)
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

# buying_gift = EnvironmentProfile.get("01H2REANWK8METSXCW6AP244H3")
# borrowing_money = EnvironmentProfile.get("01H2REANWE5XSSSBE1X9KHSF5E")
# prison_dillema = EnvironmentProfile.get("01H2REANWTJJDSF00YR897SC28")
# charity = EnvironmentProfile.get("01H2REANWP9QSXWM8VNV0CGC5H")
# obtain env profiles

env_pks = EnvironmentProfile.all_pks()
env_candidates = []
for pk in env_pks:
    env_candidates.append(EnvironmentProfile.get(pk))

agents = [
    AgentProfile.get("01H49HPQJ0S3J76KW94JZYFS1D"),
    AgentProfile.get("01H49HPQKS32HSJ2XSMWRA8S7G"),
]

push_to_db = sys.argv[1]
assert push_to_db in ["True", "False"], "push_to_db should be True or False"
push_to_db_bool = push_to_db == "True"

# env_candidates = [EnvironmentProfile.get("01H4EFKY8VCAJJJM8WACW3KYWE")]

for _ in range(10):
    messages = asyncio.run(
        run_async_server(
            model_dict=model_names,
            action_order="round-robin",
            push_to_db=push_to_db_bool,
            env_candidates=env_candidates,
            #        agent_candidates=agents,
        )
    )
