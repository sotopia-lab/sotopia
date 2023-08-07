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
from sotopia.messages import AgentAction, Observation
from sotopia.samplers import UniformSampler
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

env_pks = EnvironmentProfile.all_pks()
env_candidates: list[EnvironmentProfile] = []
for pk in env_pks:
    env_candidates.append(EnvironmentProfile.get(pk))

all_agent_pks = list(AgentProfile.all_pks())
assert (
    len(all_agent_pks) >= 2
), "There should be at least 2 agents in the database"
agents = [
    all_agent_pks[0],
    all_agent_pks[1],
]

push_to_db = sys.argv[1]
assert push_to_db in ["True", "False"], "push_to_db should be True or False"
push_to_db_bool = push_to_db == "True"

# env_candidates = [EnvironmentProfile.get("01H4EFKY8VCAJJJM8WACW3KYWE")]
sampler = UniformSampler[Observation, AgentAction](
    env_candidates=env_candidates,
    # agent_candidates=agent_candidates,
)

for _ in range(10):
    messages = asyncio.run(
        run_async_server(
            model_dict=model_names,
            action_order="round-robin",
            push_to_db=push_to_db_bool,
            sampler=sampler,
        )
    )
