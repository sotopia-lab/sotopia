import asyncio
import logging
import sys
from logging import FileHandler
from typing import Literal

from rich import print
from rich.logging import RichHandler

from sotopia.agents import LLMAgent
from sotopia.database import (
    AgentProfile,
    EnvAgentComboStorage,
    EnvironmentProfile,
)
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
)
from sotopia.generation_utils.generate import LLM_Name
from sotopia.messages import AgentAction, Observation
from sotopia.samplers import (
    BaseSampler,
    ConstraintBasedSampler,
    EnvAgentCombo,
)
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
    "agent2": "human",
}

push_to_db = sys.argv[1]
assert push_to_db in ["True", "False"], "push_to_db should be True or False"
push_to_db_bool = push_to_db == "True"
env_ids: list[str] = []

for code_name in ["borrow_money"]:
    envs_with_code_name = EnvironmentProfile.find(
        EnvironmentProfile.codename == code_name
    ).all()
    assert len(envs_with_code_name)
    assert (env_id := envs_with_code_name[0].pk)
    env_ids.append(env_id)


for env_id in env_ids:
    assert env_id is not None, "env_id should not be None"
    env_agent_combo_storage_list = EnvAgentComboStorage.find(
        EnvAgentComboStorage.env_id == env_id
    ).all()

    sampler = (
        ConstraintBasedSampler[Observation, AgentAction](
            env_candidates=[env_id],
        )
        if len(env_agent_combo_storage_list) == 0
        else BaseSampler[Observation, AgentAction]()
    )

    env_agent_combo_list: list[EnvAgentCombo[Observation, AgentAction]] = []

    for env_agent_combo_storage in env_agent_combo_storage_list:
        assert isinstance(env_agent_combo_storage, EnvAgentComboStorage)
        env_profile = EnvironmentProfile.get(env_agent_combo_storage.env_id)
        env = ParallelSotopiaEnv(
            env_profile=env_profile,
            model_name=model_names["env"],
            action_order="round-robin",
            evaluators=[
                RuleBasedTerminatedEvaluator(
                    max_turn_number=10, max_stale_turn=2
                ),
            ],
            terminal_evaluators=[
                ReachGoalLLMEvaluator(model_names["env"]),
            ],
        )
        agent_profiles = [
            AgentProfile.get(id) for id in env_agent_combo_storage.agent_ids
        ]

        agents = [
            LLMAgent(agent_profile=agent_profile, model_name=agent_model)
            for agent_profile, agent_model in zip(
                agent_profiles, [model_names["agent1"], model_names["agent2"]]
            )
        ]

        env_agent_combo_list.append((env, agents))

    asyncio.run(
        run_async_server(
            model_dict=model_names,
            action_order="round-robin",
            sampler=sampler,
            env_agent_combo_list=env_agent_combo_list,
            push_to_db=push_to_db_bool,
            using_async=False,
        )
    )
