import asyncio
import logging
import sys
from logging import FileHandler
from typing import Literal, cast

from rich import print
from rich.logging import RichHandler

from sotopia.agents import LLMAgent
from sotopia.database import (
    AgentProfile,
    EnvAgentComboStorage,
    EnvironmentProfile,
    EpisodeLog,
)
from sotopia.envs.evaluators import (
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
)
from sotopia.envs.parallel import ParallelSotopiaEnv
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

agent_model_1: LLM_Name = "gpt-3.5-turbo"
agent_model_2: LLM_Name = "gpt-4"


model_names: dict[str, LLM_Name] = {
    "env": "gpt-4",
    "agent1": agent_model_1,
    "agent2": agent_model_2,
}

model_names_dual: dict[str, LLM_Name] = {
    "env": "gpt-4",
    "agent1": agent_model_2,
    "agent2": agent_model_1,
}

model_triple_list: tuple[dict[str, LLM_Name], dict[str, LLM_Name]] = (
    model_names,
    model_names_dual,
)

env_ids: list[str] = list(EnvironmentProfile.all_pks())
assert all(
    isinstance(env_id, str) for env_id in env_ids
), "env_ids should be a list of strings"

push_to_db = sys.argv[1]
assert push_to_db in ["True", "False"], "push_to_db should be True or False"
push_to_db_bool = push_to_db == "True"
tag = "6_initial_aug5"

sampler = ConstraintBasedSampler[Observation, AgentAction]()


def check_existing_episodes(
    env_id: str, agent_ids: list[str], tag: str, models: dict[str, LLM_Name]
) -> bool:
    existing_episode = EpisodeLog.find(
        (EpisodeLog.environment == env_id) & (EpisodeLog.tag == tag)
    ).all()
    if existing_episode:
        for episode in existing_episode:
            assert isinstance(
                episode, EpisodeLog
            ), "episode should be an EpisodeLog"
            if episode.agents == agent_ids and episode.models == list(
                models.values()
            ):
                return True
        return False
    else:
        return False


def _sample_env_agent_combo_and_push_to_db(env_id: str) -> None:
    sampler = ConstraintBasedSampler[Observation, AgentAction](
        env_candidates=[env_id]
    )
    env_agent_combo_list = list(
        sampler.sample(agent_classes=[LLMAgent] * 2, replacement=False)
    )
    for env, agent in env_agent_combo_list:
        EnvAgentComboStorage(
            env_id=env.profile.pk,
            agent_ids=[agent[0].profile.pk, agent[1].profile.pk],
        ).save()


episode_number = 0
for env_id in env_ids:
    for model_names in model_triple_list:
        assert env_id is not None, "env_id should not be None"
        env_agent_combo_storage_list = list(
            EnvAgentComboStorage.find(
                EnvAgentComboStorage.env_id == env_id
            ).all()
        )
        if not env_agent_combo_storage_list:
            _sample_env_agent_combo_and_push_to_db(env_id)
            env_agent_combo_storage_list = list(
                EnvAgentComboStorage.find(
                    EnvAgentComboStorage.env_id == env_id
                ).all()
            )
            assert env_agent_combo_storage_list
        agents_not_in_log: list[list[str]] = []
        env_agent_combo_list: list[
            EnvAgentCombo[Observation, AgentAction]
        ] = []
        for env_agent_combo_storage in env_agent_combo_storage_list:
            env_agent_combo_storage = cast(
                EnvAgentComboStorage, env_agent_combo_storage
            )
            agent_ids = env_agent_combo_storage.agent_ids
            if check_existing_episodes(env_id, agent_ids, tag, model_names):
                episode_number += 1
                print(
                    f"Episode {episode_number} for {env_id} with agents {agent_ids} using {list(model_names.values())} already exists"
                )
                continue

            env_profile = EnvironmentProfile.get(env_id)
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
            agent_profiles = [AgentProfile.get(id) for id in agent_ids]

            agents = [
                LLMAgent(agent_profile=agent_profile, model_name=agent_model)
                for agent_profile, agent_model in zip(
                    agent_profiles,
                    [model_names["agent1"], model_names["agent2"]],
                )
            ]

            env_agent_combo_list.append((env, agents))

        asyncio.run(
            run_async_server(
                model_dict=model_names,
                action_order="round-robin",
                sampler=BaseSampler[Observation, AgentAction](),
                env_agent_combo_list=env_agent_combo_list,
                tag=tag,
                push_to_db=push_to_db_bool,
            )
        )
