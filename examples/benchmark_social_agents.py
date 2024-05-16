from datasets import load_dataset
import typer
from sotopia.database.persistent_profile import EnvironmentList
from sotopia.database.env_agent_combo_storage import EnvAgentComboStorage
import asyncio
import logging
import os
import subprocess
from typing import Any, Generator, cast

import gin
from absl import flags
from tqdm import tqdm

from sotopia.agents import LLMAgent
from sotopia.database import (
    AgentProfile,
    EnvAgentComboStorage,
    EnvironmentProfile,
    EpisodeLog,
)
from sotopia.database.serialization import get_rewards_from_episode
from sotopia.envs.evaluators import (
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.generation_utils.generate import LLM_Name
from sotopia.messages import AgentAction, Observation
from sotopia.samplers import (
    BaseSampler,
    EnvAgentCombo,
)
from sotopia.server import run_async_server
from sotopia_conf.gin_utils import parse_gin_flags, run
from experiment_eval import check_existing_episodes

app = typer.Typer()

_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

DEFAULT_GIN_FILES = [
    "sotopia_conf/generation_utils_conf/generate.gin",
    "sotopia_conf/server_conf/server.gin",
    "sotopia_conf/run_async_server_in_batch.gin",
]

DEFAULT_GIN_BINDINGS = [
    '--gin.ENV_IDS=[]',
    '--gin.AGENT1_MODEL="groq/llama3-70b-8192"',
    '--gin.PUSH_TO_DB=True',
    '--gin.OMNISCIENT=False',
    '--gin.VERBOSE=False',
]

process = subprocess.Popen(
    ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
)
git_head_hash = process.communicate()[0].strip()

def initilize_benchmark_combo(dataset)->list[EnvAgentComboStorage]: # type: ignore
    list_of_env_agent_combo_storage = []
    for combo in dataset["train"]:
        env_agent_combo_storage = EnvAgentComboStorage(
            env_id=combo["env_id"],
            agent_ids=combo["agent_ids"]
        )
        list_of_env_agent_combo_storage.append(env_agent_combo_storage)
    return list_of_env_agent_combo_storage


def get_avg_reward(episodes: list[EpisodeLog], model_name: str) -> dict[str, float]:
    rewards_list = []   
    avg_reward_dict = {}
    for episode in episodes:
        assert episode.models is not None, "episode.models should not be None"
        if episode.models[1] == model_name:
            reward = get_rewards_from_episode(episode)[0]
        else:
            reward = get_rewards_from_episode(episode)[1]
        rewards_list.append(reward)
    for dimension in rewards_list[0].keys():
        rewards = [reward[dimension] for reward in rewards_list]
        avg_reward = sum(rewards) / len(rewards)
        avg_reward_dict[dimension] = avg_reward
    return avg_reward_dict



@gin.configurable
def _iterate_env_agent_combo_not_in_db(
    model_names: dict[str, LLM_Name],
    env_agent_combo_storage_list: list[EnvAgentComboStorage],
    tag: str | None = None,
) -> Generator[EnvAgentCombo[Observation, AgentAction], None, None]:
    """We iterate over each environment and return the **first** env-agent combo that is not in the database."""
    hard_envs = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").environments
    agent_index = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").agent_index
    assert isinstance(agent_index, list), "agent_index should be a list"
    for index, env_id in zip(agent_index, hard_envs):
        assert env_id is not None, "env_id should not be None"
        first_env_agent_combo_storage_to_run: EnvAgentComboStorage | None = None
        for env_agent_combo_storage in env_agent_combo_storage_list:
            agent_ids = env_agent_combo_storage.agent_ids
            if check_existing_episodes(env_id, agent_ids, model_names, tag):
                logging.info(
                    f"Episode for {env_id} with agents {agent_ids} using {list(model_names.values())} already exists"
                )
                continue
            first_env_agent_combo_storage_to_run = env_agent_combo_storage
            break
        if first_env_agent_combo_storage_to_run:
            env_profile = EnvironmentProfile.get(env_id)
            env = ParallelSotopiaEnv(
                env_profile=env_profile,
                model_name=model_names["env"],
                action_order="round-robin",
                evaluators=[
                    RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
                ],
                terminal_evaluators=[
                    ReachGoalLLMEvaluator(model_names["env"]),
                ],
            )
            agent_profiles = [AgentProfile.get(id) for id in agent_ids]
            # make sure the second agent (i.e., the agent being benchmarked) is always the indexed agent
            if int(index) == 0:
                model_names["agent1"], model_names["agent2"] = model_names["agent2"], model_names["agent1"]
            agents = [
                LLMAgent(agent_profile=agent_profile, model_name=agent_model)
                for agent_profile, agent_model in zip(
                    agent_profiles,
                    [model_names["agent1"], model_names["agent2"]],
                )
            ]
            yield env, agents

@gin.configurable
def run_async_benchmark_in_batch(
        *,
    batch_size: int = 1,
    model_names: dict[str, LLM_Name] = {
        "env": "gpt-4",
        "agent1": "gpt-3.5-turbo",
        "agent2": "gpt-3.5-turbo",
    },
    tag: str | None = None,
    verbose: bool = False,
) -> None:
    dataset = load_dataset("cmu-lti/sotopia", data_files="benchmark_agents.json")
    benchmark_combo = initilize_benchmark_combo(dataset)
    env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(model_names=model_names, tag=tag ,env_agent_combo_storage_list=benchmark_combo)
    env_agent_combo_iter_length = sum(1 for _ in env_agent_combo_iter)
    env_agent_combo_batch: list[EnvAgentCombo[Observation, AgentAction]] = []
    number_of_fix_turns = 0
    while True:
        for env_agent_combo in tqdm(
            env_agent_combo_iter,
            total=env_agent_combo_iter_length,
            desc="Running all envs in batch",
        ):
            env_agent_combo_batch.append(env_agent_combo)
            if len(env_agent_combo_batch) == batch_size:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                    )
                )
                env_agent_combo_batch = []
        else:
            if env_agent_combo_batch:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                    )
                )
            # remove episodes that has bad rewards
            simulated_episodes = EpisodeLog.find(EpisodeLog.tag == tag).all() 
            valid_episodes = [
                not isinstance(relevant_episode.rewards[0], float) # type: ignore
                for relevant_episode in simulated_episodes
            ]
            for valid, episode in zip(valid_episodes, simulated_episodes):
                if not valid:
                    pk = episode.pk
                    assert isinstance(pk, str)
                    EpisodeLog.delete(pk)
            
            env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(model_names=model_names, tag=tag ,env_agent_combo_storage_list=benchmark_combo)
            env_agent_combo_iter_length = sum(1 for _ in env_agent_combo_iter)
            env_agent_combo_batch = [] 
            number_of_fix_turns += 1
            if env_agent_combo_iter_length == 0 or number_of_fix_turns >= 5:
                rewards_dict = get_avg_reward(simulated_episodes, model_names["agent2"]) # type: ignore
                rewards_dict["model_name"] = model_names["agent2"] # type: ignore
                print(rewards_dict)
                return

@app.command()
def main(
    eval_model: str = "gpt-4o-2024-05-13",
    batch_size: int = 10,
    gin_file: list[str] = typer.Option(
        DEFAULT_GIN_FILES, help="Path to gin configuration file. Multiple paths may be passed and will be imported in the given order, with later configurations overriding earlier ones."
    ),
    gin_search_paths: list[str] = typer.Option(
        _DEFAULT_GIN_SEARCH_PATHS, help="Comma-separated list of gin config path prefixes to be prepended to suffixes given via `--gin_file`. Only the first prefix that produces a valid path for each suffix will be used."
    ),
)-> None:
    gin_bindings = DEFAULT_GIN_BINDINGS + [
        f'--gin.AGENT2_MODEL="{eval_model}"',
        f'--gin.BATCH_SIZE={batch_size}',
        f'--gin.TAG="benchmark_{eval_model}"',
        f'--gin.TAG_TO_CHECK_EXISTING_EPISODES="benchmark_{eval_model}"'
    ]
    
    parse_gin_flags(
        gin_search_paths,
        gin_file,
        gin_bindings,
    )
    run_async_benchmark_in_batch()

if __name__ == "__main__":
    app()


