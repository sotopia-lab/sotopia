from datetime import datetime
from datasets import load_dataset
from sotopia.database.persistent_profile import EnvironmentList
import asyncio
import logging
from typing import Generator, cast

from logging import FileHandler
from rich.logging import RichHandler

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

import typer
from pathlib import Path

app = typer.Typer()


def check_existing_episodes(
    env_id: str,
    agent_ids: list[str],
    models: dict[str, LLM_Name],
    tag: str | None = None,
) -> bool:
    if tag:
        existing_episode = EpisodeLog.find(
            (EpisodeLog.environment == env_id) & (EpisodeLog.tag == tag)
        ).all()
    else:
        existing_episode = EpisodeLog.find(EpisodeLog.environment == env_id).all()
    if existing_episode:
        for episode in existing_episode:
            assert isinstance(episode, EpisodeLog), "episode should be an EpisodeLog"
            if episode.agents == agent_ids and episode.models == list(models.values()):
                return True
        return False
    else:
        return False


def initilize_benchmark_combo(dataset) -> list[EnvAgentComboStorage]:  # type: ignore
    list_of_env_agent_combo_storage = []
    for combo in dataset["train"]:
        env_agent_combo_storage = EnvAgentComboStorage(
            env_id=combo["env_id"], agent_ids=combo["agent_ids"]
        )
        list_of_env_agent_combo_storage.append(env_agent_combo_storage)
    return list_of_env_agent_combo_storage


def get_avg_reward(episodes: list[EpisodeLog], model_name: str) -> dict[str, float]:
    breakpoint()
    rewards_list = []
    avg_reward_dict = {}
    for episode in episodes:
        assert episode.models is not None, "episode.models should not be None"
        if episode.models[1] == model_name:
            reward = get_rewards_from_episode(episode)[0][1]
        else:
            reward = get_rewards_from_episode(episode)[1][1]
        rewards_list.append(reward)
    for dimension in rewards_list[0].keys():
        rewards = [reward[dimension] for reward in rewards_list]
        avg_reward = sum(rewards) / len(rewards)
        avg_reward_dict[dimension] = avg_reward
    return avg_reward_dict


def _iterate_all_env_agent_combo_not_in_db(
    model_names: dict[str, LLM_Name],
    env_agent_combo_storage_list: list[EnvAgentComboStorage],
    tag: str | None = None,
) -> Generator[EnvAgentCombo[Observation, AgentAction], None, None]:
    """We iterate over each environment and return the **first** env-agent combo that is not in the database."""
    hard_envs = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").environments
    agent_index = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").agent_index
    assert isinstance(agent_index, list), "agent_index should be a list"
    envs_index_mapping: dict[str, list[str]] = {env_id: [] for env_id in set(hard_envs)}
    # Repeat 10 times to match the number of combos
    for _ in range(10):
        for index, env_id in zip(agent_index, hard_envs):
            envs_index_mapping[env_id].append(index)

    for env_agent_combo_storage in env_agent_combo_storage_list:
        agent_ids = env_agent_combo_storage.agent_ids
        env_id = env_agent_combo_storage.env_id
        if check_existing_episodes(env_id, agent_ids, model_names, tag):
            logging.info(
                f"Episode for {env_id} with agents {agent_ids} using {list(model_names.values())} already exists"
            )
            continue
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
        index = envs_index_mapping[env_id].pop(0)
        if int(index) == 0:
            model_names["agent1"], model_names["agent2"] = (
                model_names["agent2"],
                model_names["agent1"],
            )
        agents = [
            LLMAgent(agent_profile=agent_profile, model_name=agent_model)
            for agent_profile, agent_model in zip(
                agent_profiles,
                [model_names["agent1"], model_names["agent2"]],
            )
        ]
        yield env, agents


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
    env_agent_combo_iter = _iterate_all_env_agent_combo_not_in_db(
        model_names=model_names, tag=tag, env_agent_combo_storage_list=benchmark_combo
    )
    env_agent_combo_iter_length = sum(1 for _ in env_agent_combo_iter)
    env_agent_combo_iter = _iterate_all_env_agent_combo_not_in_db(
        model_names=model_names, tag=tag, env_agent_combo_storage_list=benchmark_combo
    )
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
                        push_to_db=True,
                        tag=tag,
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
                        push_to_db=True,
                        tag=tag,
                    )
                )
            # remove episodes that has bad rewards
            simulated_episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()
            valid_episodes = [
                not isinstance(relevant_episode.rewards[0], float)  # type: ignore
                for relevant_episode in simulated_episodes
            ]
            for valid, episode in zip(valid_episodes, simulated_episodes):
                if not valid:
                    pk = episode.pk
                    assert isinstance(pk, str)
                    EpisodeLog.delete(pk)

            env_agent_combo_iter = _iterate_all_env_agent_combo_not_in_db(
                model_names=model_names,
                tag=tag,
                env_agent_combo_storage_list=benchmark_combo,
            )
            env_agent_combo_iter_length = sum(1 for _ in env_agent_combo_iter)
            env_agent_combo_batch = []
            number_of_fix_turns += 1
            if env_agent_combo_iter_length == 0 or number_of_fix_turns >= 5:
                rewards_dict = get_avg_reward(simulated_episodes, model_names["agent2"])  # type: ignore
                rewards_dict["model_name"] = model_names["agent2"]  # type: ignore
                print(rewards_dict)
                return


def _set_up_logs(
    *,
    log_file_level: int = logging.DEBUG,
    log_rich_level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    log_file: str = datetime.now().strftime("./logs/%H_%M_%d_%m_%Y.log"),
    print_logs: bool = False,
) -> None:
    # date and message only
    logging_path = Path(log_file)
    logging_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_file_level,
        format=log_format,
        datefmt="[%X]",
        handlers=[
            FileHandler(logging_path),
            RichHandler(level=log_rich_level) if print_logs else RichHandler(level=100),
        ],
    )


@app.command()
def cli(
    model: str = typer.Option(..., help="The language model you want to benchmark."),
    partner_model: str = typer.Option(
        "together_ai/meta-llama/Llama-2-70b-chat-hf",
        help="The partner model you want to use.",
    ),
    evaluator_model: str = typer.Option(
        "gpt-4o", help="The evaluator model you want to use."
    ),
    batch_size: int = typer.Option(10, help="The batch size you want to use."),
    task: str = typer.Option("hard", help="The task id you want to benchmark."),
    print_logs: bool = typer.Option(False, help="Print logs."),
) -> None:
    """A simple command-line interface example."""
    _set_up_logs(print_logs=print_logs)
    typer.echo(
        f"Running benchmark for {model} and {partner_model} on task {task} with {evaluator_model} as the evaluator."
    )
    model = cast(LLM_Name, model)
    partner_model = cast(LLM_Name, partner_model)
    evaluator_model = cast(LLM_Name, evaluator_model)
    tag = f"benchmark_{model}_q"
    run_async_benchmark_in_batch(
        batch_size=batch_size,
        model_names={"agent1": model, "agent2": partner_model, "env": evaluator_model},
        tag=tag,
        verbose=False,
    )


if __name__ == "__main__":
    app()
