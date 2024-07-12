from datetime import datetime
import requests
import rich
from sotopia.database.persistent_profile import EnvironmentList
import asyncio
import logging
import json
import math
from itertools import chain
from collections import defaultdict
from typing import cast, List, Dict, OrderedDict, Tuple

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
    EvaluationForTwoAgents,
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
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
from ..app import app


def check_existing_episodes(
    env_id: str,
    agent_ids: list[str],
    models: dict[str, LLM_Name],
    index: str,
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
            models_list = (
                [models["env"], models["test_model"], models["partner_model"]]
                if index == "0"
                else [models["env"], models["partner_model"], models["test_model"]]
            )
            if episode.agents == agent_ids and episode.models == models_list:
                return True
        return False
    else:
        return False


def initilize_benchmark_combo(data: list[dict[str, str]]) -> list[EnvAgentComboStorage]:
    list_of_env_agent_combo_storage = []
    for combo in data:
        env_agent_combo_storage = EnvAgentComboStorage(
            env_id=combo["env_id"], agent_ids=combo["agent_ids"]
        )
        list_of_env_agent_combo_storage.append(env_agent_combo_storage)
    return list_of_env_agent_combo_storage


def get_avg_reward(
    episodes: list[EpisodeLog], model_name: str
) -> dict[str, tuple[float, float]]:
    # rewards_list = []
    rewards_dict = defaultdict(
        list
    )  # {pk: [rewards]}, {pk}_{i} denotes the i-th agent is the test agent
    avg_reward_dict = {}
    avg_variance_dict = {}
    var_reward_list = []
    for episode in episodes:
        assert episode.models is not None, "episode.models should not be None"
        if episode.models[1] == model_name:
            reward = get_rewards_from_episode(episode)[0][1]
            rewards_dict[f"{episode.environment}_0"].append(reward)
        else:
            reward = get_rewards_from_episode(episode)[1][1]
            rewards_dict[f"{episode.environment}_1"].append(reward)
        # rewards_list.append(reward)
    dimensions = list(rewards_dict.values())[0][0].keys()

    def calc_variance(local_rewards_list: List[Dict[str, float]]) -> Dict[str, float]:
        # get the variance within a list
        local_var_reward_dict = {}
        local_dimensions = local_rewards_list[0].keys()
        assert set(local_dimensions) == set(dimensions), "dimensions should be the same"
        for dimension in local_dimensions:
            rewards = [reward[dimension] for reward in local_rewards_list]
            avg_reward = sum(rewards) / len(rewards)
            variance = sum([(reward - avg_reward) ** 2 for reward in rewards]) / len(
                rewards
            )
            local_var_reward_dict[dimension] = variance

        return local_var_reward_dict

    rewards_list = list(chain(*rewards_dict.values()))

    var_reward_list = [calc_variance(rewards) for rewards in rewards_dict.values()]
    for dimension in rewards_list[0].keys():
        rewards = [reward[dimension] for reward in rewards_list]
        avg_reward = sum(rewards) / len(rewards)
        avg_reward_dict[dimension] = avg_reward

        variances = [variance[dimension] for variance in var_reward_list]
        # print(f"Dimension {dimension}, variances: {variances}, {len(variances)}")
        avg_variance = sum(variances) / len(variances)
        avg_variance_dict[dimension] = avg_variance

    return_rewards_dict = {
        key: (avg_reward_dict[key], math.sqrt(avg_variance_dict[key]))
        for key in avg_reward_dict.keys()
    }
    return_rewards_dict = {
        **return_rewards_dict,
        "setting_num": (float(len(var_reward_list)), 0.0),
        "episode_count": (float(len(rewards_list)), 0.0),
    }

    return return_rewards_dict


def _list_all_env_agent_combo_not_in_db(
    model_names: dict[str, LLM_Name],
    env_agent_combo_storage_list: list[EnvAgentComboStorage],
    tag: str | None = None,
) -> list[EnvAgentCombo[Observation, AgentAction]]:
    """We iterate over each environment and return the **first** env-agent combo that is not in the database."""
    hard_envs = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").environments
    agent_index = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").agent_index
    assert isinstance(agent_index, list), "agent_index should be a list"

    env_agent_combo_storage_index_list = []
    for env_id, index in zip(hard_envs, agent_index):
        for env_agent_combo_storage in env_agent_combo_storage_list:
            if env_agent_combo_storage.env_id == env_id:
                env_agent_combo_storage_index_list.append(
                    (env_agent_combo_storage, index)
                )

    list_of_env_agent_combo_storage = []
    for env_agent_combo_storage, index in env_agent_combo_storage_index_list:
        agent_ids = env_agent_combo_storage.agent_ids
        env_id = env_agent_combo_storage.env_id
        if check_existing_episodes(
            env_id=env_id, agent_ids=agent_ids, models=model_names, index=index, tag=tag
        ):
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
                ReachGoalLLMEvaluator(
                    model_names["env"],
                    EvaluationForTwoAgents[SotopiaDimensions],
                ),
            ],
        )
        agent_profiles = [AgentProfile.get(id) for id in agent_ids]
        # make sure the second agent (i.e., the agent being benchmarked) is always the indexed agent
        agents = [
            LLMAgent(agent_profile=agent_profile, model_name=agent_model)
            for agent_profile, agent_model in zip(
                agent_profiles,
                [model_names["test_model"], model_names["partner_model"]]
                if index == "0"
                else [model_names["partner_model"], model_names["test_model"]],
            )
        ]
        list_of_env_agent_combo_storage.append((env, agents))
    return list_of_env_agent_combo_storage  # type: ignore


def run_async_benchmark_in_batch(
    *,
    batch_size: int = 1,
    model_names: dict[str, LLM_Name] = {
        "env": "gpt-4",
        "test_model": "gpt-3.5-turbo",
        "partner_model": "gpt-3.5-turbo",
    },
    tag: str | None = None,
    push_to_db: bool = False,
    verbose: bool = False,
) -> None:
    url = "https://huggingface.co/datasets/cmu-lti/sotopia/resolve/main/benchmark_agents.json?raw=true"
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        print("Data fetched successfully")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        raise ValueError("Failed to fetch data")
    benchmark_combo = initilize_benchmark_combo(data)
    env_agent_combo_list = _list_all_env_agent_combo_not_in_db(
        model_names=model_names, tag=tag, env_agent_combo_storage_list=benchmark_combo
    )
    env_agent_combo_batch: list[EnvAgentCombo[Observation, AgentAction]] = []
    number_of_fix_turns = 0
    loop = asyncio.get_event_loop()
    while True:
        for env_agent_combo in tqdm(
            env_agent_combo_list,
            desc="Running all envs in batch",
        ):
            env_agent_combo_batch.append(env_agent_combo)
            if len(env_agent_combo_batch) == batch_size:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                loop.run_until_complete(
                    run_async_server(
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                        push_to_db=push_to_db,
                        tag=tag,
                    )
                )
                env_agent_combo_batch = []
        else:
            if env_agent_combo_batch:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                loop.run_until_complete(
                    run_async_server(
                        model_dict=model_names,
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                        push_to_db=push_to_db,
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

            env_agent_combo_list = _list_all_env_agent_combo_not_in_db(
                model_names=model_names,
                tag=tag,
                env_agent_combo_storage_list=benchmark_combo,
            )
            env_agent_combo_batch = []
            number_of_fix_turns += 1
            if len(env_agent_combo_list) == 0 or number_of_fix_turns >= 5:
                rewards_dict = get_avg_reward(
                    simulated_episodes,  # type: ignore
                    model_names["test_model"],
                )
                rewards_dict["model_name"] = model_names["test_model"]  # type: ignore
                rewards_dict["episode_count"] = len(simulated_episodes)  # type: ignore
                rich.print(rewards_dict)
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


def save_to_jsonl(
    model_rewards_dict: Dict[str, Dict[str, Tuple[float, float]]],
    partner_model: str,
) -> None:
    simplified_model_name = partner_model.split("/")[-1]
    output_fn = f"./models_vs_{simplified_model_name}.jsonl"
    outputs: List[str] = []
    for model, rewards in model_rewards_dict.items():
        formatted_reward = OrderedDict(
            {
                "model_name": model,
                **{
                    f"{v[0]} {v[1]}": rewards[k][0]
                    for k, v in dimension_range_mapping.items()
                },
            }
        )
        outputs.append(json.dumps(formatted_reward))
    with open(output_fn, "w") as f:
        f.write("\n".join(outputs))

    print(f"Output saved to {output_fn}")


default_model_list: List[str] = [
    "gpt-4o",
]
dimension_range_mapping = OrderedDict(
    {
        "social_rules": ["SOC", [-10, 0]],
        "secret": ["SEC", [-10, 0]],
        "financial_and_material_benefits": ["FIN", [-5, 5]],
        "relationship": ["REL", [-5, 5]],
        "knowledge": ["KNO", [0, 10]],
        "goal": ["GOAL", [0, 10]],
        "believability": ["BEL", [0, 10]],
    }
)


def display_in_table(
    model_rewards_dict: Dict[str, Dict[str, Tuple[float, float]]], partner_model: str
) -> None:
    table = rich.table.Table(
        title="Model Performance when facing {}".format(partner_model)
    )
    table.add_column("Model")
    for dimension in dimension_range_mapping.keys():
        table.add_column(dimension)
    table.add_column("Settings")
    table.add_column("Episodes")
    for model, rewards in model_rewards_dict.items():
        table.add_row(
            model,
            *(
                [
                    f"{rewards[k][0]:.2f} ± {rewards[k][1]:.2f}"
                    for k in dimension_range_mapping.keys()
                ]
                + [f"{rewards[k][0]:.0f}" for k in ["setting_num", "episode_count"]]
            ),
        )
    rich.print(table)


def benchmark_display(
    model_list: List[str] = default_model_list,
    partner_model: str = "together_ai/meta-llama/Llama-3-70b-chat-hf",
    evaluator_model: str = "gpt-4o",
    task: str = "hard",
    output_to_jsonl: bool = False,
) -> None:
    """
    Usage: sotopia benchmark-display --model-list gpt-4o --model-list together_ai/meta-llama-Llama-3-70b-chat-hf
    Aggregate all the results for the benchmark, as described in https://github.com/sotopia-lab/sotopia-space/blob/main/data_dir/models_vs_gpt35.jsonl
    """

    print(f"Displaying performance for {model_list} vs {partner_model} on task {task}")
    model_rewards_dict = dict()
    for model in model_list:
        tag = f"benchmark_{model}_{partner_model}_{evaluator_model}_{task}_trial0"
        episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()
        if len(episodes) == 0:
            print(f"No episodes found for {model}")
            continue
        avg_rewards = get_avg_reward(episodes, model)  # type: ignore
        model_rewards_dict[model] = avg_rewards
        # print(f"Model: {model}, episodes: {len(episodes)}, Avg Rewards: {avg_rewards}")

    display_in_table(model_rewards_dict, partner_model)
    if output_to_jsonl:
        save_to_jsonl(model_rewards_dict, partner_model)


@app.command()
def benchmark(
    models: List[str] = typer.Option(
        default_model_list,
        help=f"All the language model you want to benchmark. Default is the pre-loaded model list {default_model_list}.",
    ),
    partner_model: str = typer.Option(
        "together_ai/meta-llama/Llama-3-70b-chat-hf",
        help="The partner model you want to use.",
    ),
    evaluator_model: str = typer.Option(
        "gpt-4o", help="The evaluator model you want to use."
    ),
    batch_size: int = typer.Option(10, help="The batch size you want to use."),
    task: str = typer.Option("hard", help="The task id you want to benchmark."),
    print_logs: bool = typer.Option(False, help="Print logs."),
    only_show_performance: bool = typer.Option(False, help="Only show performance."),
    output_to_jsonl: bool = typer.Option(False, help="Output to jsonl."),
) -> None:
    if only_show_performance:
        benchmark_display(models, partner_model, evaluator_model, task, output_to_jsonl)
        return

    """A simple command-line interface example."""
    _set_up_logs(print_logs=print_logs)

    for model in models:
        typer.echo(
            f"Running benchmark for {model} chatting with {partner_model} on task {task} with {evaluator_model} as the evaluator."
        )
        model = cast(LLM_Name, model)
        partner_model = cast(LLM_Name, partner_model)
        evaluator_model = cast(LLM_Name, evaluator_model)
        tag = f"benchmark_{model}_{partner_model}_{evaluator_model}_{task}_trial0"
        run_async_benchmark_in_batch(
            batch_size=batch_size,
            model_names={
                "env": evaluator_model,
                "test_model": model,
                "partner_model": partner_model,
            },
            tag=tag,
            verbose=False,
            push_to_db=True,
        )
    benchmark_display(
        models, partner_model, evaluator_model, task, output_to_jsonl=output_to_jsonl
    )
