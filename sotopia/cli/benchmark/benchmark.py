import asyncio
import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import OrderedDict

import numpy as np
import requests
import rich
import typer
from rich.logging import RichHandler
from tqdm import tqdm

from sotopia.agents import LLMAgent
from sotopia.database import (
    AgentProfile,
    EnvAgentComboStorage,
    EnvironmentProfile,
    EpisodeLog,
    SotopiaDimensions,
)
from sotopia.database.persistent_profile import EnvironmentList
from sotopia.database.serialization import get_rewards_from_episode
from sotopia.envs.evaluators import (
    EpisodeLLMEvaluator,
    EvaluationForAgents,
    RuleBasedTerminatedEvaluator,
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.logging import FileHandler
from sotopia.messages import AgentAction, Observation
from sotopia.samplers import (
    BaseSampler,
    EnvAgentCombo,
)
from sotopia.server import run_async_server

from ..app import app
from typing import Annotated

# date and message only
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

logging.basicConfig(
    level=20,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(),
    ],
)

default_model_list: list[str] = [
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


def get_avg_reward(
    episodes: list[EpisodeLog],
    model_name: str,
    agent_class: str = "",
) -> dict[str, tuple[float, float]]:
    """
    input: list of EpisodeLog, model_name, agent_class

    return: dictionary of {dimension: (avg_reward, margin_of_error (in 95% confidence interval))}, plus the distinct setting number and episode count (in the same format, but with 0 margin of error)
    """
    rewards_dict = defaultdict(
        list
    )  # {pk: [rewards]}, {pk}_{i} denotes the i-th agent is the test agent
    avg_reward_dict = {}
    avg_margin_dict = {}
    avg_variance_dict = {}

    for episode in episodes:
        assert episode.models is not None, "episode.models should not be None"
        agent_classes = getattr(episode, "agent_classes", None)
        if episode.models[1] == model_name and (
            not agent_class or (agent_classes and agent_classes[0] == agent_class)
        ):
            reward = get_rewards_from_episode(episode)[0][1]
            rewards_dict[f"{episode.environment}_0"].append(reward)
        else:
            reward = get_rewards_from_episode(episode)[1][1]
            rewards_dict[f"{episode.environment}_1"].append(reward)
    dimensions = list(rewards_dict.values())[0][0].keys()

    def calc_variance(
        local_rewards_list: list[dict[str, float]],
    ) -> dict[str, float]:
        # get the variance within a list, discarded
        local_var_reward_dict = {}
        local_dimensions = local_rewards_list[0].keys()
        assert set(local_dimensions) == set(dimensions), "dimensions should be the same"
        for dimension in local_dimensions:
            rewards = [reward[dimension] for reward in local_rewards_list]
            avg_reward = sum(rewards) / len(rewards)
            if len(rewards) == 1:
                variance = 0.0
            else:
                variance = sum([(reward - avg_reward) ** 2 for reward in rewards]) / (
                    len(rewards) - 1
                )
            local_var_reward_dict[dimension] = variance

        return local_var_reward_dict

    def calc_average(list_to_average: list[float]) -> float:
        return sum(list_to_average) / len(list_to_average)

    rewards_list = list(chain(*rewards_dict.values()))

    variance_reward_list = [calc_variance(rewards) for rewards in rewards_dict.values()]
    for dimension in rewards_list[0].keys():
        avg_reward_dict[dimension] = calc_average(
            [reward[dimension] for reward in rewards_list]
        )
        avg_variance_dict[dimension] = calc_average(
            [variance[dimension] for variance in variance_reward_list]
        )  # average the variances for an estimation of the variance

    for dimension in rewards_list[0].keys():
        # calculate the margin of error by the averaged mean and variance
        avg_variance = avg_variance_dict[dimension]

        combined_variance = avg_variance
        combined_sem = math.sqrt(
            combined_variance / len(rewards_list)
        )  # sem = sqrt(variance / n), we use the averaged variance under different settings

        confidence_level = 0.95
        t_samples = np.random.standard_t(df=len(rewards_list) - 1, size=1000000)

        overall_t_value = np.percentile(
            t_samples, 100 * (1 - (1 - confidence_level) / 2)
        )

        margin = overall_t_value * combined_sem
        avg_margin_dict[dimension] = margin

    return_rewards_dict = {
        key: (avg_reward_dict[key], avg_margin_dict[key])
        for key in avg_reward_dict.keys()
    }
    return_rewards_dict = {
        **return_rewards_dict,
        "setting_num": (float(len(variance_reward_list)), 0.0),
        "episode_count": (float(len(rewards_list)), 0.0),
    }

    return return_rewards_dict


def initialize_benchmark_combo(url: str) -> list[EnvAgentComboStorage]:
    if url:
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            print("Data fetched successfully")
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            raise ValueError("Failed to fetch data")
        list_of_env_agent_combo_storage = []
        for combo in data:
            env_agent_combo_storage = EnvAgentComboStorage(
                env_id=combo["env_id"], agent_ids=combo["agent_ids"]
            )
            list_of_env_agent_combo_storage.append(env_agent_combo_storage)
    else:
        list_of_env_agent_combo_storage = EnvAgentComboStorage.find().all()  # type: ignore
    return list_of_env_agent_combo_storage


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


def preprocess_episode_data(
    episode_list_with_tag: list[EpisodeLog],
) -> dict[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, str]], bool]:
    """Preprocess episodes into a dictionary for quick lookup."""
    episode_dict = {}
    for episode in episode_list_with_tag:
        assert isinstance(episode, EpisodeLog), "episode should be an EpisodeLog"
        assert episode.models is not None, "episode.models should not be None"
        agent_classes = getattr(episode, "agent_classes", None)
        episode_key = (
            episode.environment,
            tuple(episode.agents),
            tuple(episode.models),
            tuple(agent_classes) if agent_classes else ("", ""),
        )
        episode_dict[episode_key] = True
    return episode_dict


def check_existing_episodes(
    env_id: str,
    agent_ids: list[str],
    models: dict[str, str],
    index: str,
    episode_dict: dict[
        tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, str]], bool
    ],
    agent_classes: dict[str, str],
) -> bool:
    models_list = (
        (models["env"], models["test_model"], models["partner_model"])
        if index == "0"
        else (models["env"], models["partner_model"], models["test_model"])
    )
    if agent_classes:
        agent_classes_list = (
            (agent_classes["test_model"], agent_classes["partner_model"])
            if index == "0"
            else (agent_classes["partner_model"], agent_classes["test_model"])
        )
    else:
        agent_classes_list = ("", "")
    episode_key = (env_id, tuple(agent_ids), models_list, agent_classes_list)
    return episode_dict.get(episode_key, False)


def run_async_benchmark_in_batch(
    *,
    env_agent_combo_list: list[EnvAgentCombo[Observation, AgentAction]],
    batch_size: int = 1,
    tag: str = "",
    push_to_db: bool = False,
    verbose: bool = False,
) -> None:
    env_agent_combo_batch: list[EnvAgentCombo[Observation, AgentAction]] = []
    loop = asyncio.get_event_loop()
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
                if pk is not None:
                    EpisodeLog.delete(pk)


def benchmark_display(
    model_list: list[str] = default_model_list,
    partner_model: str = "together_ai/meta-llama/Llama-3-70b-chat-hf",
    evaluator_model: str = "gpt-4o",
    task: str = "hard",
    output_to_jsonl: bool = False,
    save_dir: str = ".",
    agent_class: str = "",
    tag: str = "",
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Usage: sotopia benchmark-display --model-list gpt-4o --model-list together_ai/meta-llama-Llama-3-70b-chat-hf
    Aggregate all the results for the benchmark, as described in https://github.com/sotopia-lab/sotopia-space/blob/main/data_dir/models_vs_gpt35.jsonl
    """

    print(f"Displaying performance for {model_list} vs {partner_model} on task {task}")
    model_rewards_dict = dict()
    for model in model_list:
        episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()
        if len(episodes) == 0:
            print(f"No episodes found for {model}")
            continue
        # Get test model results
        test_model_rewards = get_avg_reward(episodes, model, agent_class)  # type: ignore
        # Get partner model results
        partner_model_rewards = get_avg_reward(
            episodes,  # type: ignore
            partner_model,
            LLMAgent.__name__,
        )
        model_rewards_dict[f"{model} (test) {evaluator_model} as the evaluator"] = (
            test_model_rewards
        )
        model_rewards_dict[
            f"{partner_model} (partner) {evaluator_model} as the evaluator"
        ] = partner_model_rewards
        rich.print(model_rewards_dict)
    if model_rewards_dict:
        display_in_table(model_rewards_dict)
        if output_to_jsonl:
            save_to_jsonl(model_rewards_dict, partner_model, save_dir)
    else:
        print("No episodes found for any model")
    return model_rewards_dict


def _list_all_env_agent_combo_not_in_db(
    model_names: dict[str, str],
    env_agent_combo_storage_index_list: list[tuple[EnvAgentComboStorage, str]],
    tag: str = "",
    task: str = "",
    agent_class: type[LLMAgent] = LLMAgent,
) -> list[EnvAgentCombo[Observation, AgentAction]]:
    """Iterate over each environment and return the first env-agent combo not in the database."""
    assert tag, "tag should not be empty"

    # Preprocess the episodes for fast lookup
    episode_list_with_tag: list[EpisodeLog] = EpisodeLog.find(
        EpisodeLog.tag == tag
    ).all()  # type: ignore
    episode_dict = preprocess_episode_data(episode_list_with_tag)
    list_of_env_agent_combo_storage = []
    for env_agent_combo_storage, index in env_agent_combo_storage_index_list:
        agent_ids = env_agent_combo_storage.agent_ids
        env_id = env_agent_combo_storage.env_id
        if check_existing_episodes(
            env_id=env_id,
            agent_ids=agent_ids,
            models=model_names,
            index=index,
            episode_dict=episode_dict,
            agent_classes={
                "test_model": agent_class.__name__,
                "partner_model": LLMAgent.__name__,
            },
        ):
            logging.info(
                f"Episode for {env_id} with agents {agent_ids} using {list(model_names.values())} already exists"
            )
            continue
        env_profile = EnvironmentProfile.get(env_id)
        env = ParallelSotopiaEnv(
            env_profile=env_profile,
            action_order="round-robin",
            model_name=model_names["env"],
            evaluators=[
                RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
            ],
            terminal_evaluators=[
                EpisodeLLMEvaluator(
                    model_names["env"],
                    EvaluationForAgents[SotopiaDimensions],
                ),
            ],
        )
        agent_profiles = [AgentProfile.get(id) for id in agent_ids]
        agents = [
            agent_class(agent_profile=agent_profile, model_name=agent_model)
            if (index == "0" and i == 0) or (index == "1" and i == 1)
            else LLMAgent(agent_profile=agent_profile, model_name=agent_model)
            for i, (agent_profile, agent_model) in enumerate(
                zip(
                    agent_profiles,
                    [model_names["test_model"], model_names["partner_model"]]
                    if index == "0"
                    else [model_names["partner_model"], model_names["test_model"]],
                )
            )
        ]
        list_of_env_agent_combo_storage.append((env, agents))
    return list_of_env_agent_combo_storage  # type: ignore


def display_in_table(
    model_rewards_dict: dict[str, dict[str, tuple[float, float]]],
) -> None:
    # Group models by their type (test/partner)
    test_models = {k: v for k, v in model_rewards_dict.items() if "(test)" in k}
    partner_models = {k: v for k, v in model_rewards_dict.items() if "(partner)" in k}

    # Create a table for test models
    if test_models:
        test_table = rich.table.Table(
            title="Test Model Performance (+/- CI bounds)",
            show_header=True,
            header_style="bold magenta",
        )
        test_table.add_column("Model")
        for dimension in list(test_models.values())[0].keys():
            if dimension not in ["setting_num", "episode_count"]:
                test_table.add_column(dimension)
        test_table.add_column("Settings")
        test_table.add_column("Episodes")

        for model, rewards in test_models.items():
            test_table.add_row(
                model.replace(" (test)", ""),
                *(
                    [
                        f"{rewards[k][0]:.2f} ± {rewards[k][1]:.2f}"
                        for k in rewards.keys()
                        if k not in ["setting_num", "episode_count"]
                    ]
                    + [
                        f"{rewards['setting_num'][0]:.0f}",
                        f"{rewards['episode_count'][0]:.0f}",
                    ]
                ),
            )
        rich.print(test_table)

    # Create a table for partner models
    if partner_models:
        partner_table = rich.table.Table(
            title="Partner Model Performance (+/- CI bounds)",
            show_header=True,
            header_style="bold cyan",
        )
        partner_table.add_column("Model")
        for dimension in list(partner_models.values())[0].keys():
            if dimension not in ["setting_num", "episode_count"]:
                partner_table.add_column(dimension)
        partner_table.add_column("Settings")
        partner_table.add_column("Episodes")

        for model, rewards in partner_models.items():
            partner_table.add_row(
                model.replace(" (partner)", ""),
                *(
                    [
                        f"{rewards[k][0]:.2f} ± {rewards[k][1]:.2f}"
                        for k in rewards.keys()
                        if k not in ["setting_num", "episode_count"]
                    ]
                    + [
                        f"{rewards['setting_num'][0]:.0f}",
                        f"{rewards['episode_count'][0]:.0f}",
                    ]
                ),
            )
        rich.print(partner_table)


def save_to_jsonl(
    model_rewards_dict: dict[str, dict[str, tuple[float, float]]],
    partner_model: str,
    save_dir: str,
) -> None:
    simplified_model_name = partner_model.split("/")[-1]
    output_fn = os.path.join(save_dir, f"models_vs_{simplified_model_name}.jsonl")
    outputs: list[str] = []
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


def _benchmark_impl(
    models: list[str] = default_model_list,
    agent_class: type[LLMAgent] = LLMAgent,
    partner_model: str = "together_ai/meta-llama/Llama-3-70b-chat-hf",
    evaluator_model: str = "gpt-4o",
    batch_size: int = 10,
    task: str = "hard",
    url: str = "",
    print_logs: bool = False,
    only_show_performance: bool = False,
    output_to_jsonl: bool = False,
    push_to_db: bool = False,
    save_dir: str = ".",
    tag: str = "",
) -> None:
    """Internal implementation of benchmark logic."""
    _set_up_logs(print_logs=print_logs)
    benchmark_combo = initialize_benchmark_combo(url)
    if task == "hard":
        hard_envs = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").environments
        agent_index = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").agent_index
        assert isinstance(agent_index, list), "agent_index should be a list"
        env_agent_combo_storage_index_list = []
        for env_id, index in zip(hard_envs, agent_index):
            for env_agent_combo_storage in benchmark_combo:
                if env_agent_combo_storage.env_id == env_id:
                    env_agent_combo_storage_index_list.append(
                        (env_agent_combo_storage, index)
                    )
    elif task == "cooperative":
        # Filter environments that have "mutual" in the codename
        cooperative_combo = []
        for env_agent_combo_storage in benchmark_combo:
            env_id = env_agent_combo_storage.env_id
            try:
                env_profile = EnvironmentProfile.get(env_id)
                if "mutual" in env_profile.codename.lower():
                    cooperative_combo.append(env_agent_combo_storage)
            except Exception as e:
                logging.warning(f"Failed to get environment profile for {env_id}: {e}")

        # Use the filtered environments for cooperative tasks
        benchmark_combo = cooperative_combo
        env_agent_combo_storage_index_list = [
            (env_agent_combo_storage, "0")
            for env_agent_combo_storage in benchmark_combo
        ] + [
            (env_agent_combo_storage, "1")
            for env_agent_combo_storage in benchmark_combo
        ]
    elif task == "competitive":
        competitive_combo = []
        for env_agent_combo_storage in benchmark_combo:
            env_id = env_agent_combo_storage.env_id
            try:
                env_profile = EnvironmentProfile.get(env_id)
                if "craigslist" in env_profile.codename.lower():
                    competitive_combo.append(env_agent_combo_storage)
            except Exception as e:
                logging.warning(f"Failed to get environment profile for {env_id}: {e}")
        benchmark_combo = competitive_combo
        env_agent_combo_storage_index_list = [
            (env_agent_combo_storage, "0")
            for env_agent_combo_storage in benchmark_combo
        ] + [
            (env_agent_combo_storage, "1")
            for env_agent_combo_storage in benchmark_combo
        ]
    else:
        env_agent_combo_storage_index_list = [
            (env_agent_combo_storage, "1")
            for env_agent_combo_storage in benchmark_combo
        ] + [
            (env_agent_combo_storage, "0")
            for env_agent_combo_storage in benchmark_combo
        ]
    if models is None:
        models = default_model_list
    for model in models:
        typer.echo(
            f"Running benchmark for {model} chatting with {partner_model} on task {task} with {evaluator_model} as the evaluator."
        )
        if partner_model == model and agent_class.__name__ == LLMAgent.__name__:
            typer.echo(
                typer.style(
                    "Partner model and test model, and their agent classes are the same. Please use different models.",
                    fg=typer.colors.RED,
                    bold=True,
                )
            )
            continue
        tag = (
            f"benchmark_{model}_{partner_model}_{evaluator_model}_{task}_trial0"
            if tag == ""
            else tag
        )
        typer.echo(typer.style(f"Tag: {tag}", fg=typer.colors.GREEN, bold=True))
        model_names = {
            "env": evaluator_model,
            "test_model": model,
            "partner_model": partner_model,
        }
        env_agent_combo_list = _list_all_env_agent_combo_not_in_db(
            model_names=model_names,
            tag=tag,
            env_agent_combo_storage_index_list=env_agent_combo_storage_index_list,
            task=task,
            agent_class=agent_class,
        )
        number_of_fix_turns = 0
        while True:
            run_async_benchmark_in_batch(
                env_agent_combo_list=env_agent_combo_list,
                batch_size=batch_size,
                tag=tag,
                verbose=False,
                push_to_db=push_to_db,
            )
            env_agent_combo_list = _list_all_env_agent_combo_not_in_db(
                model_names=model_names,
                tag=tag,
                env_agent_combo_storage_index_list=env_agent_combo_storage_index_list,
                task=task,
                agent_class=agent_class,
            )
            number_of_fix_turns += 1
            if len(env_agent_combo_list) == 0 or number_of_fix_turns >= 5:
                break

        benchmark_display(
            [model],
            partner_model,
            evaluator_model,
            task,
            output_to_jsonl=output_to_jsonl,
            agent_class=agent_class.__name__,
            tag=tag,
        )


@app.command()
def benchmark(
    models: Annotated[
        list[str] | None,
        typer.Option(
            "--models",
            "-m",
            help=f"Language models to benchmark (default: {default_model_list})",
        ),
    ] = None,
    partner_model: Annotated[
        str,
        typer.Option(help="The partner model you want to use."),
    ] = "together_ai/meta-llama/Llama-3-70b-chat-hf",
    evaluator_model: Annotated[
        str,
        typer.Option(help="The evaluator model you want to use."),
    ] = "gpt-4o",
    batch_size: Annotated[
        int,
        typer.Option(help="The batch size you want to use."),
    ] = 10,
    task: Annotated[
        str,
        typer.Option(help="The task id you want to benchmark."),
    ] = "hard",
    url: Annotated[
        str,
        typer.Option(help="The url to fetch the benchmark combo."),
    ] = "",
    print_logs: Annotated[
        bool,
        typer.Option(help="Print logs."),
    ] = False,
    only_show_performance: Annotated[
        bool,
        typer.Option(help="Only show performance."),
    ] = False,
    output_to_jsonl: Annotated[
        bool,
        typer.Option(help="Output to jsonl."),
    ] = False,
    push_to_db: Annotated[
        bool,
        typer.Option(help="Push to db."),
    ] = False,
    save_dir: Annotated[
        str,
        typer.Option(help="The directory to save the output."),
    ] = ".",
    tag: Annotated[
        str,
        typer.Option(help="The tag for the experiment."),
    ] = "",
) -> None:
    """Run sotopia benchmark using LLMAgent (CLI interface)."""
    # Handle default for models (can't use default with list[str] in typer.Option)
    if models is None:
        models = default_model_list

    # Call the implementation with LLMAgent hard-coded
    _benchmark_impl(
        models=models,
        agent_class=LLMAgent,  # Hard-coded for CLI
        partner_model=partner_model,
        evaluator_model=evaluator_model,
        batch_size=batch_size,
        task=task,
        url=url,
        print_logs=print_logs,
        only_show_performance=only_show_performance,
        output_to_jsonl=output_to_jsonl,
        push_to_db=push_to_db,
        save_dir=save_dir,
        tag=tag,
    )
