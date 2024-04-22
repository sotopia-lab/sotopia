import logging
import sys
from collections import defaultdict
from typing import List

import pandas as pd
import rich

from sotopia.database.env_agent_combo_storage import (
    EnvAgentComboStorage,
)
from sotopia.database.logs import EpisodeLog
from sotopia.database.persistent_profile import EnvironmentProfile


def get_avg_reward_for_models(episodes: List[EpisodeLog]) -> pd.DataFrame:
    """
    Get the average reward for each model in the episodes.

    Args:
        episodes (List[EpisodeLog]): A list of episodes.

    Returns:
        pd.DataFrame: A DataFrame mapping model names to average rewards.
    """
    model_rewards = defaultdict(list)
    model_rewards_avg = {}

    for episode in episodes:
        assert isinstance(episode.models, list), "models should be a list"
        for idx, model in enumerate(episode.models[1:]):  # skip env
            if isinstance(episode.rewards[idx], tuple):
                model_rewards[model].append(episode.rewards[idx])
            else:
                print(episode.pk)

    for model, rewards in model_rewards.items():
        model_rewards[model] = [reward[1] for reward in rewards]
        model_rewards_avg[model] = (
            pd.DataFrame.from_dict(model_rewards[model]).mean(axis=0).to_dict()
        )

    return pd.DataFrame.from_dict(model_rewards_avg)


def get_avg_success_rate_for_models(
    episodes: List[EpisodeLog],
) -> pd.DataFrame:
    """
    Get the average success rate for each model in the episodes.

    Args:
        episodes (List[EpisodeLog]): A list of episodes.

    Returns:
        pd.DataFrame: A DataFrame with model names and their corresponding average success rates.
    """
    model_rewards = defaultdict(list)
    model_success_rate_avg = {}

    for episode in episodes:
        assert isinstance(episode.models, list), "models should be a list"
        for idx, model in enumerate(episode.models[1:]):  # skip env
            if isinstance(episode.rewards[idx], tuple):
                model_rewards[model].append(episode.rewards[idx])
            else:
                print(episode.pk)

    for model, rewards in model_rewards.items():
        model_success_rate_avg[model] = [reward[1] for reward in rewards]
        model_success_rate_avg[model] = pd.DataFrame.from_dict(
            model_success_rate_avg[model]
        )

    assert len(model_success_rate_avg) == 2, "There should be two models"

    model_list = list(model_success_rate_avg.keys())
    model_one_success_rate = (
        model_success_rate_avg[model_list[0]] > model_success_rate_avg[model_list[1]]
    )
    model_two_success_rate = (
        model_success_rate_avg[model_list[0]] < model_success_rate_avg[model_list[1]]
    )
    model_on_par_success_rate = (
        model_success_rate_avg[model_list[0]] == model_success_rate_avg[model_list[1]]
    )

    return pd.DataFrame.from_dict(
        {
            model_list[0]: model_one_success_rate.mean(axis=0).to_dict(),
            "on_par": model_on_par_success_rate.mean(axis=0).to_dict(),
            model_list[1]: model_two_success_rate.mean(axis=0).to_dict(),
        }
    )


def is_symmetric_epilogs(epilogs: List[EpisodeLog]) -> bool:
    """
    Check if the given episode logs are symmetric.

    Args:
        epilogs (List[EpisodeLog]): A list of episode logs.

    Returns:
        bool: True if the episode logs are symmetric, False otherwise.
    """
    asymmetric_epilogs = []
    gpt35_llama2_epilogs_dict = {}

    for ep in epilogs:
        assert isinstance(ep.models, list), "models should be a list"
        hash_key = (
            ep.environment,
            ep.agents[0],
            ep.agents[1],
            ep.models[0],
            ep.models[1],
            ep.models[2],
        )
        gpt35_llama2_epilogs_dict[hash_key] = ep.pk

    for hash_key in gpt35_llama2_epilogs_dict:
        reverse_key = (
            hash_key[0],
            hash_key[1],
            hash_key[2],
            hash_key[3],
            hash_key[5],
            hash_key[4],
        )
        if reverse_key not in gpt35_llama2_epilogs_dict:
            asymmetric_epilogs.append(gpt35_llama2_epilogs_dict[hash_key])

    if not asymmetric_epilogs:
        return True
    else:
        logging.warning(
            f"Found {len(asymmetric_epilogs)} asymmetric epilogs: {asymmetric_epilogs}"
        )
        return False


def extract_fixed_episode_set(
    episodes: List[EpisodeLog], models: List[str]
) -> List[EpisodeLog]:
    """
    Extract a fixed set of episodes based on the given models.

    Args:
        episodes (List[EpisodeLog]): A list of episodes.
        models (List[str]): A list of models.

    Returns:
        List[EpisodeLog]: A list of filtered episodes.
    """
    env_ids = list(EnvironmentProfile.all_pks())
    fixed_env_agent_combo = []

    for env_id in env_ids:
        assert env_id is not None, "env_id should not be None"
        env_agent_combo_storage_list = list(
            EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env_id).all()
        )
        fixed_env_agent_combo += env_agent_combo_storage_list[:5]

    print("Number of fixed env_agent_combo:", len(fixed_env_agent_combo))

    filtered_episodes = []
    filtered_episodes_verse = []
    fixed_env_agent_combo_pair_one = fixed_env_agent_combo.copy()
    fixed_env_agent_combo_pair_two = fixed_env_agent_combo.copy()

    for episode in episodes:
        assert isinstance(episode, EpisodeLog), "episode should be an EpisodeLog"
        for combo in fixed_env_agent_combo_pair_one:
            assert isinstance(
                combo, EnvAgentComboStorage
            ), "combo should be an EnvAgentComboStorage"
            if (
                episode.environment == combo.env_id
                and episode.agents == combo.agent_ids
                and isinstance(episode.rewards[0], tuple)
            ):
                if episode.models == models:
                    filtered_episodes.append(episode)
                    fixed_env_agent_combo_pair_one.remove(combo)
                break

    for episode in episodes:
        assert isinstance(episode, EpisodeLog), "episode should be an EpisodeLog"
        for combo in fixed_env_agent_combo_pair_two:
            assert isinstance(
                combo, EnvAgentComboStorage
            ), "combo should be an EnvAgentComboStorage"
            if (
                episode.environment == combo.env_id
                and episode.agents == combo.agent_ids
                and isinstance(episode.rewards[0], tuple)
            ):
                if episode.models == [models[0], models[2], models[1]]:
                    filtered_episodes_verse.append(episode)
                    fixed_env_agent_combo_pair_two.remove(combo)
                break

    filtered_episodes += filtered_episodes_verse
    return filtered_episodes


def main(episodes_to_evaluate: str, model_1: str, model_2: str) -> None:
    """
    Main function to evaluate the episodes.

    Args:
        episodes_to_evaluate (str): Tag for the episodes to evaluate.
        model_1 (str): First model identifier.
        model_2 (str): Second model identifier.

    Outputs analysis results to the console using rich formatting.
    """
    Episodes = EpisodeLog.find(EpisodeLog.tag == episodes_to_evaluate).all()
    print("Number of episodes:", len(Episodes))

    models = [
        "gpt-4",
        model_1,
        model_2,
    ]

    print(f"Evaluating model 1: {models[1]}, model 2: {models[2]}")

    filtered_episodes = extract_fixed_episode_set(Episodes, models=models)
    print("Number of filtered episodes:", len(filtered_episodes))

    avg_rewards = get_avg_reward_for_models(filtered_episodes)
    avg_rewards = avg_rewards.reindex(
        [
            "believability",
            "relationship",
            "knowledge",
            "secret",
            "social_rules",
            "financial_and_material_benefits",
            "goal",
            "overall_score",
        ]
    )
    rich.print(avg_rewards)

    if models[1] != models[2]:
        avg_success_rate = get_avg_success_rate_for_models(filtered_episodes)
        rich.print(avg_success_rate)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <episodes_to_evaluate> <model_1> <model_2>")
        sys.exit(1)

    episodes_to_evaluate = sys.argv[1]
    model_1 = sys.argv[2]
    model_2 = sys.argv[3]
    assert isinstance(
        episodes_to_evaluate, str
    ), "episodes_to_evaluate should be a string"

    main(episodes_to_evaluate, model_1, model_2)
