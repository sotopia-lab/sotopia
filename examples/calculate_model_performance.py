import logging
import sys
from collections import Counter, defaultdict
from typing import cast

import pandas as pd
import rich

from sotopia.database.env_agent_combo_storage import (
    EnvAgentComboStorage,
)
from sotopia.database.logs import EpisodeLog
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipProfile,
)


def get_avg_reward_for_models(episodes: list[EpisodeLog]) -> pd.DataFrame:
    """Get the average reward for each model in the episodes.

    Args:
        episodes (list[EpisodeLog]): A list of episodes.

    Returns:
        dict[str, float]: A dictionary mapping model names to average rewards.
    """
    model_rewards = defaultdict(list)
    model_rewards_avg = {}
    for episode in episodes:
        assert isinstance(episode.models, list), "models should be a list"
        for idx, model in enumerate(episode.models[1:]):  # skip env
            if isinstance(episode.rewards[idx], tuple):
                model_rewards[model].append(episode.rewards[idx])
            else:
                # rich.print(episode.render_for_humans())
                print(episode.pk)
                # print(episode.rewards[idx])

    for model in model_rewards:
        model_rewards[model] = [rewards[1] for rewards in model_rewards[model]]  # type: ignore
        model_rewards_avg[model] = pd.DataFrame.from_dict(model_rewards[model])  # type: ignore
        model_rewards_avg[model] = model_rewards_avg[model].mean(axis=0).to_dict()  # type: ignore
    return pd.DataFrame.from_dict(model_rewards_avg)


def get_avg_successRate_for_models(
    episodes: list[EpisodeLog],
) -> dict[str, dict[str, float]]:
    """Get the average success rate for each model in the episodes.

    Args:
        episodes (list[EpisodeLog]): A list of episodes.

    Returns:
        dict[str, float]: A dictionary mapping model names to average success rates.
    """
    model_rewards = defaultdict(list)
    model_successRate_avg = {}
    for episode in episodes:
        assert isinstance(episode.models, list), "models should be a list"
        for idx, model in enumerate(episode.models[1:]):  # skip env
            if isinstance(episode.rewards[idx], tuple):
                model_rewards[model].append(episode.rewards[idx])
            else:
                print(episode.pk)

    for model in model_rewards:
        model_successRate_avg[model] = [rewards[1] for rewards in model_rewards[model]]  # type: ignore
        model_successRate_avg[model] = pd.DataFrame.from_dict(model_successRate_avg[model])  # type: ignore
    assert len(model_successRate_avg) == 2, "There should be two models"
    model_list = list(model_successRate_avg.keys())
    model_one_successRate = (
        model_successRate_avg[model_list[0]]
        > model_successRate_avg[model_list[1]]
    )
    model_two_successRate = (
        model_successRate_avg[model_list[0]]
        < model_successRate_avg[model_list[1]]
    )
    model_on_par_successRate = (
        model_successRate_avg[model_list[0]]
        == model_successRate_avg[model_list[1]]
    )
    assert isinstance(
        model_one_successRate, pd.DataFrame
    ), "model_one_successRate should be a dataframe"
    assert isinstance(
        model_two_successRate, pd.DataFrame
    ), "model_two_successRate should be a dataframe"
    assert isinstance(
        model_on_par_successRate, pd.DataFrame
    ), "model_on_par_successRate should be a dataframe"
    return pd.DataFrame.from_dict(
        {
            model_list[0]: model_one_successRate.mean(axis=0).to_dict(),
            "on_par": model_on_par_successRate.mean(axis=0).to_dict(),
            model_list[1]: model_two_successRate.mean(axis=0).to_dict(),
        }
    )


def is_symmetric_epilogs(epilogs: list[EpisodeLog]) -> bool:
    asymetric_epilogs = []
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
        if (
            hash_key[0],
            hash_key[1],
            hash_key[2],
            hash_key[3],
            hash_key[5],
            hash_key[4],
        ) not in gpt35_llama2_epilogs_dict:
            asymetric_epilogs.append(gpt35_llama2_epilogs_dict[hash_key])

    if len(asymetric_epilogs) == 0:
        return True
    else:
        logging.warning(
            f"Found {len(asymetric_epilogs)} asymetric epilogs: {asymetric_epilogs}"
        )
        return False


def extract_fixed_episode_set(
    episodes: list[EpisodeLog], models: list[str]
) -> list[EpisodeLog]:
    env_ids = list(EnvironmentProfile.all_pks())
    fixed_env_agent_combo = []
    for env_id in env_ids:
        assert env_id is not None, "env_id should not be None"
        env_agent_combo_storage_list = list(
            EnvAgentComboStorage.find(
                EnvAgentComboStorage.env_id == env_id
            ).all()
        )
        fixed_env_agent_combo += env_agent_combo_storage_list[:5]
    print("Number of fixed env_agent_combo:", len(fixed_env_agent_combo))
    # Filter out the episode logs that are not in the fixed set
    filtered_episodes = []
    filtered_episodes_verse = []
    fixed_env_agent_combo_pair_one = fixed_env_agent_combo.copy()
    fixed_env_agent_combo_pair_two = fixed_env_agent_combo.copy()
    for episode in Episodes:
        assert isinstance(
            episode, EpisodeLog
        ), "episode should be a EpisodeLog"
        for combo in fixed_env_agent_combo_pair_one:
            assert isinstance(
                combo, EnvAgentComboStorage
            ), "combo should be a EnvAgentComboStorage"
            if (
                episode.environment == combo.env_id
                and episode.agents == combo.agent_ids
                and isinstance(episode.rewards[0], tuple)
            ):
                if episode.models == models:
                    filtered_episodes.append(episode)
                    fixed_env_agent_combo_pair_one.remove(combo)
                break

    for episode in Episodes:
        assert isinstance(
            episode, EpisodeLog
        ), "episode should be a EpisodeLog"
        for combo in fixed_env_agent_combo_pair_two:
            assert isinstance(
                combo, EnvAgentComboStorage
            ), "combo should be a EnvAgentComboStorage"
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


episodes_to_evaluate = sys.argv[1]
assert isinstance(
    episodes_to_evaluate, str
), "episodes_to_evaluate should be a string"
Episodes = EpisodeLog.find(EpisodeLog.tag == episodes_to_evaluate).all()
# Episodes = EpisodeLog.find((EpisodeLog.tag == "gpt3.5_gpt4_v0.0.1_hzhu2") | (EpisodeLog.tag == "gpt4_gpt3.5_v0.0.1_hzhu2")).all()
print("Number of episodes:", len(Episodes))
filtered_episodes = extract_fixed_episode_set(Episodes, models=models)  # type: ignore
print("Number of filtered episodes:", len(filtered_episodes))
# check if the epilogs are symmetric
if is_symmetric_epilogs(filtered_episodes):
    avg_rewards = get_avg_reward_for_models(filtered_episodes)
    avg_rewards = avg_rewards.reindex(
        [
            "believability",
            "relationship",
            "knowledge",
            "secret",
            "social_rules",
            "financial_and_material_benefits",
            "overall_score",
        ]
    )
    rich.print(avg_rewards)
    if (
        episodes_to_evaluate.split("_")[0]
        != episodes_to_evaluate.split("_")[1]
    ):
        avg_successRate = get_avg_successRate_for_models(filtered_episodes)
        rich.print(avg_successRate)
