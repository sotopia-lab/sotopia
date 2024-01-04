from collections import defaultdict
from typing import cast

import numpy as np
import pandas as pd
import rich
from scipy import stats

from sotopia.database import EpisodeLog, AnnotationForEpisode


def get_dimension_correlation(dimension: str) -> dict[str, float]:
    annotated_episodes = [AnnotationForEpisode.get(annotation_pk) for annotation_pk in AnnotationForEpisode.all_pks()]
    relevant_episode_ids = [annotation.episode for annotation in annotated_episodes]

    # get relevant episodes
    relevant_episodes = [
        EpisodeLog.get(relevant_episode_id)
        for relevant_episode_id in relevant_episode_ids
    ]

    # search for the corresponding episode in tag "reeval_gpt4_improved_prompts"
    tagged_episodes = EpisodeLog.find(EpisodeLog.tag == "reeval_llama2").all()
    ordered_tagged_episodes = []
    for relevant_episode in relevant_episodes:
        for tagged_episode in tagged_episodes:
            assert isinstance(tagged_episode, EpisodeLog)
            if (
                relevant_episode.environment == tagged_episode.environment
                and relevant_episode.agents == tagged_episode.agents
                and relevant_episode.models == tagged_episode.models
            ):
                ordered_tagged_episodes.append(tagged_episode)
                break
    relevant_episodes = ordered_tagged_episodes

    # check the data is present
    with_dimension_list = [
        not isinstance(relevant_episode.rewards[0], float)
        for relevant_episode in relevant_episodes
    ]

    # list of episodes for which dimension is present
    relevant_episodes_with_dimension = [
        relevant_episode
        for relevant_episode, with_dimension in zip(
            relevant_episodes, with_dimension_list
        )
        if with_dimension
    ]

    for annotation in annotated_episodes:
        human_rewards = annotation.rewards
        human_rewards_list: list[tuple[float, float]] = []
        
        human_rewards_list.append(
            (
                float(human_rewards[0][1][dimension]), # type: ignore
                float(human_rewards[1][1][dimension]), # type: ignore
            )
        )
    dimension_scores_agent1 = [human_rewards[0] for human_rewards in human_rewards_list]
    dimension_scores_agent2 = [human_rewards[1] for human_rewards in human_rewards_list]
       
    dimension_machine = dimension
    if dimension == "financial":
        dimension_machine = "financial_and_material_benefits"
    elif dimension == "socialrules":
        dimension_machine = "social_rules"

    if dimension == "overall":
        dimension_scores_agent1_machine = [relevant_episode.rewards[0][0] for relevant_episode in relevant_episodes_with_dimension]  # type: ignore
        dimension_scores_agent2_machine = [relevant_episode.rewards[1][0] for relevant_episode in relevant_episodes_with_dimension]  # type: ignore
    else:
        dimension_scores_agent1_machine = [relevant_episode.rewards[0][1][dimension_machine] for relevant_episode in relevant_episodes_with_dimension]  # type: ignore
        dimension_scores_agent2_machine = [relevant_episode.rewards[1][1][dimension_machine] for relevant_episode in relevant_episodes_with_dimension]  # type: ignore
    x = dimension_scores_agent2 + dimension_scores_agent1
    y = dimension_scores_agent2_machine + dimension_scores_agent1_machine
    # average
    # x = [agent1 + agent2 for agent1, agent2 in zip(dimension_scores_agent1, dimension_scores_agent2)]
    # y = [agent1 + agent2 for agent1, agent2 in zip(dimension_scores_agent1_machine, dimension_scores_agent2_machine)]
    res = stats.pearsonr(x, y)
    spearman_res = stats.spearmanr(x, y)
    mse = ((np.array(x) - np.array(y)) ** 2).mean()

    return {
        "pearson_correlation": res.statistic,
        "pearson_pvalue": res.pvalue,
        "spearman_correlation": spearman_res.correlation,
        "spearman_pvalue": spearman_res.pvalue,
        "mse": mse,
    }

relevant_dimension = [
        "believability",
        "relationship",
        "knowledge",
        "secret",
        "socialrules",
        "financial",
        "goal",
    ]

correlation_dict = {}
for dimension in relevant_dimension:
    correlation_dict[dimension] = get_dimension_correlation(dimension)

