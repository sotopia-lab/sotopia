from collections import defaultdict
from typing import cast

import numpy as np
import pandas as pd
import rich
from scipy import stats

from sotopia.database import EpisodeLog, AnnotationForEpisode
from sotopia.envs.evaluators import EvaluationBySocialDimensions


def average_human_rewards(
        annotation_list: list[AnnotationForEpisode]
) -> list[tuple[float, dict[str, float]]]:
    # average human rewards
    if len(annotation_list) == 1:
        return annotation_list[0].rewards # type: ignore
    elif len(annotation_list) == 2:
        assert isinstance(annotation_list[0].rewards[0], tuple)
        assert isinstance(annotation_list[0].rewards[1], tuple)
        assert isinstance(annotation_list[1].rewards[0], tuple)
        assert isinstance(annotation_list[1].rewards[1], tuple)
        return [
            (
                (annotation_list[0].rewards[0][0] + annotation_list[1].rewards[0][0]) / 2,
                {
                    dimension: (
                        annotation_list[0].rewards[0][1][dimension]
                        + annotation_list[1].rewards[0][1][dimension]
                    ) / 2
                    for dimension in annotation_list[0].rewards[0][1]
                },
            ),
            (
                (annotation_list[0].rewards[1][0] + annotation_list[1].rewards[1][0]) / 2,
                {
                    dimension: (
                        annotation_list[0].rewards[1][1][dimension]
                        + annotation_list[1].rewards[1][1][dimension]
                    ) / 2
                    for dimension in annotation_list[0].rewards[1][1]
                },
            ),
        ] 
    else:
        raise NotImplementedError

                                                                               
def group_annotated_episodes(
    annotated_episodes: list[AnnotationForEpisode],
) -> dict[str, list[AnnotationForEpisode]]:
    # only keep eval for machine interactions
    grouped_episodes: dict[str, list[AnnotationForEpisode]] = defaultdict(list)
    for annotation in annotated_episodes:
        episode = EpisodeLog.get(annotation.episode)
        assert episode.models
        if 'human' in episode.models or 'redis' in episode.models:
            continue
        grouped_episodes[annotation.episode].append(annotation)
    return grouped_episodes


def get_dimension_correlation(dimension: str) -> dict[str, float]:
    annotated_episodes = [AnnotationForEpisode.get(annotation_pk) for annotation_pk in AnnotationForEpisode.all_pks()]

    grouped_episodes = group_annotated_episodes(annotated_episodes)
    relevant_episode_ids = grouped_episodes.keys()

    # get relevant episodes
    relevant_episodes = [
        EpisodeLog.get(relevant_episode_id)
        for relevant_episode_id in relevant_episode_ids
    ]

    # search for the corresponding episode in tag "reeval_gpt4_improved_prompts"
    tagged_episodes = EpisodeLog.find(EpisodeLog.tag == "reeval_gpt4_turbo").all()
    ordered_tagged_episodes = []
    for relevant_episode in relevant_episodes:
        for tagged_episode in tagged_episodes:
            assert isinstance(tagged_episode, EpisodeLog)
            if (
                relevant_episode.environment == tagged_episode.environment
                and relevant_episode.agents == tagged_episode.agents
                and relevant_episode.models[1:] == tagged_episode.models[1:] # type: ignore
            ):
                assert isinstance(tagged_episode, EpisodeLog)
                tagged_episode.pk = relevant_episode.pk
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

    human_rewards_list: list[tuple[float, float]] = []
    for episode in relevant_episodes_with_dimension:
        annotation_list = grouped_episodes[episode.pk] # type: ignore
        human_rewards = average_human_rewards(annotation_list)
        human_rewards_list.append(
            (
                float(human_rewards[0][1][dimension]), # type: ignore
                float(human_rewards[1][1][dimension]), # type: ignore
            )
        )
    dimension_scores_agent1 = [human_rewards[0] for human_rewards in human_rewards_list]
    dimension_scores_agent2 = [human_rewards[1] for human_rewards in human_rewards_list]
       
    if dimension == "overall":
        dimension_scores_agent1_machine = [relevant_episode.rewards[0][0] for relevant_episode in relevant_episodes_with_dimension]  # type: ignore
        dimension_scores_agent2_machine = [relevant_episode.rewards[1][0] for relevant_episode in relevant_episodes_with_dimension]  # type: ignore
    else:
        dimension_scores_agent1_machine = [relevant_episode.rewards[0][1][dimension] for relevant_episode in relevant_episodes_with_dimension]  # type: ignore
        dimension_scores_agent2_machine = [relevant_episode.rewards[1][1][dimension] for relevant_episode in relevant_episodes_with_dimension]  # type: ignore
    x = dimension_scores_agent2[:179] + dimension_scores_agent1[:179]
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

SOCIAL_DIMENSIONS: list[str] = list(EvaluationBySocialDimensions.__fields__.keys()) 

correlation_dict = {}
for dimension in SOCIAL_DIMENSIONS:
    correlation_dict[dimension] = get_dimension_correlation(dimension)
df = pd.DataFrame.from_dict(correlation_dict, orient="index")
df.to_csv("./logs/correlation.csv")
print(df) 

