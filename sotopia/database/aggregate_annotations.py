from sotopia.database import EpisodeLog, AnnotationForEpisode
from typing import List, Tuple, Dict
from copy import deepcopy
from collections import defaultdict


def aggregate_reasoning(reasonings: List[str]) -> str:
    return "\n".join(reasonings)


def aggregate_rewards(
    rewards: List[Tuple[float, Dict[str, float]]],
) -> Tuple[float, Dict[str, float]]:
    def average_dict(item: List[Dict[str, float]]) -> Dict[str, float]:
        keys = item[0].keys()
        return {k: sum(d[k] for d in item) / len(item) for k in keys}

    def average_list(item: List[float]) -> float:
        return sum(item) / len(item)

    ret_rewards = (
        average_list([r[0] for r in rewards]),
        average_dict([r[1] for r in rewards]),
    )
    return ret_rewards


def human_annotation_to_episodelog(
    human_annotation: list[AnnotationForEpisode],
    return_model_episodes: bool = False,
    aggregate: bool = False,
) -> dict[str, EpisodeLog | tuple[EpisodeLog, EpisodeLog]]:
    """
    retrieve related episodes and return the {pk: EpisodeLog}
    if return_model_episodes is True, return {pk: (human_episode, model_episode)}
    if aggregate==False, the primary key here is the AnnotationForEpisode's episode
    if aggregate==True, the primary key here is the model episode
    """

    model_human_pk_mapping: Dict[str, List[str]] = defaultdict(list)
    for annotation in human_annotation:
        model_episode_pk = annotation.episode
        human_episode_pk = annotation.pk
        assert human_episode_pk is not None
        model_human_pk_mapping[model_episode_pk].append(human_episode_pk)

    ep_dict: Dict[str, EpisodeLog | Tuple[EpisodeLog, EpisodeLog]] = {}

    if aggregate:
        for model_episode_pk, human_episode_pks in model_human_pk_mapping.items():
            all_human_rewards = []
            all_human_reasonings = []

            for human_episode_pk in human_episode_pks:
                annotation = AnnotationForEpisode.get(pk=human_episode_pk)

                human_reasoning = annotation.reasoning
                human_rewards = annotation.rewards

                if any(isinstance(r, float) for r in human_rewards):
                    print("Rewards are not in the correct format")
                    print(human_rewards)
                    continue

                all_human_rewards.append(human_rewards)
                all_human_reasonings.append(human_reasoning)

            episode = EpisodeLog.get(pk=model_episode_pk)
            human_episode = deepcopy(episode)
            human_reasoning = aggregate_reasoning(all_human_reasonings)

            human_rewards = [
                aggregate_rewards([r[0] for r in all_human_rewards]),  # type: ignore
                aggregate_rewards([r[1] for r in all_human_rewards]),  # type: ignore
            ]
            if human_reasoning:
                human_episode.reasoning = human_reasoning
            if human_rewards:
                human_episode.rewards = human_rewards

            if return_model_episodes:
                ep_dict[model_episode_pk] = (human_episode, episode)
            else:
                ep_dict[model_episode_pk] = human_episode
        return ep_dict

    for annotation in human_annotation:
        episode_pk = annotation.episode
        episode = EpisodeLog.get(pk=episode_pk)
        human_episode_pk = annotation.pk
        assert human_episode_pk is not None

        human_reasoning = annotation.reasoning
        human_rewards = annotation.rewards

        human_episode = deepcopy(episode)
        human_episode.reasoning = human_reasoning
        human_episode.rewards = human_rewards

        if return_model_episodes:
            ep_dict[human_episode_pk] = (human_episode, episode)
        else:
            ep_dict[human_episode_pk] = human_episode

    return ep_dict
