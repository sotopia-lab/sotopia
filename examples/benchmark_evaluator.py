from evaluate_existing_episode import run_async_server_in_batch_aevaluate
from sotopia.database import (
    human_annotation_to_episodelog,
    AnnotationForEpisode,
    EpisodeLog,
)
from typer import Typer
import numpy as np
import pandas as pd
import scipy.stats as stats

app = Typer()

target_model_patterns: list[list[str]] = [
    ["gpt-4", "gpt-4", "gpt-3.5-turbo"],
    ["gpt-4", "gpt-3.5-turbo", "gpt-4"],
    ["gpt-4", "gpt-3.5-turbo", "togethercomputer/llama-2-70b-chat"],
    ["gpt-4", "togethercomputer/llama-2-70b-chat", "gpt-3.5-turbo"],
]


def get_human_annotations(
    target_model_patterns: list[list[str]],
) -> list[AnnotationForEpisode]:
    human_episodes: list[AnnotationForEpisode] = []
    for pk in AnnotationForEpisode.all_pks():
        episode_human = AnnotationForEpisode.get(pk)
        episode_model = EpisodeLog.get(episode_human.episode)
        if episode_model.models in target_model_patterns:
            human_episodes.append(episode_human)
    return human_episodes


def get_dimension_correlation(
    human_annotations: list[EpisodeLog],
    machine_annotations: list[EpisodeLog],
    dimension: str,
) -> dict[str, float]:
    # check the data is present
    with_dimension_list = [
        int(not isinstance(relevant_episode.rewards[0], float))
        for relevant_episode in machine_annotations
    ]
    assert sum(with_dimension_list) == len(human_annotations), "Data is missing"
    if dimension == "overall":
        dimension_scores_agent1_human = [
            relevant_episode.rewards[0][0] for relevant_episode in human_annotations
        ]  # type: ignore
        dimension_scores_agent2_human = [
            relevant_episode.rewards[1][0] for relevant_episode in human_annotations
        ]  # type: ignore
        dimension_scores_agent1_machine = [
            relevant_episode.rewards[0][0] for relevant_episode in machine_annotations
        ]  # type: ignore
        dimension_scores_agent2_machine = [
            relevant_episode.rewards[1][0] for relevant_episode in machine_annotations
        ]  # type: ignore
    else:
        dimension_scores_agent1_human = [
            relevant_episode.rewards[0][1][dimension]
            for relevant_episode in human_annotations
        ]  # type: ignore
        dimension_scores_agent2_human = [
            relevant_episode.rewards[1][1][dimension]
            for relevant_episode in human_annotations
        ]  # type: ignore
        dimension_scores_agent1_machine = [
            relevant_episode.rewards[0][1][dimension]
            for relevant_episode in machine_annotations
        ]  # type: ignore
        dimension_scores_agent2_machine = [
            relevant_episode.rewards[1][1][dimension]
            for relevant_episode in machine_annotations
        ]  # type: ignore
    x = dimension_scores_agent1_human + dimension_scores_agent2_human
    y = dimension_scores_agent1_machine + dimension_scores_agent2_machine
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


@app.command()
def evaluate_evaluator(
    batch_size: int = 10,
    model: str = "gpt-4",
    tag: str = "reeval_gpt4",
    push_to_db: bool = False,
    verbose: bool = False,
) -> None:
    relevant_dimension = [
        "believability",
        "relationship",
        "knowledge",
        "secret",
        "social_rules",
        "financial_and_material_benefits",
        "goal",
    ]
    human_annotations = get_human_annotations(target_model_patterns)
    human_annotation_dict = human_annotation_to_episodelog(
        human_annotations, return_model_episodes=False, aggregate=True
    )
    re_annotate_list = list(human_annotation_dict.keys())
    aggregate_human_annotations: list[EpisodeLog] = list(human_annotation_dict.values())  # type: ignore
    # Call the function with the specified parameters

    re_evaluate_episodes: list[EpisodeLog] = EpisodeLog.find(
        EpisodeLog.tag == tag
    ).all()  # type: ignore
    if len(re_evaluate_episodes) < len(re_annotate_list):
        run_async_server_in_batch_aevaluate(
            tag=tag,
            model=model,  # type: ignore
            batch_size=batch_size,
            push_to_db=push_to_db,
            verbose=verbose,
            reeval_list=re_annotate_list,
        )
    else:
        valid_episodes = [
            not isinstance(relevant_episode.rewards[0], float)
            for relevant_episode in re_evaluate_episodes
        ]
        re_annotate_list = [
            episode.pk
            for valid, episode in zip(valid_episodes, re_evaluate_episodes)
            if not valid
        ]  # type: ignore
        while re_annotate_list:
            run_async_server_in_batch_aevaluate(
                tag=tag,
                model=model,  # type: ignore
                batch_size=batch_size,
                push_to_db=push_to_db,
                verbose=verbose,
                reeval_list=re_annotate_list,
            )
            re_annotate_list = []
            re_evaluate_episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()  # type: ignore
            valid_episodes = [
                not isinstance(relevant_episode.rewards[0], float)
                for relevant_episode in re_evaluate_episodes
            ]
            for valid, episode in zip(valid_episodes, re_evaluate_episodes):
                if not valid:
                    pk = episode.pk
                    assert isinstance(pk, str)
                    EpisodeLog.delete(pk)
                    re_annotate_list.append(pk)

    correlation_list = []
    ordered_re_eval_episodes = []

    for human_annotated_episode in aggregate_human_annotations:
        for re_eval_episode in re_evaluate_episodes:
            assert isinstance(re_eval_episode, EpisodeLog)
            if (
                human_annotated_episode.environment == re_eval_episode.environment
                and human_annotated_episode.agents == re_eval_episode.agents
                and human_annotated_episode.models[1:] == re_eval_episode.models[1:]  # type: ignore
            ):
                ordered_re_eval_episodes.append(re_eval_episode)
                break

    for dimension in relevant_dimension:
        correlation = get_dimension_correlation(
            aggregate_human_annotations, ordered_re_eval_episodes, dimension
        )
        correlation_list.append(correlation)
    print("Correlation between human and machine")
    print(pd.DataFrame(correlation_list, index=relevant_dimension))


if __name__ == "__main__":
    app()
