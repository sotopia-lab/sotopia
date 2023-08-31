from collections import defaultdict
from typing import cast

import numpy as np
import pandas as pd
import rich
from scipy import stats

from annotation.agreement import computeAlpha, computeFleissKappa
from sotopia.database import EpisodeLog


def _get_per_episode_reward_for_models(
    annotation_df: pd.DataFrame, relevant_dimension: list[str]
) -> dict[str, pd.DataFrame]:
    episode_ids = annotation_df["Input.episode_id"]
    episodes = [EpisodeLog.get(episode_id) for episode_id in episode_ids]
    model_rewards = defaultdict(list)  # type: ignore
    model_df_rewards = {}
    for episode in episodes:
        models = episode.models
        assert isinstance(models, list)
        assert len(models) == 3
        agent_1_dimension = [f"Answer.agent1_{i}" for i in relevant_dimension]
        model_rewards[models[1]] += (
            annotation_df[annotation_df["Input.episode_id"] == episode.pk][
                agent_1_dimension
            ]
            .rename(
                columns={
                    i: j for i, j in zip(agent_1_dimension, relevant_dimension)
                }
            )
            .to_dict("records")
        )
        agent_2_dimension = [f"Answer.agent2_{i}" for i in relevant_dimension]
        model_rewards[models[2]] += (
            annotation_df[annotation_df["Input.episode_id"] == episode.pk][
                agent_2_dimension
            ]
            .rename(
                columns={
                    i: j for i, j in zip(agent_2_dimension, relevant_dimension)
                }
            )
            .to_dict("records")
        )
    for model in model_rewards:
        model_df_rewards[model] = pd.DataFrame.from_dict(model_rewards[model])
    return model_df_rewards


def get_avg_reward_for_models(
    annotation_df: pd.DataFrame, relevant_dimension: list[str]
) -> pd.DataFrame:
    model_rewards = _get_per_episode_reward_for_models(
        annotation_df, relevant_dimension
    )
    model_rewards_avg = {}
    for model in sorted(list(model_rewards.keys())):
        model_rewards_avg[model] = model_rewards[model].mean(axis=0).to_dict()
    return pd.DataFrame.from_dict(model_rewards_avg)


def get_avg_successRate_for_models(
    annotation_df: pd.DataFrame, relevant_dimension: list[str]
) -> pd.DataFrame:
    model_rewards = _get_per_episode_reward_for_models(
        annotation_df, relevant_dimension
    )
    model_list = sorted(list(model_rewards.keys()))
    model_one_successRate = (
        model_rewards[model_list[0]] > model_rewards[model_list[1]]
    )
    model_two_successRate = (
        model_rewards[model_list[0]] < model_rewards[model_list[1]]
    )
    model_on_par_successRate = (
        model_rewards[model_list[0]] == model_rewards[model_list[1]]
    )
    return pd.DataFrame.from_dict(
        {
            model_list[0]: model_one_successRate.mean(axis=0).to_dict(),
            "on_par": model_on_par_successRate.mean(axis=0).to_dict(),
            model_list[1]: model_two_successRate.mean(axis=0).to_dict(),
        }
    )


def get_dimension_correlation(
    dimension: str, annotation_df: pd.DataFrame
) -> dict[str, float]:
    relevant_episode_ids = annotation_df["Input.episode_id"].values

    # get relevant episodes
    relevant_episodes = [
        EpisodeLog.get(relevant_episode_id)
        for relevant_episode_id in relevant_episode_ids
    ]

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
    annotation_df_with_dimension = annotation_df[with_dimension_list]
    dimension_scores_agent1 = (
        annotation_df_with_dimension[f"Answer.agent1_{dimension}"]
        .astype(float)
        .values
    )
    dimension_scores_agent2 = (
        annotation_df_with_dimension[f"Answer.agent2_{dimension}"]
        .astype(float)
        .values
    )
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
    x = dimension_scores_agent2.tolist() + dimension_scores_agent1.tolist()
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


def average_human_annotation(
    annotation_df: pd.DataFrame, relevant_dimension: list[str]
) -> pd.DataFrame:
    # average the scores of the same episode_id

    relevant_dimension = [
        f"Answer.agent1_{dimension}" for dimension in relevant_dimension
    ] + [f"Answer.agent2_{dimension}" for dimension in relevant_dimension]
    mean_df = (
        annotation_df[relevant_dimension + ["Input.episode_id"]]
        .groupby("Input.episode_id")
        .mean()
    )
    # make the index as a column
    mean_df.reset_index(level=0, inplace=True)
    return mean_df


def inter_annotator_agreement(
    annotation_df: pd.DataFrame, value_column: str
) -> tuple[dict[str, float | int], pd.DataFrame, float]:
    annotation_df = annotation_df.copy()
    annotation_df[value_column] = (
        annotation_df[value_column] / 10
    )  # normalize the scale
    scores = computeAlpha(
        annotation_df, value_column, groupCol="Input.episode_id"
    )
    randolfa = computeFleissKappa(
        annotation_df, value_column, "Input.episode_id", 2, method="fleiss"
    )
    scores_annotator_wise = {}
    workerIds = annotation_df["WorkerId"].unique()
    for workerId in workerIds:
        worker_df = annotation_df[annotation_df["WorkerId"] == workerId]
        worker_df = annotation_df[
            annotation_df["Input.episode_id"].isin(
                worker_df["Input.episode_id"]
            )
        ]
        assert isinstance(worker_df, pd.DataFrame)
        if len(worker_df) > 1:
            scores_annotator_wise[workerId] = computeAlpha(
                worker_df, value_column, groupCol="Input.episode_id"
            )
    scores_annotator_wise_df = pd.DataFrame.from_dict(
        scores_annotator_wise, orient="index"
    )
    return scores, scores_annotator_wise_df, randolfa


def analyze_perHitTime(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["WorkTimeInSeconds"] = (
        df["Answer.clickedSubmitTime"] - df["Answer.clickedConsentTime"]
    ) / 1000
    df["WorkTimeInMinutes"] = df["WorkTimeInSeconds"] / 60
    line = df["WorkTimeInMinutes"].describe()
    line_ignoreMax_df = (
        df.sort_values("WorkTimeInSeconds")
        .groupby("WorkerId")
        .apply(lambda x: x[:-1])
    )
    line_ignoreMax = line_ignoreMax_df["WorkTimeInMinutes"].describe()
    items = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    time_dict = defaultdict(list)
    for i in items:
        time_dict[i].append(line[i])
        time_dict[i].append(line_ignoreMax[i])
    df_time_overall = pd.DataFrame(time_dict, index=["all", "ignoreMax"])
    df_time_workerwise = df.groupby("WorkerId")["WorkTimeInMinutes"].describe()
    return df_time_overall, df_time_workerwise


def filter_patch_df(
    patch_df: pd.DataFrame, annotation_df: pd.DataFrame
) -> list[bool]:
    is_valid = []
    for index, row in patch_df.iterrows():
        annotation_df_row = annotation_df[
            annotation_df["Input.episode_id"] == row["Input.episode_id"]
        ]
        if row["WorkerId"] in annotation_df_row["WorkerId"].values:
            is_valid.append(False)
        else:
            is_valid.append(True)
    return is_valid


if __name__ == "__main__":
    annotation_df = pd.read_csv(
        "./annotation/gpt3.5_gpt4/gpt3.5_gpt4_results.csv"
    )
    # get rid of problematic data
    annotation_df = annotation_df[
        ~annotation_df["WorkerId"].isin(
            ["A1PR0PSNNM1WTX", "A1YFFKN3QVV54D", "A34ZJFQ9UCP1CR"]
        )
    ]
    # get the patch data
    patch_df = pd.read_csv(
        "./annotation/gpt3.5_gpt4/gpt3.5_gpt4_results_patch.csv"
    )
    # filter patch data
    patch_df = patch_df[filter_patch_df(patch_df, annotation_df)]
    # combine the data
    annotation_df = pd.concat([annotation_df, patch_df], axis=0)

    annotation_df_2 = pd.read_csv(
        "./annotation/gpt3.5_llama2/gpt3.5_llama2_results_patched.csv"
    )
    annotation_df = pd.concat([annotation_df, annotation_df_2], axis=0)
    print(len(annotation_df))
    annotation_df["Answer.agent1_overall"] = (
        annotation_df[
            [
                f"Answer.agent1_{dimension}"
                for dimension in [
                    "believability",
                    "relationship",
                    "knowledge",
                    "secret",
                    "socialrules",
                    "financial",
                    "goal",
                ]
            ]
        ]
        .astype(float)
        .sum(axis=1)
        / 7
    )
    annotation_df["Answer.agent2_overall"] = (
        annotation_df[
            [
                f"Answer.agent2_{dimension}"
                for dimension in [
                    "believability",
                    "relationship",
                    "knowledge",
                    "secret",
                    "socialrules",
                    "financial",
                    "goal",
                ]
            ]
        ]
        .astype(float)
        .sum(axis=1)
        / 7
    )

    hittime_overall, hittime_annotator_wise = analyze_perHitTime(annotation_df)
    # Analyze the agreement between annotators
    rich.print("[bold italic blue_violet on red blink]Annotator agreement:")
    relevant_dimension = [
        "believability",
        "relationship",
        "knowledge",
        "secret",
        "socialrules",
        "financial",
        "goal",
    ]
    overall_scores, overall_scores_annotator_wise = [], []
    for dimension in relevant_dimension:
        for index in range(1, 3):
            dimension_mturk = f"Answer.agent{index}_{dimension}"
            (
                scores,
                scores_annotator_wise,
                randolph_alpha,
            ) = inter_annotator_agreement(annotation_df, dimension_mturk)
            assert isinstance(scores, dict)
            assert isinstance(scores_annotator_wise, pd.DataFrame)
            scores["randolph_alpha"] = randolph_alpha
            overall_scores.append(scores)
            overall_scores_annotator_wise.append(scores_annotator_wise)
    overall_scores_df = pd.DataFrame(
        overall_scores,
        index=[
            f"agent{index}_{dimension}"
            for dimension in relevant_dimension
            for index in range(1, 3)
        ],
    )
    # average within agent
    overall_scores_df = overall_scores_df.groupby(
        overall_scores_df.index.str.split("_").str[1]
    ).mean()
    overall_scores_avg = overall_scores_df.mean(axis=0)
    rich.print("overall_scores_df:")
    rich.print(overall_scores_df.round(3))
    rich.print("overall_scores_avg:")
    rich.print(overall_scores_avg)
    overall_scores_annotator_wise_avg = sum(
        overall_scores_annotator_wise
    ) / len(overall_scores_annotator_wise)
    assert isinstance(overall_scores_annotator_wise_avg, pd.DataFrame)

    print("")
    rich.print("[bold italic yellow on red blink]GPT-4 w annotator agreement:")
    # Analyze the correlation between human and machine
    relevant_dimension += ["overall"]
    annotation_df_avg = average_human_annotation(
        annotation_df, relevant_dimension
    )
    correlation_dict = {}
    for dimension in relevant_dimension:
        correlation_dict[dimension] = get_dimension_correlation(
            dimension, annotation_df_avg
        )
    correlation_df = pd.DataFrame.from_dict(correlation_dict, orient="index")
    rich.print(correlation_df.round(3))

    # Analyze the annotator behaviors
    print("")
    rich.print("[bold italic yellow on red blink]Annotator behaviors:")
    worker_data = pd.concat(
        [overall_scores_annotator_wise_avg, hittime_annotator_wise], axis=1
    ).sort_values("ppa")
    rich.print(worker_data.round(3))

    # Models performance by Annotators
    print("")
    rich.print(
        "[bold italic yellow on red blink]Models performance by Annotators:"
    )
    model_rewards_avg = get_avg_reward_for_models(
        annotation_df_avg, relevant_dimension
    )
    rich.print(model_rewards_avg.round(3))

    # Models success rate by Annotators
    print("")
    rich.print(
        "[bold italic yellow on red blink]Models success rate by Annotators:"
    )
    model_successRate_avg = get_avg_successRate_for_models(
        annotation_df_avg, relevant_dimension
    )
    rich.print(model_successRate_avg.round(3))
