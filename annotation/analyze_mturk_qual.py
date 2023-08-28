import json

import numpy as np
import pandas as pd
from scipy import stats

from sotopia.database import EpisodeLog


def calc_mae(ground_truth: list[int], predictions: list[int]) -> float:
    ground_truth_array, predictions_array = np.array(ground_truth), np.array(
        predictions
    )
    return float(np.mean(np.abs(ground_truth_array - predictions_array)))


if __name__ == "__main__":
    annotation_df = pd.read_csv("annotation/sotopiaqual_results.csv")
    # reading the data from the file
    with open("annotation/ground_truth_qual.txt") as f:
        data = f.read()

    ground_truths = json.loads(data)
    # list of ground truth values
    ground_truth_list = list(ground_truths.values())

    pearson_coefficients = []
    spearman_coefficients = []
    mses = []
    maes = []

    passing_ids = (
        []
    )  # worker ids with >= .80 spearman correlation and <= 10 mse

    repeated_worder_ids = []
    for worker_id in annotation_df["WorkerId"].unique():
        curr_df = annotation_df[annotation_df["WorkerId"] == worker_id]
        if len(curr_df) > 1:
            repeated_worder_ids.append(worker_id)

    annotation_df = annotation_df[
        ~annotation_df.WorkerId.isin(repeated_worder_ids)
    ]
    print(len(annotation_df))
    print(repeated_worder_ids)

    for worker_id in annotation_df["WorkerId"]:
        curr_df = annotation_df[annotation_df["WorkerId"] == worker_id]
        # get list of worker predictions in order of [agent1_believability, agent2_believability...agent1_goal, agent2_goal]
        worker_predictions = []
        for key in ground_truths:
            worker_predictions.append(curr_df["Answer." + key].values[0])

        # compute metrics
        pearson_res = stats.pearsonr(ground_truth_list, worker_predictions)
        spearman_res = stats.spearmanr(ground_truth_list, worker_predictions)
        mse = (
            (np.array(ground_truth_list) - np.array(worker_predictions)) ** 2
        ).mean()
        mae = calc_mae(ground_truth_list, worker_predictions)
        pearson_val = pearson_res.statistic
        spearman_val = spearman_res.statistic
        # print(f'pearson: {res_val}, spearman: {spearman_val}, mse: {mse}, mae: {mae}')
        pearson_coefficients.append(pearson_val)
        spearman_coefficients.append(spearman_val)
        mses.append(mse)
        maes.append(mae)
        if spearman_val >= 0.60 and mse <= 15:
            passing_ids.append(worker_id)

    annotation_df["pearson_coeff"] = pearson_coefficients
    annotation_df["spearman_coeff"] = spearman_coefficients
    annotation_df["mse"] = mses
    annotation_df["mae"] = maes

    annotation_df.to_csv(
        "annotation/sotopiaqual_8.18_with_metrics.csv", index=False
    )

    print("mean pearson: ", np.mean(pearson_coefficients))
    print("median pearson: ", np.median(pearson_coefficients))
    print("std deviation pearson: ", np.std(pearson_coefficients))
    print("max pearson: ", np.max(pearson_coefficients))
    print("min pearson: ", np.min(pearson_coefficients))
    print()
    print("mean spearman: ", np.mean(spearman_coefficients))
    print("median spearman: ", np.median(spearman_coefficients))
    print("std deviation spearman: ", np.std(spearman_coefficients))
    print("max spearman: ", np.max(spearman_coefficients))
    print("min spearman: ", np.min(spearman_coefficients))
    print()
    print("mean mse: ", np.mean(mses))
    print("median mse: ", np.median(mses))
    print("std deviation mse: ", np.std(mses))
    print("max mse: ", np.max(mses))
    print("min mse ", np.min(mses))
    print()
    print("mean mae: ", np.mean(maes))
    print("median mae: ", np.median(maes))
    print("std deviation mae: ", np.std(maes))
    print("max mae: ", np.max(maes))
    print("min mae: ", np.min(maes))

    print("number of passing ids: ", len(passing_ids))
    print(passing_ids)
