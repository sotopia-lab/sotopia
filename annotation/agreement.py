#!/usr/bin/env python3

from collections import Counter
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd


def distNumeric(l1: float, l2: float) -> float:
    return float(np.abs(l1 - l2))


def computePairwiseAgreement(
    df: pd.DataFrame,
    valCol: str,
    groupCol: str = "HITId",
    minN: int = 2,
    distF: Callable[[float, float], float] = distNumeric,
) -> tuple[float, int, pd.Series]:  # type: ignore[type-arg]
    """Computes pairwise agreement.
    valCol: the column with the answers (e.g., Lickert scale values)
    groupCol: the column identifying the rated item (e.g., HITId, post Id, etc)
    """
    g = df.groupby(groupCol)[valCol]
    ppas = {}
    n = 0
    for s, votes in g:
        if len(votes) >= minN:
            pa = np.mean([1 - distF(*v) for v in combinations(votes, r=2)])
            ppas[s] = pa
            n += 1
            if pd.isnull(pa):  # type: ignore
                print("Pairwise agreement is null for group: ")
                print(g)
                # embed()
        # else: print(len(votes))
    if len(ppas) == 0:
        return np.nan, n, pd.Series(ppas)
    else:
        ppa = float(np.mean(list(ppas.values())))
        if pd.isnull(ppa):
            print(f"Pairwise agreement probs for column {valCol}")
            # embed()

    return ppa, n, pd.Series(ppas)


def computeRandomAgreement(
    df: pd.DataFrame,
    valCol: str,
    distF: Callable[[float, float], float] = distNumeric,
) -> float:
    distrib = Counter(df[valCol])
    agree = 0.0
    tot = 0.0
    i = 0
    for p1 in distrib:
        for p2 in distrib:
            a1 = p1
            a2 = p2
            num, denom = 1 - distF(a1, a2), 1
            if p1 == p2:
                agree += distrib[p1] * (distrib[p2] - 1) * num / denom
                tot += distrib[p1] * (distrib[p2] - 1)
            else:
                agree += distrib[p1] * (distrib[p2]) * num / denom
                tot += distrib[p1] * distrib[p2]
            i += 1
    return agree / tot


def fleiss_kappa(
    df: pd.DataFrame,
    valCol: str,
    groupCol: str = "HITId",
    method: str = "fleiss",
) -> float:
    """
    Computes Fleiss' kappa for group of annotators.
    Use method="rand" for Randolph's kappa agreement.
    See Randolph, Justus J. 2005 "Free-Marginal Multirater Kappa (multirater
    K [free]): An Alternative to Fleiss' Fixed-Marginal Multirater Kappa."
    Presented at the Joensuu Learning and Instruction Symposium, vol. 2005
    https://eric.ed.gov/?id=ED490661
    """
    # TODO: add support for non-binary ratings
    df = df.copy()
    df = df[[groupCol, valCol]]
    df = df.groupby(by=[groupCol]).sum()
    table = df.to_numpy()

    table = 1.0 * np.asarray(table)  # avoid integer division
    n_sub, n_cat = table.shape
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    # assume fully ranked
    assert n_total == n_sub * n_rat

    # marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.0))
    p_mean = p_rat.mean()

    if method == "fleiss":
        p_mean_exp = (p_cat * p_cat).sum()
    elif method.startswith("rand") or method.startswith("unif"):
        p_mean_exp = 1 / n_cat

    kappa = (p_mean - p_mean_exp) / (1 - p_mean_exp)
    return float(kappa)


def computeAlpha(
    df: pd.DataFrame,
    valCol: str,
    groupCol: str = "HITId",
    minN: int = 2,
    distF: Callable[[float, float], float] = distNumeric,
) -> dict[str, float | int]:
    """Computes Krippendorf's Alpha"""
    d = df[~df[valCol].isnull()]
    ppa, n, groups = computePairwiseAgreement(
        d, valCol, groupCol=groupCol, minN=minN, distF=distF
    )

    d2 = d[d[groupCol].isin(groups.index)]

    # Only computing random agreement on HITs that
    # we computed pairwise agreement for.
    if len(groups):
        rnd = computeRandomAgreement(d2, valCol, distF=distF)

        # Skew: computes how skewed the answers are; Krippendorf's Alpha
        # behaves terribly under skewed distributions.
        if d2[valCol].dtype == float or d2[valCol].dtype == int:
            skew = d2[valCol].mean()
        else:
            if isinstance(d2[valCol].iloc[0], list) or isinstance(
                d2[valCol].iloc[0], set
            ):
                skew = 0
            else:
                skew = d2[valCol].describe()["freq"] / len(d2)
    else:
        rnd = np.nan
        skew = 0
    if rnd == 1:
        alpha = np.nan
    else:
        alpha = 1 - ((1 - ppa) / (1 - rnd))
    return dict(alpha=alpha, ppa=ppa, rnd_ppa=rnd, skew=skew, n=n)


if __name__ == "__main__":
    # creates fake data
    # 5-point lickert scale
    # rater1 is normal rater
    rater1 = pd.Series(np.random.randint(0, 5, 100))

    # rater2 agrees with rater1 most of the time
    rater2 = np.random.uniform(size=rater1.shape)
    rater2 = pd.Series((rater2 > 0.1).astype(int) * rater1)

    # rater3 should be random
    rater3 = pd.Series(np.random.randint(0, 5, 100))

    df = pd.DataFrame([rater1, rater2, rater3], index=["r1", "r2", "r3"]).T
    df.index.name = "id"
    df = df.reset_index()
    longDf = df.melt(
        id_vars=["id"],
        value_vars=["r1", "r2", "r3"],
        var_name="raterId",
        value_name="rating",
    )
    longDf["ratingBinary"] = (longDf["rating"] / longDf["rating"].max()).round(
        0
    )

    # metrics = computeKappa(longDf)
    ppa = computePairwiseAgreement(longDf, "ratingBinary", groupCol="id")
    rndPpa = computeRandomAgreement(longDf, "ratingBinary")
    scores = computeAlpha(longDf, "ratingBinary", groupCol="id")
