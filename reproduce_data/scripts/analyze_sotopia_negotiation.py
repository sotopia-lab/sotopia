import json
import os
import re
import sys
from collections import Counter
from math import e

import numpy as np
import pandas as pd
import rich
import typer
from numpy import cast, mat

# from redis_om import Migrator
from sotopia.database.env_agent_combo_storage import (
    EnvAgentComboStorage,
)
from sotopia.database.logs import EpisodeLog
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipProfile,
)
from tqdm import tqdm

# from analyze.utils import rewards_table

import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel, pearsonr, spearmanr, zscore, norm

def ttestSummary(df, condition_col, measure_col,paired=None):
  # conds = sorted(list(df[condition_col].unique()))
  conds = sorted(filter(lambda x: not pd.isnull(x),df[condition_col].unique()))

  conds = conds[:2]
  assert len(conds) == 2, "Not supported for more than 2 conditions "+str(conds)
  
  a = conds[0]
  b = conds[1]
  
  ix = ~df[measure_col].isnull()
  if paired:
    # merge and remove items that don't have two pairs
    pair_counts = df[ix].groupby(by=paired)[measure_col].count()
    pair_ids = pair_counts[pair_counts == 2].index
    ix = df[paired].isin(pair_ids)
    
  s_a = df.loc[(df[condition_col] == a) & ix,measure_col]
  s_b = df.loc[(df[condition_col] == b) & ix,measure_col]
  a = a.split("_")[-1]
  b = b.split("_")[-1]
    
  out = {
    # f"mean_{a}": s_a.mean(),
    # f"mean_{b}": s_b.mean(),
    # f"std_{a}": s_a.std(),
    # f"std_{b}": s_b.std(),
    # f"n_{a}": len(s_a),
    # f"n_{b}": len(s_b),    
  }
  if paired:    
    t, p = ttest_rel(s_a,s_b)
  else:
    t, p = ttest_ind(s_a,s_b)
    
#   out["t"] = t
  out["p"] = p

  # Cohen's d  
  out["d"] = (s_a.mean() - s_b.mean()) / (np.sqrt(( s_a.std() ** 2 + s_b.std() ** 2) / 2))
  
  return out

app = typer.Typer()

def add_to_dict(
    d: dict, key: str, value: float | str
) -> dict[str, float | str]:
    d[key] = value
    return d

def rewards_table(episodes: list[EpisodeLog]) -> pd.DataFrame:
    rewards = []
    success_episodes = []

    for ep in episodes:
        if isinstance(ep.rewards[0], float):
            continue

        ep_rewards = [add_to_dict(r[1], "character", f"agent_{index+1}") for index, r in enumerate(ep.rewards)]  # type: ignore
        ep_rewards = [add_to_dict(r, "environment", ep.environment) for r in ep_rewards]
        rewards += ep_rewards

        # Add this part to return only success episodes?
        success_episodes.append((ep))

    rewards_df = pd.DataFrame(rewards)
    # print(rewards_df)
    # rewards_df = rewards_df[rewards_df["character"] == "agent_1"]
    # print(rewards_df)

    # print("The number of valid episodes:", len(success_episodes))
    return rewards_df

import scipy.stats as stats
def prepare_corresponding_episodes(
    eps: list[EpisodeLog], corr_eps: list[EpisodeLog]
) -> list[tuple[EpisodeLog, EpisodeLog]]:
    """Prepare corresponding episodes for comparison.
    Args:
        eps (list[EpisodeLog]): List of episodes.
        corr_eps (list[EpisodeLog]): List of corresponding episodes.
    Returns:
        list[tuple[EpisodeLog, EpisodeLog]]: List of corresponding episodes.
    """
    episode_pairs: list[tuple[EpisodeLog, EpisodeLog]] = []
    episode_infos: list[tuple[str, str, list[str], list[str]]] = []
    # pair and sort
    paired_episodes = set()
    
    for ep in eps:
        for corr_ep in corr_eps:
            if (
                ep.environment == corr_ep.environment and 
                corr_ep.pk not in paired_episodes
            ):
                episode_pairs.append((ep, corr_ep))
                paired_episodes.add(ep.pk)
                paired_episodes.add(corr_ep.pk)
                break
    return episode_pairs


def t_significance(
    score_1: list[float] or np.ndarray,
    score_2: list[float] or np.ndarray,
    alternative: str = "greater",
):
    if isinstance(score_1, list):
        score_1 = np.array(score_1)
    if isinstance(score_2, list):
        score_2 = np.array(score_2)

    significance = stats.ttest_rel(score_1, score_2, alternative=alternative)
    return significance


def calc_significance(
    episodes: list[EpisodeLog],
    corr_episodes: list[EpisodeLog],
    alternative: str = "greater",
    goal_only: bool = False,
    key_words: str = "",
    agent_idx: int = 1,
    dimensions: list[str] = [],
) -> dict[str, float]:
    """
    Calculate the significance of the difference between the rewards of the episodes and their corresponding episodes.
    return: {dimension: significance}
    """
    matched_episodes = prepare_corresponding_episodes(episodes, corr_episodes)
    # print("Number of matched episodes:", len(matched_episodes))
    episodes = [ep for ep, _ in matched_episodes]
    corr_episodes = [corr_ep for _, corr_ep in matched_episodes]

    ep_rewards = rewards_table(episodes)
    corr_ep_rewards = rewards_table(corr_episodes)
    ep_rewards = ep_rewards[ep_rewards["character"] == f"agent_{agent_idx}"]
    corr_ep_rewards = corr_ep_rewards[corr_ep_rewards["character"] == f"agent_{agent_idx}"]

    if dimensions == []:
        dimensions = ep_rewards.columns.to_list()
        dimensions.remove("character")
    # dimensions.remove("environment")
    if goal_only:
        dimensions = ["goal"]

    significance_dict = {}
    for dims in dimensions:
        # ep_reward = ep_rewards[[dims, "environment"]]
        # corr_ep_reward = corr_ep_rewards[[dims, "environment"]]
        # sort first by environment then by dims
        ep_reward = ep_rewards[[dims, "environment"]].sort_values(by=["environment", dims])
        corr_ep_reward = corr_ep_rewards[[dims, "environment"]].sort_values(by=["environment", dims])
        ep_reward = ep_reward[dims].to_numpy()
        corr_ep_reward = corr_ep_reward[dims].to_numpy()
        # print(ep_reward)
        # print(corr_ep_reward)
        # exit(0)
        
        # ep_reward = ep_rewards[[dims, ].to_numpy()
        # corr_ep_reward = corr_ep_rewards[dims].to_numpy()
        
        
        significance = t_significance(ep_reward, corr_ep_reward, alternative)
        print(
            f"Significance for {dims} on alternative {alternative}:",
            significance,
        )
        significance_dict[dims] = significance

    significance_dict = {
        k: {"t_statistic": v[0], "p_value": v[1]}
        for k, v in significance_dict.items()
    }

    ep: EpisodeLog = episodes[0]
    corr_ep: EpisodeLog = corr_episodes[0]
    significance_df = pd.DataFrame(significance_dict).transpose()
    
    output_name = f"{ep.tag}_{corr_ep.tag}_significance.csv"
    if key_words:
        output_name = f"{ep.tag}_{corr_ep.tag}_{key_words}_significance.csv"
    # significance_df.to_csv(output_name)
    return significance_dict

def calc_significance_with_more(
    episodes: list[EpisodeLog],
    corr_episodes: list[EpisodeLog],
    alternative: str = "greater",
    goal_only: bool = False,
    key_words: str = "",
    agent_idx: int = 1,
    dimensions: list[str] = [],
):
    """
    Calculate the significance of the difference between the rewards of the episodes and their corresponding episodes.
    return: {dimension: {statistics}}
    """
    matched_episodes = prepare_corresponding_episodes(episodes, corr_episodes)
    episodes = [ep for ep, _ in matched_episodes]
    corr_episodes = [corr_ep for _, corr_ep in matched_episodes]

    ep_rewards = rewards_table(episodes)
    corr_ep_rewards = rewards_table(corr_episodes)
    
    # Add condition and ensure environment is included
    ep_rewards['condition'] = episodes[0].tag
    corr_ep_rewards['condition'] = corr_episodes[0].tag
    
    # Combine the two reward tables
    combined_rewards = pd.concat([ep_rewards, corr_ep_rewards])
    
    # Filter for the specific agent
    combined_rewards = combined_rewards[combined_rewards["character"] == f"agent_{agent_idx}"]

    if not dimensions:
        dimensions = combined_rewards.columns.to_list()
        dimensions = [d for d in dimensions if d not in ["character", "condition", "environment"]]

    significance_dict = {}
    for dim in dimensions:
        # Call ttestSummary for each dimension, using environment for pairing
        result = ttestSummary(combined_rewards, 'condition', dim)
        significance_dict[dim] = result

    return significance_dict

def average_episode_length(
    episodes: list[EpisodeLog], mode: str = "agent"
) -> tuple[float, float]:
    episode_length: list[int] = []
    turns: list[int] = []
    for ep in episodes:
        if isinstance(ep.rewards[0], float):
            continue
        if mode == "agent":
            agent_interaction_list = ep.render_for_humans()[1][:-2]
            agent_interaction_list[0] = agent_interaction_list[0].split(
                "Conversation Starts:\n\n"
            )[-1]
            episode_length += [
                len(agent_interaction.split())
                for agent_interaction in agent_interaction_list
            ]
            turns.append(len(agent_interaction_list))
            
        elif mode == "script":
            script_interaction_list = ep.render_for_humans()[1][1:-2]
            episode_length += [
                len(script_interaction.split())
                for script_interaction in script_interaction_list
            ]
            turns.append(len(script_interaction_list))
    return float(np.mean(episode_length)), float(np.mean(turns))



def match(key_word_list: list[str], target_list: list[str]) -> bool:
    for key_word in key_word_list:
        if key_word in target_list:
            return True
    return False


@app.command()
def analyze(tag: str | list[str], mode: str = "agent", key_words: str = "", agent_idx: int = 1):
    Episodes: list[EpisodeLog] = []
    if isinstance(tag, str):
        Episodes = list(EpisodeLog.find(EpisodeLog.tag == tag).all())  # type: ignore
    else:
        for t in tag:
            Episodes += list(EpisodeLog.find(EpisodeLog.tag == t).all())
    
    if key_words:
        key_word_list = key_words.split(",")
        Episodes = [
            ep
            for ep in Episodes
            if match(
                key_word_list,
                EnvironmentProfile.get(ep.environment).codename.split("_"),
            )
        ]
    # Episodes is a list EpisodeLog objects; typing correction
    rewards_df = rewards_table(Episodes)  # type: ignore
    # overall average rewards
    rewards_df = rewards_df[rewards_df["character"] == f"agent_{agent_idx}"]
    # print(rewards_df)
    avg_rewards = rewards_df.drop("character", axis=1).drop("environment", axis=1).mean(axis=0)
    avg_rewards["samples"] = len(rewards_df)
    # print("Average rewards:")
    # print(avg_rewards)

    # avg length of episodes
    episode_length, turns = average_episode_length(Episodes, mode=mode)
    # print("Average length of episodes:", episode_length)
    return avg_rewards, episode_length, turns


def agent_calc(agent_idx):
    modes = ["cooperative", "competitive"]
    agents = ["Agreeableness", "Low_Agreeableness", "Extraversion", "Introversion"]
    all_dimensions = ["Transparency", "Competence", "Adaptability"]
    # degree = ["High", "Low"]
    degree = ["High"]
    ai_agents = []
    # all combinations of ai_agents: 2^3 = 8
    for transparency in degree:
        for competence in degree:
            for adaptability in degree:
                ai_agents.append(
                    f"{transparency}_{all_dimensions[0]}-{competence}_{all_dimensions[1]}-{adaptability}_{all_dimensions[2]}"
                )
            
    
    
    # ai_agents = ["High_Transparency-High_Competence-High_Adaptability", "Low_Transparency-High_Competence-High_Adaptability",]
    
    aggregated_result = {}
    
    for mode in modes:
        for agent in agents:
            for ai_idx, ai_agent in enumerate(ai_agents):
                tag_base = f"1019_hiring_equal_{mode}_salary_start_date_trust-bigfive"
                tag = f"{tag_base}-{ai_agent}-{agent}"
                if len(list(EpisodeLog.find(EpisodeLog.tag == tag).all())) == 0:
                    print(f"Tag {tag} not found, skipping")
                    continue
                # print(f"Analyzing tag: {tag}")
                
                avg_rewards, episode_length, turns = analyze(tag, agent_idx=agent_idx)
                # print(f"Agent: {agent}, rewards: {avg_rewards}, episode_length: {episode_length}, turns: {turns}")
                
                avg_rewards["samples"] = len(EpisodeLog.find(EpisodeLog.tag == tag).all())
                avg_rewards["episode_length"] = episode_length
                avg_rewards["turns"] = turns
                ai_agents_dimensions = {d.split("_")[-1]: d.split("_")[0] for d in ai_agent.split("-")}
                avg_rewards = {**ai_agents_dimensions, **avg_rewards}
                
                aggregated_result[f"{mode}_{agent}_{ai_idx}"] = avg_rewards
    
    # only get ["goal", "overall_scores", "samples", "episode_length", "turns"]
    aggregated_result = pd.DataFrame(aggregated_result).transpose()
    # aggregated_result.index.name = "Placeholder"
    aggregated_result.columns.name = "Placeholder"
    # aggregated_result = aggregated_result[["deal_made", "point", "episode_length", "turns", "samples"]]
    print(aggregated_result)
    aggregated_result.to_csv(f"result_{agent_idx}.csv")
    
    # now get the stat tests for goal dimension, in the form of pair-wise matrix
    # significance_matrix = np.ones((len(agents), len(agents)))
    # for idx1, agent1 in enumerate(agents):
    #     for idx2, agent2 in enumerate(agents):
    #         if idx1 >= idx2:
    #             continue
    #         tag1 = f"{tag_base}_{agent1}"
    #         tag2 = f"{tag_base}_{agent2}"
    #         episodes1 = list(EpisodeLog.find(EpisodeLog.tag == tag1).all())
    #         episodes2 = list(EpisodeLog.find(EpisodeLog.tag == tag2).all())
    #         significance_dict = calc_significance_with_more(episodes1, episodes2, dimensions=["deal_made"])
    #         # print(f"Significance between {agent1} and {agent2}:", significance_dict)
            
    #         significance_matrix[idx1, idx2] = significance_dict["goal"]["p"]
    #         significance_matrix[idx2, idx1] = significance_dict["goal"]["p"]
    
    # # now add the agents as columns and rows, use the format of 2 digits
    # significance_matrix = pd.DataFrame(significance_matrix, columns=agents, index=agents)
    # significance_matrix = significance_matrix.applymap(lambda x: f"{x:.4f}")
    
    
    
    # print(significance_matrix)
    

from collections import defaultdict
if __name__ == "__main__":
    # agents = ["Agreeableness", "Low_Agreeableness", "Extraversion", "Introversion"]
    
    print("Agent 1 performance")
    agent_calc(agent_idx=1)
    
    print("Agent 2 performance")
    agent_calc(agent_idx=2)