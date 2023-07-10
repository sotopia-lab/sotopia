from typing import Generator, cast

import pandas as pd
import rich

from sotopia.database.logs import EpisodeLog

# get all the episode logs
episode_log_pks: Generator[str, None, None] = EpisodeLog.all_pks()  # type: ignore[attr-defined]

episode_dict: dict[str, list[str]] = {
    "episode_log_pk": [],
    "agents": [],
    "environment": [],
    "messages": [],
    "reasoning": [],
    "rewards": [],
    "rewards_prompt": [],
}

try:
    df = pd.read_csv("./data/episodes.csv")
    episode_dict["episode_log_pk"] = list(df["episode_log_pk"])
    episode_dict["agents"] = list(df["agents"])
    episode_dict["environment"] = list(df["environment"])
    episode_dict["messages"] = list(df["messages"])
    episode_dict["reasoning"] = list(df["reasoning"])
    episode_dict["rewards"] = list(df["rewards"])
    episode_dict["rewards_prompt"] = list(df["rewards_prompt"])
except:
    pass

episode_log_pks_list = [
    pk for pk in episode_log_pks if pk not in episode_dict["episode_log_pk"]
]

for episode_log_pk in episode_log_pks_list:
    try:
        episode_log = cast(EpisodeLog, EpisodeLog.get(episode_log_pk))
    except:
        rich.print(
            f"[red] Episode log {episode_log_pk} not found. Skipping..."
        )
        continue
    agent_profiles, messages_and_rewards = episode_log.render_for_humans()
    episode_dict["episode_log_pk"].append(episode_log_pk)
    env = "\n\n".join(messages_and_rewards[0].split("\n\n")[:-1])
    messages = "\n".join(
        [messages_and_rewards[0].split("\n\n")[-1]]
        + messages_and_rewards[1:-2]
    )
    reasoning = messages_and_rewards[-2]
    rewards = messages_and_rewards[-1]
    episode_dict["environment"].append(env)
    episode_dict["agents"].append(
        "\n".join([str(agent) for agent in agent_profiles])
    )
    episode_dict["messages"].append(messages)
    episode_dict["reasoning"].append(reasoning)
    episode_dict["rewards"].append(rewards)
    episode_dict["rewards_prompt"].append(episode_log.rewards_prompt)

df = pd.DataFrame(episode_dict)
df.to_csv("./data/episodes.csv")
