import json

import pandas as pd

from .logs import EpisodeLog
from .persistent_profile import AgentProfile, EnvironmentProfile
from pydantic import ConstrainedList, conlist, root_validator
from redis_om import HashModel, JsonModel
from redis_om.model.model import Field

class TwoAgentEpisodeWithScenarioBackgroundGoals(JsonModel):
    episode_id: str = Field(index=True)
    scenario: str = Field(index=True)
    codename: str = Field(index=True)
    agents_background: dict[str, str] = Field(index=True)
    social_goals: dict[str, str] = Field(index=True)
    social_interactions: str = Field(index=True)

def _map_gender_to_adj(gender: str) -> str:
    gender_to_adj = {
        "Man": "male",
        "Woman": "female",
        "Nonbinary": "nonbinary",
    }
    if gender:
        return gender_to_adj[gender]
    else:
        return ""


def get_rewards_from_episode(episode: EpisodeLog) -> list[dict[str, float]]:
    assert (
        len(episode.rewards) == 2
        and (not isinstance(episode.rewards[0], float))
        and (not isinstance(episode.rewards[1], float))
    )
    return [episode.rewards[0][1], episode.rewards[1][1]]


def get_scenario_from_episode(episode: EpisodeLog) -> str:
    """Get the scenario from the episode.

    Args:
        episode (EpisodeLog): The episode.

    Returns:
        str: The scenario.
    """
    return EnvironmentProfile.get(pk=episode.environment).scenario


def get_codename_from_episode(episode: EpisodeLog) -> str:
    """Get the codename from the episode.

    Args:
        episode (EpisodeLog): The episode.

    Returns:
        str: The codename.
    """
    return EnvironmentProfile.get(pk=episode.environment).codename


def get_agents_background_from_episode(episode: EpisodeLog) -> dict[str, str]:
    """Get the agents' background from the episode.

    Args:
        episode (EpisodeLog): The episode.

    Returns:
        list[str]: The agents' background.
    """
    agents = [AgentProfile.get(pk=agent) for agent in episode.agents]

    return {
        f"{profile.first_name} {profile.last_name}": f"{profile.first_name} {profile.last_name} is a {profile.age}-year-old {_map_gender_to_adj(profile.gender)} {profile.occupation.lower()}. {profile.gender_pronoun} pronouns. {profile.public_info} Personality and values description: {profile.personality_and_values} {profile.first_name}'s secrets: {profile.secret}"
        for profile in agents
    }


def get_agent_name_to_social_goal_from_episode(
    episode: EpisodeLog,
) -> dict[str, str]:
    agents = [AgentProfile.get(agent) for agent in episode.agents]
    agent_names = [
        agent.first_name + " " + agent.last_name for agent in agents
    ]
    environment = EnvironmentProfile.get(episode.environment)
    agent_goals = {
        agent_names[0]: environment.agent_goals[0],
        agent_names[1]: environment.agent_goals[1],
    }
    return agent_goals


def get_social_interactions_from_episode(
    episode: EpisodeLog,
) -> str:
    assert isinstance(episode.tag, str)
    list_of_social_interactions = episode.render_for_humans()[1]
    if len(list_of_social_interactions) < 3:
        return ""
    if "script" in episode.tag.split("_"):
        overall_social_interaction = list_of_social_interactions[1:-3]
    else:
        overall_social_interaction = list_of_social_interactions[0:-3]
        # only get the sentence after "Conversation Starts:\n\n"
        starter_msg_list = overall_social_interaction[0].split(
            "Conversation Starts:\n\n"
        )
        if len(starter_msg_list) < 3:
            overall_social_interaction = list_of_social_interactions[1:-3]
            # raise ValueError("The starter message is not in the expected format")
        else:
            overall_social_interaction[0] = starter_msg_list[-1]
    return  "\n\n".join(overall_social_interaction)


def episodes_to_csv(
    episodes: list[EpisodeLog], filepath: str = "episodes.csv"
) -> None:
    """Save episodes to a csv file.

    Args:
        episodes (list[EpisodeLog]): List of episodes.
        filepath (str, optional): The file path. Defaults to "episodes.csv".
    """
    data = {
        "episode_id": [episode.pk for episode in episodes],
        "scenario": [
            get_scenario_from_episode(episode) for episode in episodes
        ],
        "codename": [
            get_codename_from_episode(episode) for episode in episodes
        ],
        "agents_background": [
            get_agents_background_from_episode(episode) for episode in episodes
        ],
        "social_goals": [
            get_agent_name_to_social_goal_from_episode(episode) for episode in episodes
        ],
        "social_interactions": [
            get_social_interactions_from_episode(episode) for episode in episodes
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def episodes_to_json(
    episodes: list[EpisodeLog], filepath: str = "episodes.json"
) -> None:
    """Save episodes to a json file.

    Args:
        episodes (list[EpisodeLog]): List of episodes.
        filepath (str, optional): The file path. Defaults to "episodes.json".
    """
    with open(filepath, "w") as f:
        for episode in episodes:
            data = TwoAgentEpisodeWithScenarioBackgroundGoals(
                episode_id=episode.pk,
                scenario=get_scenario_from_episode(episode),
                codename=get_codename_from_episode(episode),
                agents_background=get_agents_background_from_episode(episode),
                social_goals=get_agent_name_to_social_goal_from_episode(episode),
                social_interactions=get_social_interactions_from_episode(episode),
            )
            json.dump(dict(data), f)
            f.write("\n")
