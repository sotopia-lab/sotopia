import json

import pandas as pd
from pydantic import BaseModel, Field

from .logs import EpisodeLog
from .persistent_profile import AgentProfile, EnvironmentProfile


class TwoAgentEpisodeWithScenarioBackgroundGoals(BaseModel):
    episode_id: str = Field(required=True)
    scenario: str = Field(required=True)
    codename: str = Field(required=True)
    agents_background: dict[str, str] = Field(required=True)
    social_goals: dict[str, str] = Field(required=True)
    social_interactions: str = Field(required=True)
    reasoning: str = Field(required=False)
    rewards: list[dict[str, float]] = Field(required=False)


class AgentProfileWithPersonalInformation(BaseModel):
    agent_id: str = Field(required=True)
    first_name: str = Field(required=True)
    last_name: str = Field(required=True)
    age: int = Field(required=True)
    occupation: str = Field(required=True)
    gender: str = Field(required=True)
    gender_pronoun: str = Field(required=True)
    public_info: str = Field(required=True)
    big_five: str = Field(required=True)
    moral_values: list[str] = Field(required=True)
    schwartz_personal_values: list[str] = Field(required=True)
    personality_and_values: str = Field(required=True)
    decision_making_style: str = Field(required=True)
    secret: str = Field(required=True)
    mbti: str = Field(required=True)


class EnvironmentProfileWithTwoAgentRequirements(BaseModel):
    scenario_id: str = Field(required=True)
    codename: str = Field(required=True)
    source: str = Field(required=True)
    scenario: str = Field(required=True)
    agent_goals: list[str] = Field(required=True)
    relationship: str = Field(required=True)
    age_constraint: str = Field(required=True)
    occupation_constraint: str = Field(required=True)
    agent_constraint: str = Field(required=True)


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
    return "\n\n".join(overall_social_interaction)


def episodes_to_csv(
    episodes: list[EpisodeLog], csv_file_path: str = "episodes.csv"
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
            get_agent_name_to_social_goal_from_episode(episode)
            for episode in episodes
        ],
        "social_interactions": [
            get_social_interactions_from_episode(episode)
            for episode in episodes
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)


def episodes_to_jsonl(
    episodes: list[EpisodeLog], jsonl_file_path: str = "episodes.jsonl"
) -> None:
    """Save episodes to a json file.

    Args:
        episodes (list[EpisodeLog]): List of episodes.
        filepath (str, optional): The file path. Defaults to "episodes.json".
    """
    with open(jsonl_file_path, "w") as f:
        for episode in episodes:
            data = TwoAgentEpisodeWithScenarioBackgroundGoals(
                episode_id=episode.pk,
                scenario=get_scenario_from_episode(episode),
                codename=get_codename_from_episode(episode),
                agents_background=get_agents_background_from_episode(episode),
                social_goals=get_agent_name_to_social_goal_from_episode(
                    episode
                ),
                social_interactions=get_social_interactions_from_episode(
                    episode
                ),
                reasoning=episode.reasoning,
                rewards=get_rewards_from_episode(episode),
            )
            json.dump(dict(data), f)
            f.write("\n")


def agentprofiles_to_csv(
    agent_profiles: list[AgentProfile],
    csv_file_path: str = "agent_profiles.csv",
) -> None:
    """Save agent profiles to a csv file.

    Args:
        agent_profiles (list[AgentProfile]): List of agent profiles.
        filepath (str, optional): The file path. Defaults to "agent_profiles.csv".
    """
    data = {
        "agent_id": [profile.pk for profile in agent_profiles],
        "first_name": [profile.first_name for profile in agent_profiles],
        "last_name": [profile.last_name for profile in agent_profiles],
        "age": [profile.age for profile in agent_profiles],
        "occupation": [profile.occupation for profile in agent_profiles],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)


def agentprofiles_to_jsonl(
    agent_profiles: list[AgentProfile],
    jsonl_file_path: str = "agent_profiles.jsonl",
) -> None:
    """Save agent profiles to a json file.

    Args:
        agent_profiles (list[AgentProfile]): List of agent profiles.
        filepath (str, optional): The file path. Defaults to "agent_profiles.json".
    """
    with open(jsonl_file_path, "w") as f:
        for profile in agent_profiles:
            data = AgentProfileWithPersonalInformation(
                agent_id=profile.pk,
                first_name=profile.first_name,
                last_name=profile.last_name,
                age=profile.age,
                occupation=profile.occupation,
                gender=profile.gender,
                gender_pronoun=profile.gender_pronoun,
                public_info=profile.public_info,
                big_five=profile.big_five,
                moral_values=profile.moral_values,
                schwartz_personal_values=profile.schwartz_personal_values,
                personality_and_values=profile.personality_and_values,
                decision_making_style=profile.decision_making_style,
                secret=profile.secret,
                mbti=profile.mbti,
            )
            json.dump(dict(data), f)
            f.write("\n")


def environmentprofiles_to_csv(
    environment_profiles: list[EnvironmentProfile],
    csv_file_path: str = "environment_profiles.csv",
) -> None:
    """Save environment profiles to a csv file.

    Args:
        environment_profiles (list[EnvironmentProfile]): List of environment profiles.
        filepath (str, optional): The file path. Defaults to "environment_profiles.csv".
    """
    data = {
        "scenario_id": [profile.pk for profile in environment_profiles],
        "codename": [profile.codename for profile in environment_profiles],
        "source": [profile.source for profile in environment_profiles],
        "scenario": [profile.scenario for profile in environment_profiles],
        "agent_goals": [
            profile.agent_goals for profile in environment_profiles
        ],
        "relationship": [
            profile.relationship for profile in environment_profiles
        ],
        "age_constraint": [
            profile.age_constraint for profile in environment_profiles
        ],
        "occupation_constraint": [
            profile.occupation_constraint for profile in environment_profiles
        ],
        "agent_constraint": [
            profile.agent_constraint for profile in environment_profiles
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)


def environmentprofiles_to_jsonl(
    environment_profiles: list[EnvironmentProfile],
    jsonl_file_path: str = "environment_profiles.jsonl",
) -> None:
    """Save environment profiles to a json file.

    Args:
        environment_profiles (list[EnvironmentProfile]): List of environment profiles.
        filepath (str, optional): The file path. Defaults to "environment_profiles.json".
    """
    with open(jsonl_file_path, "w") as f:
        for profile in environment_profiles:
            data = EnvironmentProfileWithTwoAgentRequirements(
                scenario_id=profile.pk,
                codename=profile.codename,
                source=profile.source,
                scenario=profile.scenario,
                agent_goals=profile.agent_goals,
                relationship=profile.relationship,
                age_constraint=profile.age_constraint,
                occupation_constraint=profile.occupation_constraint,
                agent_constraint=profile.agent_constraint
                if profile.agent_constraint
                else "nan",
            )
            json.dump(dict(data), f)
            f.write("\n")


def jsonl_to_episodes(
    jsonl_file_path: str,
) -> list[TwoAgentEpisodeWithScenarioBackgroundGoals]:
    """Load episodes from a jsonl file.

    Args:
        jsonl_file_path (str): The file path.

    Returns:
        list[TwoAgentEpisodeWithScenarioBackgroundGoals]: List of episodes.
    """
    episodes = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            episode = TwoAgentEpisodeWithScenarioBackgroundGoals(**data)
            episodes.append(episode)
    return episodes


def jsonl_to_agentprofiles(
    jsonl_file_path: str,
) -> list[AgentProfileWithPersonalInformation]:
    """Load agent profiles from a jsonl file.

    Args:
        jsonl_file_path (str): The file path.

    Returns:
        list[AgentProfileWithPersonalInformation]: List of agent profiles.
    """
    agent_profiles = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            agent_profile = AgentProfileWithPersonalInformation(**data)
            agent_profiles.append(agent_profile)
    return agent_profiles


def jsonl_to_environmentprofiles(
    jsonl_file_path: str,
) -> list[EnvironmentProfileWithTwoAgentRequirements]:
    """Load environment profiles from a jsonl file.

    Args:
        jsonl_file_path (str): The file path.

    Returns:
        list[EnvironmentProfileWithTwoAgentSettings]: List of environment profiles.
    """
    environment_profiles = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            environment_profile = EnvironmentProfileWithTwoAgentRequirements(
                **data
            )
            environment_profiles.append(environment_profile)
    return environment_profiles
