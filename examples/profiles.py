import ast
import sys
from typing import Any, cast

import pandas as pd
from redis_om import Migrator  # type: ignore

from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
)


def add_agent_to_database(**kwargs: dict[str, Any]) -> None:
    agent = AgentProfile(**kwargs)
    agent.save()


def add_agents_to_database(agents: list[dict[str, Any]]) -> None:
    for agent in agents:
        add_agent_to_database(**agent)


def retrieve_agent_by_first_name(first_name: str) -> AgentProfile:
    result = AgentProfile.find(AgentProfile.first_name == first_name).all()
    if len(result) == 0:
        raise ValueError(f"Agent with first name {first_name} not found")
    elif len(result) > 1:
        raise ValueError(f"Multiple agents with first name {first_name} found")
    else:
        assert isinstance(result[0], AgentProfile)
        return result[0]


def add_env_profile(**kwargs: dict[str, Any]) -> None:
    env_profile = EnvironmentProfile(**kwargs)
    env_profile.save()


def add_env_profiles(env_profiles: list[dict[str, Any]]) -> None:
    for env_profile in env_profiles:
        add_env_profile(**env_profile)


def delete_all_agents() -> None:
    pks = AgentProfile.all_pks()
    for id in pks:
        AgentProfile.delete(id)


def delete_all_env_profiles() -> None:
    pks = EnvironmentProfile.all_pks()
    for id in pks:
        EnvironmentProfile.delete(id)


if __name__ == "__main__":
    assert (
        len(sys.argv) == 3
    ), "Please provide a csv file with agent or environment profiles, and the type of profile (agent or environment)"
    df = pd.read_csv(sys.argv[1])
    type = sys.argv[2]
    if type == "agent":
        # delete_all_agents()
        agents = cast(list[dict[str, Any]], df.to_dict(orient="records"))
        for agent in agents:
            agent["age"] = int(agent["age"])
            agent["moral_values"] = agent["moral_values"].split(",")
            agent["schwartz_personal_values"] = agent[
                "schwartz_personal_values"
            ].split(",")
        add_agents_to_database(agents)
    elif type == "environment":
        # drop columns that are not needed
        delete_all_env_profiles()
        df = df.drop("more comments", axis=1)
        df = df.drop("comments", axis=1)
        envs = cast(list[dict[str, Any]], df.to_dict(orient="records"))
        for env in envs:
            env["agent_goals"] = ast.literal_eval(env["agent_goals"])
        add_env_profiles(envs)
        Migrator().run()
