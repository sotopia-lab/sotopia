import sys
from typing import Any, cast

import pandas as pd

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
    result = AgentProfile.find(AgentProfile.first_name == first_name).all()  # type: ignore[attr-defined]
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


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please provide a csv file with agent profiles"
    df = pd.read_csv(sys.argv[1])
    agents = cast(list[dict[str, Any]], df.to_dict(orient="records"))
    for agent in agents:
        agent["age"] = int(agent["age"])
        agent["moral_values"] = agent["moral_values"].split(",")
        agent["schwartz_personal_values"] = agent[
            "schwartz_personal_values"
        ].split(",")
    add_agents_to_database(agents)
