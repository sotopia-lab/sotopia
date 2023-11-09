import ast
import sys
from typing import Any, cast

import pandas as pd
from redis_om import Migrator

from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipProfile,
)
from sotopia.database.env_agent_combo_storage import EnvAgentComboStorage
from sotopia.samplers import ConstraintBasedSampler
from sotopia.messages import AgentAction, Observation
from sotopia.agents import LLMAgent



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


def add_relationship_profile(**kwargs: dict[str, Any]) -> None:
    relationship_profile = RelationshipProfile(**kwargs)
    relationship_profile.save()


def add_relationship_profiles(
    relationship_profiles: list[dict[str, Any]]
) -> None:
    for relationship_profile in relationship_profiles:
        add_relationship_profile(**relationship_profile)


def delete_all_agents() -> None:
    pks = AgentProfile.all_pks()
    pks_list = list(pks)
    #for id in pks:
    #    AgentProfile.delete(id)


def delete_all_env_profiles() -> None:
    pks = EnvironmentProfile.all_pks()
    #for id in pks:
    #    EnvironmentProfile.delete(id)


def delete_all_relationships() -> None:
    pks = list(RelationshipProfile.all_pks())
    #for id in pks:
    #    RelationshipProfile.delete(id)
    pks = list(RelationshipProfile.all_pks())
    print("Relationships deleted, all relationships: ", len(list(pks)))


def sample_env_agent_combo_and_push_to_db(env_id: str) -> None:
    sampler = ConstraintBasedSampler[Observation, AgentAction](
        env_candidates=[env_id]
    )
    try:
        env_agent_combo_list = list(
            sampler.sample(agent_classes=[LLMAgent] * 2, replacement=False)
        )
    except:
        return
    print(len(env_agent_combo_list))
    for env, agent in env_agent_combo_list:
        EnvAgentComboStorage(
            env_id=env.profile.pk,
            agent_ids=[agent[0].profile.pk, agent[1].profile.pk],
        ).save()


def relationship_map(relationship: str) -> int:
    return int(eval(relationship))


if __name__ == "__main__":
    assert (
        len(sys.argv) == 3
    ), "Please provide a csv file with agent or environment profiles, and the type of profile (agent or environment)"
    df = pd.read_csv(sys.argv[1])
    type = sys.argv[2]
    if type == "agent":
        agents = cast(list[dict[str, Any]], df.to_dict(orient="records"))
        for agent in agents:
            agent["age"] = int(agent["age"])
            agent["moral_values"] = agent["moral_values"].split(",")
            agent["schwartz_personal_values"] = agent[
                "schwartz_personal_values"
            ].split(",")
        add_agents_to_database(agents)
        Migrator().run()
    elif type == "environment":
        df = df[
            [
                "codename",
                "scenario",
                "agent_goals",
                "relationship",
                "age_constraint",
                "occupation_constraint",
                "source",
            ]
        ]
        envs = cast(list[dict[str, Any]], df.to_dict(orient="records"))
        for env in envs:
            env["agent_goals"] = ast.literal_eval(env["agent_goals"])
            assert isinstance(env["relationship"], int)
        add_env_profiles(envs)
        Migrator().run()
    elif type == "relationship":
        relationships = cast(
            list[dict[str, Any]], df.to_dict(orient="records")
        )
        for relationship in relationships:
            assert isinstance(relationship["relationship"], int)
        add_relationship_profiles(relationships)
        Migrator().run()
    elif type == 'agentenvcombo':
        env_ids = list(EnvironmentProfile.all_pks())
        for env_id in env_ids:
            sample_env_agent_combo_and_push_to_db(env_id)