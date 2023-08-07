import asyncio
import random
from itertools import combinations
from typing import TypeVar

import pandas as pd
import rich
from pydantic import BaseModel

from sotopia.database import AgentProfile, RelationshipProfile
from sotopia.generation_utils.generate import (
    agenerate_relationship_profile,
)

random.seed(41)

# randomly select 100 pairs of agents without repetition
agent_pks = AgentProfile.all_pks()
agent_pairs = combinations(agent_pks, 2)
# generate_agent_pairs = random.sample(list(agent_pairs), 120)
generate_agent_pairs = list(agent_pairs)

T = TypeVar("T", bound=BaseModel)


def pydantics_to_csv(filename: str, data: list[T]) -> None:
    pd.DataFrame([item.dict() for item in data]).to_csv(filename, index=False)


def delete_all_relationships() -> None:
    pks = list(RelationshipProfile.all_pks())
    for id in pks:
        RelationshipProfile.delete(id)
    pks = list(RelationshipProfile.all_pks())
    print("Relationships deleted, all relationships: ", len(list(pks)))


def generate_relationships(pairs: list[tuple[str, str]]) -> None:
    for agents in pairs:
        agent_profiles = []
        for pk in agents:
            agent = AgentProfile.get(pk)
            agent_profiles.append(agent.json())
        relationship, prompt_full = asyncio.run(
            agenerate_relationship_profile(
                model_name="gpt-4",
                agents_profiles=agent_profiles,
            )
        )
        # reconscturct the relationship profile to avoid generated pk value
        new_relationship = RelationshipProfile(
            agent_1_id=relationship.agent_1_id,
            agent_2_id=relationship.agent_2_id,
            relationship=relationship.relationship,
            background_story=relationship.background_story,
        )
        rich.print(prompt_full)
        rich.print(new_relationship)
        new_relationship.save()


delete_all_relationships()
generate_relationships(generate_agent_pairs)
new_relationships = [
    RelationshipProfile.get(pk=pk) for pk in RelationshipProfile.all_pks()
]
pydantics_to_csv("data/relationships.csv", new_relationships)
