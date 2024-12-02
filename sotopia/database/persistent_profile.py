from enum import IntEnum
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from pydantic import model_validator
from redis_om import JsonModel
from redis_om.model.model import Field


class RelationshipType(IntEnum):
    stranger = 0
    know_by_name = 1
    acquaintance = 2
    friend = 3
    romantic_relationship = 4
    family_member = 5


class AgentProfile(JsonModel):
    first_name: str = Field(index=True, description="The first name of the character")
    last_name: str = Field(index=True, description="The last name of the character")
    age: int = Field(
        index=True, default_factory=lambda: 0, description="The age of the character"
    )
    occupation: str = Field(
        index=True,
        default_factory=lambda: "",
        description="The occupation of the character",
    )
    gender: str = Field(
        index=True,
        default_factory=lambda: "",
        description="The gender of the character",
    )
    gender_pronoun: str = Field(
        index=True,
        default_factory=lambda: "",
        description="The gender pronoun of the character",
    )
    public_info: str = Field(
        index=True,
        default_factory=lambda: "",
        description="Public information about the character",
    )
    big_five: str = Field(
        index=True,
        default_factory=lambda: "",
        description="The Big Five personality traits of the character",
    )
    moral_values: list[str] = Field(
        index=False,
        default_factory=lambda: [],
        description="The moral values of the character, use the categories from Moral Foundations Theory (https://en.wikipedia.org/wiki/Moral_foundations_theory)",
    )
    schwartz_personal_values: list[str] = Field(
        index=False,
        default_factory=lambda: [],
        description="The Schwartz personal values of the character, use the categories from Schwartz's Theory of Basic Values (https://scholarworks.gvsu.edu/cgi/viewcontent.cgi?article=1116&context=orpc)",
    )
    decision_making_style: str = Field(
        index=True,
        default_factory=lambda: "",
        description="The decision-making style of the character (categories from The Development and Validation of the Rational and Intuitive Decision Styles Scale)",
    )
    personality_and_values: str = Field(
        index=True,
        default_factory=lambda: "",
        description="A summary of the character's personality and values",
    )
    secret: str = Field(
        default_factory=lambda: "", description="Secrets about the character"
    )
    model_id: str = Field(default_factory=lambda: "", description="?")
    mbti: str = Field(
        default_factory=lambda: "",
        description="The MBTI personality type of the character",
    )
    # New added fields to merge with the Stanford 1000 agent profiles (https://github.com/joonspk-research/genagents)
    ethnicity: str = Field(
        default_factory=lambda: "",
        description="The ethnicity of the character, for example, 'American', 'Portugal', 'Chinese', etc.",
    )
    race: str = Field(
        default_factory=lambda: "", description="The race of the character"
    )
    detailed_race: str = Field(
        default_factory=lambda: "",
        description="The detailed race of the character",
    )
    hispanic_origin: str = Field(
        default_factory=lambda: "",
        description="The hispanic origin of the character, e.g., 'hispanic', 'not hispanic'",
    )
    street_address: str = Field(
        default_factory=lambda: "",
        description="The street address of the character, e.g., '123 Elmwood Drive'",
    )
    city: str = Field(
        default_factory=lambda: "",
        description="The city where the character resides, e.g., 'Little Rock'",
    )
    state: str = Field(
        default_factory=lambda: "",
        description="The state where the character resides, e.g., 'AR'",
    )
    political_views: str = Field(
        default_factory=lambda: "",
        description="The political views of the character, e.g., 'Slightly liberal'",
    )
    party_identification: str = Field(
        default_factory=lambda: "",
        description="The party identification of the character, e.g., 'Independent, close to democrat'",
    )
    residence_at_16: str = Field(
        default_factory=lambda: "",
        description="The region where the character resided at age 16, e.g., 'West South Central'",
    )
    same_residence_since_16: str = Field(
        default_factory=lambda: "",
        description="Whether the character has lived in the same state since age 16, e.g., 'Different state'",
    )
    family_structure_at_16: str = Field(
        default_factory=lambda: "",
        description="The family structure of the character at age 16, e.g., 'Lived with parents'",
    )
    family_income_at_16: str = Field(
        default_factory=lambda: "",
        description="The family income of the character at age 16, e.g., 'Average'",
    )
    fathers_highest_degree: str = Field(
        default_factory=lambda: "",
        description="The highest degree obtained by the character's father, e.g., 'High school'",
    )
    mothers_highest_degree: str = Field(
        default_factory=lambda: "",
        description="The highest degree obtained by the character's mother, e.g., 'High school'",
    )
    mothers_work_history: str = Field(
        default_factory=lambda: "",
        description="The work history of the character's mother, e.g., 'No'",
    )
    marital_status: str = Field(
        default_factory=lambda: "",
        description="The marital status of the character, e.g., 'Never married'",
    )
    work_status: str = Field(
        default_factory=lambda: "",
        description="The work status of the character, e.g., 'With a job, but not at work because of temporary illness, vacation, strike'",
    )
    military_service_duration: str = Field(
        default_factory=lambda: "",
        description="The duration of military service of the character, e.g., 'No active duty'",
    )
    religion: str = Field(
        default_factory=lambda: "",
        description="The religion of the character, e.g., 'Catholic'",
    )
    religion_at_16: str = Field(
        default_factory=lambda: "",
        description="The religion of the character at age 16, e.g., 'Catholic'",
    )
    born_in_us: str = Field(
        default_factory=lambda: "",
        description="Whether the character was born in the US, e.g., 'Yes'",
    )
    us_citizenship_status: str = Field(
        default_factory=lambda: "",
        description="The US citizenship status of the character, e.g., 'A U.S. citizen'",
    )
    highest_degree_received: str = Field(
        default_factory=lambda: "",
        description="The highest degree received by the character, e.g., 'High school'",
    )
    speak_other_language: str = Field(
        default_factory=lambda: "",
        description="Whether the character speaks another language, e.g., 'No'",
    )
    total_wealth: str = Field(
        default_factory=lambda: "",
        description="The total wealth of the character, e.g., 'Less than $5,000'. Make sure to include a certain value with US dollars as the unit.",
    )

    tag: str = Field(
        index=True,
        default_factory=lambda: "",
        description="The tag of the agent, used for searching, could be convenient to document agent profiles from different works and sources",
    )


class EnvironmentProfile(JsonModel):
    codename: str = Field(
        index=True,
        default_factory=lambda: "",
        description="The codename of the environment",
    )
    source: str = Field(
        index=True,
        default_factory=lambda: "",
        description="The source of the environment",
    )
    scenario: str = Field(
        index=True,
        default_factory=lambda: "",
        description="A concrete scenario of where the social interaction takes place, the scenario should have two agents (agent1 and agent2), and you should illustrate the relationship between the two agents, and for what purpose agent1 is interacting with agent2. Please avoid mentioning specific names and occupations in the scenario and keep all the mentions gender-neutral. Also avoid generating scenarios that requires childrend (below 18) or elderly (above 70) to be involved.",
    )
    agent_goals: list[str] = Field(
        default_factory=lambda: [],
        description="The social goals of each agent, which could include <extra_info>...</extra_info>, <clarification_hint>...</clarification_hint>, and <strategy_hint>...</strategy_hint> to help the agent achieve the goal. Avoid providing too specific strategy hint, try to be as abstract as possible. For example, use 'you can provide financial benefits to achieve your goal' instead of 'you can buy him a boba tea to achieve your goal.'",
    )
    relationship: RelationshipType = Field(
        index=True,
        default_factory=lambda: RelationshipType.stranger,
        description="The relationship between the two agents, choose from: stranger, know_by_name, acquaintance, friend, romantic_relationship, family_member. Do not make up a relationship, but choose from the list, 0 means stranger, 1 means know_by_name, 2 means acquaintance, 3 means friend, 4 means romantic_relationship, 5 means family_member",
    )
    age_constraint: str | None = Field(
        default_factory=lambda: None,
        description="The age constraint of the environment, a list of tuples, each tuple is a range of age, e.g., '[(18, 25), (30, 40)]' means the environment is only available to agent one between 18 and 25, and agent two between 30 and 40",
    )
    occupation_constraint: str | None = Field(
        default_factory=lambda: None,
        description="The occupation constraint of the environment, a list of lists, each list is a list of occupations, e.g., '[['student', 'teacher'], ['doctor', 'nurse']]' means the environment is only available to agent one if agent one is a student or a teacher, and agent two is a doctor or a nurse",
    )
    agent_constraint: list[list[str]] | None = Field(
        default_factory=lambda: None,
    )
    tag: str = Field(
        index=True,
        default_factory=lambda: "",
        description="The tag of the environment, used for searching, could be convenient to document environment profiles from different works and sources",
    )


class RelationshipProfile(JsonModel):
    agent_1_id: str = Field(index=True)
    agent_2_id: str = Field(index=True)
    relationship: RelationshipType = Field(
        index=True,
        description="0 means stranger, 1 means know_by_name, 2 means acquaintance, 3 means friend, 4 means romantic_relationship, 5 means family_member",
    )  # this could be improved by limiting str to a relationship Enum
    background_story: str | None = Field(default_factory=lambda: None)


class EnvironmentList(JsonModel):
    name: str = Field(index=True)
    environments: list[str] = Field(default_factory=lambda: [])
    agent_index: list[str] | None = Field(default_factory=lambda: None)

    # validate the length of agent_index should be same as environments
    @model_validator(mode="after")
    def the_length_agent_index_matches_environments(self) -> Self:
        environments, agent_index = (
            self.environments,
            self.agent_index,
        )
        if agent_index is None:
            return self
        assert (
            len(environments) == len(agent_index)
        ), f"Number of environments {len(environments)} and agent_index {len(agent_index)} do not match"
        return self
