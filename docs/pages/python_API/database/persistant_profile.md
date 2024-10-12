# Documentation for AgentProfile, EnvironmentProfile, RelationshipProfile, and EnvironmentList Classes

## Overview

This documentation provides details about four classes: `AgentProfile`, `EnvironmentProfile`, `RelationshipProfile`, and `EnvironmentList`. All classes are built using the `pydantic` and `redis_om` libraries to create data models that are stored and managed within a Redis database.

## Dependencies

- `IntEnum` from `enum`
- `Any` from `typing`
- `root_validator` from `pydantic`
- `JsonModel` from `redis_om`
- `Field` from `redis_om.model.model`

## RelationshipType Enum

The `RelationshipType` enum defines various types of relationships between agents.

```python
class RelationshipType(IntEnum):
    stranger = 0
    know_by_name = 1
    acquaintance = 2
    friend = 3
    romantic_relationship = 4
    family_member = 5
```

## AgentProfile Class

The `AgentProfile` class is designed to store profile details of agents.

### Class Definition

```python
class AgentProfile(JsonModel):
```

### Attributes

- `first_name` (str): First name of the agent.
- `last_name` (str): Last name of the agent.
- `age` (int): Age of the agent. Default is 0.
- `occupation` (str): Occupation of the agent. Default is an empty string.
- `gender` (str): Gender of the agent. Default is an empty string.
- `gender_pronoun` (str): Gender pronoun for the agent. Default is an empty string.
- `public_info` (str): Public information about the agent. Default is an empty string.
- `big_five` (str): Big Five personality traits. Default is an empty string.
- `moral_values` (list[str]): List of moral values. Default is an empty list.
- `schwartz_personal_values` (list[str]): Schwartz personal values. Default is an empty list.
- `personality_and_values` (str): Combination of personality and values. Default is an empty string.
- `decision_making_style` (str): Decision-making style of the agent. Default is an empty string.
- `secret` (str): Secret information about the agent. Default is an empty string.
- `model_id` (str): Associated model ID. Default is an empty string.
- `mbti` (str): MBTI personality type. Default is an empty string.

## EnvironmentProfile Class

The `EnvironmentProfile` class is designed to store profile details of environments.

### Class Definition

```python
class EnvironmentProfile(JsonModel):
```

### Attributes

- `codename` (str): Codename of the environment.
- `source` (str): Source of the environment.
- `scenario` (str): A concrete scenario where the social interaction takes place.
- `agent_goals` (list[str]): Social goals of each agent.
- `relationship` (RelationshipType): Relationship between the two agents.
- `age_constraint` (Optional[str]): Age constraint ranges for agents.
- `occupation_constraint` (Optional[str]): Occupation constraint for agents.
- `agent_constraint` (Optional[list[list[str]]]): List of agent constraints.

## RelationshipProfile Class

The `RelationshipProfile` class is designed to store relationship details between two agents.

### Class Definition

```python
class RelationshipProfile(JsonModel):
```

### Attributes

- `agent_1_id` (str): ID of the first agent.
- `agent_2_id` (str): ID of the second agent.
- `relationship` (RelationshipType): Relationship type between the two agents.
- `background_story` (Optional[str]): Background story of the relationship.

## EnvironmentList Class

The `EnvironmentList` class is designed to store lists of environments.

### Class Definition

```python
class EnvironmentList(JsonModel):
```

### Attributes

- `name` (str): Name of the environment list.
- `environments` (list[str]): List of environments.
- `agent_index` (Optional[list[str]]): List of agent indices.

### Validators

#### `the_length_agent_index_matches_environments`

Ensures that the length of `agent_index` matches the length of `environments`.

**Parameters**
- `cls`: The class being validated.
- `values` (Any): Values provided to the class.

**Returns**
- `values` (Any): Validated values.

```python
@root_validator
def the_length_agent_index_matches_environments(cls, values: Any) -> Any:
```

## Summary

- The `AgentProfile` class manages profile details of individual agents.
- The `EnvironmentProfile` class captures detailed information about environments.
- The `RelationshipProfile` class stores relationship details between agents.
- The `EnvironmentList` class manages lists of environments and validates the length of agent indices against environments.

All classes utilize the `redis_om` and `pydantic` libraries to facilitate storage, retrieval, and validation of data within a Redis database.

## Example Usage

```python
from database.persistant_profile import AgentProfile, EnvironmentProfile, RelationshipProfile, EnvironmentList

# Create an AgentProfile
agent_profile = AgentProfile(
    first_name="John",
    last_name="Doe",
    age=30,
    occupation="Engineer",
    gender="Male",
    gender_pronoun="He/Him",
    public_info="John is a software engineer.",
    big_five="Extroverted",
    moral_values=["Honesty", "Justice"],
    schwartz_personal_values=["Self-Direction", "Stimulation"],
    personality_and_values="John is a friendly and honest person.",
    decision_making_style="Analytical",
    secret="John is a secret agent.",
    model_id="1234567890",
    mbti="ENFP"
)

# Create an EnvironmentProfile
environment_profile = EnvironmentProfile(
    codename="Mission1",
    source="Scenario1",
    scenario="John and Jane are tasked with stopping a cyber attack.",
    agent_goals=["Stop the attack", "Save the data"],
    relationship=RelationshipType.acquaintance,
    age_constraint="25-35",
    occupation_constraint="Engineer",
)

# Create a RelationshipProfile
relationship_profile = RelationshipProfile(
    agent_1_id="1234567890",
    agent_2_id="0987654321",
    relationship=RelationshipType.friend,
    background_story="John and Jane have been friends since childhood."
)

# Create an EnvironmentList
environment_list = EnvironmentList(
    name="Mission1",
    environments=["550e8400-e29b-41d4-a716-446655440000", "f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    agent_index=["1234567890", "0987654321"]
)
```

# Save the profiles to Redis

```python
agent_profile.save()
environment_profile.save()
relationship_profile.save()
environment_list.save()
```


# Retrieve the profiles from Redis

```python
retrieved_agent_profile = AgentProfile.find(AgentProfile.model_id == "1234567890").all()
retrieved_environment_profile = EnvironmentProfile.find(EnvironmentProfile.codename == "Mission1").all()
retrieved_relationship_profile = RelationshipProfile.find(RelationshipProfile.agent_1_id == "1234567890").all()
```
