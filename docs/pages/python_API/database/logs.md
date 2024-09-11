# Documentation for EpisodeLog and AnnotationForEpisode Classes

## Overview

This documentation provides details about two classes: `EpisodeLog` and `AnnotationForEpisode`. Both classes are built using the `pydantic` and `redis_om` libraries to create data models that are stored and managed within a Redis database.

## EpisodeLog Class

The `EpisodeLog` class is designed to store and validate logs of episodes that involve interactions between agents and environments.

### Dependencies

- `Any` from `typing`
- `root_validator` from `pydantic`
- `JsonModel` from `redis_om`
- `Field` from `redis_om.model.model`
- `AgentProfile` from `sotopia.database.persistent_profile`

### Class Definition

```python
class EpisodeLog(JsonModel):
```

### Attributes

- `environment` (str): The environment in which the episode takes place. Indexed for quick search.
- `agents` (list[str]): List of agent IDs involved in the episode. Indexed for quick search.
- `tag` (Optional[str]): Additional tags for the episode. Indexed for quick search.
- `models` (Optional[list[str]]): List of models used in the episode. Indexed for quick search.
- `messages` (list[list[tuple[str, str, str]]]): Messages exchanged during the episode, organized by turn.
- `reasoning` (str): Reasoning or explanations involved in the episode.
- `rewards` (list[tuple[float, dict[str, float]] | float]): Rewards given during each turn.
- `rewards_prompt` (str): Instructions or prompts for rewards.

### Validators

#### `agent_number_message_number_reward_number_turn_number_match`

Ensures that the number of agents matches the number of rewards provided.

**Parameters**
- `cls`: The class being validated.
- `values` (Any): Values provided to the class.

**Returns**
- `values` (Any): Validated values.

```python
@root_validator(skip_on_failure=True)
def agent_number_message_number_reward_number_turn_number_match(cls, values: Any) -> Any:
```

### Methods

#### `render_for_humans`

Generates a human-readable version of the episode log.

**Returns**
- `tuple[list[AgentProfile], list[str]]`: A tuple containing a list of agent profiles and messages and rewards by turn.

```python
def render_for_humans(self) -> tuple[list[AgentProfile], list[str]]:
    """Generate a human readable version of the episode log.

    Returns:
        A tuple of (a list of agent_profiles, a list of str): The agent profiles, and the messages and rewards in each turn.
    """
```

## AnnotationForEpisode Class

The `AnnotationForEpisode` class is designed to store annotations for specific episodes.

### Dependencies

- `JsonModel` from `redis_om`
- `Field` from `redis_om.model.model`

### Class Definition

```python
class AnnotationForEpisode(JsonModel):
```

### Attributes

- `episode` (str): The primary key ID of the episode log. Indexed for quick search.
- `annotator_id` (str): The ID of the annotator. Indexed for quick search and full-text search.
- `rewards` (list[tuple[float, dict[str, float]] | float]): Annotator's assessment of rewards.
- `reasoning` (str): Annotator's reasoning or explanations.

## Summary

- The `EpisodeLog` class is used to manage logs of episodic interactions with detailed validation and human-readable rendering capabilities.
- The `AnnotationForEpisode` class captures annotations, including rewards and reasoning, for specific episodes.

Both classes utilize the `redis_om` and `pydantic` libraries to facilitate storage, retrieval, and validation of data within a Redis database.
