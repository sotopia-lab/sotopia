## Adding new characters and environments
You can use the following function with the `**kwargs` being the properties of the `AgentProfile` class. This is the same for the scenarios/environments.
```python
class AgentProfile(JsonModel):
    first_name: str = Field(index=True)
    last_name: str = Field(index=True)
    age: int = Field(index=True, default_factory=lambda: 0)
    occupation: str = Field(index=True, default_factory=lambda: "")
    gender: str = Field(index=True, default_factory=lambda: "")
    gender_pronoun: str = Field(index=True, default_factory=lambda: "")
    public_info: str = Field(index=True, default_factory=lambda: "")
    big_five: str = Field(index=True, default_factory=lambda: "")
    moral_values: list[str] = Field(index=False, default_factory=lambda: [])
    schwartz_personal_values: list[str] = Field(index=False, default_factory=lambda: [])
    personality_and_values: str = Field(index=True, default_factory=lambda: "")
    decision_making_style: str = Field(index=True, default_factory=lambda: "")
    secret: str = Field(default_factory=lambda: "")
    model_id: str = Field(default_factory=lambda: "")

class EnvironmentProfile(JsonModel):
    codename: str = Field(...)
    source: str = Field(...)
    scenario: str = Field(...)
    agent_goals: list[str] = Field(...)
    ...
```

```python

from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile

def add_agent_to_database(**kwargs: dict[str, Any]) -> None:
    agent = AgentProfile(**kwargs)
    agent.save()

def add_env_profile(**kwargs: dict[str, Any]) -> None:
    env_profile = EnvironmentProfile(**kwargs)
    env_profile.save()
```

## Serialization and Deserialization

It is very easy to serialize any database structures to JSON or CSV.

#### Serialize simulated episodes
```python
from sotopia.database import episodes_to_jsonl, EpisodeLog

episodes: list[EpisodeLog] = [
    EpisodeLog.get(pk=pk)
    for pk in EpisodeLog.all_pks()
]

episodes_to_jsonl(episodes, "episodes.jsonl")
```


#### Serialize env/agent profiles
```python
from sotopia.database import environmentprofiles_to_jsonl, agentprofiles_to_jsonl

agent_profiles: list[AgentProfile] = [
    AgentProfile.get(pk=pk)
    for pk in AgentProfile.all_pks()
]

environment_profiles: list[EnvironmentProfile] = [
    EnvironmentProfile.get(pk=pk)
    for pk in EnvironmentProfile.all_pks()
]

agentprofiles_to_jsonl(agent_profiles, "agent_profiles.jsonl")
environmentprofiles_to_jsonl(environment_profiles, "environment_profiles.jsonl")
```

#### Other utilities
| Function Name | Description | Arguments | Return Type |
|---------------|-------------|-----------|-------------|
| `agentprofiles_to_csv` | Saves agent profiles to a CSV file. | `agent_profiles: list[AgentProfile]`, `csv_file_path: str = "agent_profiles.csv"` | `None` |
| `agentprofiles_to_jsonl` | Saves agent profiles to a JSONL file. | `agent_profiles: list[AgentProfile]`, `jsonl_file_path: str = "agent_profiles.jsonl"` | `None` |
| `envagnetcombostorage_to_csv` | Saves environment-agent combo storages to a CSV file. | `env_agent_combo_storages: list[EnvAgentComboStorage]`, `csv_file_path: str = "env_agent_combo_storages.csv"` | `None` |
| `envagnetcombostorage_to_jsonl` | Saves environment-agent combo storages to a JSONL file. | `env_agent_combo_storages: list[EnvAgentComboStorage]`, `jsonl_file_path: str = "env_agent_combo_storages.jsonl"` | `None` |
| `environmentprofiles_to_csv` | Saves environment profiles to a CSV file. | `environment_profiles: list[EnvironmentProfile]`, `csv_file_path: str = "environment_profiles.csv"` | `None` |
| `environmentprofiles_to_jsonl` | Saves environment profiles to a JSONL file. | `environment_profiles: list[EnvironmentProfile]`, `jsonl_file_path: str = "environment_profiles.jsonl"` | `None` |
| `episodes_to_csv` | Saves episodes to a CSV file. | `episodes: list[EpisodeLog]`, `csv_file_path: str = "episodes.csv"` | `None` |
| `episodes_to_jsonl` | Saves episodes to a JSONL file. | `episodes: list[EpisodeLog]`, `jsonl_file_path: str = "episodes.jsonl"` | `None` |
| `get_rewards_from_episode` | Retrieves rewards from an episode. | `episode: EpisodeLog` | `list[tuple[float, dict[str, float]]]` |
| `jsonl_to_agentprofiles` | Loads agent profiles from a JSONL file. | `jsonl_file_path: str` | `list[AgentProfile]` |
| `jsonl_to_envagnetcombostorage` | Loads environment-agent combo storages from a JSONL file. | `jsonl_file_path: str` | `list[EnvAgentComboStorage]` |
| `jsonl_to_environmentprofiles` | Loads environment profiles from a JSONL file. | `jsonl_file_path: str` | `list[EnvironmentProfile]` |
| `jsonl_to_episodes` | Loads episodes from a JSONL file. | `jsonl_file_path: str` | `list[EpisodeLog]` |
| `jsonl_to_relationshipprofiles` | Loads relationship profiles from a JSONL file. | `jsonl_file_path: str` | `list[RelationshipProfile]` |
| `relationshipprofiles_to_csv` | Saves relationship profiles to a CSV file. | `relationship_profiles: list[RelationshipProfile]`, `csv_file_path: str = "relationship_profiles.csv"` | `None` |
| `relationshipprofiles_to_jsonl` | Saves relationship profiles to a JSONL file. | `relationship_profiles: list[RelationshipProfile]`, `jsonl_file_path: str = "relationship_profiles.jsonl"` | `None` |
