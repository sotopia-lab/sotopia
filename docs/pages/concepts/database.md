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
