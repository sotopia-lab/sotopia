import uuid
from typing import Any, cast

from pydantic import ConstrainedList, Field, conlist
from redis_om import HashModel, JsonModel


class AgentProfile(JsonModel):
    first_name: str = Field(index=True)
    last_name: str = Field(index=True)
    age: int = Field(index=True, default_factory=lambda: 0)
    occupation: str = Field(index=True, default_factory=lambda: "")
    gender: str = Field(index=True, default_factory=lambda: "")
    big_five: str = Field(index=True, default_factory=lambda: "")
    moral_values: list[str] = Field(index=False, default_factory=lambda: [])
    schwartz_personal_values: list[str] = Field(
        index=False, default_factory=lambda: []
    )
    decision_making_style: str = Field(index=True, default_factory=lambda: "")
    secret: str = Field(default_factory=lambda: "")
    model_id: str = Field(default_factory=lambda: "")

    @classmethod
    def get(cls, pk: Any) -> "AgentProfile":
        return cast(AgentProfile, super().get(pk))


class EnvironmentProfile(JsonModel):
    scenario: str = Field(index=True, default_factory=lambda: "")
    agent_goals: list[str] = Field(default_factory=lambda: [])

    @classmethod
    def get(cls, pk: Any) -> "EnvironmentProfile":
        return cast(EnvironmentProfile, super().get(pk))
