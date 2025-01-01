from redis_om import JsonModel
from redis_om.model.model import Field


class AgentProfile(JsonModel):
    name: str = Field(index=True)
    background: str = Field(default="")
    traits: str = Field(default="")
    model_name: str = Field(default="")


class EpisodeLog(JsonModel):
    environment: str = Field(index=True)
    agents_pk: list[str] = Field(index=True)
    tag: str | None = Field(index=True, default="")
    models: list[str] | None = Field(index=True, default=[])
    messages: list[tuple[str, str, str]]  # Messages [sender,receiver,content]
    rewards: (
        list[tuple[float, dict[str, float]] | float] | None
    )  # None if there is no evaluation
    rewards_prompt: str | None
