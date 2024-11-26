from fastapi import FastAPI
from typing import Literal, cast, Dict
from sotopia.database import EnvironmentProfile, AgentProfile, EpisodeLog
from pydantic import BaseModel
import uvicorn

app = FastAPI()


class AgentProfileWrapper(BaseModel):
    """
    Wrapper for AgentProfile to avoid pydantic v2 issues
    """

    first_name: str
    last_name: str
    age: int = 0
    occupation: str = ""
    gender: str = ""
    gender_pronoun: str = ""
    public_info: str = ""
    big_five: str = ""
    moral_values: list[str] = []
    schwartz_personal_values: list[str] = []
    personality_and_values: str = ""
    decision_making_style: str = ""
    secret: str = ""
    model_id: str = ""
    mbti: str = ""
    tag: str = ""


class EnvironmentProfileWrapper(BaseModel):
    """
    Wrapper for EnvironmentProfile to avoid pydantic v2 issues
    """

    codename: str
    source: str = ""
    scenario: str = ""
    agent_goals: list[str] = []
    relationship: Literal[0, 1, 2, 3, 4, 5] = 0
    age_constraint: str | None = None
    occupation_constraint: str | None = None
    agent_constraint: list[list[str]] | None = None
    tag: str = ""


@app.get("/scenarios", response_model=list[EnvironmentProfile])
async def get_scenarios_all() -> list[EnvironmentProfile]:
    return EnvironmentProfile.all()


@app.get("/scenarios/{get_by}/{value}", response_model=list[EnvironmentProfile])
async def get_scenarios(
    get_by: Literal["id", "codename"], value: str
) -> list[EnvironmentProfile]:
    # Implement logic to fetch scenarios based on the parameters
    scenarios: list[EnvironmentProfile] = []  # Replace with actual fetching logic
    if get_by == "id":
        scenarios.append(EnvironmentProfile.get(pk=value))
    elif get_by == "codename":
        json_models = EnvironmentProfile.find(
            EnvironmentProfile.codename == value
        ).all()
        scenarios.extend(cast(list[EnvironmentProfile], json_models))
    return scenarios


@app.get("/agents", response_model=list[AgentProfile])
async def get_agents_all() -> list[AgentProfile]:
    return AgentProfile.all()


@app.get("/agents/{get_by}/{value}", response_model=list[AgentProfile])
async def get_agents(
    get_by: Literal["id", "gender", "occupation"], value: str
) -> list[AgentProfile]:
    agents_profiles: list[AgentProfile] = []
    if get_by == "id":
        agents_profiles.append(AgentProfile.get(pk=value))
    elif get_by == "gender":
        json_models = AgentProfile.find(AgentProfile.gender == value).all()
        agents_profiles.extend(cast(list[AgentProfile], json_models))
    elif get_by == "occupation":
        json_models = AgentProfile.find(AgentProfile.occupation == value).all()
        agents_profiles.extend(cast(list[AgentProfile], json_models))
    return agents_profiles


@app.get("/episodes/{get_by}/{value}", response_model=list[EpisodeLog])
async def get_episodes(get_by: Literal["id", "tag"], value: str) -> list[EpisodeLog]:
    episodes: list[EpisodeLog] = []
    if get_by == "id":
        episodes.append(EpisodeLog.get(pk=value))
    elif get_by == "tag":
        json_models = EpisodeLog.find(EpisodeLog.tag == value).all()
        episodes.extend(cast(list[EpisodeLog], json_models))
    return episodes


@app.post("/agents/")
async def create_agent(agent: AgentProfileWrapper) -> str:
    agent_profile = AgentProfile(**agent.model_dump())
    agent_profile.save()
    pk = agent_profile.pk
    assert pk is not None
    return pk


@app.post("/scenarios/", response_model=str)
async def create_scenario(scenario: EnvironmentProfileWrapper) -> str:
    print(scenario)
    scenario_profile = EnvironmentProfile(**scenario.model_dump())
    scenario_profile.save()
    pk = scenario_profile.pk
    assert pk is not None
    return pk


@app.delete("/agents/{agent_id}", response_model=str)
async def delete_agent(agent_id: str) -> str:
    AgentProfile.delete(agent_id)
    return agent_id


@app.delete("/scenarios/{scenario_id}", response_model=str)
async def delete_scenario(scenario_id: str) -> str:
    EnvironmentProfile.delete(scenario_id)
    return scenario_id


active_simulations: Dict[
    str, bool
] = {}  # TODO check whether this is the correct way to store the active simulations


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8800)
