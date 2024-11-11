from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, cast
from sotopia.database import EnvironmentProfile, AgentProfile, EpisodeLog

app = FastAPI()


class SimulationEpisodeInitiation(BaseModel):
    scenario_id: str
    agent_ids: list[str]
    episode_tag: str
    models: list[str]


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


@app.get("/episodes", response_model=list[EpisodeLog])
async def get_episodes_all() -> list[EpisodeLog]:
    return EpisodeLog.all()


@app.get("/episodes/{get_by}/{value}", response_model=list[EpisodeLog])
async def get_episodes(get_by: Literal["id", "tag"], value: str) -> list[EpisodeLog]:
    episodes: list[EpisodeLog] = []
    if get_by == "id":
        episodes.append(EpisodeLog.get(pk=value))
    elif get_by == "tag":
        json_models = EpisodeLog.find(EpisodeLog.tag == value).all()
        episodes.extend(cast(list[EpisodeLog], json_models))
    return episodes


@app.post("/agents/", response_model=str)
async def create_agent(agent: AgentProfile) -> str:
    agent.save()
    assert agent.pk is not None
    return agent.pk


@app.post("/scenarios/", response_model=str)
async def create_scenario(scenario: EnvironmentProfile) -> str:
    scenario.save()
    assert scenario.pk is not None
    return scenario.pk


@app.put("/agents/{agent_id}", response_model=str)
async def update_agent(agent_id: str, agent: AgentProfile) -> str:
    old_agent = AgentProfile.get(pk=agent_id)
    old_agent.update(**agent.model_dump())  # type: ignore
    assert old_agent.pk is not None
    return old_agent.pk


@app.put("/scenarios/{scenario_id}", response_model=str)
async def update_scenario(scenario_id: str, scenario: EnvironmentProfile) -> str:
    old_scenario = EnvironmentProfile.get(pk=scenario_id)
    old_scenario.update(**scenario.model_dump())  # type: ignore
    assert old_scenario.pk is not None
    return old_scenario.pk


@app.delete("/agents/{agent_id}", response_model=str)
async def delete_agent(agent_id: str) -> str:
    agent = AgentProfile.get(pk=agent_id)
    AgentProfile.delete(agent.pk)
    assert agent.pk is not None
    return agent.pk


@app.delete("/scenarios/{scenario_id}", response_model=str)
async def delete_scenario(scenario_id: str) -> str:
    scenario = EnvironmentProfile.get(pk=scenario_id)
    EnvironmentProfile.delete(scenario.pk)
    assert scenario.pk is not None
    return scenario.pk
