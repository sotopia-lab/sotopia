from fastapi import FastAPI, WebSocket, HTTPException
from typing import Literal, cast, Dict
from fastapi.middleware.cors import CORSMiddleware

from sotopia.database import EnvironmentProfile, AgentProfile, EpisodeLog
import json
from sotopia.ui.websocket_utils import (
    WebSocketSotopiaSimulator,
    WSMessageType,
    ErrorType,
    WSMessage,
)
import uvicorn
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)  # TODO: Whether allowing CORS for all origins


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

    if not scenarios:
        raise HTTPException(
            status_code=404, detail=f"No scenarios found with {get_by}={value}"
        )

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

    if not agents_profiles:
        raise HTTPException(
            status_code=404, detail=f"No agents found with {get_by}={value}"
        )

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

    if not episodes:
        raise HTTPException(
            status_code=404, detail=f"No episodes found with {get_by}={value}"
        )
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
    try:
        old_agent = AgentProfile.get(pk=agent_id)
    except Exception:  # TODO Check the exception type
        raise HTTPException(
            status_code=404, detail=f"Agent with id={agent_id} not found"
        )
    old_agent.update(**agent.model_dump())  # type: ignore
    assert old_agent.pk is not None
    return old_agent.pk


@app.put("/scenarios/{scenario_id}", response_model=str)
async def update_scenario(scenario_id: str, scenario: EnvironmentProfile) -> str:
    try:
        old_scenario = EnvironmentProfile.get(pk=scenario_id)
    except Exception:  # TODO Check the exception type
        raise HTTPException(
            status_code=404, detail=f"Scenario with id={scenario_id} not found"
        )
    old_scenario.update(**scenario.model_dump())  # type: ignore
    assert old_scenario.pk is not None
    return old_scenario.pk


@app.delete("/agents/{agent_id}", response_model=str)
async def delete_agent(agent_id: str) -> str:
    try:
        agent = AgentProfile.get(pk=agent_id)
    except Exception:  # TODO Check the exception type
        raise HTTPException(
            status_code=404, detail=f"Agent with id={agent_id} not found"
        )
    AgentProfile.delete(agent.pk)
    assert agent.pk is not None
    return agent.pk


@app.delete("/scenarios/{scenario_id}", response_model=str)
async def delete_scenario(scenario_id: str) -> str:
    try:
        scenario = EnvironmentProfile.get(pk=scenario_id)
    except Exception:  # TODO Check the exception type
        raise HTTPException(
            status_code=404, detail=f"Scenario with id={scenario_id} not found"
        )
    EnvironmentProfile.delete(scenario.pk)
    assert scenario.pk is not None
    return scenario.pk


active_simulations: Dict[
    str, bool
] = {}  # TODO check whether this is the correct way to store the active simulations


@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket, token: str) -> None:
    if not token:
        await websocket.close(code=1008, reason="Missing token")
        return

    # TODO check the validity of the token

    await websocket.accept()
    simulation_started = False

    while True:
        raw_message = await websocket.receive_text()
        client_start_msg = json.loads(raw_message)
        msg_type = client_start_msg.get("type")

        if msg_type == WSMessageType.START_SIM.value:
            if simulation_started:
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.ERROR,
                        data={"type": ErrorType.SIMULATION_ALREADY_STARTED},
                    ).to_json()
                )
                continue

            simulation_started = True
            active_simulations[token] = True

            try:
                simulator = WebSocketSotopiaSimulator(
                    env_id=client_start_msg["data"]["env_id"],
                    agent_ids=client_start_msg["data"]["agent_ids"],
                )
            except Exception:
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.ERROR, data={"type": ErrorType.OTHER}
                    ).to_json()
                )
                break

            try:
                async for message in simulator.run_one_step():
                    agent_message_to_pass_back = WSMessage(
                        type=WSMessageType.SERVER_MSG,
                        data=message,
                    ).to_json()
                    await websocket.send_json(agent_message_to_pass_back)

                    try:
                        data = await asyncio.wait_for(
                            websocket.receive_text(), timeout=0.01
                        )
                        msg = json.loads(data)
                        if msg.get("type") == WSMessageType.FINISH_SIM.value:
                            print("----- FINISH -----")
                            break
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print("Error in receiving message from client", e)

            except Exception:
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.ERROR,
                        data={"type": ErrorType.SIMULATION_ISSUE},
                    ).to_json()
                )

            end_simulation_message = WSMessage(
                type=WSMessageType.END_SIM, data={}
            ).to_json()
            await websocket.send_json(end_simulation_message)
            simulation_started = False
            active_simulations[token] = False

        else:
            # TODO if that is not a start message, check other possibilities
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8800)
