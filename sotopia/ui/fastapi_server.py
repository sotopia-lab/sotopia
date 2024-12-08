from typing import Literal, cast, Dict
from sotopia.database import (
    EnvironmentProfile,
    AgentProfile,
    EpisodeLog,
    RelationshipProfile,
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.server import arun_one_episode
from sotopia.agents import LLMAgent, Agents
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from typing import Optional, Any
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sotopia.ui.websocket_utils import (
    WebSocketSotopiaSimulator,
    WSMessageType,
    ErrorType,
)
import uvicorn
import asyncio

from contextlib import asynccontextmanager
from typing import AsyncIterator
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)  # TODO: Whether allowing CORS for all origins


class RelationshipWrapper(BaseModel):
    agent_1_id: str = ""
    agent_2_id: str = ""
    relationship: Literal[0, 1, 2, 3, 4, 5] = 0
    backstory: str = ""


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


class SimulateRequest(BaseModel):
    env_id: str
    agent_ids: list[str]
    models: list[str]
    max_turns: int


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


@app.get("/relationship/{agent_1_id}/{agent_2_id}", response_model=str)
async def get_relationship(agent_1_id: str, agent_2_id: str) -> str:
    relationship_profiles = RelationshipProfile.find(
        (RelationshipProfile.agent_1_id == agent_1_id)
        & (RelationshipProfile.agent_2_id == agent_2_id)
    ).all()
    assert len(relationship_profiles) == 1
    relationship_profile = relationship_profiles[0]
    assert isinstance(relationship_profile, RelationshipProfile)
    return str(relationship_profile.relationship)


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


@app.post("/scenarios/", response_model=str)
async def create_scenario(scenario: EnvironmentProfileWrapper) -> str:
    scenario_profile = EnvironmentProfile(**scenario.model_dump())
    scenario_profile.save()
    pk = scenario_profile.pk
    assert pk is not None
    return pk


@app.post("/agents/", response_model=str)
async def create_agent(agent: AgentProfileWrapper) -> str:
    agent_profile = AgentProfile(**agent.model_dump())
    agent_profile.save()
    pk = agent_profile.pk
    assert pk is not None
    return pk


@app.post("/relationship/", response_model=str)
async def create_relationship(relationship: RelationshipWrapper) -> str:
    relationship_profile = RelationshipProfile(**relationship.model_dump())
    relationship_profile.save()
    pk = relationship_profile.pk
    assert pk is not None
    return pk


@app.post("/simulate/", response_model=str)
async def simulate(simulate_request: SimulateRequest) -> str:
    env_profile: EnvironmentProfile = EnvironmentProfile.get(pk=simulate_request.env_id)
    env = ParallelSotopiaEnv(env_profile=env_profile)

    agents = Agents(
        {
            "agent1": LLMAgent(
                "agent1",
                model_name=simulate_request.models[0],
                agent_profile=AgentProfile.get(pk=simulate_request.agent_ids[0]),
            ),
            "agent2": LLMAgent(
                "agent2",
                model_name=simulate_request.models[1],
                agent_profile=AgentProfile.get(pk=simulate_request.agent_ids[1]),
            ),
        }
    )

    episode_pk = await arun_one_episode(
        env=env, agent_list=list(agents.values()), only_return_episode_pk=True
    )
    assert isinstance(episode_pk, str)
    return episode_pk


@app.put("/agents/{agent_id}", response_model=str)
async def update_agent(agent_id: str, agent: AgentProfileWrapper) -> str:
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
async def update_scenario(scenario_id: str, scenario: EnvironmentProfileWrapper) -> str:
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


@app.delete("/relationship/{relationship_id}", response_model=str)
async def delete_relationship(relationship_id: str) -> str:
    RelationshipProfile.delete(relationship_id)
    return relationship_id


@app.delete("/episodes/{episode_id}", response_model=str)
async def delete_episode(episode_id: str) -> str:
    EpisodeLog.delete(episode_id)
    return episode_id


active_simulations: Dict[
    str, bool
] = {}  # TODO check whether this is the correct way to store the active simulations


@app.get("/models", response_model=list[str])
async def get_models() -> list[str]:
    # TODO figure out how to get the available models
    return ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]


class SimulationState:
    _instance: Optional["SimulationState"] = None
    _lock = asyncio.Lock()
    _active_simulations: dict[str, bool] = {}

    def __new__(cls) -> "SimulationState":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._active_simulations = {}
        return cls._instance

    async def try_acquire_token(self, token: str) -> tuple[bool, str]:
        async with self._lock:
            if not token:
                return False, "Invalid token"

            if self._active_simulations.get(token):
                return False, "Token is active already"

            self._active_simulations[token] = True
            return True, "Token is valid"

    async def release_token(self, token: str) -> None:
        async with self._lock:
            self._active_simulations.pop(token, None)

    @asynccontextmanager
    async def start_simulation(self, token: str) -> AsyncIterator[bool]:
        try:
            yield True
        finally:
            await self.release_token(token)


class SimulationManager:
    def __init__(self) -> None:
        self.state = SimulationState()

    async def verify_token(self, token: str) -> dict[str, Any]:
        is_valid, msg = await self.state.try_acquire_token(token)
        return {"is_valid": is_valid, "msg": msg}

    async def create_simulator(
        self, env_id: str, agent_ids: list[str]
    ) -> WebSocketSotopiaSimulator:
        try:
            return WebSocketSotopiaSimulator(env_id=env_id, agent_ids=agent_ids)
        except Exception as e:
            error_msg = f"Failed to create simulator: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def handle_client_message(
        self,
        websocket: WebSocket,
        simulator: WebSocketSotopiaSimulator,
        message: dict[str, Any],
        timeout: float = 0.1,
    ) -> bool:
        try:
            msg_type = message.get("type")
            if msg_type == WSMessageType.FINISH_SIM.value:
                return True
            # TODO handle other message types
            return False
        except Exception as e:
            msg = f"Error handling client message: {e}"
            logger.error(msg)
            await self.send_error(websocket, ErrorType.INVALID_MESSAGE, msg)
            return False

    async def run_simulation(
        self, websocket: WebSocket, simulator: WebSocketSotopiaSimulator
    ) -> None:
        try:
            async for message in simulator.arun():
                await self.send_message(websocket, WSMessageType.SERVER_MSG, message)

                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                    if await self.handle_client_message(websocket, simulator, data):
                        break
                except asyncio.TimeoutError:
                    continue

        except Exception as e:
            msg = f"Error running simulation: {e}"
            logger.error(msg)
            await self.send_error(websocket, ErrorType.SIMULATION_ISSUE, msg)
        finally:
            await self.send_message(websocket, WSMessageType.END_SIM, {})

    @staticmethod
    async def send_message(
        websocket: WebSocket, msg_type: WSMessageType, data: dict[str, Any]
    ) -> None:
        await websocket.send_json({"type": msg_type.value, "data": data})

    @staticmethod
    async def send_error(
        websocket: WebSocket, error_type: ErrorType, details: str = ""
    ) -> None:
        await websocket.send_json(
            {
                "type": WSMessageType.ERROR.value,
                "data": {"type": error_type.value, "details": details},
            }
        )


@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket, token: str) -> None:
    manager = SimulationManager()

    token_status = await manager.verify_token(token)
    if not token_status["is_valid"]:
        await websocket.close(code=1008, reason=token_status["msg"])
        return

    try:
        await websocket.accept()

        while True:
            start_msg = await websocket.receive_json()
            if start_msg.get("type") != WSMessageType.START_SIM.value:
                continue

            async with manager.state.start_simulation(token):
                simulator = await manager.create_simulator(
                    env_id=start_msg["data"]["env_id"],
                    agent_ids=start_msg["data"]["agent_ids"],
                )
                await manager.run_simulation(websocket, simulator)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {token}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await manager.send_error(websocket, ErrorType.SIMULATION_ISSUE, str(e))
    finally:
        try:
            await websocket.close()
        except Exception as e:
            logger.error(f"Error closing websocket: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8800)
