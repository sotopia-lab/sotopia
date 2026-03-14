from typing import Literal, cast, Dict
import sys
import random
import json
from pathlib import Path

# Load .env file for API keys
from dotenv import load_dotenv
load_dotenv()

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from redis_om import get_redis_connection
import rq
from sotopia.database import (
    EnvironmentProfile,
    AgentProfile,
    EpisodeLog,
    RelationshipProfile,
    RelationshipType,
    NonStreamingSimulationStatus,
    CustomEvaluationDimensionList,
    CustomEvaluationDimension,
    BaseEnvironmentProfile,
    BaseAgentProfile,
    BaseRelationshipProfile,
    SotopiaDimensions,
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    RuleBasedTerminatedEvaluator,
    EpisodeLLMEvaluator,
    EvaluationForAgents,
)
from sotopia.server import arun_one_episode
from sotopia.agents import LLMAgent, Agents
from fastapi import (
    FastAPI,
    WebSocket,
    HTTPException,
    WebSocketDisconnect,
)
from typing import Optional, Any
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator, field_validator, Field

from sotopia.api.websocket_utils import (
    WebSocketSotopiaSimulator,
    WSMessageType,
    ErrorType,
)
from sotopia.api.werewolf_runner import run_werewolf_simulation
from sotopia.api.auth import router as auth_router, decode_access_token
from sotopia.api.oauth import router as oauth_router
from sotopia.api.leaderboard import router as leaderboard_router
from sotopia.api.profile import router as profile_router
import uvicorn
import asyncio

from contextlib import asynccontextmanager
from typing import AsyncIterator
import logging
from fastapi.responses import Response
import time

logger = logging.getLogger(__name__)


# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )  # TODO: Whether allowing CORS for all origins

active_simulations: Dict[
    str, bool
] = {}  # TODO check whether this is the correct way to store the active simulations


class CustomEvaluationDimensionsWrapper(BaseModel):
    pk: str = ""
    name: str = Field(
        default="", description="The name of the custom evaluation dimension list"
    )
    dimensions: list[CustomEvaluationDimension] = Field(
        default=[], description="The dimensions of the custom evaluation dimension list"
    )


class SimulationRequest(BaseModel):
    env_id: str
    agent_ids: list[str]
    models: list[str]
    max_turns: int
    tag: str

    @field_validator("agent_ids")
    @classmethod
    def validate_agent_ids(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError(
                "At least 2 agents are required for a simulation"
            )
        return v

    @model_validator(mode="after")
    def validate_models(self) -> Self:
        models = self.models
        agent_ids = self.agent_ids
        if len(models) != len(agent_ids) + 1:
            raise ValueError(
                f"models must have exactly {len(agent_ids) + 1} elements, if there are {len(agent_ids)} agents, the first model is the evaluator model"
            )
        return self


class SimulationState:
    _instance: Optional["SimulationState"] = None
    _lock = asyncio.Lock()
    _active_simulations: dict[str, bool] = {}
    _human_players: dict[str, str] = {}  # token -> agent_name
    _pending_actions: dict[str, asyncio.Queue[dict[str, Any]]] = {}  # token -> action queue
    _pack_chats: dict[str, list[dict[str, Any]]] = {}
    _game_types: dict[str, str] = {}  # token -> game_type (e.g., "werewolf", "sotopia")

    def __new__(cls) -> "SimulationState":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._active_simulations = {}
            cls._instance._human_players = {}
            cls._instance._pending_actions = {}
            cls._instance._pack_chats = {}
            cls._instance._game_types = {}
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
            self._human_players.pop(token, None)
            self._pending_actions.pop(token, None)
            self._game_types.pop(token, None)

    def get_action_queue(self, token: str) -> Optional[asyncio.Queue[dict[str, Any]]]:
        return self._pending_actions.get(token)

    def append_pack_chat(self, token: str, sender: str, message: str, recorded_at: int | None = None) -> None:
        """Persist a pack chat message for the session token and append to in-memory store."""
        if token not in self._pack_chats:
            self._pack_chats[token] = []
        entry = {"from": sender, "message": message, "recordedAt": recorded_at or int(time.time())}
        self._pack_chats[token].append(entry)
        # Also write to a server-side append-only log for persistence
        try:
            logs_dir = Path(__file__).parent.parent.parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            logfile = logs_dir / f"pack_chat_{token}.log"
            with open(logfile, "a", encoding="utf-8") as f:
                f.write(json.dumps({**entry, "ts": int(time.time())}) + "\n")
        except Exception:
            logger.exception("Failed to persist pack chat to file")

    def get_and_clear_pack_chats(self, token: str) -> list[dict[str, Any]]:
        msgs = self._pack_chats.get(token, [])
        self._pack_chats[token] = []
        return msgs

    def set_human_player(self, token: str, agent_name: str, game_type: str = "sotopia") -> None:
        self._human_players[token] = agent_name
        self._pending_actions[token] = asyncio.Queue()
        self._game_types[token] = game_type

    def get_game_type(self, token: str) -> str:
        return self._game_types.get(token, "sotopia")

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
        self,
        env_id: str,
        agent_ids: list[str],
        agent_models: list[str],
        evaluator_model: str,
        evaluation_dimension_list_name: str,
        env_profile_dict: dict[str, Any],
        agent_profile_dicts: list[dict[str, Any]],
        max_turns: int = 20,
    ) -> WebSocketSotopiaSimulator:
        try:
            return WebSocketSotopiaSimulator(
                env_id=env_id,
                agent_ids=agent_ids,
                agent_models=agent_models,
                evaluator_model=evaluator_model,
                evaluation_dimension_list_name=evaluation_dimension_list_name,
                env_profile_dict=env_profile_dict,
                agent_profile_dicts=agent_profile_dicts,
                max_turns=max_turns,
            )
        except Exception as e:
            error_msg = f"Failed to create simulator: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def handle_client_message(
        self,
        websocket: WebSocket,
        simulator: WebSocketSotopiaSimulator,
        message: dict[str, Any],
        token: str,
        timeout: float = 0.1,
    ) -> bool:
        try:
            msg_type = message.get("type")
            if msg_type == WSMessageType.FINISH_SIM.value:
                return True
            elif msg_type == WSMessageType.CLIENT_MSG.value:
                # Handle human player action
                action_queue = self.state.get_action_queue(token)
                if action_queue is not None:
                    action_data = message.get("data", {})
                    await action_queue.put({
                        "action_type": action_data.get("action_type", "action"),
                        "argument": action_data.get("argument", ""),
                        "participant_id": action_data.get("participant_id", ""),
                    })
                    logger.info(f"Queued human action: {action_data}")
                return False
            return False
        except Exception as e:
            msg = f"Error handling client message: {e}"
            logger.error(msg)
            await self.send_error(websocket, ErrorType.INVALID_MESSAGE, msg)
            return False

    async def run_simulation(
        self, websocket: WebSocket, simulator: WebSocketSotopiaSimulator, token: str
    ) -> None:
        try:
            async for message in simulator.arun():
                await self.send_message(websocket, WSMessageType.SERVER_MSG, message)

                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                    if await self.handle_client_message(websocket, simulator, data, token):
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
        try:
            await websocket.send_json(
                {
                    "type": WSMessageType.ERROR.value,
                    "data": {"type": error_type.value, "details": details},
                }
            )
        except Exception:
            # WebSocket may be closed, ignore send errors
            pass


async def nonstreaming_simulation(
    episode_pk: str,
    simulation_request: SimulationRequest,
    simulation_status: NonStreamingSimulationStatus,
) -> None:
    try:
        env_profile: EnvironmentProfile = EnvironmentProfile.get(
            pk=simulation_request.env_id
        )
    except Exception:  # TODO Check the exception type
        raise HTTPException(
            status_code=404,
            detail=f"Environment with id={simulation_request.env_id} not found",
        )
    try:
        agent_1_profile = AgentProfile.get(pk=simulation_request.agent_ids[0])
    except Exception:  # TODO Check the exception type
        raise HTTPException(
            status_code=404,
            detail=f"Agent with id={simulation_request.agent_ids[0]} not found",
        )
    try:
        agent_2_profile = AgentProfile.get(pk=simulation_request.agent_ids[1])
    except Exception:  # TODO Check the exception type
        raise HTTPException(
            status_code=404,
            detail=f"Agent with id={simulation_request.agent_ids[1]} not found",
        )

    env_params: dict[str, Any] = {
        "model_name": simulation_request.models[0],
        "action_order": "round-robin",
        "evaluators": [
            RuleBasedTerminatedEvaluator(
                max_turn_number=simulation_request.max_turns, max_stale_turn=2
            ),
        ],
        "terminal_evaluators": [
            EpisodeLLMEvaluator(
                simulation_request.models[0],
                EvaluationForAgents[SotopiaDimensions],
            ),
        ],
    }
    env = ParallelSotopiaEnv(env_profile=env_profile, **env_params)
    agents = Agents(
        {
            "agent1": LLMAgent(
                "agent1",
                model_name=simulation_request.models[1],
                agent_profile=agent_1_profile,
            ),
            "agent2": LLMAgent(
                "agent2",
                model_name=simulation_request.models[2],
                agent_profile=agent_2_profile,
            ),
        }
    )

    await arun_one_episode(
        env=env,
        agent_list=list(agents.values()),
        push_to_db=True,
        tag=simulation_request.tag,
        episode_pk=episode_pk,
        simulation_status=simulation_status,
    )


async def get_scenarios_all() -> list[EnvironmentProfile]:
    scenarios = EnvironmentProfile.all()
    if not scenarios:
        # Create a pseudo scenario if none exist
        pseudo_scenario = EnvironmentProfile(
            codename="Sample Scenario",
            scenario="Sample scenario description",
            agent_goals=["Sample agent 1", "Sample agent 2"],
        )
        scenarios = [pseudo_scenario]
    return scenarios


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


async def get_agents_all() -> list[AgentProfile]:
    agents = AgentProfile.all()
    if not agents:
        # Create a pseudo agent if none exist
        pseudo_agent = AgentProfile(
            first_name="Sample Agent",
            last_name="",
        )
        agents = [pseudo_agent]
    return agents


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


async def get_relationship(agent_1_id: str, agent_2_id: str) -> str:
    relationship_profiles = RelationshipProfile.find(
        (RelationshipProfile.agent_1_id == agent_1_id)
        & (RelationshipProfile.agent_2_id == agent_2_id)
    ).all()
    assert (
        len(relationship_profiles) == 1
    ), f"{len(relationship_profiles)} relationship profiles found for agents {agent_1_id} and {agent_2_id}, expected 1"
    relationship_profile = relationship_profiles[0]
    assert isinstance(relationship_profile, RelationshipProfile)
    return f"{str(relationship_profile.relationship)}: {RelationshipType(relationship_profile.relationship).name}"


async def get_episodes_all() -> list[EpisodeLog]:
    episodes = EpisodeLog.all()
    if not episodes:
        # Create a pseudo episode if none exist
        pseudo_episode = EpisodeLog(
            environment="Sample Environment",
            agents=["Sample Agent 1", "Sample Agent 2"],
            models=["gpt-4o", "gpt-4o"],
            messages=[
                [
                    ("Environment", "Agent 1", "Welcome to the sample environment."),
                    ("Environment", "Agent 2", "This is a sample conversation."),
                ]
            ],
            reasoning="This is a sample reasoning about the interaction between the agents.",
            rewards=[
                (0.5, {"cooperation": 0.7, "empathy": 0.3}),
                (0.6, {"cooperation": 0.5, "empathy": 0.7}),
            ],
            rewards_prompt="Evaluate the agents based on cooperation and empathy.",
            tag="sample",
        )
        episodes = [pseudo_episode]
    return episodes


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


async def get_evaluation_dimensions() -> dict[str, list[CustomEvaluationDimension]]:
    custom_evaluation_dimensions: dict[str, list[CustomEvaluationDimension]] = {}
    all_custom_evaluation_dimension_list = CustomEvaluationDimensionList.all()

    if not all_custom_evaluation_dimension_list:
        # Create a pseudo evaluation dimension if none exist
        pseudo_dimension = CustomEvaluationDimension(
            name="Sample Dimension",
            description="This is a sample evaluation dimension",
            range_high=5,
            range_low=1,
        )
        custom_evaluation_dimensions["sample_dimensions"] = [pseudo_dimension]
    else:
        for custom_evaluation_dimension_list in all_custom_evaluation_dimension_list:
            assert isinstance(
                custom_evaluation_dimension_list, CustomEvaluationDimensionList
            )
            dimensions = [
                CustomEvaluationDimension.get(pk=pk)
                for pk in custom_evaluation_dimension_list.dimension_pks
            ]
            custom_evaluation_dimensions[custom_evaluation_dimension_list.name] = (
                dimensions
            )
    return custom_evaluation_dimensions


async def get_models() -> list[str]:
    # TODO figure out how to get the available models
    return ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]


class SotopiaFastAPI(FastAPI):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # Include auth router for user authentication
        self.include_router(auth_router)
        # Include OAuth router for Google/GitHub/Discord login
        self.include_router(oauth_router)
        # Include leaderboard router for rankings
        self.include_router(leaderboard_router)
        # Include profile router for user profiles and history
        self.include_router(profile_router)
        self.setup_routes()

    def setup_routes(self) -> None:
        @self.get("/health", status_code=200)
        async def health_check() -> dict[str, Any]:
            """Comprehensive health check endpoint"""
            health_status: dict[str, Any] = {
                "status": "ok",
                "message": "All systems operational",
                "components": {},
            }
            # Check Redis connection
            try:
                redis_conn = get_redis_connection()
                redis_conn.ping()
                health_status["components"]["redis"] = "connected"
            except Exception as e:
                health_status["status"] = "degraded"
                health_status["components"]["redis"] = f"error: {str(e)}"

            # Check database connections by attempting a simple query
            try:
                # Simple test query that should be fast
                _ = EnvironmentProfile.all()
                health_status["components"]["database"] = "connected"
            except Exception as e:
                health_status["status"] = "degraded"
                health_status["components"]["database"] = f"error: {str(e)}"

            return health_status

        self.get("/scenarios", response_model=list[EnvironmentProfile])(
            get_scenarios_all
        )
        self.get(
            "/scenarios/{get_by}/{value}", response_model=list[EnvironmentProfile]
        )(get_scenarios)
        self.get("/agents", response_model=list[AgentProfile])(get_agents_all)
        self.get("/agents/{get_by}/{value}", response_model=list[AgentProfile])(
            get_agents
        )
        self.get("/relationship/{agent_1_id}/{agent_2_id}", response_model=str)(
            get_relationship
        )
        self.get("/episodes", response_model=list[EpisodeLog])(get_episodes_all)
        self.get("/episodes/{get_by}/{value}", response_model=list[EpisodeLog])(
            get_episodes
        )
        self.get("/models", response_model=list[str])(get_models)
        self.get(
            "/evaluation_dimensions",
            response_model=dict[str, list[CustomEvaluationDimension]],
        )(get_evaluation_dimensions)

        @self.post("/scenarios", response_model=str)
        async def create_scenario(scenario: BaseEnvironmentProfile) -> str:
            scenario_profile = EnvironmentProfile(**scenario.model_dump())
            scenario_profile.save()
            pk = scenario_profile.pk
            assert pk is not None
            return pk

        @self.post("/agents", response_model=str)
        async def create_agent(agent: BaseAgentProfile) -> str:
            agent_profile = AgentProfile(**agent.model_dump())
            agent_profile.save()
            pk = agent_profile.pk
            assert pk is not None
            return pk

        @self.post("/relationship", response_model=str)
        async def create_relationship(relationship: BaseRelationshipProfile) -> str:
            relationship_profile = RelationshipProfile(**relationship.model_dump())
            relationship_profile.save()
            pk = relationship_profile.pk
            assert pk is not None
            return pk

        @self.post("/evaluation_dimensions", response_model=str)
        async def create_evaluation_dimensions(
            evaluation_dimensions: CustomEvaluationDimensionsWrapper,
        ) -> str:
            dimension_list = CustomEvaluationDimensionList.find(
                CustomEvaluationDimensionList.name == evaluation_dimensions.name
            ).all()

            if len(dimension_list) == 0:
                all_dimensions_pks = []
                for dimension in evaluation_dimensions.dimensions:
                    find_dimension = CustomEvaluationDimension.find(
                        CustomEvaluationDimension.name == dimension.name
                    ).all()
                    if len(find_dimension) == 0:
                        dimension.save()
                        all_dimensions_pks.append(dimension.pk)
                    elif len(find_dimension) == 1:
                        all_dimensions_pks.append(find_dimension[0].pk)
                    else:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Evaluation dimension with name={dimension.name} already exists",
                        )

                custom_evaluation_dimension_list = CustomEvaluationDimensionList(
                    pk=evaluation_dimensions.pk,
                    name=evaluation_dimensions.name,
                    dimension_pks=all_dimensions_pks,
                )
                custom_evaluation_dimension_list.save()
                logger.info(
                    f"Created evaluation dimension list {evaluation_dimensions.name}"
                )
            else:
                raise HTTPException(
                    status_code=409,
                    detail=f"Evaluation dimension list with name={evaluation_dimensions.name} already exists",
                )

            pk = custom_evaluation_dimension_list.pk
            assert pk is not None
            return pk

        @self.post("/simulate", response_model=str)
        def simulate(simulation_request: SimulationRequest) -> Response:
            try:
                _: EnvironmentProfile = EnvironmentProfile.get(
                    pk=simulation_request.env_id
                )
            except Exception:  # TODO Check the exception type
                raise HTTPException(
                    status_code=404,
                    detail=f"Environment with id={simulation_request.env_id} not found",
                )
            try:
                __ = AgentProfile.get(pk=simulation_request.agent_ids[0])
            except Exception:  # TODO Check the exception type
                raise HTTPException(
                    status_code=404,
                    detail=f"Agent with id={simulation_request.agent_ids[0]} not found",
                )
            try:
                ___ = AgentProfile.get(pk=simulation_request.agent_ids[1])
            except Exception:  # TODO Check the exception type
                raise HTTPException(
                    status_code=404,
                    detail=f"Agent with id={simulation_request.agent_ids[1]} not found",
                )

            episode_pk = EpisodeLog(
                environment="",
                agents=[],
                models=[],
                messages=[],
                reasoning="",
                rewards=[],  # Pseudorewards
                rewards_prompt="",
            ).pk
            try:
                simulation_status = NonStreamingSimulationStatus(
                    episode_pk=episode_pk,
                    status="Started",
                )
                simulation_status.save()
                queue = rq.Queue("default", connection=get_redis_connection())
                queue.enqueue(
                    nonstreaming_simulation,
                    episode_pk=episode_pk,
                    simulation_request=simulation_request,
                    simulation_status=simulation_status,
                )

            except Exception as e:
                logger.error(f"Error starting simulation: {e}")
                simulation_status.status = "Error"
                simulation_status.save()
            return Response(content=episode_pk, status_code=202)

        @self.get("/simulation_status/{episode_pk}", response_model=str)
        async def get_simulation_status(episode_pk: str) -> str:
            status = NonStreamingSimulationStatus.find(
                NonStreamingSimulationStatus.episode_pk == episode_pk
            ).all()[0]
            assert isinstance(status, NonStreamingSimulationStatus)
            return status.status

        @self.delete("/agents/{agent_id}", response_model=str)
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

        @self.delete("/scenarios/{scenario_id}", response_model=str)
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

        @self.delete("/relationship/{relationship_id}", response_model=str)
        async def delete_relationship(relationship_id: str) -> str:
            RelationshipProfile.delete(relationship_id)
            return relationship_id

        @self.delete("/episodes/{episode_id}", response_model=str)
        async def delete_episode(episode_id: str) -> str:
            EpisodeLog.delete(episode_id)
            return episode_id

        @self.delete(
            "/evaluation_dimensions/{evaluation_dimension_list_name}",
            response_model=str,
        )
        async def delete_evaluation_dimension_list(
            evaluation_dimension_list_name: str,
        ) -> str:
            CustomEvaluationDimensionList.delete(evaluation_dimension_list_name)
            return evaluation_dimension_list_name

        # ============= Game Queue Endpoints =============

        @self.get("/games/queue")
        async def get_queue_overview() -> dict[str, Any]:
            """Get queue overview for matchmaking (stub for frontend compatibility)."""
            import time
            return {
                "globalStats": {
                    "avgWaitSeconds": 0,
                    "activeSessions": len(active_simulations),
                    "queueDepth": 0,
                    "serverStatus": "online",
                    "lastUpdated": time.time(),
                    "issues": [],
                },
                "games": [
                    {
                        "slug": "werewolf",
                        "title": "Werewolf",
                        "queueDepth": 0,
                        "avgWaitSeconds": 0,
                        "runningSessions": len(active_simulations),
                        "lastMatch": None,
                        "gamesPlayed": 0,
                        "avgSessionSeconds": None,
                        "enabled": True,
                    }
                ],
            }

        @self.get("/games/leaderboard")
        async def get_games_leaderboard() -> dict[str, Any]:
            """Get game-level leaderboard stats (for frontend compatibility)."""
            import time
            return {
                "entries": [
                    {
                        "game": "Werewolf",
                        "totalMatches": 0,
                        "humanWins": 0,
                        "aiWins": 0,
                        "humanWinRate": 0.0,
                        "avgDurationSeconds": 0,
                    }
                ],
                "lastUpdated": time.time(),
            }

        @self.get("/games/history/{participant_id}")
        async def get_games_history(participant_id: str) -> dict[str, Any]:
            """Get game history for a participant (for frontend compatibility)."""
            return {
                "participantId": participant_id,
                "history": [],
            }

        @self.get("/memory/{participant_id}")
        async def get_player_memory(participant_id: str) -> dict[str, Any]:
            """Get player memory data for dossier panel (stub for now)."""
            return {
                "participantId": participant_id,
                "memories": [],
                "relationships": {},
                "stats": {
                    "gamesPlayed": 0,
                    "wins": 0,
                    "losses": 0,
                },
            }

        # ============= Werewolf Game Endpoints =============

        @self.get("/games/werewolf/config")
        async def get_werewolf_config() -> dict[str, Any]:
            """Return available roles and game settings for Werewolf"""
            return {
                "roles": ["Villager", "Werewolf", "Seer", "Witch"],
                "teams": ["Villagers", "Werewolves"],
                "min_players": 6,
                "max_players": 12,
                "default_ai_model": "gpt-4o-mini",
                "phases": [
                    "Night_werewolf",
                    "Night_seer", 
                    "Night_witch",
                    "Day_discussion",
                    "Day_vote",
                ],
            }

        @self.post("/games/werewolf/sessions/create")
        async def create_werewolf_session(
            participant_id: str,
            random_role: bool = True,
        ) -> dict[str, Any]:
            """Create a new Werewolf game session with role assignment"""
            import uuid as uuid_module
            from pathlib import Path
            import json as json_module
            
            # Load agents from the actual werewolf config file  
            config_path = Path(__file__).parent.parent.parent / "examples" / "experimental" / "werewolves" / "config.json"
            try:
                with open(config_path) as f:
                    config = json_module.load(f)
                agents = config.get("agents", [])
            except Exception:
                # Fallback to default if config not found
                agents = [
                    {"name": "Aurora", "role": "Villager", "team": "Villagers"},
                    {"name": "Bram", "role": "Werewolf", "team": "Werewolves"},
                    {"name": "Celeste", "role": "Seer", "team": "Villagers"},
                    {"name": "Dorian", "role": "Werewolf", "team": "Werewolves"},
                    {"name": "Elise", "role": "Witch", "team": "Villagers"},
                    {"name": "Finn", "role": "Villager", "team": "Villagers"},
                ]
            
            if random_role:
                # Create a list of indices and shuffle them
                indices = list(range(len(agents)))
                random.shuffle(indices)
                # Human gets a random position
                human_idx = indices[0]
            else:
                human_idx = 0
            
            human_agent = agents[human_idx]
            
            session_id = str(uuid_module.uuid4())
            
            return {
                "session_id": session_id,
                "participant_id": participant_id,
                "human_agent": {
                    "name": human_agent["name"],
                    "role": human_agent["role"],
                    "team": human_agent.get("team", "Villagers"),
                    "index": human_idx,
                },
                "all_agents": [
                    {
                        "name": a["name"],
                        "role": a["role"] if i == human_idx else "Hidden",
                        "team": a.get("team", "Villagers") if i == human_idx else "Hidden",
                        "is_human": i == human_idx,
                    }
                    for i, a in enumerate(agents)
                ],
                "game_config": {
                    "total_players": len(agents),
                    "werewolf_count": sum(1 for a in agents if a["role"] == "Werewolf"),
                    "villager_count": sum(1 for a in agents if a.get("team") == "Villagers"),
                },
            }

        async def _run_werewolf_game(
            websocket: WebSocket,
            token: str,
            manager: SimulationManager,
            human_agent_index: Optional[int],
            human_agent_name: Optional[str],
            ai_model: str = "gpt-4o-mini",
            user_id: Optional[str] = None,
        ) -> None:
            """Run a Werewolf game with WebSocket streaming."""
            try:
                # Create action queue getter that accesses the manager's state
                def action_queue_getter() -> asyncio.Queue[dict[str, Any]]:
                    queue = manager.state.get_action_queue(token)
                    if queue is None:
                        # Create queue if not exists
                        manager.state._pending_actions[token] = asyncio.Queue()
                        return manager.state._pending_actions[token]
                    return queue

                # Initialize the queue
                if manager.state.get_action_queue(token) is None:
                    manager.state._pending_actions[token] = asyncio.Queue()

                # Run the werewolf simulation
                async for message in run_werewolf_simulation(
                    human_agent_index=human_agent_index,
                    human_agent_name=human_agent_name,
                    action_queue_getter=action_queue_getter,
                    pack_chat_getter=lambda: manager.state.get_and_clear_pack_chats(token),
                    ai_model=ai_model,
                    user_id=user_id,
                ):
                    # Send game state to frontend
                    await SimulationManager.send_message(
                        websocket, 
                        WSMessageType.SERVER_MSG, 
                        message
                    )
                    
                    # If waiting for human action, listen for CLIENT_MSG
                    if message.get("type") == "waiting_for_action":
                        while True:
                            try:
                                client_msg = await asyncio.wait_for(
                                    websocket.receive_json(),
                                    timeout=300.0  # 5 min timeout
                                )
                                
                                if client_msg.get("type") == WSMessageType.CLIENT_MSG.value:
                                    action_data = client_msg.get("data", {})
                                    # Special-case pack_chat: persist and broadcast to client UI, do not queue as a game action
                                    if action_data.get("action_type") == "pack_chat":
                                        try:
                                            sender = (
                                                action_data.get("participant_id")
                                                or manager.state._human_players.get(token)
                                                or "You"
                                            )
                                        except Exception:
                                            sender = action_data.get("participant_id") or "You"

                                        # Persist pack chat in SimulationState
                                        try:
                                            manager.state.append_pack_chat(
                                                token, sender, action_data.get("content", ""), int(time.time())
                                            )
                                        except Exception:
                                            logger.exception("Failed to append pack chat to state")

                                        # Broadcast to client UI(s)
                                        await SimulationManager.send_message(
                                            websocket,
                                            WSMessageType.SERVER_MSG,
                                            {
                                                "type": "pack_chat",
                                                "from": sender,
                                                "message": action_data.get("content", ""),
                                                "recordedAt": int(time.time()),
                                            },
                                        )
                                        # continue waiting for real action
                                        continue

                                    action_queue = manager.state.get_action_queue(token)
                                    if action_queue:
                                        await action_queue.put({
                                            "action_type": action_data.get("action_type", "action"),
                                            "content": action_data.get("content", action_data.get("argument", "")),
                                        })
                                        break  # Action received, continue game loop
                                elif client_msg.get("type") == WSMessageType.FINISH_SIM.value:
                                    # User wants to end game
                                    return
                                    
                            except asyncio.TimeoutError:
                                # Send timeout message and continue with default action
                                await SimulationManager.send_message(
                                    websocket,
                                    WSMessageType.SERVER_MSG,
                                    {"type": "timeout", "message": "Action timed out"}
                                )
                                # Put a skip action
                                action_queue = manager.state.get_action_queue(token)
                                if action_queue:
                                    await action_queue.put({
                                        "action_type": "action",
                                        "content": "skip",
                                    })
                                break

                # Send end message
                await SimulationManager.send_message(
                    websocket, WSMessageType.END_SIM, {}
                )
                
            except Exception as e:
                logger.error(f"Error in werewolf game: {e}")
                import traceback
                traceback.print_exc()
                await SimulationManager.send_error(
                    websocket, ErrorType.SIMULATION_ISSUE, str(e)
                )

        # Store the method for use in websocket endpoint
        self._run_werewolf_game = _run_werewolf_game

        @self.websocket("/ws/simulation")
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
                        # Handle CLIENT_MSG even before simulation starts
                        if start_msg.get("type") == WSMessageType.CLIENT_MSG.value:
                            action_queue = manager.state.get_action_queue(token)
                            if action_queue:
                                await action_queue.put(start_msg.get("data", {}))
                        continue
                    # Extract game configuration
                    game_type = start_msg["data"].get("game_type", "sotopia")
                    human_agent_name = start_msg["data"].get("human_agent_name")
                    human_agent_index = start_msg["data"].get("human_agent_index")
                    
                    # Extract user_id from auth token if provided
                    user_id: Optional[str] = None
                    auth_token = start_msg["data"].get("auth_token")
                    if auth_token:
                        payload = decode_access_token(auth_token)
                        if payload:
                            user_id = payload.get("sub")
                    
                    # Setup human player tracking if specified
                    if human_agent_name or human_agent_index is not None:
                        manager.state.set_human_player(
                            token, 
                            human_agent_name or f"player_{human_agent_index}", 
                            game_type
                        )
                    async with manager.state.start_simulation(token):
                        # Route to appropriate game runner based on game_type
                        if game_type == "werewolf":
                            await self._run_werewolf_game(
                                websocket=websocket,
                                token=token,
                                manager=manager,
                                human_agent_index=human_agent_index,
                                human_agent_name=human_agent_name,
                                ai_model=start_msg["data"].get("agent_models", ["gpt-4o-mini"])[0],
                                user_id=user_id,
                            )
                        else:
                            # Default: run standard Sotopia simulation
                            simulator = await manager.create_simulator(
                                env_id=start_msg["data"]["env_id"],
                                agent_ids=start_msg["data"]["agent_ids"],
                                agent_models=start_msg["data"].get(
                                    "agent_models", ["gpt-4o-mini", "gpt-4o-mini"]
                                ),
                                env_profile_dict=start_msg["data"].get(
                                    "env_profile_dict", {}
                                ),
                                agent_profile_dicts=start_msg["data"].get(
                                    "agent_profile_dicts", []
                                ),
                                evaluator_model=start_msg["data"].get(
                                    "evaluator_model", "gpt-4o"
                                ),
                                evaluation_dimension_list_name=start_msg["data"].get(
                                    "evaluation_dimension_list_name", "sotopia"
                                ),
                                max_turns=start_msg["data"].get("max_turns", 20),
                            )
                            await manager.run_simulation(websocket, simulator, token)

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


app = SotopiaFastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8800)
