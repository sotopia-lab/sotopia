from typing import Literal, cast, Dict
import sys
import asyncio
import logging
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator, field_validator, Field
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Any
import uvicorn
import rq

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# Import Sotopia components
from redis_om import get_redis_connection
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
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    RuleBasedTerminatedEvaluator,
    EpisodeLLMEvaluator,
    EvaluationForTwoAgents,
    SotopiaDimensions,
)
from sotopia.server import arun_one_episode
from sotopia.agents import LLMAgent, Agents

# Import the enhanced WebSocket utilities
from sotopia.api.websocket_utils import (
    WebSocketSotopiaSimulator,
    WSMessageType,
    ErrorType,
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Active simulations tracking
active_simulations: Dict[str, bool] = {}


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
        if len(v) != 2:
            raise ValueError(
                "Currently only 2 agents are supported, we are working on supporting more agents"
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


class NPCGroupSimulationRequest(BaseModel):
    env_id: str
    npcs: list[str]
    groups: dict[str, list[str]]
    models: list[str] = Field(default=["gpt-4o-mini", "gpt-4o-mini"])
    evaluator_model: str = Field(default="gpt-4o")
    max_turns: int = Field(default=20)
    tag: str = Field(default="npc_group_simulation")

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        # Check that all NPCs in groups are in the npcs list
        all_group_npcs = set()
        for group_members in self.groups.values():
            all_group_npcs.update(group_members)

        if not all_group_npcs.issubset(set(self.npcs)):
            missing_npcs = all_group_npcs - set(self.npcs)
            raise ValueError(
                f"These NPCs are in groups but not in the npcs list: {missing_npcs}"
            )

        return self


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
        self,
        env_id: str,
        agent_ids: list[str],
        agent_models: list[str],
        evaluator_model: str,
        evaluation_dimension_list_name: str,
        env_profile_dict: dict[str, Any] = {},
        agent_profile_dicts: list[dict[str, Any]] = [],
        max_turns: int = 20,
        npcs: list[str] = [],
        groups: dict[str, list[str]] = {},
    ) -> WebSocketSotopiaSimulator:
        try:
            simulator = WebSocketSotopiaSimulator(
                env_id=env_id,
                agent_ids=agent_ids,
                agent_models=agent_models,
                evaluator_model=evaluator_model,
                evaluation_dimension_list_name=evaluation_dimension_list_name,
                env_profile_dict=env_profile_dict,
                agent_profile_dicts=agent_profile_dicts,
                max_turns=max_turns,
            )

            # If NPCs and groups are provided, initialize them
            if npcs or groups:
                simulator.active_npcs = set(npcs)
                simulator.npc_groups = groups
                simulator.group_mode = True

            return simulator
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

            # Finish simulation
            if msg_type == WSMessageType.FINISH_SIM.value:
                return True

            # Process client message for NPC routing
            elif msg_type == WSMessageType.CLIENT_MSG.value:
                try:
                    client_response = await simulator.process_client_message(
                        message.get("data", {})
                    )
                    await self.send_message(
                        websocket,
                        WSMessageType.SERVER_MSG,
                        {"type": "npc_responses", "data": client_response},
                    )
                except Exception as e:
                    logger.error(f"Error processing client message: {e}")
                    await self.send_error(
                        websocket,
                        ErrorType.SIMULATION_ISSUE,
                        str(e),
                    )

            # Process turn request
            elif msg_type == WSMessageType.TURN_REQUEST.value:
                try:
                    turn_response = await simulator.process_turn(
                        message.get("data", {})
                    )
                    await self.send_message(
                        websocket,
                        WSMessageType.TURN_RESPONSE,
                        turn_response,
                    )
                except Exception as e:
                    logger.error(f"Error processing turn: {e}")
                    await self.send_error(
                        websocket,
                        ErrorType.SIMULATION_ISSUE,
                        str(e),
                    )

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
            # Check if this is a group-based simulation
            if simulator.group_mode:
                # Send initialization with NPCs and groups
                await self.send_message(
                    websocket,
                    WSMessageType.SERVER_MSG,
                    {
                        "type": "initialization",
                        "data": {
                            "status": "ready",
                            "npcs": list(simulator.active_npcs),
                            "groups": {
                                name: members
                                for name, members in simulator.npc_groups.items()
                            },
                        },
                    },
                )

                # Enter message handling loop for group-based mode
                while True:
                    try:
                        data = await websocket.receive_json()
                        if await self.handle_client_message(websocket, simulator, data):
                            break
                    except asyncio.TimeoutError:
                        pass
                    except WebSocketDisconnect:
                        logger.info("Client disconnected during group-based simulation")
                        break
            # Regular simulation mode
            else:
                # Run the standard simulation and stream messages
                async for message in simulator.arun():
                    await self.send_message(
                        websocket, WSMessageType.SERVER_MSG, message
                    )

                    try:
                        data = await asyncio.wait_for(
                            websocket.receive_json(), timeout=0.1
                        )
                        if await self.handle_client_message(websocket, simulator, data):
                            break
                    except asyncio.TimeoutError:
                        pass
                    except WebSocketDisconnect:
                        logger.info("Client disconnected")
                        break

        except WebSocketDisconnect:
            logger.info("Client disconnected during simulation")
        except Exception as e:
            msg = f"Error running simulation: {e}"
            logger.error(msg)
            await self.send_error(websocket, ErrorType.SIMULATION_ISSUE, msg)
        finally:
            # Send end simulation message
            try:
                await self.send_message(websocket, WSMessageType.END_SIM, {})
                logger.info("Simulation ended")
            except Exception as e:
                logger.error(f"Error sending end simulation message: {e}")

    @staticmethod
    async def send_message(
        websocket: WebSocket, msg_type: WSMessageType, data: dict[str, Any]
    ) -> None:
        """Send a message to the WebSocket client"""
        try:
            await websocket.send_json({"type": msg_type.value, "data": data})
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            raise

    @staticmethod
    async def send_error(
        websocket: WebSocket, error_type: ErrorType, details: str = ""
    ) -> None:
        """Send an error message to the WebSocket client"""
        try:
            await websocket.send_json(
                {
                    "type": WSMessageType.ERROR.value,
                    "data": {"type": error_type.value, "details": details},
                }
            )
            logger.error(f"Sent error to client: {error_type.value} - {details}")
        except Exception as e:
            logger.error(f"Error sending error message to WebSocket: {e}")


async def nonstreaming_simulation(
    episode_pk: str,
    simulation_request: SimulationRequest,
    simulation_status: NonStreamingSimulationStatus,
) -> None:
    """Run a non-streaming simulation and update its status"""
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
                EvaluationForTwoAgents[SotopiaDimensions],
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
    """FastAPI application for Sotopia with WebSocket support"""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)
        # Add CORS middleware for cross-origin requests
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # Set up API routes
        self.setup_routes()

    def setup_routes(self) -> None:
        """Set up all API routes"""

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

        @self.websocket("/ws/simulation")
        async def websocket_endpoint(websocket: WebSocket, token: str) -> None:
            """
            WebSocket endpoint for running simulations

            This endpoint supports multiple modes:
            - Standard simulation: Run a standard two-agent conversation
            - Turn-based mode: Process simulation on a turn-by-turn basis
            - Group-based mode: Handle message routing between NPCs and groups

            Parameters:
            - websocket: The WebSocket connection
            - token: Authentication token for the session
            """
            manager = SimulationManager()
            token_status = await manager.verify_token(token)
            if not token_status["is_valid"]:
                await websocket.close(code=1008, reason=token_status["msg"])
                return

            try:
                await websocket.accept()

                # Wait for the START_SIM message
                start_msg = await websocket.receive_json()
                if start_msg.get("type") != WSMessageType.START_SIM.value:
                    await manager.send_error(
                        websocket,
                        ErrorType.INVALID_MESSAGE,
                        "First message must be START_SIM",
                    )
                    return

                # Extract mode information and NPC/group data if available
                npcs = start_msg["data"].get("npcs", [])
                groups = start_msg["data"].get("groups", {})

                async with manager.state.start_simulation(token):
                    simulator = await manager.create_simulator(
                        env_id=start_msg["data"]["env_id"],
                        agent_ids=start_msg["data"]["agent_ids"],
                        agent_models=start_msg["data"].get(
                            "agent_models", ["gpt-4o-mini", "gpt-4o-mini"]
                        ),
                        evaluator_model=start_msg["data"].get(
                            "evaluator_model", "gpt-4o"
                        ),
                        evaluation_dimension_list_name=start_msg["data"].get(
                            "evaluation_dimension_list_name", "sotopia"
                        ),
                        max_turns=start_msg["data"].get("max_turns", 20),
                        npcs=npcs,
                        groups=groups,
                    )

                    # Run the simulation based on the mode
                    await manager.run_simulation(websocket, simulator)

            except WebSocketDisconnect:
                logger.info("Client disconnected during simulation")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                try:
                    await manager.send_error(websocket, ErrorType.SIMULATION_ISSUE, str(e))
                except Exception as e:
                    pass
            finally:
                try:
                    await websocket.close()
                except Exception as e:
                    logger.error(f"Error closing websocket: {e}")


app = SotopiaFastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8800)
