import asyncio
import json
import logging
import uuid
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TypedDict, Literal, cast
from contextlib import asynccontextmanager

from redis_om import get_redis_connection
import rq
import redis.asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, model_validator, field_validator, Field

from sotopia.api.websocket_utils import (
    WebSocketSotopiaSimulator,
    WSMessageType,
    WSMessage,
    ErrorType
)
from sotopia.database import (
    EnvironmentProfile,
    AgentProfile,
    EpisodeLog,
    NonStreamingSimulationStatus,
    RelationshipProfile,
    RelationshipType,
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

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token state manager for simulation sessions
class SimulationState:
    _instance = None
    _lock = asyncio.Lock()
    _active_simulations = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._active_simulations = {}
        return cls._instance

    async def try_acquire_token(self, token: str) -> tuple[bool, str]:
        """Try to acquire a token for simulation"""
        async with self._lock:
            if not token:
                return False, "Invalid token"

            if self._active_simulations.get(token):
                return False, "Token is already active"

            self._active_simulations[token] = True
            return True, "Token is valid"

    async def release_token(self, token: str) -> None:
        """Release a token after simulation ends"""
        async with self._lock:
            self._active_simulations.pop(token, None)

    @asynccontextmanager
    async def start_simulation(self, token: str):
        """Context manager for simulation session"""
        try:
            yield True
        finally:
            await self.release_token(token)

# Data models for API endpoints
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

# Simulation manager to handle WebSocket connections
class SimulationManager:
    def __init__(self, redis_host="localhost", redis_port=6379, redis_db=0):
        self.state = SimulationState()
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify if the token is valid for simulation
        
        Args:
            token: Authentication token
            
        Returns:
            dict: Status of token verification
        """
        is_valid, msg = await self.state.try_acquire_token(token)
        return {"is_valid": is_valid, "msg": msg}

    async def handle_client_message(
        self,
        websocket: WebSocket,
        simulator: WebSocketSotopiaSimulator,
        message: Dict[str, Any]
    ) -> bool:
        """
        Process a message from a WebSocket client
        
        Args:
            websocket: The WebSocket connection
            simulator: The simulation manager
            message: The message from the client
            
        Returns:
            bool: True if the simulation should end, False otherwise
        """
        try:
            msg_type = message.get("type")
            data = message.get("data", {})
            
            # Handle simulation finish request
            if msg_type == WSMessageType.FINISH_SIM.value:
                logger.info("Client requested to finish simulation")
                return True
            
            # Handle mode switching
            if "mode" in data:
                mode = data["mode"]
                if mode not in ["full", "group"]:
                    await self.send_error(
                        websocket, 
                        ErrorType.INVALID_MESSAGE, 
                        f"Invalid mode: {mode}. Must be 'full' or 'group'"
                    )
                    return False
                    
                logger.info(f"Setting communication mode to: {mode}")
                await simulator.set_mode(mode)
                
                # Acknowledge the mode change
                await self.send_message(
                    websocket,
                    WSMessageType.SERVER_MSG,
                    {"status": "mode_updated", "mode": mode}
                )
            
            # Handle group configuration
            if "groups" in data:
                logger.info(f"Setting up groups: {data['groups']}")
                await simulator.set_groups(data["groups"])
                
                # Acknowledge the groups configuration
                await self.send_message(
                    websocket,
                    WSMessageType.SERVER_MSG,
                    {"status": "groups_updated", "groups": data["groups"]}
                )
                
            # Handle messages
            if msg_type == WSMessageType.CLIENT_MSG.value:
                # Check if this is a message with content
                if "message" in data or "content" in data:
                    # Get the mode (either from the message or use the current mode)
                    mode = data.get("mode", simulator.mode)
                    
                    if mode == "full":
                        # In full mode, just forward the message
                        message_content = data.get("content", "")
                        if "message" in data:
                            message_content = data["message"].get("content", message_content)
                        
                        if not message_content:
                            await self.send_error(
                                websocket,
                                ErrorType.INVALID_MESSAGE,
                                "Message must include content"
                            )
                            return False
                        
                        # Forward to simulator
                        await simulator.send_message({
                            "content": message_content,
                            "sender": data.get("sender", "websocket_user")
                        })
                        
                    else:  # group mode
                        # In group mode, we need target information
                        message_content = data.get("content", "")
                        if "message" in data:
                            message_content = data["message"].get("content", message_content)
                        
                        if not message_content:
                            await self.send_error(
                                websocket,
                                ErrorType.INVALID_MESSAGE,
                                "Message must include content"
                            )
                            return False
                        
                        # Check for target agents or groups
                        target_agents = data.get("target_agents", [])
                        if "message" in data:
                            target_agents = data["message"].get("target_agents", target_agents)
                            
                        target_groups = data.get("target_groups", [])
                        if "message" in data:
                            target_groups = data["message"].get("target_groups", target_groups)
                        
                        if not target_agents and not target_groups:
                            await self.send_error(
                                websocket,
                                ErrorType.INVALID_MESSAGE,
                                "Group mode message must specify either target_agents or target_groups"
                            )
                            return False
                        
                        # Forward to simulator
                        await simulator.process_group_message({
                            "content": message_content,
                            "sender": data.get("sender", "websocket_user"),
                            "target_agents": target_agents,
                            "target_groups": target_groups
                        })
                    
                    logger.info(f"Processed {mode} mode message: {message_content[:30]}...")
            
            return False
        except Exception as e:
            error_msg = f"Error handling client message: {e}"
            logger.error(error_msg)
            await self.send_error(websocket, ErrorType.INVALID_MESSAGE, error_msg)
            return False

    async def run_simulation(
        self, websocket: WebSocket, simulator: WebSocketSotopiaSimulator
    ) -> None:
        """Run the simulation and process client messages"""
        try:
            # Start the simulation tasks
            sim_task = asyncio.create_task(self._process_simulation(websocket, simulator, None))
            client_task = asyncio.create_task(self._process_client_messages(websocket, simulator))
            
            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [sim_task, client_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel the remaining task
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
        except Exception as e:
            msg = f"Error running simulation: {e}"
            logger.error(msg)
            await self.send_error(websocket, ErrorType.SIMULATION_ISSUE, msg)
        finally:
            # Always send END_SIM message
            await self.send_message(websocket, WSMessageType.END_SIM, {})
    
    async def run_simulation(
        self, websocket: WebSocket, simulator: WebSocketSotopiaSimulator
    ) -> None:
        """Run the simulation and process client messages"""
        try:
            # Start the simulation tasks
            sim_task = asyncio.create_task(self._process_simulation(websocket, simulator, None))
            client_task = asyncio.create_task(self._process_client_messages(websocket, simulator))
            
            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [sim_task, client_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel the remaining task
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
        except Exception as e:
            msg = f"Error running simulation: {e}"
            logger.error(msg)
            await self.send_error(websocket, ErrorType.SIMULATION_ISSUE, msg)
        finally:
            # Always send END_SIM message
            await self.send_message(websocket, WSMessageType.END_SIM, {})
    
    async def _process_simulator_epilogs(self, websocket: WebSocket, simulator: WebSocketSotopiaSimulator) -> None:
        """Process epilog updates from the simulator"""
        try:
            async for message in simulator.arun():
                await self.send_message(websocket, WSMessageType.SERVER_MSG, message)
        except Exception as e:
            logger.error(f"Error processing simulator epilogs: {e}")
            raise
    
    async def _process_redis_epilogs(self, websocket: WebSocket, pubsub: redis.asyncio.client.PubSub) -> None:
        """Process epilog updates directly from Redis"""
        try:
            while True:
                # Get the next message from Redis
                message = await pubsub.get_message(ignore_subscribe_messages=True)
                
                if message and message["type"] == "message":
                    try:
                        # Parse the message
                        data = json.loads(message["data"].decode())
                        
                        # Forward SERVER_MSG epilog updates to the client
                        if data.get("type") == "SERVER_MSG" and data.get("data", {}).get("type") == "episode_log":
                            await websocket.send_json(data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse Redis message: {message['data'][:100]}...")
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
                
                # Short delay to avoid CPU spinning
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in Redis epilog processor: {e}")
            raise

    async def _process_client_messages(self, websocket: WebSocket, simulator: WebSocketSotopiaSimulator) -> None:
        """Process messages from the client"""
        try:
            while True:
                data = await websocket.receive_json()
                should_end = await self.handle_client_message(websocket, simulator, data)
                if should_end:
                    break
        except Exception as e:
            logger.error(f"Error processing client messages: {e}")
            raise
            
    async def create_simulator(
        self,
        env_id: str,
        agent_ids: List[str],
        agent_models: Optional[List[str]] = None,
        evaluator_model: str = "gpt-4o",
        evaluation_dimension_list_name: str = "sotopia",
        env_profile_dict: Optional[Dict[str, Any]] = None,
        agent_profile_dicts: Optional[List[Dict[str, Any]]] = None,
        max_turns: int = 20,
    ) -> WebSocketSotopiaSimulator:
        """Create and initialize a WebSocketSotopiaSimulator"""
        try:
            # Set defaults for optional parameters
            if agent_models is None:
                agent_models = ["gpt-4o-mini"] * len(agent_ids)
                
            if env_profile_dict is None:
                env_profile_dict = {}
                
            if agent_profile_dicts is None:
                agent_profile_dicts = []
                
            # Create simulator with Redis configuration
            simulator = WebSocketSotopiaSimulator(
                env_id=env_id,
                agent_ids=agent_ids,
                agent_models=agent_models,
                evaluator_model=evaluator_model,
                evaluation_dimension_list_name=evaluation_dimension_list_name,
                env_profile_dict=env_profile_dict,
                agent_profile_dicts=agent_profile_dicts,
                max_turns=max_turns,
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                redis_db=self.redis_db
            )
            return simulator
        except Exception as e:
            error_msg = f"Failed to create simulator: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
    @staticmethod
    async def send_message(
        websocket: WebSocket, msg_type: WSMessageType, data: Dict[str, Any]
    ) -> None:
        """Send a message to the WebSocket client"""
        await websocket.send_json({"type": msg_type.value, "data": data})

    @staticmethod
    async def send_error(
        websocket: WebSocket, error_type: ErrorType, details: str = ""
    ) -> None:
        """Send an error message to the WebSocket client"""
        await websocket.send_json(
            {
                "type": WSMessageType.ERROR.value,
                "data": {"type": error_type.value, "details": details},
            }
        )

# Non-streaming simulation functions and helper functions
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

# API helper functions
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

# Create FastAPI app with routes
app = FastAPI(title="Sotopia Group Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
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

# Scenario endpoints
@app.get("/scenarios", response_model=list[EnvironmentProfile])
async def scenarios_all():
    return await get_scenarios_all()

@app.get("/scenarios/{get_by}/{value}", response_model=list[EnvironmentProfile])
async def scenarios_filtered(get_by: Literal["id", "codename"], value: str):
    return await get_scenarios(get_by, value)

@app.post("/scenarios", response_model=str)
async def create_scenario(scenario: BaseEnvironmentProfile) -> str:
    scenario_profile = EnvironmentProfile(**scenario.model_dump())
    scenario_profile.save()
    pk = scenario_profile.pk
    assert pk is not None
    return pk

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

# Agent endpoints
@app.get("/agents", response_model=list[AgentProfile])
async def agents_all():
    return await get_agents_all()

@app.get("/agents/{get_by}/{value}", response_model=list[AgentProfile])
async def agents_filtered(get_by: Literal["id", "gender", "occupation"], value: str):
    return await get_agents(get_by, value)

@app.post("/agents", response_model=str)
async def create_agent(agent: BaseAgentProfile) -> str:
    agent_profile = AgentProfile(**agent.model_dump())
    agent_profile.save()
    pk = agent_profile.pk
    assert pk is not None
    return pk

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

# Relationship endpoints
@app.get("/relationship/{agent_1_id}/{agent_2_id}", response_model=str)
async def relationship_get(agent_1_id: str, agent_2_id: str):
    return await get_relationship(agent_1_id, agent_2_id)

@app.post("/relationship", response_model=str)
async def create_relationship(relationship: BaseRelationshipProfile) -> str:
    relationship_profile = RelationshipProfile(**relationship.model_dump())
    relationship_profile.save()
    pk = relationship_profile.pk
    assert pk is not None
    return pk

@app.delete("/relationship/{relationship_id}", response_model=str)
async def delete_relationship(relationship_id: str) -> str:
    RelationshipProfile.delete(relationship_id)
    return relationship_id

# Episode endpoints
@app.get("/episodes", response_model=list[EpisodeLog])
async def episodes_all():
    return await get_episodes_all()

@app.get("/episodes/{get_by}/{value}", response_model=list[EpisodeLog])
async def episodes_filtered(get_by: Literal["id", "tag"], value: str):
    return await get_episodes(get_by, value)

@app.delete("/episodes/{episode_id}", response_model=str)
async def delete_episode(episode_id: str) -> str:
    EpisodeLog.delete(episode_id)
    return episode_id

# Evaluation dimension endpoints
@app.get("/evaluation_dimensions", response_model=dict[str, list[CustomEvaluationDimension]])
async def evaluation_dimensions_all():
    return await get_evaluation_dimensions()

@app.post("/evaluation_dimensions", response_model=str)
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

@app.delete(
    "/evaluation_dimensions/{evaluation_dimension_list_name}",
    response_model=str,
)
async def delete_evaluation_dimension_list(
    evaluation_dimension_list_name: str,
) -> str:
    CustomEvaluationDimensionList.delete(evaluation_dimension_list_name)
    return evaluation_dimension_list_name

# Model endpoints
@app.get("/models", response_model=list[str])
async def models_all():
    return await get_models()

# Simulation endpoints
@app.post("/simulate", response_model=str)
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

@app.get("/simulation_status/{episode_pk}", response_model=str)
async def get_simulation_status(episode_pk: str) -> str:
    status = NonStreamingSimulationStatus.find(
        NonStreamingSimulationStatus.episode_pk == episode_pk
    ).all()[0]
    assert isinstance(status, NonStreamingSimulationStatus)
    return status.status

# WebSocket endpoint for group chat simulation
@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket, token: str) -> None:
    """
    WebSocket endpoint for Sotopia simulations with group chat support
    
    Args:
        websocket: The WebSocket connection
        token: Authentication token
    """
    manager = SimulationManager()

    # Verify token
    token_status = await manager.verify_token(token)
    if not token_status["is_valid"]:
        await websocket.close(code=1008, reason=token_status["msg"])
        return

    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted for token: {token}")

        # Wait for the first message (should be START_SIM)
        start_msg = await websocket.receive_json()
        if start_msg.get("type") != WSMessageType.START_SIM.value:
            await manager.send_error(
                websocket,
                ErrorType.INVALID_MESSAGE,
                "First message must be of type START_SIM"
            )
            await websocket.close(code=1008)
            return

        # Extract simulation parameters
        sim_data = start_msg.get("data", {})
        env_id = sim_data.get("env_id", "")
        agent_ids = sim_data.get("agent_ids", [])

        # Create and run the simulation
        async with manager.state.start_simulation(token):
            try:
                # Create the simulator
                simulator = await manager.create_simulator(
                    env_id=env_id,
                    agent_ids=agent_ids,
                    agent_models=sim_data.get("agent_models", ["gpt-4o-mini"] * len(agent_ids)),
                    evaluator_model=sim_data.get("evaluator_model", "gpt-4o"),
                    evaluation_dimension_list_name=sim_data.get("evaluation_dimension_list_name", "sotopia"),
                    env_profile_dict=sim_data.get("env_profile_dict", {}),
                    agent_profile_dicts=sim_data.get("agent_profile_dicts", []),
                    max_turns=sim_data.get("max_turns", 20),
                )
                
                # Ensure simulator connects to Redis
                await simulator.connect_to_redis()
                
                # Configure groups and mode if provided in START_SIM
                initial_mode = sim_data.get("mode", "full")
                logger.info(f"Initial mode: {initial_mode}")
                await simulator.set_mode(initial_mode)
                
                if "groups" in sim_data:
                    await simulator.set_groups(sim_data["groups"])
                
                # Initial message to client
                await manager.send_message(
                    websocket,
                    WSMessageType.SERVER_MSG,
                    {
                        "status": "simulation_started",
                        "env_id": simulator.env_id,
                        "agent_ids": simulator.agent_ids,
                        "mode": initial_mode,
                        "groups": sim_data.get("groups", {}),
                        "connection_id": simulator.connection_id
                    }
                )
                logger.info("WebSocket start sim message confirmation sent.")
                # Run the simulation
                await manager.run_simulation(websocket, simulator)
                
            except Exception as e:
                logger.error(f"Error creating or running simulator: {e}")
                await manager.send_error(
                    websocket,
                    ErrorType.SIMULATION_ISSUE,
                    f"Error in simulation: {str(e)}"
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {token}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection: {e}")
    finally:
        try:
            await websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8800)