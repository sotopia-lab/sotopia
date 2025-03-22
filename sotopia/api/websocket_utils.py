from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    EpisodeLLMEvaluator,
    RuleBasedTerminatedEvaluator,
)
from sotopia.agents import Agents, LLMAgent
from sotopia.messages import Observation
from sotopia.envs import ParallelSotopiaEnv
from sotopia.database import (
    EnvironmentProfile,
    AgentProfile,
    EpisodeLog,
    EvaluationDimensionBuilder,
)
from sotopia.server import arun_one_episode

from enum import Enum
from typing import Type, TypedDict, Any, AsyncGenerator, List, Dict, Optional, Set
from pydantic import BaseModel
import uuid
import json
import asyncio
import redis.asyncio
import hashlib
import logging

logger = logging.getLogger(__name__)

class WSMessageType(str, Enum):
    SERVER_MSG = "SERVER_MSG"
    CLIENT_MSG = "CLIENT_MSG"
    ERROR = "ERROR"
    START_SIM = "START_SIM"
    END_SIM = "END_SIM"
    FINISH_SIM = "FINISH_SIM"


class ErrorType(str, Enum):
    NOT_AUTHORIZED = "NOT_AUTHORIZED"
    SIMULATION_ALREADY_STARTED = "SIMULATION_ALREADY_STARTED"
    SIMULATION_NOT_STARTED = "SIMULATION_NOT_STARTED"
    SIMULATION_ISSUE = "SIMULATION_ISSUE"
    INVALID_MESSAGE = "INVALID_MESSAGE"
    OTHER = "OTHER"


class MessageForRendering(TypedDict):
    role: str
    type: str
    content: str


class WSMessage(BaseModel):
    type: WSMessageType
    data: Dict[str, Any]

    model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "data": self.data,
        }


def get_env_agents(
    env_id: str,
    agent_ids: List[str],
    agent_models: List[str],
    evaluator_model: str,
    evaluation_dimension_list_name: str,
) -> tuple[ParallelSotopiaEnv, Agents, Dict[str, Observation]]:
    assert len(agent_ids) == len(
        agent_models
    ), f"Provided {len(agent_ids)} agent_ids but {len(agent_models)} agent_models"
    try:
        environment_profile: EnvironmentProfile = EnvironmentProfile.get(env_id)
        agent_profiles: List[AgentProfile] = [
            AgentProfile.get(agent_id) for agent_id in agent_ids
        ]
    except Exception:
        environment_profile = EnvironmentProfile(
            codename=f"env_{env_id}",
            scenario="Just chat (finish the conversation in 2 turns)",
            agent_goals=["Just chat"] * len(agent_ids),
        )
        agent_profiles = [
            AgentProfile(
                first_name=f"agent_{agent_id}",
                last_name=f"agent_{agent_id}",
            )
            for agent_id in agent_ids
        ]

    agent_list = [
        LLMAgent(
            agent_profile=agent_profile,
            model_name=agent_models[idx],
        )
        for idx, agent_profile in enumerate(agent_profiles)
    ]
    for idx, goal in enumerate(environment_profile.agent_goals):
        if idx < len(agent_list):
            agent_list[idx].goal = goal

    evaluation_dimensions: Type[BaseModel] = (
        EvaluationDimensionBuilder.select_existing_dimension_model_by_list_name(
            list_name=evaluation_dimension_list_name
        )
    )

    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env = ParallelSotopiaEnv(
        action_order="round-robin",
        evaluators=[
            RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
        ],
        terminal_evaluators=[
            EpisodeLLMEvaluator(
                evaluator_model,
                EvaluationForTwoAgents[evaluation_dimensions],  # type: ignore
            ),
        ],
        env_profile=environment_profile,
    )
    
    # Initialize environment with agents
    environment_messages = {}
    if agent_list:  # Handle case with at least one agent
        environment_messages = env.reset(agents=agents, omniscient=False)
    agents.reset()

    return env, agents, environment_messages


def parse_reasoning(reasoning: str, num_agents: int) -> tuple[List[str], str]:
    """Parse the reasoning string into a dictionary."""
    sep_token = "SEPSEP"
    for i in range(1, num_agents + 1):
        reasoning = (
            reasoning.replace(f"Agent {i} comments:\n", sep_token)
            .strip(" ")
            .strip("\n")
        )
    all_chunks = reasoning.split(sep_token)
    general_comment = all_chunks[0].strip(" ").strip("\n")
    comment_chunks = all_chunks[-num_agents:]

    return comment_chunks, general_comment


class WebSocketSotopiaSimulator:
    def __init__(
        self,
        env_id: str,
        agent_ids: List[str],
        env_profile_dict: Dict[str, Any] = {},
        agent_profile_dicts: List[Dict[str, Any]] = [],
        agent_models: List[str] = ["gpt-4o-mini", "gpt-4o-mini"],
        evaluator_model: str = "gpt-4o",
        evaluation_dimension_list_name: str = "sotopia",
        max_turns: int = 20,
        redis_host: str = "localhost", 
        redis_port: int = 6379,
        redis_db: int = 0,
    ) -> None:
        # Set common attributes regardless of number of agents
        self.env_id = env_id
        self.agent_ids = agent_ids
        self.agent_models = agent_models
        self.evaluator_model = evaluator_model
        self.evaluation_dimension_list_name = evaluation_dimension_list_name
        self.connection_id = str(uuid.uuid4())
        self.max_turns = max_turns
        
        # Group messaging support
        self.mode = "full"  # Communication mode: "full" or "group"
        self.groups = {}  # Dictionary mapping group names to lists of agent names
        
        # Redis connection details
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        self.redis_client = None
        self.redis_pubsub = None
        
        # Redis channel patterns
        self.epilog_channel = f"epilog:{self.connection_id}"
        self.command_channel = f"command:{self.connection_id}"
        
        # Async tasks
        self.redis_subscriber_task = None
        
        # Track the latest epilog to pass to clients
        self.latest_epilog = None
        self.last_epilog_hash = None
        
        # Initialize environment and agents
        try:
            # Use a unified approach for any number of agents
            if not env_profile_dict and not agent_profile_dicts:
                # Use the standard approach with database profiles
                self.env, self.agents, self.environment_messages = get_env_agents(
                    env_id,
                    agent_ids,
                    agent_models,
                    evaluator_model,
                    evaluation_dimension_list_name,
                )
            else:
                # Use provided profile dictionaries
                self.env_profile = EnvironmentProfile(**env_profile_dict)
                self.agent_profiles = [
                    AgentProfile(**agent_profile_dict)
                    for agent_profile_dict in agent_profile_dicts
                ]
        except Exception as e:
            logger.error(f"Error initializing environment or agents: {e}")
            raise

    async def connect_to_redis(self) -> None:
        """Establish connection to Redis and subscribe to channels"""
        if self.redis_client is None:
            try:
                # Connect to Redis
                self.redis_client = redis.asyncio.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db
                )
                
                # Create pubsub interface
                self.redis_pubsub = self.redis_client.pubsub()
                
                # Subscribe to epilog channel
                await self.redis_pubsub.subscribe(self.epilog_channel)
                logger.info(f"Subscribed to Redis channel: {self.epilog_channel}")
                
                # Start subscriber task
                if self.redis_subscriber_task is None or self.redis_subscriber_task.done():
                    self.redis_subscriber_task = asyncio.create_task(self._redis_subscriber())
                    logger.info("Started Redis subscriber task")
                
                # Register with RedisAgent
                await self.send_to_redis({
                    "type": "register",
                    "connection_id": self.connection_id
                })
                
                logger.info(f"Connected to Redis server at {self.redis_url}")
            except Exception as e:
                logger.error(f"Error connecting to Redis: {e}")
                raise

    async def _redis_subscriber(self) -> None:
        """Listen for messages from Redis channels"""
        try:
            while True:
                try:
                    # Wait for a message from Redis
                    message = await self.redis_pubsub.get_message(ignore_subscribe_messages=True)
                    
                    if message and message["type"] == "message":
                        # Process the message
                        try:
                            data = json.loads(message["data"].decode())
                            
                            # Handle epilog updates
                            if data.get("type") == "SERVER_MSG" and data.get("data", {}).get("type") == "episode_log":
                                logger.info("Received epilog update from Redis")
                                
                                # Store the latest epilog
                                self.latest_epilog = data["data"]["log"]
                                
                                # Calculate hash for deduplication
                                epilog_str = json.dumps(self.latest_epilog)
                                self.last_epilog_hash = hashlib.md5(epilog_str.encode()).hexdigest()
                                
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse Redis message: {message['data'][:100]}...")
                        except Exception as e:
                            logger.error(f"Error processing Redis message: {e}")
                
                except Exception as e:
                    logger.error(f"Error in Redis subscriber: {e}")
                    await asyncio.sleep(1)  # Avoid tight loop on error
                
                # Short delay to avoid CPU spinning
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            logger.info("Redis subscriber task cancelled")
        except Exception as e:
            logger.error(f"Fatal error in Redis subscriber: {e}")

    async def send_to_redis(self, message: Dict) -> None:
        """
        Send a message to Redis for the RedisAgent
        
        Args:
            message: The message to send
        """
        if self.redis_client is None:
            await self.connect_to_redis()
        
        try:
            # Publish message to the command channel
            await self.redis_client.publish(
                self.command_channel,
                json.dumps(message)
            )
            logger.info(f"Sent message type: {message.get('type', 'command')} to Redis")
        except Exception as e:
            logger.error(f"Error sending message to Redis: {e}")

    async def set_mode(self, mode: str) -> None:
        """
        Set the communication mode
        
        Args:
            mode: Either "full" for normal operation or "group" for group messaging
        """
        self.mode = mode
        logger.info(f"Set communication mode to: {mode}")
        
        # Notify RedisAgent of mode change
        await self.send_to_redis({"mode": mode})

    async def set_groups(self, groups: Dict) -> None:
        """
        Configure agent groups
        
        Args:
            groups: Dictionary mapping group names to lists of agent names
        """
        self.groups = groups
        logger.info(f"Set groups: {groups}")
        
        # Notify RedisAgent of group configuration
        await self.send_to_redis({"groups": groups})

    async def send_message(self, message: Dict) -> None:
        """
        Send a regular message in full mode
        
        Args:
            message: The message data containing:
                - content: The message content
                - sender: The sender (defaults to "websocket_user")
        """
        # Validate message
        if "content" not in message:
            logger.error("Error: Message must contain 'content' field")
            return
            
        # Set default sender if not provided
        sender = message.get("sender", "websocket_user")
            
        # Create message payload for full mode
        payload = {
            "message": {
                "content": message["content"],
                "sender": sender
            }
        }
        
        # Send to RedisAgent via Redis
        await self.send_to_redis(payload)
        logger.info(f"Sent regular message: {message['content'][:30]}...")

    async def process_group_message(self, message: Dict) -> None:
        """
        Process a client message and send to appropriate agents in group mode
        
        Args:
            message: The message data containing:
                - content: The message content
                - sender: The sender (defaults to "websocket_user")
                - target_agents: List of specific agents to receive this message
                - target_groups: List of groups to receive this message
        """
        # Validate message
        if "content" not in message:
            logger.error("Error: Message must contain 'content' field")
            return
            
        # Set default sender if not provided
        sender = message.get("sender", "websocket_user")
        
        # Get targets
        target_agents = message.get("target_agents", [])
        target_groups = message.get("target_groups", [])
        
        # If no valid targets in group mode, return with error
        if self.mode == "group" and not target_agents and not target_groups:
            logger.error("In group mode, must specify target_agents or target_groups")
            return
            
        # Create the message payload
        payload = {
            "content": message["content"],
            "sender": sender,
            "target_agents": target_agents,
            "target_groups": target_groups
        }
        
        # Send to RedisAgent via Redis
        await self.send_to_redis(payload)
        
        target_description = ""
        if target_agents:
            target_description += f"agents: {target_agents} "
        if target_groups:
            target_description += f"groups: {target_groups}"
            
        logger.info(f"Sent group message to {target_description}: {message['content'][:30]}...")

    async def handle_client_message(self, message: Dict) -> None:
        """
        Handle an incoming client message based on current mode
        
        Args:
            message: The message data from the client
        """
        if self.mode == "full":
            await self.send_message(message)
        else:  # group mode
            await self.process_group_message(message)
    
    async def handle_client_command(self, command: Dict) -> None:
        """
        Handle commands from client such as mode changes or group settings
        
        Args:
            command: The command data
        """
        # Handle mode setting
        if "mode" in command:
            await self.set_mode(command["mode"])
            
        # Handle group configuration
        if "groups" in command:
            await self.set_groups(command["groups"])
    
    async def arun(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the simulation and yield messages
        
        Yields:
            dict: Message data for the WebSocket client
        """
        # Connect to Redis
        await self.connect_to_redis()
        
        try:
            # Set up variables to track epilog updates
            epilog_check_interval = 0.2  # Check for new epilog every 200ms
            last_epilog_hash = None
            
            # Handle different initialization scenarios in a unified way
            if hasattr(self, 'env') and hasattr(self, 'agents'):
                # Standard simulation with agents from database
                generator = await arun_one_episode(
                    env=self.env,
                    agent_list=list(self.agents.values()),
                    push_to_db=False,
                    streaming=True,
                )
                
                # Instead of yielding every individual message,
                # just check for epilog updates periodically
                async for _ in generator:
                    # We don't use the individual messages from the generator,
                    # instead we rely on epilog updates from the RedisAgent
                    
                    # Check if we have a new epilog
                    if self.latest_epilog:
                        # Get current hash
                        current_hash = self.last_epilog_hash
                        
                        if current_hash != last_epilog_hash:
                            # Only yield if it's a new epilog
                            yield {
                                "type": "episode_log",
                                "messages": self.latest_epilog,
                            }
                            last_epilog_hash = current_hash
                    
                    # Allow other tasks to run
                    await asyncio.sleep(epilog_check_interval)
                    
            elif hasattr(self, 'env_profile') and hasattr(self, 'agent_profiles'):
                # Use the multi-agent generator with profile dictionaries
                from sotopia.api.websocket_utils import arun_server_adaptor
                
                multi_agent_generator = arun_server_adaptor(
                    env=self.env_profile,
                    agent_list=self.agent_profiles,
                    agent_models=self.agent_models,
                    evaluator_model=self.evaluator_model,
                    evaluation_dimension_list_name=self.evaluation_dimension_list_name,
                    push_to_db=False,
                    streaming=True,
                    connection_id=self.connection_id,
                    max_turns=self.max_turns,
                )
                
                # Again, don't yield every message, just check for epilog updates
                try:
                    async for _ in multi_agent_generator:
                        # Check if we have a new epilog
                        if self.latest_epilog:
                            # Get current hash
                            current_hash = self.last_epilog_hash
                            
                            if current_hash != last_epilog_hash:
                                # Only yield if it's a new epilog
                                yield {
                                    "type": "episode_log",
                                    "messages": self.latest_epilog,
                                }
                                last_epilog_hash = current_hash
                        
                        # Allow other tasks to run
                        await asyncio.sleep(epilog_check_interval)
                        
                except StopAsyncIteration:
                    # End of regular messaging
                    logger.info("Multi-agent generator completed")
            
            # Continue checking for epilog updates from Redis even after agent generation is done
            while True:
                try:
                    # Check if we have a new epilog from Redis subscriber
                    if self.latest_epilog:
                        # Get current hash
                        current_hash = self.last_epilog_hash
                        
                        if current_hash != last_epilog_hash:
                            # Only yield if it's a new epilog
                            yield {
                                "type": "episode_log",
                                "messages": self.latest_epilog
                            }
                            last_epilog_hash = current_hash
                            
                    # Short delay between checks
                    await asyncio.sleep(epilog_check_interval)
                        
                except asyncio.CancelledError:
                    # Simulation was cancelled
                    logger.info("Simulation cancelled")
                    break
                
        except Exception as e:
            logger.error(f"Error in simulation: {e}")
        finally:
            # Clean up resources
            if self.redis_pubsub:
                await self.redis_pubsub.unsubscribe(self.epilog_channel)
            
            # Unregister from RedisAgent
            if self.redis_client:
                try:
                    await self.send_to_redis({
                        "type": "unregister",
                        "connection_id": self.connection_id
                    })
                except Exception:
                    pass
                    
                await self.redis_client.close()
            
            # Cancel subscriber task
            if self.redis_subscriber_task and not self.redis_subscriber_task.done():
                self.redis_subscriber_task.cancel()
                try:
                    await self.redis_subscriber_task
                except asyncio.CancelledError:
                    pass