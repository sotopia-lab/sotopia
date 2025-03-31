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
# from sotopia.server import arun_one_episode

from enum import Enum
from typing import Type, TypedDict, Any, AsyncGenerator, List, Dict
from pydantic import BaseModel
import uuid
from sotopia.experimental.server import arun_one_episode
import logging
from redis.asyncio import Redis
import json



# Setup logging
logging.basicConfig(level=logging.INFO)
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
    data: dict[str, Any]

    model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.type.value,  # TODO check whether we want to use the enum value or the enum itself
            "data": self.data,
        }


def get_env_agents(
    env_id: str,
    agent_ids: list[str],
    agent_models: list[str],
    evaluator_model: str,
    evaluation_dimension_list_name: str,
) -> tuple[ParallelSotopiaEnv, Agents, dict[str, Observation]]:
    assert len(agent_ids) == len(
        agent_models
    ), f"Provided {len(agent_ids)} agent_ids but {len(agent_models)} agent_models"
    try:
        environment_profile: EnvironmentProfile = EnvironmentProfile.get(env_id)
        agent_profiles: list[AgentProfile] = [
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
    if len(agent_ids) == 2:
        environment_messages = env.reset(agents=agents, omniscient=False)
    agents.reset()

    return env, agents, environment_messages


def parse_reasoning(reasoning: str, num_agents: int) -> tuple[list[str], str]:
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
        agent_ids: list[str],
        env_profile_dict: dict[str, Any] = {},
        agent_profile_dicts: list[dict[str, Any]] = [],
        agent_models: list[str] = ["gpt-4o-mini", "gpt-4o-mini"],
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
        # Redis channel
        self.command_channel = f"websocket:{self.connection_id}"
        # Redis connection details
        self.redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        self.redis_client = None
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
                self.env_profile.save()
                assert self.env_profile.pk is not None
                self.env_id = self.env_profile.pk
                self.agent_profiles = [
                    AgentProfile(**agent_profile_dict)
                    for agent_profile_dict in agent_profile_dicts
                ]
                self.agent_ids = []
                for agent_profile in self.agent_profiles:
                    agent_profile.save()
                    assert agent_profile.pk is not None
                    self.agent_ids.append(agent_profile.pk)
        except Exception as e:
            logger.error(f"Error initializing environment or agents: {e}")
            raise

    async def connect_to_redis(self) -> None:
        """Establish connection to Redis and subscribe to channels"""
        if self.redis_client is None:
            try:
                # Connect to Redis
                self.redis_client = Redis.from_url(self.redis_url)
                # Create pubsub interface
                self.redis_pubsub = self.redis_client.pubsub()

                # Subscribe to epilog channel
                await self.redis_pubsub.subscribe(self.command_channel)
                logger.info(f"Subscribed to Redis channel: {self.command_channel}")
            except Exception as e:
                logger.error(f"Error connecting to Redis: {e}")
                raise

    async def send_to_redis(self, message: Dict) -> None:
        """
        Send a message to Redis for the RedisAgent

        Args:
            message: The message to send
        """
        try:
            # Publish message to the command channel
            print("From websocket")
            print(message)
            await self.redis_client.publish(self.command_channel, json.dumps(message))
            msg_type = message.get("type", "command")
            content_preview = ""
            if "content" in message:
                content_preview = f": {message['content'][:30]}..."
            elif "message" in message and "content" in message["message"]:
                content_preview = f": {message['message']['content'][:30]}..."
            logger.info(
                f"[{self.connection_id}] Sent message type: {msg_type}{content_preview}"
            )
        except Exception as e:
            logger.error(f"Error sending message to Redis: {e}")

    async def send_message(self, message: Dict) -> None:
        """
        Send a regular message in full mode

        Args:
            message: The message data containing:
                - content: The message content
                - sender: The sender (defaults to "redis_agent")
        """
        # Validate message
        if "content" not in message:
            logger.error("Error: Message must contain 'content' field")
            return

        # Set default sender if not provided
        sender = message.get("sender", "redis_agent")
        receiver =  message.get("receiver", "all")

        # Create message payload for full mode
        payload = {"message": {"content": message["content"], "sender": sender, "receiver":receiver}}

        # Send to RedisAgent via Redis
        await self.send_to_redis(payload)
        logger.info(
            f"[{self.connection_id}] Sent full mode message: {message['content'][:30]}..."
        )

    def prepare_episode_config(self) -> Dict[str, Any]:
        """
        Prepare the simulation configuration for arun_one_episode

        Returns:
            Dict: Episode configuration
        """
        # Base configuration
        config = {
            "redis_url": self.redis_url,
            "extra_modules": [
                "examples.experimental.sotopia_original_replica.llm_agent_sotopia",
            ],
            "agent_node": "llm_agent",
            "default_model": "gpt-4o-mini",
            "evaluator_model": self.evaluator_model,
            "use_pk_value": True,
            "push_to_db": False,  # TODO: check, do we need to push epilog to redis database? Probably not.
            "evaluate_episode": False,
            "max_turns": self.max_turns,
            "scenario": self.env_profile.scenario,
            "connection_id": self.connection_id,
            "redis_url": self.redis_url,
            "agents": [
                {
                    "name": profile.first_name,
                    "goal": self.env_profile.agent_goals[i],
                    "model_name": self.agent_models[i]
                    if i < len(self.agent_models)
                    else "gpt-4o-mini",
                    "agent_pk": profile.pk,
                    "background": {},
                }
                for i, profile in enumerate(self.agent_profiles)
            ],
        }       
        return config

    async def arun(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the simulation and yield message updates from the queue.

        Yields:
            dict: Message data for the WebSocket client
        """
        # Reset control flags
        episode_config = self.prepare_episode_config()

        # Use the new arun_one_episode function
        from sotopia.experimental.server import arun_one_episode

        async for message in arun_one_episode(episode_config, self.connection_id):
            yield message
