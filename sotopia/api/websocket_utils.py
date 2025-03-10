from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    EpisodeLLMEvaluator,
    RuleBasedTerminatedEvaluator,
)
from sotopia.agents import Agents, LLMAgent
from sotopia.messages import Observation, SimpleMessage
from sotopia.envs import ParallelSotopiaEnv
from sotopia.database import (
    EnvironmentProfile,
    AgentProfile,
    EpisodeLog,
    EvaluationDimensionBuilder,
)
from sotopia.server import arun_one_episode

from enum import Enum
from typing import Type, TypedDict, Any, AsyncGenerator, List
from pydantic import BaseModel
import asyncio
import json
import os
import redis.asyncio  # Use Redis async client
from sotopia.experimental import generate_executable
import uuid
from sotopia.messages import Message


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
    ) -> None:
        if len(agent_ids) == 2:
            try:
                self.env, self.agents, self.environment_messages = get_env_agents(
                    env_id,
                    agent_ids,
                    agent_models,
                    evaluator_model,
                    evaluation_dimension_list_name,
                )
            except Exception as e:
                raise Exception(f"Error in loading environment or agents profiles: {e}")

            for index, agent_name in enumerate(self.env.agents):
                self.agents[agent_name].goal = self.env.profile.agent_goals[index]
        else:
            assert (
                env_profile_dict
            ), "env_profile_dict must be provided if number of agents is greater than 2"
            assert agent_profile_dicts, "agent_profile_dicts must be provided if number of agents is greater than 2"
            self.env = EnvironmentProfile(**env_profile_dict)
            self.agents = [
                AgentProfile(**agent_profile_dict)
                for agent_profile_dict in agent_profile_dicts
            ]
            self.agent_models = agent_models
            self.evaluator_model = evaluator_model
            self.evaluation_dimension_list_name = evaluation_dimension_list_name

        self.connection_id = str(uuid.uuid4())

    async def arun(self) -> AsyncGenerator[dict[str, Any], None]:
        # Use sotopia to run the simulation
        if len(self.agents) == 2:
            generator = await arun_one_episode(
                env=self.env,
                agent_list=list(self.agents.values()),
                push_to_db=False,
                streaming=True,
            )
        elif len(self.agents) > 2:
            generator = arun_sotopia_original_replica(
                env=self.env,
                agent_list=self.agents,
                agent_models=self.agent_models,
                evaluator_model=self.evaluator_model,
                evaluation_dimension_list_name=self.evaluation_dimension_list_name,
                push_to_db=False,
                streaming=True,
                connection_id=self.connection_id,
            )
        else:
            raise ValueError("Number of agents must be 2 or greater")

        assert isinstance(
            generator, AsyncGenerator
        ), "generator should be async generator, but got {}".format(type(generator))

        async for messages in generator:
            reasoning, rewards = "", [0.0, 0.0]
            if messages[-1][0][0] == "Evaluation":
                reasoning = messages[-1][0][2].to_natural_language()
                rewards = eval(messages[-2][0][2].to_natural_language())
            try:
                epilog = EpisodeLog(
                    environment=self.env.profile.pk,
                    agents=[agent.profile.pk for agent in self.agents.values()],
                    tag="test",
                    messages=[
                        [
                            (m[0], m[1], m[2].to_natural_language())
                            for m in messages_in_turn
                        ]
                        for messages_in_turn in messages
                    ],
                    reasoning=reasoning,
                    rewards=rewards,
                    rewards_prompt="",
                ).dict()
            except Exception:
                epilog = {
                    "environment": self.env.pk,
                    "agents": [agent.pk for agent in self.agents],
                    "tag": "test",
                    "messages": [
                        [
                            (m[0], m[1], m[2].to_natural_language())
                            for m in messages_in_turn
                        ]
                        for messages_in_turn in messages
                    ],
                    "reasoning": reasoning,
                    "rewards": rewards,
                    "rewards_prompt": "",
                }
            yield {
                "type": "messages",
                "messages": epilog,
            }


async def arun_sotopia_original_replica(
    env: EnvironmentProfile,
    agent_list: List[AgentProfile],
    agent_models: List[str],
    evaluator_model: str,
    evaluation_dimension_list_name: str,
    push_to_db: bool = True,
    streaming: bool = False,
    connection_id: str = "",
) -> AsyncGenerator[List[List[tuple[str, str, Message]]], None]:
    """
    Run a Sotopia simulation using the original replica method for more than 2 agents.

    Args:
        env: The Sotopia environment
        agent_list: List of agents participating in the simulation
        push_to_db: Whether to push results to database
        streaming: Whether to stream results

    Returns:
        AsyncGenerator yielding simulation messages
    """
    # Create data folder if it doesn't exist
    data_folder = os.path.join(os.getcwd(), "data")  # Need to be careful here
    os.makedirs(data_folder, exist_ok=True)

    # Create raw config file in the data folder
    output_toml_path = os.path.join(data_folder, f"output_{connection_id}.toml")

    # Prepare raw config data
    config_data = {
        "redis_url": "redis://localhost:6379/0",
        "extra_modules": [
            "examples.experimental.sotopia_original_replica.llm_agent_sotopia",
            "sotopia.experimental.agents.redis_agent",
        ],
        "agent_node": "llm_agent",
        "default_model": "gpt-4o-mini",
        "evaluator_model": evaluator_model,
        "use_pk_value": False,
        "push_to_db": push_to_db,
        "evaluate_episode": False,
        "max_turns": 3,
        "scenario": env.scenario,
        "agents": [
            {
                "name": agent.first_name,
                "goal": env.agent_goals[i] if i < len(env.agent_goals) else "",
                "model_name": agent_models[i]
                if i < len(agent_models)
                else "gpt-4o-mini",
                "background": agent.dict(),
            }
            for i, agent in enumerate(agent_list)
        ]
        + [
            {
                "name": "redis_agent",
            }
        ],
        "connection_id": connection_id,
    }
    generate_executable(config_data, output_toml_path)
    # Run the dataflow
    run_cmd = f"aact run-dataflow {output_toml_path}"
    # Start the process and capture stdout and stderr for debugging
    proc = await asyncio.create_subprocess_shell(
        run_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    # Create tasks to read and log stdout and stderr without blocking
    async def log_stdout():
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            print(f"Dataflow output: {line.decode().strip()}")

    async def log_stderr():
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            print(f"Dataflow error: {line.decode().strip()}")

    # Start the logging tasks
    asyncio.create_task(log_stdout())
    asyncio.create_task(log_stderr())

    # Connect to Redis using the async client
    redis_client = redis.asyncio.Redis(host="localhost", port=6379, db=0)
    pubsub = redis_client.pubsub()
    channel = f"{connection_id}" if connection_id else "sotopia:simulation"
    print(f"Subscribing to channel: {channel}")
    await pubsub.subscribe(channel)

    # Parse output and yield messages using Redis pubsub

    try:
        # Process messages from pubsub using async iterator
        async for message in pubsub.listen():
            print("--------------------------------")
            print(f"Received message: {message}")
            # Skip subscription confirmation messages
            if message["type"] != "message":
                continue

            # Process the message data
            try:
                line_str = message["data"].decode()
                message_data = json.loads(line_str)
                potential_episode_log = json.loads(message_data.get("last_turn"))
                if "messages" in potential_episode_log:
                    messages = potential_episode_log["messages"]
                    processed_messages = []
                    for i, message in enumerate(messages):
                        processed_messages.append(
                            [
                                (
                                    message[0][0],
                                    message[0][1],
                                    SimpleMessage(message=message[0][2]),
                                )
                            ]
                        )
                    yield processed_messages
                else:
                    print(f"oooooook message: {message_data}")

            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                # Skip messages that aren't valid JSON or have unexpected format
                continue

    except asyncio.CancelledError:
        # Handle cancellation
        raise
    finally:
        # Clean up Redis connection
        await pubsub.unsubscribe(channel)
        await pubsub.close()
        await redis_client.close()

        # Terminate the process if it's still running
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()

        # Check for errors
        if proc.returncode and proc.returncode != 0:
            stderr_content = await proc.stderr.read()
            raise RuntimeError(f"Dataflow execution failed: {stderr_content.decode()}")
