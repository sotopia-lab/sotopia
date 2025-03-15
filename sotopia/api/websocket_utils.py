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
from enum import Enum
from typing import Type, TypedDict, Any, AsyncGenerator, List, Dict, Set, Optional, cast
from pydantic import BaseModel
import uuid
import asyncio
import logging
from sotopia.experimental.server import arun_one_episode

logger = logging.getLogger(__name__)


class WSMessageType(str, Enum):
    SERVER_MSG = "SERVER_MSG"  # Server to client message
    CLIENT_MSG = "CLIENT_MSG"  # Client to server message
    ERROR = "ERROR"  # Error notification
    START_SIM = "START_SIM"  # Initialize simulation
    TURN_REQUEST = "TURN_REQUEST"  # Request for next turn
    TURN_RESPONSE = "TURN_RESPONSE"  # Response with turn results
    NPC_RESPONSE = "NPC_RESPONSE"  # Response from NPC
    END_SIM = "END_SIM"  # End simulation notification
    FINISH_SIM = "FINISH_SIM"  # Terminate simulation


class ErrorType(str, Enum):
    NOT_AUTHORIZED = "NOT_AUTHORIZED"
    SIMULATION_ALREADY_STARTED = "SIMULATION_ALREADY_STARTED"
    SIMULATION_NOT_STARTED = "SIMULATION_NOT_STARTED"
    SIMULATION_ISSUE = "SIMULATION_ISSUE"
    INVALID_MESSAGE = "INVALID_MESSAGE"
    GROUP_NOT_FOUND = "GROUP_NOT_FOUND"
    NPC_NOT_FOUND = "NPC_NOT_FOUND"
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
            "type": self.type.value,
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


def build_observation(turn_number: int, history: list[dict[str, str]]) -> Observation:
    """
    Helper function to build an Observation object from the current conversation.
    Parameters:
    turn_number: The current turn number.
    history: A list of dicts, where each dict contains at least the keys:
            "role" (e.g. "client", "agent", etc.) and "content" (the message text).

    Returns:
    An Observation instance with:
        - last_turn set to the most recent message's content (or an empty string if no history)
        - turn_number set appropriately
        - available_actions a list of possible actions (e.g. ["speak", "react", "leave"])
    """
    # Use the last message's content as the observation's "last_turn"
    last_turn = history[-1]["content"] if history else ""

    # Define the available actions for the agent; adjust as needed.
    available_actions = [
        "speak",
        "non-verbal communication",
        "action",
        "leave",
    ]

    return Observation(
        last_turn=last_turn,
        turn_number=turn_number,
        available_actions=available_actions,
    )


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
    ) -> None:
        # Initialize the simulation environment
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
            self.env_profile = EnvironmentProfile(**env_profile_dict)
            self.agent_profiles = [
                AgentProfile(**agent_profile_dict)
                for agent_profile_dict in agent_profile_dicts
            ]

        self.agent_models = agent_models
        self.evaluator_model = evaluator_model
        self.evaluation_dimension_list_name = evaluation_dimension_list_name
        self.connection_id = str(uuid.uuid4())
        self.max_turns = max_turns

        # New fields for NPC and group management
        self.npc_groups: Dict[
            str, List[str]
        ] = {}  # Map from group ID to list of NPC IDs
        self.active_npcs: Set[str] = set()  # Set of active NPC IDs
        self.turn_number: int = 0  # Current turn number
        self.conversation_history: list[dict[str, str]] = []  # History of messages
        self.pending_responses: Dict[str, Dict[str, Any]] = {}  # Pending NPC responses
        self.response_queue: asyncio.Queue[Dict[str, Any]] = (
            asyncio.Queue()
        )  # Queue for NPC responses

        # Flag for group-based routing
        self.group_mode: bool = False

    async def process_start_msg(self, start_data: dict[str, Any]) -> dict[str, Any]:
        """Process start message with NPC and group information"""
        npcs = start_data.get("npcs", [])
        groups = start_data.get("groups", {})

        self.group_mode = True
        self.active_npcs = set(npcs)
        self.npc_groups = groups

        # Add metadata about NPCs to the conversation history
        self.conversation_history.append(
            {
                "role": "system",
                "content": f"Simulation started with NPCs: {', '.join(npcs)}",
            }
        )

        # For each group, add metadata
        for group_name, group_members in groups.items():
            self.conversation_history.append(
                {
                    "role": "system",
                    "content": f"Group '{group_name}' created with members: {', '.join(group_members)}",
                }
            )

        return {
            "status": "initialized",
            "npcs": list(self.active_npcs),
            "groups": {name: members for name, members in self.npc_groups.items()},
            "conversation_initialized": True,
        }

    async def process_client_message(
        self, client_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process a message from the client, route it to specific NPCs

        client_data should contain:
        - content: message content
        - target_npcs: optional list of specific NPC IDs to message
        - target_group: optional group ID to message all NPCs in that group
        """
        content = client_data.get("content", "")
        target_npcs = client_data.get("target_npcs", [])
        target_group = client_data.get("target_group", None)

        # Add message to conversation history
        self.conversation_history.append({"role": "client", "content": content})

        # Determine which NPCs to message
        npcs_to_message: Set[str] = set()

        # Add specifically targeted NPCs
        if target_npcs:
            npcs_to_message.update(target_npcs)

        # Add all NPCs in the target group
        if target_group and target_group in self.npc_groups:
            npcs_to_message.update(self.npc_groups[target_group])

        # If no targeting specified, message all active NPCs
        if not target_npcs and not target_group:
            npcs_to_message.update(self.active_npcs)

        # Validate that all targeted NPCs exist
        for npc_id in list(npcs_to_message):
            if npc_id not in self.active_npcs:
                return {
                    "status": "error",
                    "error_type": "NPC_NOT_FOUND",
                    "message": f"NPC with ID {npc_id} not found",
                }

        # Increment turn number
        self.turn_number += 1

        # Create observation for each NPC and collect responses
        npc_responses: Dict[str, Any] = {}

        for npc_id in npcs_to_message:
            if npc_id in self.agents:
                # Build observation for this NPC
                agent = self.agents[npc_id]
                observation = build_observation(
                    turn_number=self.turn_number,
                    history=self.conversation_history,
                )

                # Get response from NPC
                try:
                    agent_action = await agent.aact(observation)
                    npc_responses[npc_id] = {
                        "content": agent_action.argument,
                        "action_type": agent_action.action_type,
                    }

                    # Add response to conversation history
                    self.conversation_history.append(
                        {
                            "role": npc_id,
                            "type": agent_action.action_type,
                            "content": agent_action.argument,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error getting response from NPC {npc_id}: {e}")
                    npc_responses[npc_id] = {
                        "status": "error",
                        "message": f"Error getting response: {str(e)}",
                    }

        return {
            "status": "success",
            "turn": self.turn_number,
            "responses": npc_responses,
        }

    async def process_turn(self, client_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process a single turn where client_data contains fields like:
        - agent_id: which agent should act.
        - content: the client-provided input that might update the simulation state.
        """
        # Append the client's input to a conversation log
        self.conversation_history.append(
            {"role": "client", "content": client_data.get("content", "")}
        )

        # Identify the specific agent by its ID provided by the client.
        agent_id = client_data.get("agent_id")
        if not isinstance(agent_id, str):
            raise ValueError("agent_id must be provided as a string")
        if agent_id not in self.agents:
            raise ValueError(f"Agent with id {agent_id} not found")

        # Build an Observation object required by the agent.
        observation = build_observation(
            turn_number=len(self.conversation_history),
            history=self.conversation_history,
        )

        # Call the agent's asynchronous action function.
        agent = self.agents[agent_id]
        agent_action = await agent.aact(observation)

        # Append the agent's response to the conversation log.
        self.conversation_history.append(
            {"role": "agent", "content": agent_action.argument}
        )

        # Return a dict that will be sent back over the websocket.
        return {
            "turn": len(self.conversation_history),
            "agent_id": agent_id,
            "agent_response": agent_action.argument,
            "action_type": agent_action.action_type,
        }

    async def arun(self) -> AsyncGenerator[dict[str, Any], None]:
        """Run the simulation and yield messages"""
        # Handle different simulation modes
        if self.group_mode:
            # Group-based routing mode
            # Just return initial state and let process_client_message handle the rest
            yield {
                "type": "initialization",
                "status": "ready",
                "npcs": list(self.active_npcs),
                "groups": {name: members for name, members in self.npc_groups.items()},
            }

            # Keep simulation alive until shutdown
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass

        elif len(self.agent_models) == 2:
            # Standard two-agent simulation
            generator = await arun_one_episode(
                episode_config={
                    "env_profile": self.env.profile,
                    "agents": [
                        {
                            "agent_profile": agent.profile,
                            "model_name": agent.model_name
                        } for agent in self.agents.values()
                    ],
                    "evaluator_model": self.evaluator_model,
                    "evaluation_dimension_list_name": self.evaluation_dimension_list_name,
                    "push_to_db": False,
                    "streaming": True
                }
            )

            async for messages in generator:
                reasoning, rewards = "", [0.0, 0.0]
                if isinstance(messages, list) and len(messages) > 0 and messages[-1][0][0] == "Evaluation":
                    reasoning = messages[-1][0][2].to_natural_language()
                    rewards = eval(messages[-2][0][2].to_natural_language())

                epilog = EpisodeLog(
                    environment=self.env.profile.pk,
                    agents=[agent.profile.pk for agent in self.agents.values()],
                    tag="test",
                    models=["gpt-4o", "gpt-4o", "gpt-4o-mini"],
                    messages=[
                        [
                            (m[0], m[1], m[2].to_natural_language())
                            for m in messages_in_turn
                        ]
                        for messages_in_turn in messages
                    ] if isinstance(messages, list) else [],
                    reasoning=reasoning,
                    rewards=rewards,
                    rewards_prompt="",
                ).dict()

                yield {
                    "type": "messages",
                    "messages": epilog,
                }
        elif len(self.agent_models) > 2:
            # Multi-agent simulation
            config_data = {
                "redis_url": "redis://localhost:6379/0",
                "extra_modules": [
                    "examples.experimental.sotopia_original_replica.llm_agent_sotopia",
                    "sotopia.experimental.agents.redis_agent",
                ],
                "agent_node": "llm_agent",
                "default_model": "gpt-4o-mini",
                "evaluator_model": self.evaluator_model,
                "use_pk_value": False,
                "push_to_db": False,
                "evaluate_episode": False,
                "max_turns": self.max_turns,
                "scenario": self.env_profile.scenario,
                "agents": [
                    {
                        "name": agent.first_name,
                        "goal": self.env_profile.agent_goals[i] if i < len(self.env_profile.agent_goals) else "",
                        "model_name": self.agent_models[i]
                        if i < len(self.agent_models)
                        else "gpt-4o-mini",
                        "background": agent.dict(),
                    }
                    for i, agent in enumerate(self.agent_profiles)
                ],
                "connection_id": self.connection_id,
            }

            # Add redis_agent to handle WebSocket communication
            config_data["agents"].append({"name": "redis_agent"})

            # Run the multi-agent simulation
            generator = await arun_one_episode(
                episode_config=config_data,
                connection_id=self.connection_id,
            )
            
            async for episode_data in generator:
                yield {
                    "type": "messages",
                    "messages": episode_data,
                }
        else:
            raise ValueError("Number of agents must be 2 or greater")


async def arun_server_adaptor(
    env: EnvironmentProfile,
    agent_list: List[AgentProfile],
    agent_models: List[str],
    evaluator_model: str,
    evaluation_dimension_list_name: str,
    max_turns: int = 20,
    push_to_db: bool = True,
    streaming: bool = False,
    connection_id: str = "",
) -> AsyncGenerator[dict[str, Any], None]:
    # Configure the episode
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
        "max_turns": max_turns,
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
        ],
        "connection_id": connection_id,
    }

    # Add redis_agent to handle WebSocket communication
    config_data["agents"].append({"name": "redis_agent"})

    # Run the episode
    async for episode_data in arun_one_episode(
        episode_config=config_data, connection_id=connection_id
    ):
        yield episode_data