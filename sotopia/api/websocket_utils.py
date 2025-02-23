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
from typing import Type, TypedDict, Any, AsyncGenerator, Literal
from pydantic import BaseModel

ActionType = Literal["none", "speak", "non-verbal communication", "action", "leave"]


class WSMessageType(str, Enum):
    SERVER_MSG = "SERVER_MSG"
    CLIENT_MSG = "CLIENT_MSG"
    ERROR = "ERROR"
    START_SIM = "START_SIM"
    TURN_REQUEST = "TURN_REQUEST"
    TURN_RESPONSE = "TURN_RESPONSE"
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
    # environment_profile = EnvironmentProfile.find().all()[0]
    # agent_profiles = AgentProfile.find().all()[:2]
    assert len(agent_ids) == len(
        agent_models
    ), f"Provided {len(agent_ids)} agent_ids but {len(agent_models)} agent_models"

    environment_profile: EnvironmentProfile = EnvironmentProfile.get(env_id)
    agent_profiles: list[AgentProfile] = [
        AgentProfile.get(agent_id) for agent_id in agent_ids
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
        model_name="gpt-4o-mini",
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
    available_actions: list[ActionType] = [
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
        agent_models: list[str] = ["gpt-4o-mini", "gpt-4o-mini"],
        evaluator_model: str = "gpt-4o",
        evaluation_dimension_list_name: str = "sotopia",
    ) -> None:
        self.env, self.agents, self.environment_messages = get_env_agents(
            env_id,
            agent_ids,
            agent_models,
            evaluator_model,
            evaluation_dimension_list_name,
        )
        # Initialize a structured conversation history for turn-by-turn exchanges.
        # Each entry records the role (e.g., "Environment", "Client", or "Agent"),
        # the agent identifier, and its message content.
        self.conversation_history: list[dict[str, str]] = []
        for agent_name, obs in self.environment_messages.items():
            self.conversation_history.append({
                "role": "environment",
                "agent": agent_name,
                "content": obs.to_natural_language(),
            })
        for index, agent_name in enumerate(self.env.agents):
            self.agents[agent_name].goal = self.env.profile.agent_goals[index]

    async def arun(self) -> AsyncGenerator[dict[str, Any], None]:
        # Use sotopia to run the simulation
        generator = await arun_one_episode(
            env=self.env,
            agent_list=list(self.agents.values()),
            push_to_db=False,
            streaming=True,
        )

        assert isinstance(
            generator, AsyncGenerator
        ), "generator should be async generator, but got {}".format(type(generator))

        async for messages in generator:
            reasoning, rewards = "", [0.0, 0.0]
            if messages[-1][0][0] == "Evaluation":
                reasoning = messages[-1][0][2].to_natural_language()
                rewards = eval(messages[-2][0][2].to_natural_language())

            epilog = EpisodeLog(
                environment=self.env.profile.pk,
                agents=[agent.profile.pk for agent in self.agents.values()],
                tag="test",
                models=["gpt-4o", "gpt-4o", "gpt-4o-mini"],
                messages=[
                    [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
                    for messages_in_turn in messages
                ],
                reasoning=reasoning,
                rewards=rewards,
                rewards_prompt="",
            )

            yield {
                "type": "messages",
                "messages": epilog.dict(),
            }

    async def process_turn(self, client_data: dict) -> dict:
        """
        Process a single turn where client_data contains fields like:
        - agent_id: which agent should act.
        - content: the client-provided input that might update the simulation state.
        """
        # Append the client's input to a conversation log
        self.conversation_history.append({
            "role": "client",
            "content": client_data.get("content", "")
        })
        
        # Identify the specific agent by its ID provided by the client.
        agent_id = client_data.get("agent_id")
        if agent_id not in self.agents:
            raise ValueError(f"Agent with id {agent_id} not found")
        
        # Build an Observation object required by the agent.
        observation = build_observation(turn_number=len(self.conversation_history),
                                        history=self.conversation_history)
        
        # Call the agent’s asynchronous action function.
        agent = self.agents[agent_id]
        agent_action = await agent.aact(observation)
        
        # Append the agent’s response to the conversation log.
        self.conversation_history.append({
            "role": "agent",
            "content": agent_action.argument
        })
        
        # Return a dict that will be sent back over the websocket.
        return {
            "turn": len(self.conversation_history),
            "agent_id": agent_id,
            "agent_response": agent_action.argument,
            "action_type": agent_action.action_type,
        }