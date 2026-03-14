import asyncio
from collections.abc import Callable
from sotopia.envs.evaluators import (
    EvaluationForAgents,
    EpisodeLLMEvaluator,
    RuleBasedTerminatedEvaluator,
)
from sotopia.agents import Agents, LLMAgent, BaseAgent
from sotopia.messages import Observation, AgentAction
from sotopia.envs import ParallelSotopiaEnv
from sotopia.database import (
    EnvironmentProfile,
    AgentProfile,
    EpisodeLog,
    EvaluationDimensionBuilder,
)
from sotopia.server import arun_one_episode

from enum import Enum
from typing import Type, TypedDict, Any, AsyncGenerator, List, Optional
from pydantic import BaseModel
import uuid


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
    environment_messages = {}
    
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
                EvaluationForAgents[evaluation_dimensions],  # type: ignore
            ),
        ],
        env_profile=environment_profile,
    )
    if len(agent_ids) == 2:
        environment_messages = env.reset(agents=agents, omniscient=False)
    else:
        environment_messages = {}
    agents.reset()
    return env, agents, environment_messages if 'environment_messages' in locals() else {}


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


class WebSocketHumanAgent(BaseAgent[Observation, AgentAction]):
    """
    A human-controlled agent that receives actions via WebSocket.
    
    This agent waits for human input from the frontend instead of
    generating actions via an LLM.
    """
    
    def __init__(
        self,
        agent_name: str,
        agent_profile: AgentProfile,
        action_queue_getter: Callable[[], asyncio.Queue[dict[str, Any]]],
        goal: str = "",
    ) -> None:
        super().__init__(agent_name=agent_name, agent_profile=agent_profile)
        self._action_queue_getter = action_queue_getter
        self.goal = goal
        self._current_observation: Optional[Observation] = None
        
    @property
    def action_queue(self) -> asyncio.Queue[dict[str, Any]]:
        """Get the action queue dynamically"""
        return self._action_queue_getter()
    
    async def aact(self, obs: Observation) -> AgentAction:
        """
        Wait for human action from the WebSocket queue.
        
        The frontend sends actions via CLIENT_MSG, which are queued
        by the server and consumed here.
        """
        self._current_observation = obs
        
        # Wait for human to submit their action
        try:
            action_data = await asyncio.wait_for(
                self.action_queue.get(),
                timeout=300.0  # 5 minute timeout for human response
            )
        except asyncio.TimeoutError:
            # Return a timeout action if human doesn't respond
            return AgentAction(
                action_type="none",
                argument="[Timed out waiting for response]",
                to=[],
            )
        
        # Parse the action from frontend
        action_type = action_data.get("action_type", "speak")
        argument = action_data.get("content", "")
        
        # Normalize action_type to canonical form
        if action_type in ["kill", "vote", "save", "poison", "inspect", "investigate", "guard", "protect", "heal", "lynch"]:
            # Convert verb actions to 'action' type with verb in argument
            if not argument:
                argument = action_type
            else:
                argument = f"{action_type} {argument}"
            action_type = "action"
        
        return AgentAction(
            action_type=action_type,
            argument=argument,
            to=[],
        )
    
    def act(self, obs: Observation) -> AgentAction:
        """Synchronous version - not typically used for human agents"""
        raise NotImplementedError(
            "WebSocketHumanAgent requires async operation via aact()"
        )
    
    def reset(self) -> None:
        """Reset the agent state"""
        self._current_observation = None
        # Clear any pending actions
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


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
        game_type: str = "chat",
        human_agent_index: Optional[int] = None,
        action_queue_getter: Optional[Callable[[], asyncio.Queue[dict[str, Any]]]] = None,
    ) -> None:
        self.game_type = game_type
        self.human_agent_index = human_agent_index
        self.action_queue_getter = action_queue_getter
        
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

    async def arun(self) -> AsyncGenerator[dict[str, Any], None]:
        # Use sotopia to run the simulation
        if len(self.agent_models) == 2:
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
                yield {
                    "type": "messages",
                    "messages": epilog,
                }
        elif len(self.agent_models) > 2:
            multi_agent_generator: AsyncGenerator[dict[str, Any], None] = (
                arun_server_adaptor(
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
            )
            assert isinstance(
                multi_agent_generator, AsyncGenerator
            ), "generator should be async generator, but got {}".format(
                type(multi_agent_generator)
            )

            async for message_data in multi_agent_generator:
                yield {
                    "type": "messages",
                    "messages": message_data,
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
    # Prepare episode configuration
    from sotopia.experimental.server import arun_one_episode

    # TODO: Unify the API of the two agents
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
    }
    # Use the arun_one_episode function from server.py
    async for episode_data in arun_one_episode(
        episode_config=config_data,
        connection_id=connection_id,
    ):
        yield episode_data
