import abc
import logging
from collections import defaultdict
from typing import Any, Generic, TypeVar

import gin
from pydantic import BaseModel, validate_call

from sotopia.generation_utils import (
    PydanticOutputParser,
    agenerate,
    custom_temperature,
    default_temperature,
)
from sotopia.messages import (
    AgentAction,
    Message,
    ScriptEnvironmentResponse,
)

log = logging.getLogger("evaluators")

T_eval_dim = TypeVar("T_eval_dim", bound=BaseModel)


class EvaluationForAgents(BaseModel, Generic[T_eval_dim]):
    evaluations: dict[str, T_eval_dim]


class Evaluator(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError


class SocialGameEndEvaluator(Evaluator):
    """Base evaluator for social game win conditions.

    Subclasses should implement _check_win_conditions() to check
    game-specific win conditions using the environment state.
    """

    def __init__(self, max_turn_number: int = 100) -> None:
        self.max_turn_number = max_turn_number

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # Check turn limit
        if turn_number >= self.max_turn_number:
            return [("environment", (("terminated", True), "Max turns reached"))]

        # Extract environment from kwargs
        env = kwargs.get("env")
        if not env:
            return [("environment", (("terminated", False), ""))]

        # Check game-specific win conditions
        terminated, reason = self._check_win_conditions(env, turn_number, messages)
        return [("environment", (("terminated", terminated), reason))]

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        return self.__call__(turn_number, messages, **kwargs)

    def _check_win_conditions(
        self, env: Any, turn_number: int, messages: list[tuple[str, Message]]
    ) -> tuple[bool, str]:
        """Check game-specific win conditions. Override in subclasses."""
        return False, ""


class RuleBasedTerminatedEvaluator(Evaluator):
    def __init__(self, max_turn_number: int = 20, max_stale_turn: int = 2) -> None:
        self.max_turn_number = max_turn_number
        self.max_stale_turn = max_stale_turn

    @validate_call
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # Rule 1: If the conversation is too long, terminate the conversation
        conversation_too_long = turn_number >= self.max_turn_number
        # Rule 2: If fewer than two agents remain active (not left), terminate
        # Determine latest action per agent, and count those whose latest is not "leave"
        latest_action_by_agent: dict[str, str] = {}
        observed_agents: set[str] = set()
        for speaker, msg in messages:
            if speaker != "Environment":
                observed_agents.add(speaker)

        for speaker, msg in messages[::-1]:
            if speaker == "Environment":
                continue
            if not isinstance(msg, AgentAction):
                continue
            if speaker not in latest_action_by_agent:
                latest_action_by_agent[speaker] = msg.action_type

        # If we haven't observed any agent messages yet, do not terminate early
        env = kwargs.get("env")
        if env:
            all_agents = set(env.agents)
        else:
            all_agents = observed_agents

        if all_agents:
            num_active_agents = sum(
                1
                for agent in all_agents
                if latest_action_by_agent.get(agent, "speak") != "leave"
            )
        else:
            num_active_agents = 2

        too_few_agents = num_active_agents < 2
        # Rule 3: If the conversation is stale for too long, terminate the conversation
        stale_count = 0
        for message in messages[::-1]:
            if message[0] == "Environment":
                continue
            assert isinstance(message[1], AgentAction)
            if message[1].action_type == "none":
                stale_count += 1
            else:
                break
            if stale_count > self.max_stale_turn:
                break
        stale_too_long = stale_count > self.max_stale_turn
        terminated = conversation_too_long or too_few_agents or stale_too_long
        reasons_for_termination = (
            f"{'The conversation is too long; ' if conversation_too_long else ''}"
            f"{'Too few active agents; ' if too_few_agents else ''}"
            f"{'The conversation stales for too long; ' if stale_too_long else ''}"
        )
        return [
            (
                "environment",
                (("terminated", terminated), reasons_for_termination),
            )
        ]

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        return self(turn_number, messages, **kwargs)


class EpisodeLLMEvaluator(Evaluator, Generic[T_eval_dim]):
    def __init__(
        self,
        model_name: str,
        response_format_class: type[EvaluationForAgents[T_eval_dim]],
    ) -> None:
        self.model_name = model_name
        self.prompt = ""
        self.response_format_class = response_format_class

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError(
            "ReachGoalLLMEvaluator is not implemented for synchronous evaluation"
        )

    @gin.configurable
    @validate_call
    async def __acall__(
        self,
        turn_number: int,
        messages: list[tuple[str, Message]] | None,
        history: str = "",
        temperature: float | None = 0.0,
        **kwargs: Any,
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # filter did nothing
        if not history and messages:
            messages_filtered = [
                (x, y)
                for x, y in messages
                if "did nothing" not in y.to_natural_language()
            ]
            history = "\n".join(
                [
                    (
                        f"{x} {y.to_natural_language()}"
                        if x != "Environment"
                        else y.to_natural_language()
                    )
                    for x, y in messages_filtered
                ]
            )

        try:
            # Count actual participating agents (exclude Environment)
            participating_agents = set()
            if messages:
                for speaker, _ in messages:
                    if speaker != "Environment":
                        participating_agents.add(speaker)
            num_agents = len(participating_agents)

            # Build explicit agent label instruction to avoid ambiguous dynamic keys in structured output
            agent_instruction = ""
            if num_agents > 0:
                agent_instruction = (
                    "There are exactly "
                    + str(num_agents)
                    + " agents. Under the 'evaluations' field, use exactly these keys: "
                    + "["
                    + ", ".join([f'"agent_{i+1}"' for i in range(num_agents)])
                    + "] (no other keys).\n"
                )

            temperature_setting = (
                default_temperature(temperature)
                if temperature == 0.0
                else custom_temperature(temperature)
            )

            response: EvaluationForAgents[T_eval_dim] = await agenerate(
                model_name=self.model_name,
                template="""{history}
                    Based on previous interactions, evaluate how well participants achieve their goals.
                    {agent_instruction}
                    Please follow the format:
                    {format_instructions}
                """,
                input_values=dict(history=history, agent_instruction=agent_instruction),
                output_parser=PydanticOutputParser[self.response_format_class](  # type: ignore[name-defined]
                    pydantic_object=self.response_format_class
                ),
                temperature=temperature_setting,
                structured_output=self.model_name.startswith("custom/structured"),
            )
            response_list = []
            # Only process evaluations for the actual number of agents
            for i, evaluation in enumerate(
                list(response.evaluations.values())[:num_agents]
            ):
                # Map agent names to expected format (agent_1, agent_2, etc.)
                agent_key = f"agent_{i+1}"
                for dimension in evaluation.model_dump().keys():
                    response_list.append(
                        (
                            agent_key,
                            (
                                (
                                    dimension,
                                    evaluation.model_dump()[dimension][1],
                                ),
                                evaluation.model_dump()[dimension][0],
                            ),
                        )
                    )
            return response_list
        except Exception as e:
            print(e)
            log.debug(f"[red] Failed to generate environment response. {e}")
            return []


@validate_call
def _reduce(
    responses_per_reducer: list[tuple[tuple[str, float | int | bool], str]],
) -> tuple[dict[str, float | int | bool], str]:
    responses_dict = defaultdict(list)
    comments_dict: dict[str, str] = defaultdict(str)
    reduced_dict: dict[str, float | int | bool] = {}
    for response, reasoning in responses_per_reducer:
        responses_dict[response[0]].append(response[1])
        comments_dict[response[0]] += reasoning
    scores: list[float | int] = []
    for k, v in responses_dict.items():
        if k == "terminated":
            assert all([isinstance(x, bool) for x in v])
            reduced_dict[k] = any(v)
        else:
            assert all([isinstance(x, (float, int)) for x in v])
            reduced_dict[k] = sum(v) / len(v)
            scores.append(reduced_dict[k])
    if len(scores) and "overall_score" not in responses_dict:
        scores = [x for x in scores if x is not None]
        reduced_dict["overall_score"] = sum(scores) / len(scores)
    comments = "\n".join([f"{k}: {v}" for k, v in comments_dict.items() if v])
    return reduced_dict, comments


@validate_call
def unweighted_aggregate_evaluate(
    responses: list[tuple[str, tuple[tuple[str, int | float | bool], str]]],
) -> ScriptEnvironmentResponse:
    """
    Aggregate the responses from the environment

    Args:
        responses (list[tuple[str, tuple[tuple[str, int | bool], str]]]): list of responses from the environment
        Each response is a tuple of (agent_name/environment, (response, reasoning))
    """
    responses_dict: dict[str, list[tuple[tuple[str, int | float | bool], str]]] = (
        defaultdict(list)
    )
    for response in responses:
        # Relaxed assertion: allow any key for agents, not just "agent_X"
        # assert response[0] == "environment" or response[0].startswith("agent")
        responses_dict[response[0]].append(response[1])

    environment_responses: tuple[dict[str, float | int | bool], str] = ({}, "")
    agent_responses: dict[str, tuple[dict[str, float | int | bool], str]] = {}

    for k, v in responses_dict.items():
        if k == "environment":
            environment_responses = _reduce(v)
        else:
            # Support any number of agents (agent_1, agent_2, agent_3, etc.)
            agent_responses[k] = _reduce(v)

    # Build comments from all agents dynamically
    agent_comments = ""
    for agent_key, (_, comment) in agent_responses.items():
        if comment:
            agent_name = agent_key.replace("_", " ").title()
            agent_comments += f"{agent_name} comments:\n{comment}\n"

    comments = (
        f"Environment comments: {environment_responses[1]}\n"
        if environment_responses[1]
        else ""
    ) + agent_comments
    if (
        "terminated" in environment_responses[0]
        and environment_responses[0]["terminated"]
    ):
        log.debug(f"[green] The conversation is terminated. {response}")
    # Get first two agents for backward compatibility with ScriptEnvironmentResponse
    agent_1_responses = agent_responses.get("agent_1", ({}, ""))
    agent_2_responses = agent_responses.get("agent_2", ({}, ""))

    return ScriptEnvironmentResponse(
        terminated=environment_responses[0]["terminated"]
        if "terminated" in environment_responses[0]
        else False,
        p1_rate=(
            agent_1_responses[0]["overall_score"]
            if "overall_score" in agent_1_responses[0]
            else 0,
            agent_1_responses[0],
        )
        if agent_1_responses != ({}, "")
        else None,
        p2_rate=(
            agent_2_responses[0]["overall_score"]
            if "overall_score" in agent_2_responses[0]
            else 0,
            agent_2_responses[0],
        )
        if agent_2_responses != ({}, "")
        else None,
        comments=comments,
        rewards={
            k: (v[0]["overall_score"] if "overall_score" in v[0] else 0, v[0])
            for k, v in agent_responses.items()
        },
    )
