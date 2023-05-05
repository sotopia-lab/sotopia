import abc
import logging

from beartype import beartype
from langchain.output_parsers import PydanticOutputParser

from sotopia.generation_utils.generate import (
    ListOfIntOutputParser,
    LLM_Name,
    agenerate,
    generate,
)
from sotopia.messages import (
    AgentAction,
    Message,
    ScriptEnvironmentResponse,
)

log = logging.getLogger("evaluators")


class Evaluator(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> ScriptEnvironmentResponse:
        raise NotImplementedError

    @abc.abstractmethod
    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> ScriptEnvironmentResponse:
        raise NotImplementedError


@beartype
class RuleBasedTerminatedEvaluator(Evaluator):
    def __init__(
        self, max_turn_number: int = 20, max_stale_turn: int = 5
    ) -> None:
        self.max_turn_number = max_turn_number
        self.max_stale_turn = max_stale_turn

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> ScriptEnvironmentResponse:
        # Rule 1: If the conversation is too long, terminate the conversation
        conversation_too_long = turn_number > self.max_turn_number
        # Rule 2: If one of the players leaves, terminate the conversation
        p1_leaving = (
            len(messages) > 1
            and isinstance(messages[-2][1], AgentAction)
            and messages[-2][1].action_type == "leave"
        )
        p2_leaving = (
            bool(len(messages))
            and isinstance(messages[-1][1], AgentAction)
            and messages[-1][1].action_type == "leave"
        )
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
        terminated = (
            conversation_too_long or p1_leaving or p2_leaving or stale_too_long
        )
        return ScriptEnvironmentResponse(
            conversation_too_long=conversation_too_long,
            p1_leaving=p1_leaving,
            p2_leaving=p2_leaving,
            stale_too_long=stale_too_long,
            terminated=terminated,
            p1_rate=0,
            p2_rate=0,
        )

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> ScriptEnvironmentResponse:
        return self(turn_number, messages)


@beartype
class ReachGoalLLMEvaluator(Evaluator):
    def __init__(self, model_name: LLM_Name) -> None:
        self.model_name = model_name

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> ScriptEnvironmentResponse:
        history = "\n".join(
            [
                f"{x}: {y.to_natural_language()}"
                if x != "Environment"
                else y.to_natural_language()
                for x, y in messages
            ]
        )

        try:
            response = generate(
                model_name=self.model_name,
                template="""
                    {history},
                    How well do participants achieve their goals (give scores on a scale of 0-9, where 0 indicates that the participants did not achieve their goals at all and 9 indicates that the participants achieved their goals perfectly)?
                    Please following the format:
                    {format_instructions}
                """,
                input_values=dict(history=history),
                output_parser=ListOfIntOutputParser(2, (0, 9)),
            )
            return ScriptEnvironmentResponse(
                conversation_too_long=False,
                p1_leaving=False,
                p2_leaving=False,
                stale_too_long=False,
                terminated=False,
                p1_rate=response[0],
                p2_rate=response[1],
            )
        except Exception as e:
            log.debug(f"[red] Failed to generate environment response. {e}")
            print(e)
            return ScriptEnvironmentResponse(
                conversation_too_long=False,
                p1_leaving=False,
                p2_leaving=False,
                stale_too_long=False,
                terminated=False,
                p1_rate=None,
                p2_rate=None,
            )

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> ScriptEnvironmentResponse:
        history = "\n".join(
            [
                f"{x}: {y.to_natural_language()}"
                if x != "Environment"
                else y.to_natural_language()
                for x, y in messages
            ]
        )

        try:
            response = await agenerate(
                model_name=self.model_name,
                template="""{history},
                    How well do participants achieve their goals (give scores on a scale of 0-9, where 0 indicates that the participants did not achieve their goals at all and 9 indicates that the participants achieved their goals perfectly)?
                    Please following the format:
                    {format_instructions}
                """,
                input_values=dict(history=history),
                output_parser=ListOfIntOutputParser(2, (0, 9)),
            )
            return ScriptEnvironmentResponse(
                conversation_too_long=False,
                p1_leaving=False,
                p2_leaving=False,
                stale_too_long=False,
                terminated=False,
                p1_rate=response[0],
                p2_rate=response[1],
            )
        except Exception as e:
            log.debug(f"[red] Failed to generate environment response. {e}")
            return ScriptEnvironmentResponse(
                conversation_too_long=False,
                p1_leaving=False,
                p2_leaving=False,
                stale_too_long=False,
                terminated=False,
                p1_rate=None,
                p2_rate=None,
            )


@beartype
def unweighted_aggregate_evaluate(
    responses: list[ScriptEnvironmentResponse],
) -> ScriptEnvironmentResponse:
    """
    Aggregate the responses from the environment
    """
    conversation_too_long = any([x.conversation_too_long for x in responses])
    p1_leaving = any([x.p1_leaving for x in responses])
    p2_leaving = any([x.p2_leaving for x in responses])
    stale_too_long = any([x.stale_too_long for x in responses])
    terminated = (
        any([x.terminated for x in responses])
        or conversation_too_long
        or p1_leaving
        or p2_leaving
        or stale_too_long
    )
    try:
        p1_rate = sum(
            [x.p1_rate if x.p1_rate else 0 for x in responses]
        ) / sum([1 if x.p1_rate else 0 for x in responses])
    except:
        p1_rate = None
    try:
        p2_rate = sum(
            [x.p2_rate if x.p2_rate else 0 for x in responses]
        ) / sum([1 if x.p2_rate else 0 for x in responses])
    except:
        p2_rate = None
    response = ScriptEnvironmentResponse(
        conversation_too_long=conversation_too_long,
        p1_leaving=p1_leaving,
        p2_leaving=p2_leaving,
        stale_too_long=stale_too_long,
        terminated=terminated,
        p1_rate=p1_rate,
        p2_rate=p2_rate,
    )
    if terminated:
        log.debug(f"[green] The conversation is terminated. {response}")
    return response
