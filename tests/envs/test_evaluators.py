import asyncio

import pytest

from sotopia.envs.evaluators import (
    EvaluationForAgents,
    EpisodeLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    unweighted_aggregate_evaluate,
)
from sotopia.messages import AgentAction, ScriptBackground, SimpleMessage
from pydantic import BaseModel, Field


def test_rule_based_terminated_evaluator() -> None:
    evaluator = RuleBasedTerminatedEvaluator(2, 5)
    response = evaluator(1, [])
    assert len(response) == 1
    assert response[0] == ("environment", (("terminated", False), ""))
    response = evaluator(3, [])
    assert response[0][1][0] == ("terminated", True)
    response = evaluator(
        1,
        [
            ("Alice", AgentAction(action_type="leave", argument="")),
            ("Bob", AgentAction(action_type="none", argument="")),
        ],
    )
    comment = response[0][1][1]
    assert comment == "Agent 1 is leaving; "
    response = evaluator(
        1,
        [
            ("Alice", AgentAction(action_type="speak", argument="Leave!")),
            ("Bob", AgentAction(action_type="leave", argument="")),
        ],
    )
    comment = response[0][1][1]
    assert comment == "Agent 2 is leaving; "
    response = evaluator(
        3,
        [
            ("Alice", AgentAction(action_type="none", argument="")),
            ("Bob", AgentAction(action_type="none", argument="")),
        ]
        * 3,
    )
    comment = response[0][1][1]
    assert (
        comment
        == "The conversation is too long; The conversation stales for too long; "
    )


def test_unweighted_aggregate_evaluate() -> None:
    # Create some response objects
    response1 = (
        "environment",
        (("terminated", True), "nope"),
    )

    response2 = (
        "agent_1",
        (
            ("believability", 0),
            "There was no interaction to evaluate believability.",
        ),
    )
    response3 = (
        "agent_2",
        (
            ("believability", 5),
            "There was no interaction to evaluate believability.",
        ),
    )
    # Call the function being tested
    result = unweighted_aggregate_evaluate([response1, response2, response3])

    # Check that the result is correct
    assert result.terminated is True
    assert isinstance(result.p1_rate, tuple)
    assert isinstance(result.p2_rate, tuple)
    assert result.p1_rate[0] == pytest.approx(0)
    assert result.p2_rate[0] == pytest.approx(5)


# Async tests
@pytest.mark.asyncio
async def test_rule_based_terminated_evaluator_async() -> None:
    evaluator = RuleBasedTerminatedEvaluator(2, 5)
    response = await evaluator.__acall__(1, [])
    assert len(response) == 1
    assert response[0] == ("environment", (("terminated", False), ""))
    response = await evaluator.__acall__(3, [])
    assert response[0][1][0] == ("terminated", True)
    response = await evaluator.__acall__(
        1,
        [
            ("Alice", AgentAction(action_type="leave", argument="")),
            ("Bob", AgentAction(action_type="none", argument="")),
        ],
    )
    comment = response[0][1][1]
    assert comment == "Agent 1 is leaving; "
    response = await evaluator.__acall__(
        1,
        [
            ("Alice", AgentAction(action_type="speak", argument="Leave!")),
            ("Bob", AgentAction(action_type="leave", argument="")),
        ],
    )
    comment = response[0][1][1]
    assert comment == "Agent 2 is leaving; "
    response = await evaluator.__acall__(
        3,
        [
            ("Alice", AgentAction(action_type="none", argument="")),
            ("Bob", AgentAction(action_type="none", argument="")),
        ]
        * 3,
    )
    comment = response[0][1][1]
    assert (
        comment
        == "The conversation is too long; The conversation stales for too long; "
    )


class _ReachGoal(BaseModel):
    goal: tuple[str, int] = Field(
        ..., description="First output a reasoning and then a score for the goal"
    )


@pytest.mark.asyncio
async def test_reach_goal_llm_evaluator_async() -> None:
    evaluator = EpisodeLLMEvaluator(
        "custom/structured@http://localhost:8000/v1",
        response_format_class=EvaluationForAgents[_ReachGoal],
    )
    background = ScriptBackground(
        scenario="Conversation between two friends at a trivia night",
        p1_name="Samuel Anderson",
        p2_name="Giselle Rousseau",
        p1_background="Samuel Anderson is a 29-year-old male software developer. He/him pronouns. Samuel Anderson can cook very well. Personality and values description: Samuel Anderson, though somewhat impulsive and free-spirited, values enjoyment. His decision-making is often spontaneous, staying within familiar boundaries. Samuel's secrets: He was once a competitive figure skater.",
        p2_background="Giselle Rousseau is a 21-year-old nonbinary art student. They/them pronouns. Giselle Rousseau enjoys biking and photography. Personality and values description: Giselle Rousseau, open-minded and outgoing yet sensitive, advocates care and fairness. Her decision-making is intuitive and inclusive. Giselle's secrets: Sells forged paintings to wealthy clients",
        p1_goal="Greet your friends and be polite",
        p2_goal="Be rude and dismissive to your friends",
    )

    # response1,
    response2 = await asyncio.gather(
        evaluator.__acall__(
            1,
            [
                (
                    "Environment",
                    background,
                ),
                (
                    "Environment",
                    SimpleMessage(message="Turn #1"),
                ),
                (
                    "Alice",
                    AgentAction(action_type="speak", argument="Thank you so much!"),
                ),
                (
                    "Environment",
                    SimpleMessage(message="Turn #2"),
                ),
                (
                    "Bob",
                    AgentAction(action_type="speak", argument="Fuck you!"),
                ),
                (
                    "Environment",
                    SimpleMessage(message="Turn #3"),
                ),
                (
                    "Alice",
                    AgentAction(
                        action_type="speak", argument="Hope you have a great weekend."
                    ),
                ),
                ("Environment", SimpleMessage(message="Turn #4")),
                (
                    "Bob",
                    AgentAction(action_type="leave", argument="Leave"),
                ),
            ],
        ),
    )
    print("---------------------")
    print("Response after 2 turns:", response2)

    assert len(response2) == 1
    assert len(response2[0]) == 2
