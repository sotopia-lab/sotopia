import asyncio
import json
from collections.abc import Callable

import pytest

from sotopia.database import GoalDimension, SotopiaDimensions, SotopiaDimensionsPlus
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
            ("Alice", AgentAction(action_type="leave", argument="", to=[])),
            ("Bob", AgentAction(action_type="none", argument="", to=[])),
        ],
    )
    comment = response[0][1][1]
    assert comment == "Too few active agents; "
    response = evaluator(
        1,
        [
            ("Alice", AgentAction(action_type="speak", argument="Leave!", to=[])),
            ("Bob", AgentAction(action_type="leave", argument="", to=[])),
        ],
    )
    comment = response[0][1][1]
    assert comment == "Too few active agents; "
    response = evaluator(
        3,
        [
            ("Alice", AgentAction(action_type="none", argument="", to=[])),
            ("Bob", AgentAction(action_type="none", argument="", to=[])),
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
            ("Alice", AgentAction(action_type="leave", argument="", to=[])),
            ("Bob", AgentAction(action_type="none", argument="", to=[])),
        ],
    )
    comment = response[0][1][1]
    assert comment == "Too few active agents; "
    response = await evaluator.__acall__(
        1,
        [
            ("Alice", AgentAction(action_type="speak", argument="Leave!", to=[])),
            ("Bob", AgentAction(action_type="leave", argument="", to=[])),
        ],
    )
    comment = response[0][1][1]
    assert comment == "Too few active agents; "
    response = await evaluator.__acall__(
        3,
        [
            ("Alice", AgentAction(action_type="none", argument="", to=[])),
            ("Bob", AgentAction(action_type="none", argument="", to=[])),
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
@pytest.mark.parametrize(
    "dimension_class", [GoalDimension, SotopiaDimensions, SotopiaDimensionsPlus]
)
async def test_episode_llm_evaluator_preserves_object_scores(
    dimension_class: type[BaseModel], mock_llm_response: Callable[..., None]
) -> None:
    scores = {
        "believability": (7, 3),
        "relationship": (-3, 4),
        "knowledge": (6, 2),
        "secret": (-7, -1),
        "social_rules": (-2, 0),
        "financial_and_material_benefits": (-4, 5),
        "goal": (8, 2),
    }
    payload = {
        "evaluations": {
            f"agent_{i + 1}": {
                dimension: {
                    "reasoning": f"Agent {i + 1} reasoning for {dimension}",
                    "score": scores[dimension][i],
                }
                for dimension in dimension_class.model_fields
            }
            for i in range(2)
        }
    }
    mock_llm_response(json.dumps(payload))
    evaluator = EpisodeLLMEvaluator(
        "custom/structured@http://localhost:8000/v1",
        response_format_class=EvaluationForAgents[dimension_class],  # type: ignore[valid-type]
    )
    responses = await evaluator.__acall__(
        1,
        [
            (name, AgentAction(action_type="speak", argument="Hello", to=[]))
            for name in ("Alice", "Bob")
        ],
    )

    assert responses == [
        (
            f"agent_{i + 1}",
            (
                (dimension, scores[dimension][i]),
                f"Agent {i + 1} reasoning for {dimension}",
            ),
        )
        for i in range(2)
        for dimension in dimension_class.model_fields
    ]
    aggregate = unweighted_aggregate_evaluate(responses)
    for i, rate in enumerate((aggregate.p1_rate, aggregate.p2_rate)):
        assert isinstance(rate, tuple)
        expected_scores = {
            dimension: scores[dimension][i]
            for dimension in dimension_class.model_fields
        }
        assert rate[0] == pytest.approx(
            sum(expected_scores.values()) / len(expected_scores)
        )
        assert rate[1] == {**expected_scores, "overall_score": rate[0]}


@pytest.mark.asyncio
async def test_episode_llm_evaluator_preserves_tuple_scores(
    mock_llm_response: Callable[..., None],
) -> None:
    mock_llm_response(
        json.dumps(
            {
                "evaluations": {
                    "agent_1": {"goal": ["Alice achieved her goal", 8]},
                    "agent_2": {"goal": ["Bob made limited progress", 2]},
                }
            }
        )
    )
    evaluator = EpisodeLLMEvaluator(
        "custom/structured@http://localhost:8000/v1",
        response_format_class=EvaluationForAgents[_ReachGoal],
    )
    responses = await evaluator.__acall__(
        1,
        [
            (name, AgentAction(action_type="speak", argument="Hello", to=[]))
            for name in ("Alice", "Bob")
        ],
    )
    assert responses == [
        ("agent_1", (("goal", 8), "Alice achieved her goal")),
        ("agent_2", (("goal", 2), "Bob made limited progress")),
    ]


@pytest.mark.asyncio
async def test_reach_goal_llm_evaluator_async(
    structured_evaluator_model_name: str,
) -> None:
    background = ScriptBackground(
        scenario="Conversation between two friends at a trivia night",
        agent_names=["Alice", "Bob"],
        agent_backgrounds=[
            "Alice is a 29-year-old female software developer. She/her pronouns. Alice can cook very well. Personality and values description: Alice, though somewhat impulsive and free-spirited, values enjoyment. Her decision-making is often spontaneous, staying within familiar boundaries. Alice's secrets: She was once a competitive figure skater.",
            "Bob is a 21-year-old male software developer. He/him pronouns. Bob enjoys biking and photography. Personality and values description: Bob, open-minded and outgoing yet sensitive, advocates care and fairness. His decision-making is intuitive and inclusive. Bob's secrets: He was once a competitive figure skater",
        ],
        agent_goals=[
            "Greet your friends and be polite",
            "Be rude and dismissive to your friends",
        ],
    )

    messages = [
        ("Environment", background),
        ("Environment", SimpleMessage(message="Turn #1")),
        (
            "Alice",
            AgentAction(action_type="speak", argument="Thank you so much!", to=["Bob"]),
        ),
        ("Environment", SimpleMessage(message="Turn #2")),
        ("Bob", AgentAction(action_type="speak", argument="Fuck you!", to=["Alice"])),
        ("Environment", SimpleMessage(message="Turn #3")),
        (
            "Alice",
            AgentAction(
                action_type="speak",
                argument="Hope you have a great weekend.",
                to=["Bob"],
            ),
        ),
        ("Environment", SimpleMessage(message="Turn #4")),
        ("Bob", AgentAction(action_type="leave", argument="Leave", to=["Alice"])),
    ]

    evaluator = EpisodeLLMEvaluator(
        structured_evaluator_model_name,
        response_format_class=EvaluationForAgents[_ReachGoal],
    )

    response2 = await asyncio.gather(evaluator.__acall__(1, messages))
    print("---------------------")
    print("Response after 2 turns:", response2)

    assert len(response2) == 1
    assert len(response2[0]) == 2
