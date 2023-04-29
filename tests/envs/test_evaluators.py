import pytest

from sotopia.envs.evaluators import (
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    unweighted_aggregate_evaluate,
)
from sotopia.messages import (
    AgentAction,
    Observation,
    ScriptEnvironmentResponse,
)


def test_rule_based_teminated_evaluator() -> None:
    evaluator = RuleBasedTerminatedEvaluator(2, 5)
    response = evaluator(1, [])
    assert response.conversation_too_long == False
    response = evaluator(3, [])
    assert response.conversation_too_long == True
    response = evaluator(
        1,
        [
            ("Alice", AgentAction(action_type="leave", argument="")),
            ("Bob", AgentAction(action_type="none", argument="")),
        ],
    )
    assert response.p1_leaving == True
    assert response.p2_leaving == False
    response = evaluator(
        1,
        [
            ("Alice", AgentAction(action_type="speak", argument="Leave!")),
            ("Bob", AgentAction(action_type="leave", argument="")),
        ],
    )
    assert response.p1_leaving == False
    assert response.p2_leaving == True
    response = evaluator(
        1, [("Alice", AgentAction(action_type="none", argument=""))]
    )
    assert response.stale_too_long == False
    response = evaluator(
        3,
        [
            ("Alice", AgentAction(action_type="none", argument="")),
            ("Bob", AgentAction(action_type="none", argument="")),
        ]
        * 3,
    )
    assert response.stale_too_long == True


def test_reach_goal_llm_evaluator() -> None:
    evaluator = ReachGoalLLMEvaluator("gpt-4")
    response = evaluator(
        1,
        [
            (
                "Environment",
                Observation(
                    last_turn="Please say something.",
                    turn_number=0,
                    available_actions=["speak", "none"],
                ),
            ),
            ("Alice", AgentAction(action_type="speak", argument="")),
            ("Bob", AgentAction(action_type="speak", argument="")),
        ],
    )
    assert response.p1_rate == 0
    assert response.p2_rate == 0
    response = evaluator(
        1,
        [
            (
                "Environment",
                Observation(
                    last_turn="Please express gratitude to each other.",
                    turn_number=0,
                    available_actions=["speak", "none"],
                ),
            ),
            (
                "Alice",
                AgentAction(
                    action_type="speak", argument="Thank you so much!"
                ),
            ),
            ("Bob", AgentAction(action_type="speak", argument="Fuck you!")),
        ],
    )
    assert isinstance(response.p1_rate, float)
    assert isinstance(response.p2_rate, float)
    assert response.p1_rate > response.p2_rate


def test_unweighted_aggregate_evaluate() -> None:
    # Create some response objects
    response1 = ScriptEnvironmentResponse(
        conversation_too_long=False,
        p1_leaving=True,
        p2_leaving=False,
        stale_too_long=False,
        terminated=False,
        p1_rate=None,
        p2_rate=7.5,
    )
    response2 = ScriptEnvironmentResponse(
        conversation_too_long=True,
        p1_leaving=False,
        p2_leaving=False,
        stale_too_long=False,
        terminated=False,
        p1_rate=8.0,
        p2_rate=None,
    )
    response3 = ScriptEnvironmentResponse(
        conversation_too_long=False,
        p1_leaving=False,
        p2_leaving=False,
        stale_too_long=True,
        terminated=True,
        p1_rate=9.0,
        p2_rate=4.0,
    )

    # Call the function being tested
    result = unweighted_aggregate_evaluate([response1, response2, response3])

    # Check that the result is correct
    assert result.conversation_too_long == True
    assert result.p1_leaving == True
    assert result.p2_leaving == False
    assert result.stale_too_long == True
    assert result.terminated == True
    assert result.p1_rate == pytest.approx(8.5)
    assert result.p2_rate == pytest.approx(5.75)
