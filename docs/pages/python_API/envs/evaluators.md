# Sotopia Evaluation Module

## Overview

This module provides various classes and methods to evaluate social interactions in the Sotopia environment, assessing multiple dimensions such as believability, relationship, knowledge, secret, social rules, financial and material benefits, and goal achievement. The evaluations can be synchronous or asynchronous, and aggregate responses to provide a comprehensive summary.

## Classes

### SotopiaDimensions
This class represents the social dimensions used in the Sotopia paper (ICLR 2024).

#### Attributes
- **believability**: Tuple containing reasoning (str) and score (int).
- **relationship**: Tuple containing reasoning (str) and score (int).
- **knowledge**: Tuple containing reasoning (str) and score (int).
- **secret**: Tuple containing reasoning (str) and score (int).
- **social_rules**: Tuple containing reasoning (str) and score (int).
- **financial_and_material_benefits**: Tuple containing reasoning (str) and score (int).
- **goal**: Tuple containing reasoning (str) and score (int).

#### Validators
- **zero_to_ten_validator**: Ensures the score is between 0 and 10.
- **minus_five_to_five_validator**: Ensures the score is between -5 and 5.
- **minus_ten_to_zero_validator**: Ensures the score is between -10 and 0.

### SotopiaDimensionsPlus
An updated version of `SotopiaDimensions` with more detailed instructions for each dimension.

### GoalDimension
This class evaluates only the goal achievement.

#### Attributes
- **goal**: Tuple containing reasoning (str) and score (int).

#### Validators
- **zero_to_ten_validator**: Ensures the score is between 0 and 10.

### EvaluationForTwoAgents
A generic class to evaluate two agents simultaneously.

#### Attributes
- **agent_1_evaluation**: Evaluation results for agent 1.
- **agent_2_evaluation**: Evaluation results for agent 2.

### Evaluator
Abstract base class for evaluators.

#### Methods
- **__call__**: Abstract method to perform evaluation.
- **__acall__**: Abstract method to perform asynchronous evaluation.

### RuleBasedTerminatedEvaluator
This class evaluates conversations based on rule-based criteria for termination.

#### Attributes
- **max_turn_number**: Maximum number of turns before termination.
- **max_stale_turn**: Maximum number of stale turns before termination.

#### Methods
- **__call__**: Performs the evaluation and returns termination status.
- **__acall__**: Asynchronous version of the `__call__` method.

### ReachGoalLLMEvaluator
This class evaluates goal achievement using a language model.

#### Attributes
- **model_name**: Name of the language model.
- **response_format_class**: Class type for the evaluation response format.

#### Methods
- **__call__**: Not implemented for synchronous evaluation.
- **__acall__**: Asynchronous evaluation method using a language model.

## Functions

### _reduce
Reduces a list of responses by averaging the scores and aggregating comments.

#### Parameters
- **responses_per_reducer**: List of tuples containing response and reasoning.

#### Returns
- Tuple containing reduced dictionary of scores and aggregated comments.

### unweighted_aggregate_evaluate
Aggregates responses from the environment.

#### Parameters
- **responses**: List of responses from the environment.

#### Returns
- An instance of `ScriptEnvironmentResponse`.

## Usage Examples

```python
import logging
from sotopia.evaluators import (
    SotopiaDimensions,
    SotopiaDimensionsPlus,
    GoalDimension,
    EvaluationForTwoAgents,
    RuleBasedTerminatedEvaluator,
    ReachGoalLLMEvaluator,
    unweighted_aggregate_evaluate
)

log = logging.getLogger("evaluators")

# Example 1: Creating an instance of SotopiaDimensions
dimensions = SotopiaDimensions(
    believability=("Agent interacts naturally.", 8),
    relationship=("Relationship improved.", 3),
    knowledge=("Gained new knowledge.", 7),
    secret=("No secrets revealed.", 0),
    social_rules=("No rules violated.", 0),
    financial_and_material_benefits=("Marginal gain.", 2),
    goal=("Achieved most goals.", 7),
)

# Example 2: Evaluating with RuleBasedTerminatedEvaluator
evaluator = RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2)
termination_status = evaluator.__call__(
    turn_number=21,
    messages=[("Agent 1", AgentAction(action_type="talk")), ("Agent 2", AgentAction(action_type="leave"))]
)

print(termination_status)
# Output: [('environment', (('terminated', True), 'Agent 2 is leaving; '))]

# Example 3: Asynchronous evaluation with ReachGoalLLMEvaluator
import asyncio

async def evaluate():
    evaluator = ReachGoalLLMEvaluator("gpt-3", EvaluationForTwoAgents[SotopiaDimensions])
    result = await evaluator.__acall__(turn_number=10, messages=[...])
    aggregated_response = unweighted_aggregate_evaluate(result)
    print(aggregated_response)

# Running the asynchronous evaluation
asyncio.run(evaluate())
