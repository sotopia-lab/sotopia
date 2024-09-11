## Overview

The environments in Sotopia are designed to simulate and evaluate social interactions between agents. They are structured to provide a realistic and dynamic simulation environment, allowing for the testing and evaluation of various social scenarios and behaviors.

## Key Features

- **Parallel Execution**: Environments can run in parallel, allowing for the simulation of multiple agents and interactions simultaneously.
- **Dynamic Interactions**: Agents can engage in dynamic interactions, with behaviors and responses influenced by their internal states and external stimuli.
- **Evaluation Metrics**: Environments provide a set of evaluation metrics to assess the quality of social interactions, including believability, relationship, knowledge, secret, social rules, financial and material benefits, and goal achievement.
- **Customizable**: Environments can be customized to test different scenarios and behaviors, and to integrate with different types of agents and profiles.

## Types of Environments

### ParallelSotopiaEnv

The `ParallelSotopiaEnv` is a parallel environment designed for the simulation of multiple agents and interactions. It is structured to provide a realistic and dynamic simulation environment, allowing for the testing and evaluation of various social scenarios and behaviors.

#### Key Features

- **Parallel Execution**: Environments can run in parallel, allowing for the simulation of multiple agents and interactions simultaneously.
- **Dynamic Interactions**: Agents can engage in dynamic interactions, with behaviors and responses influenced by their internal states and external stimuli.

## Evaluation

The evaluation of social interactions in Sotopia is crucial for understanding the performance and effectiveness of the simulation environment. The evaluation is based on a set of metrics that are used to assess the quality of social interactions, including believability, relationship, knowledge, secret, social rules, financial and material benefits, and goal achievement.

### Custom Evaluators

Custom evaluators can be created to evaluate the quality of social interactions in Sotopia. They can be used to evaluate the performance of the simulation environment, and to test and evaluate different scenarios and behaviors. Here is an example of a custom evaluator:

```python
from sotopia.envs.evaluators import ReachGoalLLMEvaluator
from sotopia.envs.evaluation import EvaluationForAIAgent, SotopiaDimensions

class CustomEvaluation(BaseModel):
    custom_score: tuple[str, int] = Field(
        ...,
        description="your custom evaluation"
    )

class EnvResponse(BaseModel):
    agent_1_evaluation: SotopiaDimensions
    agent_2_evaluation: CustomEvaluation

evaluator = ReachGoalLLMEvaluator(
    model_name="gpt-4o",
    response_format_class=EnvResponse,
)
```
In the example above, we created a custom evaluation metric called `CustomEvaluation`. This metric is a tuple of a string and an integer, where the string is the reasoning for the score and the integer is the score itself.
Through this, we can evaluate the performance of the simulation environment, and to test and evaluate different scenarios and behaviors.
