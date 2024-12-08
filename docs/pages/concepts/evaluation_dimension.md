## Overview

Evaluation dimensions are used to evaluate the quality of social interactions.
In original Sotopia paper, there are 7 dimensions to evaluate the quality of social interactions, where we named them as `sotopia` evaluation dimensions:
- believability
- relationship
- knowledge
- secret
- social rules
- financial and material benefits
- goal

The `SotopiaDimensions` can be used directly without initializing the database. It provides a set of predefined evaluation dimensions that are ready to use for evaluating social interactions. For example,

```python
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.envs.evaluators import EvaluationForTwoAgents, ReachGoalLLMEvaluator, RuleBasedTerminatedEvaluator, SotopiaDimensions

env = ParallelSotopiaEnv(
    env_profile=env_profile,
        model_name=model_names["env"],
        action_order="round-robin",
        evaluators=[
            RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
        ],
        terminal_evaluators=[
            ReachGoalLLMEvaluator(
                model_names["env"],
                EvaluationForTwoAgents[SotopiaDimensions],  # type: ignore
                # TODO check how to do type annotation
            ),
        ],
    )
```


However we observe under many use cases people may want to evaluate with customized evaluation metrics, so we provide a way to build custom evaluation dimensions.
For a quick reference, you can directly check out the `examples/use_custom_dimensions.py`.

### CustomEvaluationDimension
The [`CustomEvaluationDimension`](/python_API/database/evaluation_dimensions) is a class that can be used to create a custom evaluation dimension.
There are four parameters:
- name: the name of the dimension
- description: the description of the dimension
- range_low: the minimum score of the dimension (should be an integer)
- range_high: the maximum score of the dimension (should be an integer)

### CustomEvaluationDimensionList
The [`CustomEvaluationDimensionList`](/python_API/database/evaluation_dimensions) is a class that can be used to create a custom evaluation dimension list based on the existing dimensions. It helps one to group multiple dimensions together for a specific use case.
There are two parameters:
- name: the name of the dimension list
- dimension_pks: the primary keys of the dimensions in the dimension list

### EvaluationDimensionBuilder
The [`EvaluationDimensionBuilder`](/python_API/database/evaluation_dimensions) is a class that can be used to generate a custom evaluation dimension model based on the existing dimensions.


## Usage
### Initialize the database
The default evaluation metric is still `SotopiaDimensions` in `sotopia.env.evaluators`.There is no `CustomEvaluationDimension` in the database by default. To initialize the database, please refer to `examples/use_custom_dimensions.py`.


### Use the custom evaluation dimensions
After you initialize your customized evaluation dimensions, you can choose to use any one of these methods provided below:

#### Method 1: Choose dimensions by names
```python
evaluation_dimensions = (
    EvaluationDimensionBuilder.select_existing_dimension_model_by_name(
        ["transactivity", "verbal_equity"]
    )
)
```

#### Method 2: Directly choose the grouped evaluation dimension list
```python
evaluation_dimensions = (
    EvaluationDimensionBuilder.select_existing_dimension_model_by_list_name(
        "sotopia"
    )
)
```

#### Method 3: Build a custom evaluation dimension model temporarily
We provide multiple ways to build a custom evaluation dimension model with `EvaluationDimensionBuilder`, specifically:
- `generate_dimension_model`: build an evaluation dimension from existing dimension primary keys.
- `generate_dimension_model_from_dict`: build an evaluation dimension from a dictionary that specifies the parameters of the `CustomEvaluationDimension`. For example
```json
[
    {
        "name": "believability",
        "description": "The believability of the interaction",
        "range_low": 0,
        "range_high": 10
    },
    ...
]
```
- `select_existing_dimension_model_by_name`: build an evaluation dimension from existing dimension names. For example `['believability', 'goal']`
- `select_existing_dimension_model_by_list_name`: build an evaluation dimension from existing `CustomEvaluationDimensionList` list names. For example, directly use `sotopia`.


After you get the evaluation dimension model, you can pass it as a parameter for the `Evaluator`, for example,
```python
evaluation_dimensions = (
    EvaluationDimensionBuilder.select_existing_dimension_model_by_list_name(
        "sotopia"
    )
)
terminal_evaluators=[
    ReachGoalLLMEvaluator(
        model_names["env"],
        EvaluationForTwoAgents[evaluation_dimensions],  # type: ignore
    ),
],
```
