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


However we observe under many use cases people may want to evaluate with customized evaluation metrics.

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
