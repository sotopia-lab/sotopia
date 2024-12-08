# Evaluation Dimensions API

## CustomEvaluationDimension

The `CustomEvaluationDimension` class is used to define custom evaluation dimensions with specific attributes such as name, description, and score range.

### Attributes
- `name`: The name of the dimension.
- `description`: A brief description of the dimension.
- `range_low`: The minimum score for the dimension.
- `range_high`: The maximum score for the dimension.

## CustomEvaluationDimensionList

The `CustomEvaluationDimensionList` class is used to group multiple custom evaluation dimensions together.

### Attributes
- `name`: The name of the dimension list.
- `dimension_pks`: A list of primary keys for the dimensions included in the list.

## EvaluationDimensionBuilder

The `EvaluationDimensionBuilder` class provides utility methods to create and manage evaluation dimension models.

### Methods
- `create_range_validator(low: int, high: int)`: Creates a validator for score ranges.
- `build_dimension_model(dimension_ids: list[str])`: Builds a dimension model from primary keys.
- `build_dimension_model_from_dict(dimensions: list[dict[str, Union[str, int]]])`: Builds a dimension model from a dictionary.
- `select_existing_dimension_model_by_name(dimension_names: list[str])`: Selects a dimension model by dimension names.
- `select_existing_dimension_model_by_list_name(list_name: str)`: Selects a dimension model by list name.