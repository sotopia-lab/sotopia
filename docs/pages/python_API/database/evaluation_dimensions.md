# `evaluation_dimensions.py`

This module provides classes and utilities for defining and managing custom evaluation dimensions within the Sotopia environment. It includes classes for individual dimensions, lists of dimensions, and a builder for creating dimension models.

## Classes

### `CustomEvaluationDimension`

Represents a custom evaluation dimension with specific attributes such as name, description, and score range.

#### Attributes
- `name`: `str`. The name of the dimension.
- `description`: `str`. A brief description of the dimension.
- `range_low`: `int`. The minimum score for the dimension.
- `range_high`: `int`. The maximum score for the dimension.

### `CustomEvaluationDimensionList`

Groups multiple custom evaluation dimensions together.

#### Attributes
- `name`: `str`. The name of the dimension list.
- `dimension_pks`: `list[str]`. A list of primary keys for the dimensions included in the list.

### `EvaluationDimensionBuilder`

Provides utility methods to create and manage evaluation dimension models.

#### Methods
- `create_range_validator(low: int, high: int)`: Creates a validator for score ranges.

  **Arguments:**
  - `low`: `int`. The minimum score allowed.
  - `high`: `int`. The maximum score allowed.

- `build_dimension_model(dimension_ids: list[str])`: Builds a dimension model from primary keys.

  **Arguments:**
  - `dimension_ids`: `list[str]`. A list of dimension primary keys.

- `build_dimension_model_from_dict(dimensions: list[dict[str, Union[str, int]]])`: Builds a dimension model from a dictionary.

  **Arguments:**
  - `dimensions`: `list[dict[str, Union[str, int]]]`. A list of dictionaries specifying dimension attributes.

- `select_existing_dimension_model_by_name(dimension_names: list[str])`: Selects a dimension model by dimension names.

  **Arguments:**
  - `dimension_names`: `list[str]`. A list of dimension names.

- `select_existing_dimension_model_by_list_name(list_name: str)`: Selects a dimension model by list name.

  **Arguments:**
  - `list_name`: `str`. The name of the dimension list.
