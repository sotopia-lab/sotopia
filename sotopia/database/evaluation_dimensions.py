from redis_om import JsonModel
from redis_om.model.model import Field
from pydantic import create_model, BaseModel, AfterValidator
from typing import Type, Callable, Tuple, Annotated, Union, cast, Any


class BaseCustomEvaluationDimension(BaseModel):
    name: str = Field(index=True)
    description: str = Field(index=True)
    range_high: int = Field(index=True)
    range_low: int = Field(index=True)


class CustomEvaluationDimension(BaseCustomEvaluationDimension, JsonModel):
    pass


class BaseCustomEvaluationDimensionList(BaseModel):
    name: str = Field(index=True)
    dimension_pks: list[str] = Field(default_factory=lambda: [], index=True)


class CustomEvaluationDimensionList(BaseCustomEvaluationDimensionList, JsonModel):
    pass


class EvaluationDimensionBuilder:
    """
    EvaluationDimensionBuilder is a utility class for creating and managing evaluation dimensions.
    It provides methods to build evaluation dimension models from various inputs such as primary keys, dictionaries, and names.
    """

    @staticmethod
    def create_range_validator(
        low: int, high: int
    ) -> Callable[[Tuple[str, int]], Tuple[str, int]]:
        def validator(x: Tuple[str, int]) -> Tuple[str, int]:
            if not isinstance(x, tuple) or len(x) != 2:
                raise ValueError("Must be a tuple of (str, int)")
            if not isinstance(x[1], int) or not low <= x[1] <= high:
                raise ValueError(f"Score must be between {low} and {high}")
            return x

        return validator

    @staticmethod
    def build_dimension_model(dimension_ids: list[str]) -> Type[BaseModel]:
        """
        Build an evaluation dimension from existing dimension primary keys.
        The returned model is a pydantic model that can be used to evaluate the conversation.
        """
        fields: dict[str, Any] = {}

        for dimension_id in dimension_ids:
            dimension = CustomEvaluationDimension.get(dimension_id)
            range_validator = EvaluationDimensionBuilder.create_range_validator(
                dimension.range_low, dimension.range_high
            )
            # Need to use AfterValidator to ensure validation happens after type checking
            field_type = Annotated[Tuple[str, int], AfterValidator(range_validator)]

            fields[dimension.name] = (
                field_type,
                Field(..., description=dimension.description),
            )

        model: Type[BaseModel] = create_model(
            "CustomEvaluationDimensionModel",
            __base__=BaseModel,
            **fields,
        )
        return model

    @staticmethod
    def build_dimension_model_from_dict(
        dimensions: list[dict[str, Union[str, int]]],
    ) -> Type[BaseModel]:
        """
        Build an evaluation dimension from a dictionary that specifies the parameters of the `CustomEvaluationDimension`.
        The returned model is a pydantic model that can be used to evaluate the conversation.
        """
        fields: dict[str, Any] = {}
        for dimension_dict in dimensions:
            dimension = CustomEvaluationDimension(**dimension_dict)
            range_validator = EvaluationDimensionBuilder.create_range_validator(
                dimension.range_low, dimension.range_high
            )
            # Need to use AfterValidator to ensure validation happens after type checking
            field_type = Annotated[Tuple[str, int], AfterValidator(range_validator)]

            fields[dimension.name] = (
                field_type,
                Field(..., description=dimension.description),
            )

        dimension_model = create_model(
            "CustomEvaluationDimensionModel",
            __base__=BaseModel,
            **fields,
        )
        return dimension_model

    @staticmethod
    def select_existing_dimension_model_by_name(
        dimension_names: list[str],
    ) -> Type[BaseModel]:
        """
        Build an evaluation dimension from existing dimension names. For example `['believability', 'goal']`
        The returned model is a pydantic model that can be used to evaluate the conversation.
        """
        fields: dict[str, Any] = {}
        for dimension_name in dimension_names:
            dimensions = CustomEvaluationDimension.find(
                CustomEvaluationDimension.name == dimension_name
            ).all()
            assert (
                len(dimensions) == 1
            ), f"Expected 1 dimension for {dimension_name}, but found {len(dimensions)}"
            dimension = cast(CustomEvaluationDimension, dimensions[0])
            range_validator = EvaluationDimensionBuilder.create_range_validator(
                dimension.range_low, dimension.range_high
            )
            # Need to use AfterValidator to ensure validation happens after type checking
            field_type = Annotated[Tuple[str, int], AfterValidator(range_validator)]

            fields[dimension.name] = (
                field_type,
                Field(..., description=dimension.description),
            )

        model: Type[BaseModel] = create_model(
            "CustomEvaluationDimensionModel",
            __base__=BaseModel,
            **fields,
        )
        return model

    @staticmethod
    def select_existing_dimension_model_by_list_name(
        list_name: str,
    ) -> Type[BaseModel]:
        """
        Build an evaluation dimension from existing `CustomEvaluationDimensionList` list names. For example, directly use `sotopia`
        The returned model is a pydantic model that can be used to evaluate the conversation.
        """
        # if list_name == "sotopia":
        #     return SotopiaDimensions # TODO see if we could make this work in `experiment_eval.py`. Right now there is a circular import

        dimensions = CustomEvaluationDimensionList.find(
            CustomEvaluationDimensionList.name == list_name
        ).all()
        assert (
            len(dimensions) == 1
        ), f"Expected 1 dimension list for {list_name}, but found {len(dimensions)}"
        dimension_list = cast(CustomEvaluationDimensionList, dimensions[0])
        dimension_ids = dimension_list.dimension_pks
        model = EvaluationDimensionBuilder.build_dimension_model(dimension_ids)
        return model
