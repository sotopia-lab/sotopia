import pytest
from pydantic import ValidationError
from sotopia.database.evaluation_dimensions import (
    EvaluationDimensionBuilder,
    CustomEvaluationDimension,
    CustomEvaluationDimensionList,
)
from typing import Generator, Callable


def test_create_range_validator() -> None:
    validator = EvaluationDimensionBuilder.create_range_validator(0, 10)
    assert validator(("test", 5)) == ("test", 5)

    with pytest.raises(ValueError):
        validator(("test", 11))

    with pytest.raises(ValueError):
        validator(("test", -1))


@pytest.fixture
def test_dimension() -> Generator[None, None, None]:
    dimension = CustomEvaluationDimension(
        pk="tmppk_test_dimension",
        name="test_dimension",
        description="A test dimension",
        range_high=10,
        range_low=0,
    )
    dimension.save()
    yield
    dimension.delete(dimension.pk)


@pytest.fixture
def test_dimension_list() -> Generator[None, None, None]:
    dimension = CustomEvaluationDimension(
        pk="tmppk_test_dimension",
        name="test_dimension",
        description="A test dimension",
        range_high=10,
        range_low=0,
    )
    dimension.save()
    dimension_list = CustomEvaluationDimensionList(
        pk="tmppk_test_dimension_list",
        name="test_list",
        dimension_pks=["tmppk_test_dimension"],
    )
    dimension_list.save()
    yield
    dimension.delete("tmppk_test_dimension")
    dimension_list.delete("tmppk_test_dimension_list")


def test_build_dimension_model(test_dimension: Callable[[], None]) -> None:
    # Test building model from dimension id
    model = EvaluationDimensionBuilder.build_dimension_model(["tmppk_test_dimension"])
    instance = model(test_dimension=("example", 5))
    assert instance.dict()["test_dimension"] == ("example", 5)
    # Test validation errors for out of range values
    with pytest.raises(ValidationError):
        model(test_dimension=("example", 11))
    with pytest.raises(ValidationError):
        model(test_dimension=("example", -1))


def test_build_dimension_model_from_dict() -> None:
    # Test building model from dictionary
    dimensions: list[dict[str, str | int]] = [
        {
            "name": "test_dim",
            "description": "A test dimension",
            "range_high": 10,
            "range_low": 0,
        }
    ]
    model = EvaluationDimensionBuilder.build_dimension_model_from_dict(dimensions)

    instance = model(test_dim=("example", 5))
    assert instance.dict()["test_dim"] == ("example", 5)

    with pytest.raises(ValidationError):
        model(test_dim=("example", 11))


def test_select_existing_dimension_model_by_name(
    test_dimension: Callable[[], None],
) -> None:
    # Test building model from dimension names
    model = EvaluationDimensionBuilder.select_existing_dimension_model_by_name(
        ["test_dimension"]
    )
    instance = model(test_dimension=("example", 5))
    assert instance.dict()["test_dimension"] == ("example", 5)

    with pytest.raises(ValidationError):
        model(test_dimension=("example", 11))


def test_select_existing_dimension_model_by_list_name(
    test_dimension_list: Callable[[], None],
) -> None:
    # Test building model from list name
    model = EvaluationDimensionBuilder.select_existing_dimension_model_by_list_name(
        "test_list"
    )
    instance = model(test_dimension=("example", 5))
    assert instance.dict()["test_dimension"] == ("example", 5)

    with pytest.raises(ValidationError):
        model(test_dimension=("example", 11))
