import pytest
from pydantic import ValidationError
from sotopia.database.evaluation_dimensions import (
    EvaluationDimensionBuilder,
    CustomEvaluationDimension,
)


def test_create_range_validator():
    validator = EvaluationDimensionBuilder.create_range_validator(0, 10)
    assert validator(("test", 5)) == ("test", 5)

    with pytest.raises(ValueError):
        validator(("test", 11))

    with pytest.raises(ValueError):
        validator(("test", -1))

    with pytest.raises(ValueError):
        validator("invalid")


def test_build_dimension_model():
    dimension = CustomEvaluationDimension(
        name="test_dimension",
        description="A test dimension",
        range_high=10,
        range_low=0
    )
    dimension.save()

    model = EvaluationDimensionBuilder.build_dimension_model([dimension.pk])
    instance = model(test_dimension=("example", 5))
    assert instance.test_dimension == ("example", 5)

    with pytest.raises(ValidationError):
        model(test_dimension=("example", 11))

    with pytest.raises(ValidationError):
        model(test_dimension=("example", -1))

    dimension.delete()