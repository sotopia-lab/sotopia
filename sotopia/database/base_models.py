"""Base model classes that support both Redis and local storage backends.

This module provides wrapper classes that delegate to either redis-om JsonModel
or a local JSON file storage backend, depending on the SOTOPIA_STORAGE_BACKEND
environment variable.

The approach uses monkey-patching of model classes at import time to add
storage methods that work with the selected backend.
"""

import sys
from typing import Any, Type, TypeVar

if sys.version_info >= (3, 11):
    pass
else:
    pass

from redis_om.model.model import NotFoundError

from .storage_backend import get_storage_backend, is_local_backend

T = TypeVar("T")


class LocalQueryResult:
    """A query result object that mimics redis-om's query interface for local storage."""

    def __init__(self, model_class: Type[T], data_list: list[dict[str, Any]]) -> None:
        """Initialize a query result.

        Args:
            model_class: The model class being queried
            data_list: List of dictionaries representing matching records
        """
        self.model_class = model_class
        self.data_list = data_list

    def all(self) -> list[Any]:
        """Execute the query and return all matching results.

        Returns:
            List of model instances matching the filters
        """
        # Convert dictionaries back to model instances
        result = [self.model_class(**data) for data in self.data_list]
        return result


class SimpleFieldDescriptor:
    """Simple field descriptor for local backend to mimic redis-om field access."""

    def __init__(self, field_name: str):
        self.name = field_name

    def __eq__(self, other: Any) -> "SimpleFieldExpression":  # type: ignore[override]
        return SimpleFieldExpression(self.name, other)


class SimpleFieldExpression:
    """Simple expression object for local backend queries."""

    def __init__(self, field_name: str, value: Any):
        self._left = SimpleFieldDescriptor(field_name)
        self._right = value

    def __and__(self, other: "SimpleFieldExpression") -> "CompoundExpression":
        """Support combining expressions with & operator."""
        return CompoundExpression(self, other, "and")


class CompoundExpression:
    """Compound expression for combining multiple field expressions."""

    def __init__(
        self,
        left: SimpleFieldExpression,
        right: SimpleFieldExpression,
        operator: str,
    ):
        self._left = left
        self._right = right
        self.operator = operator


def add_local_storage_methods(model_class: Type[T]) -> None:
    """Add local storage methods to a model class.

    This function monkey-patches a model class to add save(), get(), delete(),
    find(), and all() methods that work with the local JSON storage backend.
    It also adds field descriptors to enable the Model.field == value syntax.

    Args:
        model_class: The model class to patch
    """

    # Add field descriptors for all model fields
    if hasattr(model_class, "model_fields"):
        for field_name in model_class.model_fields.keys():  # type: ignore[attr-defined]
            if field_name != "pk" and not hasattr(model_class, field_name):
                setattr(model_class, field_name, SimpleFieldDescriptor(field_name))

    def save(self: Any) -> None:
        """Save this model instance to local storage."""
        backend = get_storage_backend()
        if not self.pk:
            self.pk = backend.generate_pk()

        # Convert model to dict for storage
        # Use mode='python' and exclude_unset=False to handle all types
        try:
            data = self.model_dump(mode="json")
        except (TypeError, ValueError):
            # Fallback for models with complex types that can't be serialized with mode="json"
            import json

            data = json.loads(self.model_dump_json())

        backend.save(self.__class__, self.pk, data)

    def get(cls: Type[T], pk: str) -> T | None:
        """Retrieve a model instance by primary key from local storage."""
        backend = get_storage_backend()
        data = backend.get(cls, pk)  # type: ignore[type-var]
        return cls(**data) if data else None

    def delete(cls: Type[T], pk: str) -> None:
        """Delete a model instance by primary key from local storage."""
        backend = get_storage_backend()
        backend.delete(cls, pk)  # type: ignore[type-var]

    def find(cls: Type[T], /, *conditions: Any, **kwargs: Any) -> "LocalQueryResult":
        """Find model instances matching the given conditions in local storage.

        For local backend, this attempts to parse redis-om style expressions
        into simple field:value filters.

        Args:
            *conditions: Filter expressions (attempts to parse redis-om style)
            **kwargs: Additional keyword argument filters

        Returns:
            LocalQueryResult object with .all() method
        """
        backend = get_storage_backend()
        filters = {}

        # Handle keyword arguments
        filters.update(kwargs)

        # Handle redis-om style conditions (Model.field == value)
        def parse_condition(condition: Any) -> None:
            """Recursively parse redis-om style conditions."""
            if isinstance(condition, CompoundExpression):
                # This is a compound expression, parse both sides
                parse_condition(condition._left)
                parse_condition(condition._right)
            elif hasattr(condition, "_left") and hasattr(condition, "_right"):
                # Check if this is a compound expression (e.g., expr1 & expr2)
                if hasattr(condition._left, "_left") or hasattr(
                    condition._right, "_left"
                ):
                    # This is a compound expression, parse both sides
                    parse_condition(condition._left)
                    parse_condition(condition._right)
                else:
                    # This is a simple field expression
                    field_name = getattr(condition._left, "name", None)
                    value = condition._right
                    if field_name and value is not None:
                        filters[field_name] = value

        for condition in conditions:
            parse_condition(condition)

        # Get all matching records
        results_data = backend.find(cls, filters)  # type: ignore[type-var]
        return LocalQueryResult(cls, results_data)

    def all(cls: Type[T]) -> list[T]:
        """Retrieve all instances of this model from local storage."""
        backend = get_storage_backend()
        results_data = backend.all(cls)  # type: ignore[type-var]
        return [cls(**data) for data in results_data]

    def all_pks(cls: Type[T]) -> list[str]:
        """Retrieve all primary keys for this model from local storage."""
        backend = get_storage_backend()
        results_data = backend.all(cls)  # type: ignore[type-var]
        return [data.get("pk", "") for data in results_data if data.get("pk")]

    # Add the methods to the class
    model_class.save = save  # type: ignore[attr-defined]
    model_class.get = classmethod(get)  # type: ignore[attr-defined]
    model_class.delete = classmethod(delete)  # type: ignore[attr-defined]
    model_class.find = classmethod(find)  # type: ignore[attr-defined]
    model_class.all = classmethod(all)  # type: ignore[attr-defined]
    model_class.all_pks = classmethod(all_pks)  # type: ignore[attr-defined]


def patch_model_for_local_storage(model_class: Type[T]) -> Type[T]:
    """Patch a model class to use local storage if the backend is local.

    This function should be called on redis-om JsonModel classes to add
    local storage support when SOTOPIA_STORAGE_BACKEND=local.

    Args:
        model_class: The model class to patch (should inherit from JsonModel)

    Returns:
        The same model class, potentially with added methods
    """
    if is_local_backend():
        add_local_storage_methods(model_class)

    return model_class


# Re-export for convenience
__all__ = [
    "NotFoundError",
    "LocalQueryResult",
    "add_local_storage_methods",
    "patch_model_for_local_storage",
]
