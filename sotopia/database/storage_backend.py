"""Storage backend abstraction layer for Sotopia.

This module provides an abstraction layer that allows Sotopia to work with
either Redis or local JSON file storage, controlled by the SOTOPIA_STORAGE_BACKEND
environment variable.
"""

import json
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel
from redis_om.model.model import NotFoundError

T = TypeVar("T", bound=BaseModel)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save(self, model_class: Type[T], pk: str, data: dict[str, Any]) -> None:
        """Save a model instance to storage.

        Args:
            model_class: The model class being saved
            pk: Primary key for the instance
            data: Dictionary representation of the model
        """
        pass

    @abstractmethod
    def get(self, model_class: Type[T], pk: str) -> dict[str, Any]:
        """Retrieve a model instance from storage.

        Args:
            model_class: The model class to retrieve
            pk: Primary key of the instance

        Returns:
            Dictionary representation of the model

        Raises:
            NotFoundError: If instance with given pk doesn't exist
        """
        pass

    @abstractmethod
    def delete(self, model_class: Type[T], pk: str) -> None:
        """Delete a model instance from storage.

        Args:
            model_class: The model class
            pk: Primary key of the instance to delete

        Raises:
            NotFoundError: If instance with given pk doesn't exist
        """
        pass

    @abstractmethod
    def find(
        self, model_class: Type[T], filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Find model instances matching the given filters.

        Args:
            model_class: The model class to search
            filters: Dictionary of field names to values

        Returns:
            List of dictionaries representing matching instances
        """
        pass

    @abstractmethod
    def all(self, model_class: Type[T]) -> list[dict[str, Any]]:
        """Retrieve all instances of a model class.

        Args:
            model_class: The model class

        Returns:
            List of dictionaries representing all instances
        """
        pass

    @abstractmethod
    def generate_pk(self) -> str:
        """Generate a new primary key.

        Returns:
            A unique primary key string
        """
        pass


class RedisBackend(StorageBackend):
    """Redis-based storage backend using redis-om."""

    def __init__(self) -> None:
        """Initialize Redis backend."""
        # Redis-om handles connection through environment variables
        # No initialization needed here - models handle their own connections
        pass

    def save(self, model_class: Type[T], pk: str, data: dict[str, Any]) -> None:
        """Save via redis-om's JsonModel.save()."""
        # This is handled by the model itself in redis-om
        # This method exists for interface compatibility
        raise NotImplementedError(
            "RedisBackend.save() should not be called directly. "
            "Use the model's save() method instead."
        )

    def get(self, model_class: Type[T], pk: str) -> dict[str, Any]:
        """Get via redis-om's JsonModel.get()."""
        raise NotImplementedError(
            "RedisBackend.get() should not be called directly. "
            "Use the model's get() class method instead."
        )

    def delete(self, model_class: Type[T], pk: str) -> None:
        """Delete via redis-om's JsonModel.delete()."""
        raise NotImplementedError(
            "RedisBackend.delete() should not be called directly. "
            "Use the model's delete() class method instead."
        )

    def find(
        self, model_class: Type[T], filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Find via redis-om's JsonModel.find()."""
        raise NotImplementedError(
            "RedisBackend.find() should not be called directly. "
            "Use the model's find() class method instead."
        )

    def all(self, model_class: Type[T]) -> list[dict[str, Any]]:
        """Get all via redis-om's custom all() method."""
        raise NotImplementedError(
            "RedisBackend.all() should not be called directly. "
            "Use the model's all() class method instead."
        )

    def generate_pk(self) -> str:
        """Generate a UUID primary key."""
        return str(uuid.uuid4())


class LocalJSONBackend(StorageBackend):
    """Local JSON file-based storage backend.

    Stores each model instance as a separate JSON file in a directory structure:
    ~/.sotopia/data/{model_class_name}/{pk}.json

    Note: This backend does not support TTL/expiration. Models with TTL fields
    (e.g., SessionTransaction, MatchingInWaitingRoom) will store the expire_time
    field but will not automatically delete expired records.
    """

    def __init__(self, base_path: str | None = None) -> None:
        """Initialize local JSON backend.

        Args:
            base_path: Base directory for storing data. Defaults to ~/.sotopia/data
        """
        if base_path is None:
            base_path = os.path.expanduser("~/.sotopia/data")
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_model_dir(self, model_class: Type[T]) -> Path:
        """Get the directory path for a model class.

        Args:
            model_class: The model class

        Returns:
            Path to the directory for this model class
        """
        model_name = model_class.__name__
        model_dir = self.base_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _get_file_path(self, model_class: Type[T], pk: str) -> Path:
        """Get the file path for a specific instance.

        Args:
            model_class: The model class
            pk: Primary key

        Returns:
            Path to the JSON file for this instance
        """
        return self._get_model_dir(model_class) / f"{pk}.json"

    def save(self, model_class: Type[T], pk: str, data: dict[str, Any]) -> None:
        """Save a model instance to a JSON file.

        Args:
            model_class: The model class being saved
            pk: Primary key for the instance
            data: Dictionary representation of the model
        """
        file_path = self._get_file_path(model_class, pk)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get(self, model_class: Type[T], pk: str) -> dict[str, Any]:
        """Retrieve a model instance from a JSON file.

        Args:
            model_class: The model class to retrieve
            pk: Primary key of the instance

        Returns:
            Dictionary representation of the model

        Raises:
            NotFoundError: If instance with given pk doesn't exist
        """
        file_path = self._get_file_path(model_class, pk)
        if not file_path.exists():
            raise NotFoundError(f"{model_class.__name__} with pk={pk} not found")

        with open(file_path, "r") as f:
            data: dict[str, Any] = json.load(f)
            return data

    def delete(self, model_class: Type[T], pk: str) -> None:
        """Delete a model instance's JSON file.

        Args:
            model_class: The model class
            pk: Primary key of the instance to delete

        Raises:
            NotFoundError: If instance with given pk doesn't exist
        """
        file_path = self._get_file_path(model_class, pk)
        if not file_path.exists():
            raise NotFoundError(f"{model_class.__name__} with pk={pk} not found")

        file_path.unlink()

    def find(
        self, model_class: Type[T], filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Find model instances matching the given filters.

        This implementation loads all instances and filters in memory.
        Not efficient for large datasets, but simple and correct.

        Args:
            model_class: The model class to search
            filters: Dictionary of field names to values

        Returns:
            List of dictionaries representing matching instances
        """
        model_dir = self._get_model_dir(model_class)
        results = []

        for file_path in model_dir.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)

            # Check if all filters match
            matches = True
            for field, value in filters.items():
                if data.get(field) != value:
                    matches = False
                    break

            if matches:
                results.append(data)

        return results

    def all(self, model_class: Type[T]) -> list[dict[str, Any]]:
        """Retrieve all instances of a model class.

        Args:
            model_class: The model class

        Returns:
            List of dictionaries representing all instances
        """
        model_dir = self._get_model_dir(model_class)
        results = []

        for file_path in model_dir.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                results.append(data)

        return results

    def generate_pk(self) -> str:
        """Generate a UUID primary key.

        Returns:
            A unique primary key string
        """
        return str(uuid.uuid4())


# Global storage backend instance
_storage_backend: StorageBackend | None = None


def get_storage_backend() -> StorageBackend:
    """Get the configured storage backend.

    Reads the SOTOPIA_STORAGE_BACKEND environment variable to determine
    which backend to use:
    - "redis" (default): Use Redis via redis-om
    - "local": Use local JSON file storage

    Returns:
        The configured storage backend instance

    Raises:
        ValueError: If SOTOPIA_STORAGE_BACKEND has an invalid value
    """
    global _storage_backend

    if _storage_backend is not None:
        return _storage_backend

    backend_type = os.environ.get("SOTOPIA_STORAGE_BACKEND", "redis").lower()

    if backend_type == "redis":
        _storage_backend = RedisBackend()
    elif backend_type == "local":
        _storage_backend = LocalJSONBackend()
    else:
        raise ValueError(
            f"Invalid SOTOPIA_STORAGE_BACKEND: {backend_type}. "
            f"Must be 'redis' or 'local'."
        )

    return _storage_backend


def is_redis_backend() -> bool:
    """Check if the current storage backend is Redis.

    Returns:
        True if using Redis backend, False otherwise
    """
    return isinstance(get_storage_backend(), RedisBackend)


def is_local_backend() -> bool:
    """Check if the current storage backend is local JSON storage.

    Returns:
        True if using local backend, False otherwise
    """
    return isinstance(get_storage_backend(), LocalJSONBackend)
