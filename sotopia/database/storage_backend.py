"""Storage backend abstraction layer for Sotopia.

This module provides an abstraction layer that allows Sotopia to work with
either Redis, local JSON file storage, or PostgreSQL, controlled by the
SOTOPIA_STORAGE_BACKEND environment variable.
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


class PostgreSQLBackend(StorageBackend):
    """PostgreSQL-based storage backend.

    Stores data in PostgreSQL tables, one table per model class.
    Uses a JSONB column for flexible schema storage.

    Requires:
    - POSTGRES_URL environment variable (or DATABASE_URL)

    Install with: pip install sotopia[postgres]
    """

    def __init__(self) -> None:
        """Initialize PostgreSQL backend."""
        try:
            import psycopg2
            from psycopg2.extras import Json, RealDictCursor
        except ImportError:
            raise ImportError(
                "PostgreSQL backend requires psycopg2. "
                "Install with: pip install sotopia[postgres]"
            )

        self._psycopg2 = psycopg2
        self._Json = Json
        self._RealDictCursor = RealDictCursor

        # Get database URL
        db_url = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
        if not db_url:
            raise ValueError(
                "PostgreSQL backend requires POSTGRES_URL or DATABASE_URL "
                "environment variable"
            )

        self._db_url = db_url
        self._conn: Any = None
        self._tables_created: set[str] = set()

    def _get_connection(self) -> Any:
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg2.connect(self._db_url)
        return self._conn

    def _ensure_table(self, model_class: Type[T]) -> str:
        """Ensure table exists for model class.

        Creates table if it doesn't exist. Uses a simple schema with:
        - pk: Primary key (TEXT)
        - data: JSONB column for all fields
        - created_at: Timestamp

        Args:
            model_class: The model class

        Returns:
            Table name
        """
        table_name = model_class.__name__.lower()

        if table_name in self._tables_created:
            return table_name

        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    pk TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Create GIN index on JSONB for faster filtering
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_data
                ON {table_name} USING GIN (data)
            """)
            conn.commit()

        self._tables_created.add(table_name)
        return table_name

    def save(self, model_class: Type[T], pk: str, data: dict[str, Any]) -> None:
        """Save a model instance to PostgreSQL.

        Args:
            model_class: The model class being saved
            pk: Primary key for the instance
            data: Dictionary representation of the model
        """
        table_name = self._ensure_table(model_class)
        conn = self._get_connection()

        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {table_name} (pk, data, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (pk) DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = CURRENT_TIMESTAMP
            """, (pk, self._Json(data)))
            conn.commit()

    def get(self, model_class: Type[T], pk: str) -> dict[str, Any]:
        """Retrieve a model instance from PostgreSQL.

        Args:
            model_class: The model class to retrieve
            pk: Primary key of the instance

        Returns:
            Dictionary representation of the model

        Raises:
            NotFoundError: If instance with given pk doesn't exist
        """
        table_name = self._ensure_table(model_class)
        conn = self._get_connection()

        with conn.cursor(cursor_factory=self._RealDictCursor) as cur:
            cur.execute(f"SELECT data FROM {table_name} WHERE pk = %s", (pk,))
            row = cur.fetchone()

        if row is None:
            raise NotFoundError(f"{model_class.__name__} with pk={pk} not found")

        return dict(row["data"])

    def delete(self, model_class: Type[T], pk: str) -> None:
        """Delete a model instance from PostgreSQL.

        Args:
            model_class: The model class
            pk: Primary key of the instance to delete

        Raises:
            NotFoundError: If instance with given pk doesn't exist
        """
        table_name = self._ensure_table(model_class)
        conn = self._get_connection()

        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name} WHERE pk = %s RETURNING pk", (pk,))
            deleted = cur.fetchone()
            conn.commit()

        if deleted is None:
            raise NotFoundError(f"{model_class.__name__} with pk={pk} not found")

    def find(
        self, model_class: Type[T], filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Find model instances matching the given filters.

        Uses PostgreSQL JSONB containment operator for efficient filtering.

        Args:
            model_class: The model class to search
            filters: Dictionary of field names to values

        Returns:
            List of dictionaries representing matching instances
        """
        table_name = self._ensure_table(model_class)
        conn = self._get_connection()

        with conn.cursor(cursor_factory=self._RealDictCursor) as cur:
            # Use JSONB containment operator @>
            cur.execute(
                f"SELECT data FROM {table_name} WHERE data @> %s",
                (self._Json(filters),)
            )
            rows = cur.fetchall()

        return [dict(row["data"]) for row in rows]

    def all(self, model_class: Type[T]) -> list[dict[str, Any]]:
        """Retrieve all instances of a model class.

        Args:
            model_class: The model class

        Returns:
            List of dictionaries representing all instances
        """
        table_name = self._ensure_table(model_class)
        conn = self._get_connection()

        with conn.cursor(cursor_factory=self._RealDictCursor) as cur:
            cur.execute(f"SELECT data FROM {table_name}")
            rows = cur.fetchall()

        return [dict(row["data"]) for row in rows]

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
    - "postgres" or "postgresql": Use PostgreSQL database

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
    elif backend_type in ("postgres", "postgresql"):
        _storage_backend = PostgreSQLBackend()
    else:
        raise ValueError(
            f"Invalid SOTOPIA_STORAGE_BACKEND: {backend_type}. "
            f"Must be 'redis', 'local', or 'postgres'."
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


def is_postgres_backend() -> bool:
    """Check if the current storage backend is PostgreSQL.

    Returns:
        True if using PostgreSQL backend, False otherwise
    """
    return isinstance(get_storage_backend(), PostgreSQLBackend)
