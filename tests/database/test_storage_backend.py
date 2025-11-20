"""Tests for storage backend selection and switching."""

import os
import tempfile

import pytest


def test_storage_backend_environment_variable():
    """Test that storage backend is selected based on environment variable."""
    # Test default (redis)
    os.environ.pop("SOTOPIA_STORAGE_BACKEND", None)

    # Need to reload modules to pick up environment changes
    import importlib
    import sotopia.database.storage_backend as sb_module

    importlib.reload(sb_module)

    from sotopia.database.storage_backend import get_storage_backend, RedisBackend

    backend = get_storage_backend()
    assert isinstance(backend, RedisBackend)


def test_local_backend_initialization():
    """Test local backend initialization with environment variable."""
    os.environ["SOTOPIA_STORAGE_BACKEND"] = "local"

    # Reload modules
    import importlib
    import sotopia.database.storage_backend as sb_module

    importlib.reload(sb_module)

    from sotopia.database.storage_backend import (
        LocalJSONBackend,
        get_storage_backend,
        is_local_backend,
    )

    backend = get_storage_backend()
    assert isinstance(backend, LocalJSONBackend)
    assert is_local_backend()

    # Clean up
    os.environ.pop("SOTOPIA_STORAGE_BACKEND", None)


def test_invalid_backend_raises_error():
    """Test that invalid backend name raises error."""
    os.environ["SOTOPIA_STORAGE_BACKEND"] = "invalid_backend"

    # Reload modules
    import importlib
    import sotopia.database.storage_backend as sb_module

    importlib.reload(sb_module)

    from sotopia.database.storage_backend import get_storage_backend

    with pytest.raises(ValueError, match="Invalid SOTOPIA_STORAGE_BACKEND"):
        get_storage_backend()

    # Clean up
    os.environ.pop("SOTOPIA_STORAGE_BACKEND", None)


def test_local_backend_base_path():
    """Test that local backend uses correct base path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["SOTOPIA_STORAGE_BACKEND"] = "local"

        from sotopia.database.storage_backend import LocalJSONBackend

        backend = LocalJSONBackend(tmpdir)
        assert str(backend.base_path) == tmpdir

        # Clean up
        os.environ.pop("SOTOPIA_STORAGE_BACKEND", None)


def test_experimental_framework_requires_redis():
    """Test that experimental framework raises error without Redis."""
    os.environ["SOTOPIA_STORAGE_BACKEND"] = "local"

    # Reload modules
    import importlib
    import sotopia.database.storage_backend as sb_module

    importlib.reload(sb_module)

    import asyncio

    from sotopia.experimental.server import arun_one_episode

    with pytest.raises(RuntimeError, match="Experimental framework requires Redis"):
        asyncio.run(arun_one_episode({}).__anext__())

    # Clean up
    os.environ.pop("SOTOPIA_STORAGE_BACKEND", None)
