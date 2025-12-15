"""Global test configuration for sotopia tests."""

import os
from typing import Final

import pytest
import requests

# Set local storage backend as default for tests unless explicitly overridden
if "SOTOPIA_STORAGE_BACKEND" not in os.environ:
    os.environ["SOTOPIA_STORAGE_BACKEND"] = "local"


# Default model identifiers used in tests
_LOCAL_STRUCTURED_MODEL: Final[str] = "custom/structured@http://localhost:8000/v1"
_LOCAL_LLAMA_MODEL: Final[str] = "custom/llama3.2:1b@http://localhost:8000/v1"

_FALLBACK_MODEL: Final[str] = "gpt-4o-mini"


def _is_local_model_available(model_id: str, timeout: float = 0.5) -> bool:
    """Best-effort check whether the local model backend is reachable.

    The `model_id` format is usually `name@BASE_URL`. We only care about whether
    the BASE_URL is accepting HTTP connections; the exact path/response code is
    not important for this health check.
    """

    if "@http" not in model_id:
        return False

    _, base_url = model_id.split("@", 1)
    base_url = base_url.rstrip("/")

    try:
        # Any HTTP response means the server is reachable; 4xx/5xx are fine.
        requests.get(base_url, timeout=timeout)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def structured_evaluator_model_name() -> str:
    """Model id for evaluator tests: prefer local, fall back to hosted model."""

    if _is_local_model_available(_LOCAL_STRUCTURED_MODEL):
        return _LOCAL_STRUCTURED_MODEL
    return _FALLBACK_MODEL


@pytest.fixture(scope="session")
def local_llama_model_name() -> str:
    """Model id for llama-based generation tests: prefer local, fall back."""

    if _is_local_model_available(_LOCAL_LLAMA_MODEL):
        return _LOCAL_LLAMA_MODEL
    return _FALLBACK_MODEL
