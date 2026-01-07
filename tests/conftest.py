"""Global test configuration for sotopia tests."""

import os
from typing import Final

import pytest
import requests
import json
from typing import Any
from litellm import ModelResponse, Choices, Message  # type: ignore[attr-defined]

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


# ============================================================================
# LLM Mocking Infrastructure
# ============================================================================


def create_mock_llm_response(content: str) -> Any:
    """Create a mock litellm ModelResponse object.

    Args:
        content: The string content to return as the LLM response

    Returns:
        A mock ModelResponse with the given content
    """

    return ModelResponse(
        choices=[
            Choices(
                message=Message(content=content, role="assistant"), finish_reason="stop"
            )
        ]
    )


def generate_mock_json_from_schema(schema: dict[str, Any]) -> str:
    """Generate valid JSON matching a Pydantic JSON schema.

    Args:
        schema: A Pydantic JSON schema dictionary

    Returns:
        JSON string matching the schema
    """

    # Extract definitions for $ref resolution
    definitions = schema.get("$defs", {})

    def resolve_ref(ref_path: str) -> dict[str, Any]:
        """Resolve a JSON schema $ref path."""
        if ref_path.startswith("#/$defs/"):
            def_name = ref_path.replace("#/$defs/", "")
            result: dict[str, Any] = definitions.get(def_name, {})
            return result
        return {}

    def generate_value(type_info: dict[str, Any], property_name: str = "field") -> Any:
        """Recursively generate values matching schema types."""
        # Handle $ref references
        if "$ref" in type_info:
            resolved = resolve_ref(type_info["$ref"])
            return generate_value(resolved, property_name)

        type_name = type_info.get("type")

        if type_name == "string":
            # Check for enum values
            if "enum" in type_info:
                return type_info["enum"][0]
            return f"mock_{property_name}"
        elif type_name == "integer":
            return 42
        elif type_name == "number":
            return 3.14
        elif type_name == "boolean":
            return True
        elif type_name == "array":
            # Check if it's a tuple (has prefixItems)
            if "prefixItems" in type_info:
                # Generate tuple with specific types for each position
                return [
                    generate_value(item_schema, f"{property_name}_{i}")
                    for i, item_schema in enumerate(type_info["prefixItems"])
                ]
            else:
                # Special case: 'to' field in AgentAction should default to empty list
                # (it's for private message recipients, empty = public)
                if property_name == "to":
                    return []
                # Regular array with same type for all items
                items = type_info.get("items", {})
                # Generate 2 items by default
                return [generate_value(items, property_name) for _ in range(2)]
        elif type_name == "object":
            properties = type_info.get("properties", {})
            result = {}

            # Handle regular properties
            for key, prop_info in properties.items():
                result[key] = generate_value(prop_info, key)

            # Handle additionalProperties (for dict[str, T] patterns)
            if "additionalProperties" in type_info and isinstance(
                type_info["additionalProperties"], dict
            ):
                # Generate at least 2 sample entries for dict types
                # Use agent_1, agent_2 as keys (common pattern in sotopia tests)
                for i in range(1, 3):
                    key = f"agent_{i}"
                    if key not in result:  # Don't override explicit properties
                        result[key] = generate_value(
                            type_info["additionalProperties"], key
                        )

            return result
        elif type_name is None and "anyOf" in type_info:
            # Handle anyOf by using the first option
            return generate_value(type_info["anyOf"][0], property_name)
        else:
            return None

    result = generate_value(schema)
    return json.dumps(result)


@pytest.fixture(autouse=True)
def mock_llm_calls(request: Any, monkeypatch: Any) -> Any:
    """Automatically mock litellm.acompletion unless test has @pytest.mark.real_llm marker.

    This fixture runs for all tests by default (autouse=True). Tests that need to make
    real API calls can opt-out by using the @pytest.mark.real_llm marker.

    The mock implementation intelligently generates responses based on:
    - Structured output schemas (for json_schema response_format)
    - Simple text responses (for other formats)

    Args:
        request: pytest request fixture
        monkeypatch: pytest monkeypatch fixture

    Yields:
        None (fixture just sets up and tears down mocking)
    """
    # Skip mocking if test is marked with real_llm
    if "real_llm" in request.keywords:
        yield
        return

    async def mock_acompletion(*args: Any, **kwargs: Any) -> Any:
        """Mock implementation of litellm.acompletion."""
        # Extract response_format to determine output type
        response_format = kwargs.get("response_format", {})

        # Check if structured output is requested
        if response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
            content = generate_mock_json_from_schema(schema)
        else:
            # Default to simple response
            content = "mock llm response"

        return create_mock_llm_response(content)

    # Patch acompletion in all locations where it's imported
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    monkeypatch.setattr(
        "sotopia.generation_utils.generate.acompletion", mock_acompletion
    )
    yield


@pytest.fixture
def mock_llm_response(monkeypatch: Any) -> Any:
    """Allow tests to configure specific mock responses.

    This fixture provides a way for tests to specify exact responses that the
    mocked LLM should return. Useful for tests that need specific outputs.

    Usage:
        def test_something(mock_llm_response):
            mock_llm_response('{"action_type": "speak", "argument": "hello"}')
            result = await agenerate(...)
            assert result.action_type == "speak"

    Args:
        monkeypatch: pytest monkeypatch fixture

    Returns:
        A function to set mock responses
    """
    responses: list[str] = []
    call_count = [0]  # Use list to allow mutation in nested function

    async def mock_acompletion(*args: Any, **kwargs: Any) -> Any:
        """Mock implementation that returns pre-configured responses."""
        if call_count[0] < len(responses):
            content = responses[call_count[0]]
            call_count[0] += 1
        else:
            content = "default mock response"
        return create_mock_llm_response(content)

    # Patch acompletion in all locations where it's imported
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    monkeypatch.setattr(
        "sotopia.generation_utils.generate.acompletion", mock_acompletion
    )

    def set_responses(*args: str) -> None:
        """Set the responses that should be returned by subsequent LLM calls."""
        responses.clear()
        responses.extend(args)
        call_count[0] = 0  # Reset counter when setting new responses

    return set_responses
