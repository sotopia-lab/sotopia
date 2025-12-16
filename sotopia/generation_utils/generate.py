import logging
import os
import json
from dataclasses import dataclass
from litellm import acompletion
from litellm.exceptions import BadRequestError
from litellm.utils import supports_response_schema
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from typing import Any, cast

import gin

from pydantic import BaseModel, validate_call
from rich import print
from rich.logging import RichHandler

from sotopia.database import EnvironmentProfile, RelationshipProfile
from sotopia.messages import ActionType, AgentAction, ScriptBackground
from sotopia.messages.message_classes import (
    ScriptInteraction,
    ScriptInteractionReturnType,
)
from sotopia.utils import format_docstring


from sotopia.generation_utils.output_parsers import (
    OutputParser,
    PydanticOutputParser,
    StrOutputParser,
    OutputType,
    EnvResponse,
    ScriptOutputParser,
)

# Configure logger
log = logging.getLogger("sotopia.generation")
log.setLevel(logging.INFO)

# Create console handler with rich formatting
console_handler = RichHandler(rich_tracebacks=True)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

# Add handler to logger
log.addHandler(console_handler)

# subject to future OpenAI changes
DEFAULT_BAD_OUTPUT_PROCESS_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
_TEMPERATURE_SENTINEL = object()

# Cache temperature support per (model, base_url) to avoid repeated bad requests
_TEMPERATURE_SUPPORT_CACHE: dict[tuple[str, str | None], bool] = {}


@dataclass(frozen=True)
class TemperatureSetting:
    value: float | None
    treat_as_default: bool = False


def default_temperature(value: float | None) -> TemperatureSetting:
    return TemperatureSetting(value=value, treat_as_default=True)


def custom_temperature(value: float | None) -> TemperatureSetting:
    return TemperatureSetting(value=value, treat_as_default=False)


@validate_call
async def format_bad_output(
    ill_formed_output: str,
    format_instructions: str,
    model_name: str,
    use_fixed_model_version: bool = True,
    base_url: str | None = None,
) -> str:
    template = """
    Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
    Original string: {ill_formed_output}

    Format instructions: {format_instructions}

    Please only generate the JSON:
    """

    input_values = {
        "ill_formed_output": ill_formed_output,
        "format_instructions": format_instructions,
    }
    content = template.format(**input_values)

    # Build completion kwargs
    completion_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
    }

    # Parse format_instructions to get the schema
    try:
        schema = json.loads(format_instructions)
        # Apply schema fixes for OpenAI compatibility
        has_tuples, has_optional_fields = _check_schema_compatibility(schema)
        use_strict = not (has_tuples or has_optional_fields)
        _apply_schema_fixes(schema)

        # Build proper json_schema response_format
        completion_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "reformatted_output",
                "schema": schema,
                "strict": use_strict,
            },
        }
    except json.JSONDecodeError:
        # Fallback to json_object if schema parsing fails
        completion_kwargs["response_format"] = {"type": "json_object"}

    # Add base_url if provided
    if base_url is not None:
        completion_kwargs["base_url"] = base_url

    response = await acompletion(**completion_kwargs)
    reformatted_output = response.choices[0].message.content
    assert isinstance(reformatted_output, str)
    log.info(f"Reformated output: {reformatted_output}")
    return reformatted_output


def _sanitize_schema_name(name: str) -> str:
    """
    Sanitize schema title to match OpenAI's naming pattern.

    OpenAI requires schema names to match: ^[a-zA-Z0-9_-]+$
    Replaces invalid characters with underscores.

    Args:
        name: Original schema title (may contain brackets, spaces, etc.)

    Returns:
        Sanitized name with only alphanumeric, underscore, and hyphen

    Example:
        >>> _sanitize_schema_name("Response[AgentAction]")
        'Response_AgentAction_'
    """
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def _check_schema_compatibility(schema: dict[str, Any]) -> tuple[bool, bool]:
    """
    Check if Pydantic schema is compatible with OpenAI strict mode.

    Checks for:
    - Tuple types (prefixItems) - not supported in strict mode
    - Optional fields (properties not in required) - not supported in strict mode

    Args:
        schema: JSON schema from pydantic_object.model_json_schema()

    Returns:
        (has_tuples, has_optional_fields) - both must be False for strict mode
    """
    has_tuples = False
    has_optional = False

    def check_recursive(obj: Any) -> tuple[bool, bool]:
        """Recursively check schema structure."""
        nonlocal has_tuples, has_optional

        if isinstance(obj, dict):
            # Check for arrays with prefixItems (tuples)
            if obj.get("type") == "array" and "prefixItems" in obj:
                has_tuples = True

            # Check for optional fields (properties not in required list)
            if obj.get("type") == "object" and "properties" in obj:
                required = set(obj.get("required", []))
                properties = set(obj["properties"].keys())
                if properties != required:
                    has_optional = True

            # Recurse into nested structures
            for value in obj.values():
                if isinstance(value, dict):
                    check_recursive(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            check_recursive(item)

        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    check_recursive(item)

        return has_tuples, has_optional

    return check_recursive(schema)


def _apply_schema_fixes(schema: dict[str, Any]) -> None:
    """
    Apply in-place fixes to make schema more OpenAI-compatible.

    Fixes applied:
    - Add 'items': {} to arrays with prefixItems (OpenAI requirement)
    - Add 'additionalProperties': False to all objects (strict mode requirement)

    Args:
        schema: JSON schema to modify in-place

    Side effects:
        Modifies schema dict in-place
    """

    def fix_recursive(obj: Any) -> None:
        """Recursively apply fixes to schema."""
        if isinstance(obj, dict):
            # Fix arrays with prefixItems
            if (
                obj.get("type") == "array"
                and "prefixItems" in obj
                and "items" not in obj
            ):
                # OpenAI requires items field even with prefixItems
                # Empty object {} means no additional items allowed
                obj["items"] = {}

            # Fix objects - add additionalProperties
            if obj.get("type") == "object":
                if "additionalProperties" not in obj:
                    obj["additionalProperties"] = False
                elif isinstance(obj.get("additionalProperties"), dict):
                    # Recurse into dict-like objects
                    fix_recursive(obj["additionalProperties"])

            # Recurse into nested structures
            for value in obj.values():
                if isinstance(value, dict):
                    fix_recursive(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            fix_recursive(item)

        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    fix_recursive(item)

    fix_recursive(schema)


def _build_json_schema_response_format(
    schema: dict[str, Any], pydantic_class: type[BaseModel]
) -> dict[str, Any]:
    """
    Build complete OpenAI response_format dict for structured output.

    Coordinates schema validation, fixing, and formatting into the
    structure expected by litellm's acompletion with json_schema mode.

    Args:
        schema: Raw JSON schema from model_json_schema()
        pydantic_class: Original Pydantic class (for name extraction)

    Returns:
        Complete response_format dict with type, json_schema, name, schema, strict
    """
    # Sanitize the schema name
    original_name = schema.get("title", pydantic_class.__name__)
    sanitized_name = _sanitize_schema_name(original_name)

    # Check compatibility with strict mode
    has_tuples, has_optional_fields = _check_schema_compatibility(schema)

    # Use strict mode only if schema doesn't contain tuples or optional fields
    use_strict = not (has_tuples or has_optional_fields)

    # Apply schema fixes
    _apply_schema_fixes(schema)

    # Build response format
    return {
        "type": "json_schema",
        "json_schema": {
            "name": sanitized_name,
            "schema": schema,
            "strict": use_strict,
        },
    }


def _resolve_temperature_setting(
    temperature: TemperatureSetting | float | None | object,
) -> tuple[float | None, bool, bool]:
    """
    Normalize temperature setting to effective value and flags.

    Args:
        temperature: Temperature setting (sentinel, TemperatureSetting, float, or None)

    Returns:
        Tuple of (effective_temperature, treat_as_default, user_provided):
        - effective_temperature: The actual temperature value to use (None means no temp)
        - treat_as_default: True if this should be treated as default (not sent to API)
        - user_provided: True if user explicitly provided temperature
    """
    if temperature is _TEMPERATURE_SENTINEL:
        return DEFAULT_TEMPERATURE, True, False
    elif isinstance(temperature, TemperatureSetting):
        return temperature.value, temperature.treat_as_default, True
    else:
        return cast(float | None, temperature), False, True


def _should_send_temperature(
    cache_key: tuple[str, str | None],
    effective_temperature: float | None,
    treat_as_default: bool,
    user_provided: bool,
    supported_params: list[str] | None,
    model_name: str,
) -> tuple[bool, bool]:
    """
    Determine if temperature should be sent to API and check model support.

    Checks cache and model capabilities to decide if temperature parameter
    should be included in the API call.

    Args:
        cache_key: (model_name, base_url) tuple for cache lookup
        effective_temperature: The temperature value to potentially send
        treat_as_default: Whether this is a default value (shouldn't be sent)
        user_provided: Whether user explicitly provided temperature
        supported_params: List of supported params for the model (None if unknown)
        model_name: Model name for logging

    Returns:
        Tuple of (send_temperature, non_default_requested):
        - send_temperature: Whether to include temperature in API call
        - non_default_requested: Whether user requested non-default temp (for warnings)
    """
    # Initial check: only send if user provided and not treating as default
    send_temperature = (
        user_provided and effective_temperature is not None and not treat_as_default
    )
    non_default_requested = send_temperature and not treat_as_default

    if send_temperature:
        # Check cache first
        if cache_key in _TEMPERATURE_SUPPORT_CACHE:
            if not _TEMPERATURE_SUPPORT_CACHE[cache_key]:
                send_temperature = False
                if non_default_requested:
                    log.warning(
                        "Model %s previously rejected temperature; ignoring temperature=%s",
                        model_name,
                        effective_temperature,
                    )
        # Check model capabilities if cache miss and we have param info
        elif supported_params is not None and "temperature" not in supported_params:
            _TEMPERATURE_SUPPORT_CACHE[cache_key] = False
            send_temperature = False
            if non_default_requested:
                log.warning(
                    "Model %s does not support temperature; ignoring temperature=%s",
                    model_name,
                    effective_temperature,
                )

    return send_temperature, non_default_requested


def _build_structured_completion_kwargs(
    model_name: str,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    base_url: str | None,
    api_key: str | None,
) -> dict[str, Any]:
    """
    Build completion kwargs for structured output mode.

    Args:
        model_name: Model identifier
        messages: List of message dicts for the API call
        response_format: Response format dict with JSON schema
        base_url: Optional base URL for custom models
        api_key: Optional API key for custom models

    Returns:
        Dict of kwargs ready for litellm acompletion
    """
    return dict(
        model=model_name,
        messages=messages,
        response_format=response_format,
        drop_params=True,  # drop params to avoid model error if the model does not support it
        base_url=base_url,
        api_key=api_key,
    )


def _build_standard_completion_kwargs(
    model_name: str,
    messages: list[dict[str, str]],
    base_url: str | None,
    api_key: str | None,
) -> dict[str, Any]:
    """
    Build completion kwargs for standard (non-structured) output mode.

    Args:
        model_name: Model identifier
        messages: List of message dicts for the API call
        base_url: Optional base URL for custom models
        api_key: Optional API key for custom models

    Returns:
        Dict of kwargs ready for litellm acompletion
    """
    return dict(
        model=model_name,
        messages=messages,
        drop_params=True,
        base_url=base_url,
        api_key=api_key,
    )


@gin.configurable
@validate_call
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: OutputParser[OutputType],
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    context: dict[str, Any] | None = None,
) -> OutputType:
    """
    Generate text using LiteLLM with optional structured output support.

    This function handles template formatting, temperature management, schema validation,
    and API calls with retry logic. It supports both structured (JSON schema) and
    standard text generation modes.

    Args:
        model_name: Model identifier (e.g., "gpt-4o-mini" or "custom/model@http://...")
        template: Template string with {variable} placeholders
        input_values: Dict mapping template variables to their values
        output_parser: Parser to convert raw output to desired type
        temperature: Temperature setting (sentinel for default, TemperatureSetting, or float)
        structured_output: If True, use JSON schema mode (requires PydanticOutputParser)
        bad_output_process_model: Model to use for reformatting bad outputs (if needed)
        use_fixed_model_version: Whether to use fixed model versioning
        context: Optional context dict passed to output parser

    Returns:
        Parsed output of type OutputType

    Example:
        >>> from sotopia.generation_utils.output_parsers import PydanticOutputParser
        >>> from pydantic import BaseModel
        >>> class Response(BaseModel):
        ...     text: str
        >>> result = await agenerate(
        ...     model_name="gpt-4o-mini",
        ...     template="Say hello to {name}",
        ...     input_values={"name": "Alice"},
        ...     output_parser=PydanticOutputParser(pydantic_object=Response),
        ...     structured_output=True,
        ... )
    """
    # Format template with input values
    if "format_instructions" not in input_values:
        input_values["format_instructions"] = output_parser.get_format_instructions()

    # Process template
    template = format_docstring(template)

    # Replace template variables
    for key, value in input_values.items():
        template = template.replace(f"{{{key}}}", str(value))

    if model_name.startswith("custom"):
        base_url, api_key = (
            model_name.split("@")[1],
            os.environ.get("CUSTOM_API_KEY", "EMPTY"),
        )
        model_name = model_name.split("@")[0].replace("custom/", "openai/")
    else:
        base_url = None
        api_key = None

    cache_key = (model_name, base_url)

    supported_params: list[str] | None = None
    if base_url is None:
        supported_params = get_supported_openai_params(model=model_name)

    # Resolve temperature setting
    effective_temperature, treat_as_default, user_provided = (
        _resolve_temperature_setting(temperature)
    )

    # Determine if temperature should be sent (checks cache and model support)
    send_temperature, non_default_requested = _should_send_temperature(
        cache_key=cache_key,
        effective_temperature=effective_temperature,
        treat_as_default=treat_as_default,
        user_provided=user_provided,
        supported_params=supported_params,
        model_name=model_name,
    )

    async def _call_with_retry(completion_kwargs: dict[str, Any]) -> Any:
        nonlocal send_temperature
        call_kwargs = dict(completion_kwargs)
        if send_temperature:
            call_kwargs["temperature"] = effective_temperature
        try:
            response = await acompletion(**call_kwargs)
            if send_temperature:
                _TEMPERATURE_SUPPORT_CACHE[cache_key] = True
            return response
        except BadRequestError as exc:
            if send_temperature and "temperature" in str(exc).lower():
                send_temperature = False
                _TEMPERATURE_SUPPORT_CACHE[cache_key] = False
                if non_default_requested:
                    log.warning(
                        "Model %s does not support temperature; ignoring temperature=%s",
                        model_name,
                        effective_temperature,
                    )
                call_kwargs.pop("temperature", None)
                return await acompletion(**call_kwargs)
            raise

    messages = [{"role": "user", "content": template}]
    if structured_output:
        if not base_url:
            assert supported_params is not None
            assert (
                "response_format" in supported_params
            ), "response_format is not supported in this model"
            assert supports_response_schema(
                model=model_name
            ), "response_schema is not supported in this model"

        assert isinstance(
            output_parser, PydanticOutputParser
        ), "structured output only supported in PydanticOutputParser"

        # Build JSON schema response format with OpenAI compatibility fixes
        schema = output_parser.pydantic_object.model_json_schema()
        response_format = _build_json_schema_response_format(
            schema, output_parser.pydantic_object
        )

        completion_kwargs = _build_structured_completion_kwargs(
            model_name=model_name,
            messages=messages,
            response_format=response_format,
            base_url=base_url,
            api_key=api_key,
        )
        response = await _call_with_retry(completion_kwargs)
    else:
        completion_kwargs = _build_standard_completion_kwargs(
            model_name=model_name,
            messages=messages,
            base_url=base_url,
            api_key=api_key,
        )
        response = await _call_with_retry(completion_kwargs)
    result = response.choices[0].message.content

    # Only PydanticOutputParser supports context parameter
    parse_kwargs = (
        {"context": context} if isinstance(output_parser, PydanticOutputParser) else {}
    )

    try:
        parsed_result = output_parser.parse(result, **parse_kwargs)
    except Exception:
        reformat_result = await format_bad_output(
            result,
            output_parser.get_format_instructions(),
            bad_output_process_model or model_name,
            use_fixed_model_version,
            base_url=base_url,
        )
        parsed_result = output_parser.parse(reformat_result, **parse_kwargs)

    # Include agent name in logs if available
    agent_name = input_values.get("agent", "")
    log_prefix = f" [{agent_name}]" if agent_name else ""
    log.info(f"Generated result{log_prefix}: {parsed_result}")
    return parsed_result


@gin.configurable
@validate_call
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> EnvironmentProfile:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please generate scenarios and goals based on the examples below as well as the inspirational prompt, when creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.
        Examples:
        {examples}
        Inspirational prompt: {inspiration_prompt}
        Please use the following format:
        {format_instructions}
        """,
        input_values=dict(
            inspiration_prompt=inspiration_prompt,
            examples=examples,
        ),
        output_parser=PydanticOutputParser(pydantic_object=EnvironmentProfile),
        temperature=temperature,
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@validate_call
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> tuple[RelationshipProfile, str]:
    """
    Using langchain to generate the background
    """
    agent_profile = "\n".join(agents_profiles)
    return await agenerate(
        model_name=model_name,
        template="""Please generate relationship between two agents based on the agents' profiles below. Note that you generate
        {agent_profile}
        Please use the following format:
        {format_instructions}
        """,
        input_values=dict(
            agent_profile=agent_profile,
        ),
        output_parser=PydanticOutputParser(pydantic_object=RelationshipProfile),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@gin.configurable
@validate_call
async def agenerate_action(
    model_name: str,
    history: str,
    turn_number: int,
    action_types: list[ActionType],
    agent: str,
    goal: str,
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    script_like: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    agent_names: list[str] | None = None,
    sender: str | None = None,
) -> AgentAction:
    """
    Using langchain to generate an example episode
    """
    try:
        if script_like:
            # model as playwright
            template = """
                Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.
                You can find {agent}'s background and goal in the 'Here is the context of the interaction' field.
                You should try your best to achieve {agent}'s goal in a way that align with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {history}.
                The script has proceeded to Turn #{turn_number}. Current available action types are
                {action_list}.
                Note: The script can be ended if 1. one agent have achieved social goals, 2. this conversation makes the agent uncomfortable, 3. the agent find it uninteresting/you lose your patience, 4. or for other reasons you think it should stop.

                Please only generate a JSON string including the action type and the argument.
                Your action should follow the given format:
                {format_instructions}
            """
        else:
            # Normal case, model as agent
            template = """
                Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
                You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
                Note that {agent}'s goal is only visible to you.
                You should try your best to achieve {agent}'s goal in a way that align with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {history}.
                You are at Turn #{turn_number}. Your available action types are
                {action_list}.
                Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

                Please only generate a JSON string including the action type and the argument.
                Your action should follow the given format:
                {format_instructions}
            """
        # Build validation context if agent_names provided
        validation_context = None
        if agent_names is not None:
            validation_context = {
                "agent_names": agent_names,
                "available_action_types": action_types,
            }
            if sender is not None:
                validation_context["sender"] = sender

        return await agenerate(
            model_name=model_name,
            template=template,
            input_values=dict(
                agent=agent,
                turn_number=str(turn_number),
                history=history,
                action_list=" ".join(action_types),
            ),
            output_parser=PydanticOutputParser(pydantic_object=AgentAction),
            temperature=temperature,
            structured_output=True,
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
            context=validation_context,
        )
    except Exception as e:
        log.warning(f"Failed to generate action due to {e}")
        return AgentAction(action_type="none", argument="")


@gin.configurable
@validate_call
async def agenerate_script(
    model_name: str,
    background: ScriptBackground,
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    agent_names: list[str] = [],
    agent_name: str = "",
    history: str = "",
    single_step: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> tuple[ScriptInteractionReturnType, str]:
    """
    Using langchain to generate an the script interactions between two agent
    The script interaction is generated in a single generation process.
    Note that in this case we do not require a json format response,
    so the failure rate will be higher, and it is recommended to use at least llama-2-70b.
    """
    try:
        if single_step:
            return await agenerate(
                model_name=model_name,
                template="""Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.

                Here are the conversation background and history:
                {background}
                {history}

                Remember that you are an independent scriptwriter and should finish the script by yourself.
                The output should only contain the script following the format instructions, with no additional comments or text.

                Here are the format instructions:
                {format_instructions}""",
                input_values=dict(
                    background=background.to_natural_language(),
                    history=history,
                    agent=agent_name,
                ),
                output_parser=ScriptOutputParser(  # type: ignore[arg-type]
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=True,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )

        else:
            return await agenerate(
                model_name=model_name,
                template="""
                Please write the script between two characters based on their social goals with a maximum of 20 turns.

                {background}
                Your action should follow the given format:
                {format_instructions}
                Remember that you are an independent scriptwriter and should finish the script by yourself.
                The output should only contain the script following the format instructions, with no additional comments or text.""",
                input_values=dict(
                    background=background.to_natural_language(),
                ),
                output_parser=ScriptOutputParser(  # type: ignore[arg-type]
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=False,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )
    except Exception as e:
        # TODO raise(e) # Maybe we do not want to return anything?
        print(f"Exception in agenerate {e}")
        return_default_value: ScriptInteractionReturnType = (
            ScriptInteraction.default_value_for_return_type()
        )
        return (return_default_value, "")


@validate_call
def process_history(
    script: ScriptBackground | EnvResponse | dict[str, AgentAction],
) -> str:
    """
    Format the script background
    """
    result = ""
    if isinstance(script, ScriptBackground | EnvResponse):
        script = script.dict()
        result = "The initial observation\n\n"
    for key, value in script.items():
        if value:
            result += f"{key}: {value} \n"
    return result


@validate_call
async def agenerate_init_profile(
    model_name: str,
    basic_info: dict[str, str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please expand a fictional background for {name}. Here is the basic information:
            {name}'s age: {age}
            {name}'s gender identity: {gender_identity}
            {name}'s pronouns: {pronoun}
            {name}'s occupation: {occupation}
            {name}'s big 5 personality traits: {bigfive}
            {name}'s moral Foundation: think {mft} is more important than others
            {name}'s Schwartz portrait value: {schwartz}
            {name}'s decision-making style: {decision_style}
            {name}'s secret: {secret}
            Include the previous information in the background.
            Then expand the personal backgrounds with concrete details (e.g, look, family, hobbies, friends and etc.)
            For the personality and values (e.g., MBTI, moral foundation, and etc.),
            remember to use examples and behaviors in the person's life to demonstrate it.
            """,
        input_values=dict(
            name=basic_info["name"],
            age=basic_info["age"],
            gender_identity=basic_info["gender_identity"],
            pronoun=basic_info["pronoun"],
            occupation=basic_info["occupation"],
            bigfive=basic_info["Big_Five_Personality"],
            mft=basic_info["Moral_Foundation"],
            schwartz=basic_info["Schwartz_Portrait_Value"],
            decision_style=basic_info["Decision_making_Style"],
            secret=basic_info["secret"],
        ),
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@validate_call
async def convert_narratives(
    model_name: str,
    narrative: str,
    text: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    if narrative == "first":
        return await agenerate(
            model_name=model_name,
            template="""Please convert the following text into a first-person narrative.
            e.g, replace name, he, she, him, her, his, and hers with I, me, my, and mine.
            {text}""",
            input_values=dict(text=text),
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    elif narrative == "second":
        return await agenerate(
            model_name=model_name,
            template="""Please convert the following text into a second-person narrative.
            e.g, replace name, he, she, him, her, his, and hers with you, your, and yours.
            {text}""",
            input_values=dict(text=text),
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    else:
        raise ValueError(f"Narrative {narrative} is not supported.")


@validate_call
async def agenerate_goal(
    model_name: str,
    background: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please generate your goal based on the background:
            {background}
            """,
        input_values=dict(background=background),
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )
