import logging
import os
import re
import json
from dataclasses import dataclass
from litellm import acompletion
from litellm.exceptions import BadRequestError
from litellm.utils import supports_response_schema
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from typing import Any, cast, Literal, overload

import gin

from pydantic import validate_call
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


def fill_template(template: str, **kwargs: str) -> str:
    """Fill template with kwargs, ignoring missing keys."""
    for k, v in kwargs.items():
        template = template.replace(f"{{{k}}}", v)
    return template


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

        def _fix_schema(s: dict[str, Any]) -> None:
            if s.get("type") == "array":
                if "prefixItems" in s:
                    # OpenAI doesn't support prefixItems (tuple validation).
                    # Convert to items: {anyOf: [...]} to satisfy "items must be a schema object"
                    # This allows valid tuple elements but loses positional validation, which is acceptable for strict=False.
                    prefix_items = s.pop("prefixItems")
                    s["items"] = {"anyOf": prefix_items}

                if "items" in s and isinstance(s["items"], dict):
                    _fix_schema(s["items"])
                elif "items" in s and isinstance(s["items"], list):
                    # Should not happen after the fix above, but handle legacy cases if any
                    for item in s["items"]:
                        _fix_schema(item)
            elif s.get("type") == "object":
                if "properties" in s:
                    for prop in s["properties"].values():
                        _fix_schema(prop)
                if "additionalProperties" in s and isinstance(
                    s["additionalProperties"], dict
                ):
                    _fix_schema(s["additionalProperties"])
                if "$defs" in s:
                    for def_schema in s["$defs"].values():
                        _fix_schema(def_schema)

        _fix_schema(schema)

        # Build proper json_schema response_format
        completion_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "reformatted_output",
                "schema": schema,
                "strict": False,
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
    log.debug(f"Model: {model_name}")
    log.debug(f"Prompt: {content}")
    log.info(f"Reformated output: {reformatted_output}")
    return reformatted_output


@overload
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: OutputParser[OutputType],
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[False] = False,
) -> OutputType: ...


@overload
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: OutputParser[OutputType],
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[True] = ...,
) -> tuple[OutputType, list[dict[str, str]], str]: ...


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
    return_prompt_and_response: bool = False,
) -> OutputType | tuple[OutputType, list[dict[str, str]], str]:
    """Generate text using LiteLLM instead of Langchain."""
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

    effective_temperature: float | None
    treat_as_default: bool
    user_provided_temperature: bool

    if temperature is _TEMPERATURE_SENTINEL:
        effective_temperature = DEFAULT_TEMPERATURE
        treat_as_default = True
        user_provided_temperature = False
    elif isinstance(temperature, TemperatureSetting):
        effective_temperature = temperature.value
        treat_as_default = temperature.treat_as_default
        user_provided_temperature = True
    else:
        effective_temperature = cast(float | None, temperature)
        treat_as_default = False
        user_provided_temperature = True

    send_temperature = (
        user_provided_temperature
        and effective_temperature is not None
        and not treat_as_default
    )
    non_default_requested = send_temperature and not treat_as_default

    if send_temperature:
        if cache_key in _TEMPERATURE_SUPPORT_CACHE:
            if not _TEMPERATURE_SUPPORT_CACHE[cache_key]:
                send_temperature = False
                if non_default_requested:
                    log.warning(
                        "Model %s previously rejected temperature; ignoring temperature=%s",
                        model_name,
                        effective_temperature,
                    )
        elif supported_params is not None and "temperature" not in supported_params:
            _TEMPERATURE_SUPPORT_CACHE[cache_key] = False
            send_temperature = False
            if non_default_requested:
                log.warning(
                    "Model %s does not support temperature; ignoring temperature=%s",
                    model_name,
                    effective_temperature,
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

    if structured_output:
        if not base_url:
            assert supported_params is not None
            assert (
                "response_format" in supported_params
            ), "response_format is not supported in this model"
            assert supports_response_schema(
                model=model_name
            ), "response_schema is not supported in this model"
        messages = [{"role": "user", "content": template}]

        assert isinstance(
            output_parser, PydanticOutputParser
        ), "structured output only supported in PydanticOutputParser"
        completion_kwargs = dict(
            model=model_name,
            messages=messages,
            response_format=output_parser.pydantic_object,
            drop_params=True,  # drop params to avoid model error if the model does not support it
            base_url=base_url,
            api_key=api_key,
        )
        response = await _call_with_retry(completion_kwargs)
        result = response.choices[0].message.content
        # Include agent name in logs if available
        agent_name = input_values.get("agent", "")
        log_prefix = f" [{agent_name}]" if agent_name else ""
        log.debug(f"Model: {model_name}")
        log.debug(f"Prompt: {messages}")
        try:
            clean_result = json.dumps(json.loads(result.strip()), ensure_ascii=False)
        except Exception:
            clean_result = result.replace("\n", "").strip()
            clean_result = re.sub(r"\s+", " ", clean_result)
        log.info(f"Generated result{log_prefix}: {clean_result}")
        assert isinstance(result, str)
        parsed = cast(OutputType, output_parser.parse(result))
        if return_prompt_and_response:
            return parsed, messages, result
        return parsed

    messages = [{"role": "user", "content": template}]

    completion_kwargs = dict(
        model=model_name,
        messages=messages,
        drop_params=True,
        base_url=base_url,
        api_key=api_key,
    )
    response = await _call_with_retry(completion_kwargs)
    result = response.choices[0].message.content

    try:
        parsed_result = output_parser.parse(result)
    except Exception as e:
        if isinstance(output_parser, ScriptOutputParser):
            raise e
        log.debug(
            f"Failed to parse result: {result}\nEncounter Exception {e}\nstart to reparse",
            extra={"markup": True},
        )
        # Handle bad output reformatting
        reformat_result = await format_bad_output(
            result,
            output_parser.get_format_instructions(),
            bad_output_process_model or model_name,
            use_fixed_model_version,
            base_url=base_url,
        )
        parsed_result = output_parser.parse(reformat_result)

    # Include agent name in logs if available
    agent_name = input_values.get("agent", "")
    log_prefix = f" [{agent_name}]" if agent_name else ""
    log.debug(f"Model: {model_name}")
    log.debug(f"Prompt: {messages}")
    try:
        clean_result = json.dumps(json.loads(result.strip()), ensure_ascii=False)
    except Exception:
        clean_result = result.replace("\n", "").strip()
        clean_result = re.sub(r"\s+", " ", clean_result)
    log.info(f"Generated result{log_prefix}: {clean_result}")
    if return_prompt_and_response:
        return parsed_result, messages, result
    return parsed_result


@overload
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[False] = False,
) -> EnvironmentProfile: ...


@overload
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[True] = ...,
) -> tuple[EnvironmentProfile, list[dict[str, str]], str]: ...


@gin.configurable
@validate_call
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: TemperatureSetting | float | None | object = _TEMPERATURE_SENTINEL,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: bool = False,
) -> EnvironmentProfile | tuple[EnvironmentProfile, list[dict[str, str]], str]:
    """
    Using langchain to generate the background
    """
    if return_prompt_and_response:
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
            return_prompt_and_response=True,
        )
    else:
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
            return_prompt_and_response=False,
        )


@overload
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[False] = False,
) -> RelationshipProfile: ...


@overload
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: Literal[True] = ...,
) -> tuple[RelationshipProfile, list[dict[str, str]], str]: ...


@validate_call
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    return_prompt_and_response: bool = False,
) -> RelationshipProfile | tuple[RelationshipProfile, list[dict[str, str]], str]:
    """
    Using langchain to generate the background
    """
    agent_profile = "\n".join(agents_profiles)
    if return_prompt_and_response:
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
            return_prompt_and_response=True,
        )
    else:
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
            return_prompt_and_response=False,
        )


@overload
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
    strict_action_constraint: bool = False,
    custom_template: str | None = None,
    return_prompt_and_response: Literal[False] = False,
) -> AgentAction: ...


@overload
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
    strict_action_constraint: bool = False,
    custom_template: str | None = None,
    return_prompt_and_response: Literal[True] = ...,
) -> tuple[AgentAction, list[dict[str, str]], str]: ...


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
    strict_action_constraint: bool = False,
    custom_template: str | None = None,
    return_prompt_and_response: bool = False,
) -> AgentAction | tuple[AgentAction, list[dict[str, str]], str]:
    """
    Using langchain to generate an example episode
    """
    try:
        if custom_template:
            if script_like:
                raise ValueError(
                    "script_like and custom_template are mutually exclusive"
                )
            template = custom_template
        elif script_like:
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

        # Create dynamic AgentAction model with restricted ActionType
        if strict_action_constraint and action_types:
            # Create a dynamic Literal for the allowed action types
            # Use __getitem__ to dynamically create Literal from list of strings
            DynamicActionType = Literal.__getitem__(tuple(action_types))

            # Create a dynamic Pydantic model
            from pydantic import create_model, Field

            DynamicAgentAction = create_model(
                "AgentAction",
                action_type=(
                    DynamicActionType,
                    Field(
                        ...,
                        description="whether to speak at this turn or choose to not do anything",
                    ),
                ),
                argument=(
                    str,
                    Field(
                        ...,
                        description="the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action",
                    ),
                ),
                __base__=AgentAction,
            )

            output_parser_obj: PydanticOutputParser[Any] = PydanticOutputParser(
                pydantic_object=DynamicAgentAction
            )
        else:
            output_parser_obj = PydanticOutputParser(pydantic_object=AgentAction)

        if return_prompt_and_response:
            return await agenerate(
                model_name=model_name,
                template=template,
                input_values=dict(
                    agent=agent,
                    turn_number=str(turn_number),
                    history=history,
                    action_list=" ".join(action_types),
                    goal=goal,
                ),
                output_parser=output_parser_obj,
                temperature=temperature,
                structured_output=True,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
                return_prompt_and_response=True,
            )
        else:
            return await agenerate(
                model_name=model_name,
                template=template,
                input_values=dict(
                    agent=agent,
                    turn_number=str(turn_number),
                    history=history,
                    action_list=" ".join(action_types),
                    goal=goal,
                ),
                output_parser=output_parser_obj,
                temperature=temperature,
                structured_output=True,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
                return_prompt_and_response=False,
            )
    except Exception as e:
        log.warning(f"Failed to generate action due to {e}")
        action = AgentAction(action_type="none", argument="")
        if return_prompt_and_response:
            return action, [], str(e)
        return action


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
            result = await agenerate(
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
                output_parser=ScriptOutputParser(
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=True,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )
            return cast(tuple[ScriptInteractionReturnType, str], result)

        else:
            result = await agenerate(
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
                output_parser=ScriptOutputParser(
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=False,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )
            return cast(tuple[ScriptInteractionReturnType, str], result)
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
