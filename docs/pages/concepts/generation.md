## Generation functions

The core of generating agent action and environment observation lies in the `agenerate` function:

```python
@gin.configurable
@beartype
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: BaseOutputParser[OutputType],
    temperature: float = 0.7,
    structured_output: bool = False,
    bad_output_process_model: str = DEFAULT_BAD_OUTPUT_PROCESS_MODEL,
    use_fixed_model_version: bool = True
) -> OutputType:
    input_variables = re.findall(r"(?<!{){([^{}]+)}(?!})", template)
```

The `agenerate` function is versatile by taking the output_parser as an argument and returning the output in the desired format.
<Callout type="info" emoji="ℹ️">
  Structured output is used to return the output in a structured format, such as a dictionary or a Pydantic object.
  Currently, the structured output is only supported for the models below:
    * `gpt-4o-mini-2024-07-18` and later
    * `gpt-4o-2024-08-06` and later

The `bad_output_process_model` is used to process the bad output. `DEFAULT_BAD_OUTPUT_PROCESS_MODEL` is set to be `gpt-4o-mini` (At the publication time of Sotopia, we used `gpt-3.5-turbo-0613`. However this model has been taken off the shelf by OpenAI.).

The `use_fixed_model_version` is used to determine whether to use the fixed model version. If set to `True`, the model version will be fixed to the version that was used in Sotopia paper. If set to `False`, the model version will be the latest version available.

Warning: As some fixed model versions might not be available in the OpenAI API, setting `use_fixed_model_version = True` might result in an error.

</Callout>

Here are a few examples of how to use the `agenerate` function:



### Automatically generate scenarios

```python
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: float = 0.7,
    bad_output_process_model: str = DEFAULT_BAD_OUTPUT_PROCESS_MODEL,
    use_fixed_model_version: bool = True
) -> tuple[EnvironmentProfile, str]:
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
        use_fixed_model_version=use_fixed_model_version
    )
```
### Other generation functions
Similarly, there are other utility functions that builds upon the `agenerate` function to generate different types of outputs.

```python
@beartype
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str = DEFAULT_BAD_OUTPUT_PROCESS_MODEL,
    use_fixed_model_version: bool = True
) -> tuple[RelationshipProfile, str]
```

```python
async def agenerate_script(
    model_name: str,
    background: ScriptBackground,
    temperature: float = 0.7,
    agent_names: list[str] = [],
    agent_name: str = "",
    history: str = "",
    single_step: bool = False,
    bad_output_process_model: str = DEFAULT_BAD_OUTPUT_PROCESS_MODEL,
    use_fixed_model_version: bool = True
) -> tuple[ScriptInteractionReturnType, str]
```
