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
) -> OutputType:
    input_variables = re.findall(r"(?<!{){([^{}]+)}(?!})", template)
```

The `agenerate` function is versatile by taking the output_parser as an argument and returning the output in the desired format.
<Callout type="info" emoji="ℹ️">
  Structured output is used to return the output in a structured format, such as a dictionary or a Pydantic object.
  Currently, the structured output is only supported for the models below:
    * `gpt-4o-mini-2024-07-18` and later
    * `gpt-4o-2024-08-06` and later

</Callout>

Here are a few examples of how to use the `agenerate` function:



### Automatically generate scenarios

```python
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: float = 0.7,
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
    )
```
### Other generation functions
Similarly, there are other utility functions that builds upon the `agenerate` function to generate different types of outputs.

```python
@beartype
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
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
) -> tuple[ScriptInteractionReturnType, str]
```
