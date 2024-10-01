import pytest
from typing import Any

from sotopia.generation_utils.generate import (
    ListOfIntOutputParser,
    agenerate,
)

from sotopia.messages import AgentAction
from langchain.output_parsers import PydanticOutputParser


@pytest.mark.asyncio
async def test_agenerate_list_integer() -> None:
    """
    async version of test_generate_list_integer
    """
    length, lower, upper = 5, -10, 10
    list_of_int = await agenerate(
        "gpt-4o-mini",
        "{format_instructions}",
        {},
        ListOfIntOutputParser(length, (lower, upper)),
    )
    assert isinstance(list_of_int, list)
    assert len(list_of_int) == length
    assert all(isinstance(i, int) for i in list_of_int)
    assert all(lower <= i <= upper for i in list_of_int)


@pytest.mark.skip(reason="togethercompute out of credit")
@pytest.mark.asyncio
async def test_agenerate_list_integer_together() -> None:
    """
    async version of test_generate_list_integer
    """
    length, lower, upper = 5, -10, 10
    list_of_int = await agenerate(
        "togethercomputer/llama-2-70b-chat",
        "{format_instructions}",
        {},
        ListOfIntOutputParser(length, (lower, upper)),
    )
    assert isinstance(list_of_int, list)
    assert len(list_of_int) == length
    assert all(isinstance(i, int) for i in list_of_int)
    assert all(lower <= i <= upper for i in list_of_int)


@pytest.mark.asyncio
async def test_logging_behavior(caplog: Any) -> None:
    # Call the function under test
    caplog.set_level(15)
    await agenerate(
        "gpt-4o-mini",
        "{format_instructions}",
        {},
        ListOfIntOutputParser(5, (-10, 10)),
    )
    # Check if any log records were captured
    assert len(caplog.records) > 0, "No log records captured"

    # Optionally, you can print the captured log records for verification
    for record in caplog.records:
        print(f"Captured log: {record.levelname} - {record.message}")


@pytest.mark.asyncio
async def test_agenerate_structured_output() -> None:
    """
    async version of test_generate_structured_output
    """
    output = await agenerate(
        "gpt-4o-2024-08-06",
        "{format_instructions}",
        {},
        PydanticOutputParser(pydantic_object=AgentAction),
        structured_output=True,
    )
    assert isinstance(output, AgentAction)
