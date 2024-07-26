import pytest
from typing import Any

from sotopia.generation_utils.generate import (
    ListOfIntOutputParser,
    agenerate,
)


@pytest.mark.asyncio
async def test_agenerate_list_integer() -> None:
    """
    async version of test_generate_list_integer
    """
    length, lower, upper = 5, -10, 10
    list_of_int = await agenerate(
        "gpt-3.5-turbo",
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
    await agenerate(
        "gpt-3.5-turbo",
        "{format_instructions}",
        {},
        ListOfIntOutputParser(5, (-10, 10)),
    )
    # Check if any log records were captured
    assert len(caplog.records) > 0, "No log records captured"

    # Optionally, you can print the captured log records for verification
    for record in caplog.records:
        print(f"Captured log: {record.levelname} - {record.message}")
