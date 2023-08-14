import pytest

from sotopia.generation_utils.generate import (
    ListOfIntOutputParser,
    agenerate,
    generate,
)
from sotopia.messages import ScriptEnvironmentResponse


def test_generate_list_integer() -> None:
    """
    Test that the integer generator works
    """
    length, lower, upper = 5, -10, 10
    l = generate(
        "gpt-3.5-turbo",
        "{format_instructions}",
        {},
        ListOfIntOutputParser(length, (lower, upper)),
    )
    assert isinstance(l, list)
    assert len(l) == length
    assert all(isinstance(i, int) for i in l)
    assert all(lower <= i <= upper for i in l)


@pytest.mark.asyncio
async def test_agenerate_list_integer() -> None:
    """
    async version of test_generate_list_integer
    """
    length, lower, upper = 5, -10, 10
    l, _ = await agenerate(
        "gpt-3.5-turbo",
        "{format_instructions}",
        {},
        ListOfIntOutputParser(length, (lower, upper)),
    )
    assert isinstance(l, list)
    assert len(l) == length
    assert all(isinstance(i, int) for i in l)
    assert all(lower <= i <= upper for i in l)


@pytest.mark.asyncio
async def test_agenerate_list_integer_together() -> None:
    """
    async version of test_generate_list_integer
    """
    length, lower, upper = 5, -10, 10
    l, _ = await agenerate(
        "togethercomputer/llama-2-70b-chat",
        "{format_instructions}",
        {},
        ListOfIntOutputParser(length, (lower, upper)),
    )
    assert isinstance(l, list)
    assert len(l) == length
    assert all(isinstance(i, int) for i in l)
    assert all(lower <= i <= upper for i in l)
