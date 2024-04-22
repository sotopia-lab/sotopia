import pytest

from sotopia.generation_utils.generate import (
    ListOfIntOutputParser,
    agenerate,
    generate,
)


def test_generate_list_integer() -> None:
    """
    Test that the integer generator works
    """
    length, lower, upper = 5, -10, 10
    list_of_int = generate(
        "gpt-3.5-turbo",
        "{format_instructions}",
        {},
        ListOfIntOutputParser(length, (lower, upper)),
    )
    assert isinstance(list_of_int, list)
    assert len(list_of_int) == length
    assert all(isinstance(i, int) for i in list_of_int)
    assert all(lower <= i <= upper for i in list_of_int)


@pytest.mark.asyncio
async def test_agenerate_list_integer() -> None:
    """
    async version of test_generate_list_integer
    """
    length, lower, upper = 5, -10, 10
    list_of_int, _ = await agenerate(
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
    list_of_int, _ = await agenerate(
        "togethercomputer/llama-2-70b-chat",
        "{format_instructions}",
        {},
        ListOfIntOutputParser(length, (lower, upper)),
    )
    assert isinstance(list_of_int, list)
    assert len(list_of_int) == length
    assert all(isinstance(i, int) for i in list_of_int)
    assert all(lower <= i <= upper for i in list_of_int)
