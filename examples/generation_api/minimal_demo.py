from sotopia.generation_utils.generate import ListOfIntOutputParser, agenerate


async def generate_first_N_prime_numbers(n: int) -> list[int]:
    return await agenerate(
        model_name="gpt-4o",
        template="Generate the first {n} prime numbers.",
        input_values={"n": str(n)},
        output_parser=ListOfIntOutputParser(n),
    )


if __name__ == "__main__":
    import asyncio

    n = 5
    prime_numbers = asyncio.run(generate_first_N_prime_numbers(n))
    print(prime_numbers)
