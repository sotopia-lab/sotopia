from sotopia.generation_utils.generate import ListOfIntOutputParser, agenerate


async def generate_first_N_prime_numbers(n: int) -> list[int]:
    return await agenerate(
        model_name="custom/llama3.2:1b@http://localhost:11434/v1",
        bad_output_process_model="custom/llama3.2:1b@http://localhost:11434/v1",
        template="Generate {n} random integer numbers",
        input_values={"n": n},
        output_parser=ListOfIntOutputParser(n),
    )


if __name__ == "__main__":
    import asyncio

    n = 5
    prime_numbers = asyncio.run(generate_first_N_prime_numbers(n))
    print(prime_numbers)
