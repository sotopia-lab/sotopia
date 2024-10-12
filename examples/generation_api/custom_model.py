# Example: Generate (2~7) random numbers using Llama3.2 model served by ollama
# To run this example, you can either
# 1. Use the sotopia devcontainer and run the following command in the terminal:
#        uv run python examples/generation_api/custom_model.py
#    This example can also serve a sanity check for your devcontainer setup.
# OR 2. after installing sotopia, install llama.cpp, and then follow llama.cpp's instructions to serve the model
# OR 3. after installing sotopia, serve your desired model on your desired port,
#    and then change the model_name of `agenerate` to point to your desired model.
#    Finally:
#        python examples/generation_api/custom_model.py
# Expected output for 1: a bunch of logs and an output [[14, 7], [14, 7, 3], [14, 7, 3, 9], [14, 7, 3, 9, 6], [14, 7, 3, 9, 6, 8]]

from sotopia.generation_utils.generate import ListOfIntOutputParser, agenerate
import logging

# Set logging to the lowest level to show all logs
logging.basicConfig(level=0)


async def generate_n_random_numbers(n: int) -> list[int]:
    return await agenerate(
        model_name="custom/llama3.2:1b@http://localhost:8000/v1",
        template="Generate {n} random integer numbers. {format_instructions}",
        input_values={"n": str(n)},
        temperature=0.0,
        output_parser=ListOfIntOutputParser(n),
    )


async def main() -> None:
    random_numbers = await asyncio.gather(
        *[generate_n_random_numbers(n) for n in range(2, 7)]
    )
    print(random_numbers)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
