# Example: Generate (2~7) random numbers using Llama3.2 model served by ollama
# To run this example, you can either
# 1. Use the sotopia devcontainer and run the following command in the terminal:
#        poetry run python examples/generation_api/custom_model.py
#    This example can also serve a sanity check for your devcontainer setup.
# OR 2. after installing sotopia, install ollama, and then:
#        ollama pull llama3.2:1b; python examples/generation_api/custom_model.py
# OR 3. after installing sotopia, serve your desired model on your desired port,
#    and then change the model_name of `agenerate` to point to your desired model.
#    Finally:
#        python examples/generation_api/custom_model.py
# Expected output for (1 and 2): a bunch of logs and an output [[14, 67], [6, 8, 3], [6, 8, 3, 9], [6, 8, 3, 9, 7], [7, 9, 6, 8, 4, 1]]

from sotopia.generation_utils.generate import ListOfIntOutputParser, agenerate
import logging

# Set logging to the lowest level to show all logs
logging.basicConfig(level=0)


async def generate_n_random_numbers(n: int) -> list[int]:
    return await agenerate(
        model_name="custom/llama3.2:1b@http://localhost:11434/v1",
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
