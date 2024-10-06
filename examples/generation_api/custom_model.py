# Example: Generate n random numbers using Llama3.2 model served by ollama
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
# Expected output: a bunch of logs and an output [6, 8, 3, 9, 7]

from sotopia.generation_utils.generate import ListOfIntOutputParser, agenerate
import logging

# Set logging to the lowest level to show all logs
logging.basicConfig(level=0)


async def generate_n_random_numbers(n: int) -> list[int]:
    return await agenerate(
        model_name="custom/llama3.2:1b@http://localhost:11434/v1",
        template="Generate {n} random integer numbers. {format_instructions}",
        input_values={"n": n},
        temperature=0.0,
        output_parser=ListOfIntOutputParser(n),
    )


if __name__ == "__main__":
    import asyncio

    n = 5
    random_numbers = asyncio.run(generate_n_random_numbers(n))
    print(random_numbers)
