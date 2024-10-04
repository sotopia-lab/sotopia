# Welcome to the documentation for the Sotopia
import os
import openai
import rich
from tqdm import tqdm


def generate_api_docs(source_dir: str, output_dir: str) -> None:
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Walk through the source directory
    total_files = sum(
        [
            len(files)
            for r, d, files in os.walk(source_dir)
            if any(f.endswith(".py") and not f.startswith("__") for f in files)
        ]
    )
    with tqdm(total=total_files, desc="Generating API docs") as pbar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    file_path = os.path.join(root, file)

                    # Read the Python file
                    with open(file_path, "r") as f:
                        code = f.read()

                    # Write the documentation to a file
                    doc_content = generate_doc_for_node(code)
                    rich.print(f"doc_content: {doc_content}")
                    relative_path = os.path.relpath(root, source_dir)
                    output_path = os.path.join(
                        output_dir, relative_path, f"{file[:-3]}.md"
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    with open(output_path, "w") as f:
                        f.write(doc_content)
            pbar.update(1)


def generate_doc_for_node(code: str) -> str:
    prompt = f"""
    Please provide a brief documentation for the following Python code:

    ```python
    {code}
    ```

    Include a short description, parameters (if any), and return value (if any) as well as usage examples. Try to prettify the markdown content as much as possible.
    Please only return the markdown content and do not include any other text.
    """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates Python documentation. You only return the markdown content and do not include any other text.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    doc = response.choices[0].message.content
    assert isinstance(doc, str)

    return doc


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    source_dir = "sotopia/samplers"
    output_dir = "docs/pages/python_API/samplers"
    generate_api_docs(source_dir, output_dir)

print("API documentation generated successfully.")
