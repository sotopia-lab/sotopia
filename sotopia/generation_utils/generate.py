import langchain
from langchain.llms import OpenAI


def generate_scenario() -> str:
    """
    Using langchain to generate a scenario
    """
    llm = OpenAI()  # type: ignore[call-arg]
    text = "Please generate a plausible scenario for a conversation between a man and a woman:"
    return llm(text)
