import langchain
from beartype import beartype
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import HumanMessage
from typing_extensions import Literal

LLM_Name = Literal["gpt-3.5-turbo", "text-davinci-003", "gpt-4"]


@beartype
def generate_scenario(model_name: LLM_Name) -> str:
    """
    Using langchain to generate a scenario
    """
    text = "Please generate a plausible scenario for a conversation between a man and a woman:"
    match model_name:
        case "gpt-3.5-turbo" | "gpt-4":
            chat_llm = ChatOpenAI(model_name=model_name)  # type: ignore[call-arg]
            message = HumanMessage(content=text)
            return chat_llm([message]).content
        case "text-davinci-003":
            llm = OpenAI(model_name="text-davinci-003")  # type: ignore[call-arg]
            return llm(text)
        case _:
            raise ValueError(f"Invalid model name: {model_name}")
