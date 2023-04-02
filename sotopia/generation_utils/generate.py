import langchain
from beartype import beartype
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema import HumanMessage
from typing_extensions import Literal

LLM_Name = Literal["gpt-3.5-turbo", "text-davinci-003", "gpt-4"]


@beartype
def obtain_chain(
    model_name: LLM_Name, template: str, input_variable: list[str]
) -> LLMChain:
    """
    Using langchain to sample profiles for participants
    """
    match model_name:
        case "gpt-3.5-turbo" | "gpt-4":
            human_message_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=template,
                    input_variables=input_variable,
                )
            )
            chat_prompt_template = ChatPromptTemplate.from_messages(
                [human_message_prompt]
            )
            chat = ChatOpenAI(model_name=model_name)  # type: ignore[call-arg]
            chain = LLMChain(llm=chat, prompt=chat_prompt_template)
            return chain
        case "text-davinci-003":
            llm = OpenAI(model_name=model_name)  # type: ignore[call-arg]
            prompt = PromptTemplate(
                input_variables=input_variable,
                template=template,
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain
        case _:
            raise ValueError(f"Invalid model name: {model_name}")


@beartype
def generate_episode(model_name: LLM_Name) -> str:
    """
    Using langchain to generate an example episode
    """
    # Obtain participants
    participants_variable = ["participants"]
    participants_template = "The participants are: {participants}, generate the participants' profiles as one would do in a movie script (include their occupations). Please use the following format: Participant 1: profile 1, Participant 2: profile 2, etc."
    participants_chain = obtain_chain(
        model_name, participants_template, participants_variable
    )

    # Obtain scenarios& goals
    openings_variable = ["participants_profiles"]
    openings_template = "The participants are: {participants_profiles}, generate a scenario and participants' goals as one would do in a movie script. Please use the following format: Scenario: , goal of participant 1: , goal of participant 2: , etc. End after the goals are generated"
    openings_chain = obtain_chain(
        model_name, openings_template, openings_variable
    )

    # Obtain pseudo conversations
    conversations_variable = ["scenarios_and_goals"]
    conversations_template = "Given the following scenarios and goals: {scenarios_and_goals}, generate the conversations as one would do in a movie script. Please use the following format: Participant 1: , Participant 2: , etc. Do not exceed 8 rounds of conversation. Reinterate the goals at the end of the conversation"
    conversations_chain = obtain_chain(
        model_name, conversations_template, conversations_variable
    )

    # Obtain rewards
    rewards_variable = ["conversations"]
    rewards_template = "Given the following conversations: {conversations}, generate a number (1-10) indicating how well participants achieve their goals. Please use the following format: Participant 1: , Participant 2: , etc."
    rewards_chain = obtain_chain(
        model_name, rewards_template, rewards_variable
    )
    overall_chain = SimpleSequentialChain(
        chains=[
            participants_chain,
            openings_chain,
            conversations_chain,
            rewards_chain,
        ],
        verbose=True,
    )
    scripts = overall_chain.run("Jack, Rose")
    return scripts


@beartype
def generate_episode_singleRound(model_name: LLM_Name) -> str:
    """
    Using langchain to generate an example episode but with a single chain
    """
    template = """
            Given {participants}, and {topic},
            generate an episode as one would do in a movie script. Please use the following format:
            Scenario:
            Participant 1's intro:
            Participant 2's intro:
            (Add more participants if given)
            The social goal of the participant 1:
            The social goal of the participant 2:
            (Add more goals of participants if given)
            Conversation:
            (In the format of Participant 1: , Participant 2: , etc.)
            If it comes to the end of the conversation, generate '--end--'
            generate a number (1-10) indicating how well participants achieve their goals.
            Please use the following format:
            Participant 1: , Participant 2: , etc.
    """
    input_variable = ["participants", "topic"]
    chain = obtain_chain(model_name, template, input_variable)
    scripts = chain.predict(
        participants="Jack (a greedy person), Rose", topic="lawsuit"
    )
    return scripts
