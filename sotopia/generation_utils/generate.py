import logging
import re
from typing import TypeVar, cast

import langchain
from beartype import beartype
from beartype.typing import Type
from langchain.chains import (
    ConversationChain,
    LLMChain,
    SimpleSequentialChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import (
    PydanticOutputParser,
    RetryWithErrorOutputParser,
)
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field, validator
from rich import print
from rich.logging import RichHandler
from typing_extensions import Literal

log = logging.getLogger("generate")

LLM_Name = Literal["gpt-3.5-turbo", "text-davinci-003", "gpt-4"]
ActionType = Literal["none", "speak", "non-verbal communication", "action"]

OutputType = TypeVar("OutputType", bound=BaseModel)


class Script(BaseModel):
    scenario: str = Field(description="scenario of the episode")
    p1_background: str = Field(description="background of participant 1")
    p2_background: str = Field(description="background of participant 2")
    p1_goal: str = Field(description="goal of participant 1")
    p2_goal: str = Field(description="goal of participant 2")
    conversation: list[tuple[int, str]] = Field(
        description="conversation between participants"
    )
    p1_rate: int = Field(
        description="rating of participant 1, on the scale of 1 to 10"
    )
    p2_rate: int = Field(
        description="rating of participant 2, on the scale of 1 to 10"
    )


class ScriptBackground(BaseModel):
    scenario: str = Field(description="scenario of the episode")
    p1_name: str = Field(description="name of participant 1")
    p2_name: str = Field(description="name of participant 2")
    p1_background: str = Field(description="background of participant 1")
    p2_background: str = Field(description="background of participant 2")
    p1_goal: str = Field(description="goal of participant 1")
    p2_goal: str = Field(description="goal of participant 2")

    def to_natural_language(self) -> str:
        return f"""
        Here is the context of this interaction: {self.scenario}
        There are two participants in this interaction: {self.p1_name} and {self.p2_name}.
        {self.p1_name} is {self.p1_background}.
        {self.p2_name} is {self.p2_background}.
        {self.p1_name}'s goal is {self.p1_goal}.
        {self.p2_name}'s goal is {self.p2_goal}.
        """


class ScriptEnvironmentResponse(BaseModel):
    conversation_too_long: bool = Field(
        description="whether the conversation is too long"
    )
    p1_leaving: bool = Field(
        description="whether participant 1 is leaving the conversation"
    )
    p2_leaving: bool = Field(
        description="whether participant 2 is leaving the conversation"
    )
    stale_too_long: bool = Field(
        description="whether the conversation is stale for too long"
    )
    terminated: bool = Field(
        description="whether the conversation is terminated",
        default_factory=lambda: False,
    )
    p1_rate: int | None = Field(
        description="rating of participant 1, on the scale of 1 to 10"
    )
    p2_rate: int | None = Field(
        description="rating of participant 2, on the scale of 1 to 10"
    )


class AgentAction(BaseModel):
    action_type: ActionType = Field(
        description="whether to speak at this turn or choose to not do anything"
    )
    argument: str = Field(
        description="the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action"
    )

    def to_natural_language(self) -> str:
        match self.action_type:
            case "none":
                return "did nothing"
            case "speak":
                return f"said {self.argument}"
            case "non-verbal communication":
                return f"{self.argument}"
            case "action":
                return f"did {self.argument}"


class ScriptPydanticOutputParser(PydanticOutputParser):
    def __init__(self, pydantic_object: Type[BaseModel] = Script) -> None:
        super(ScriptPydanticOutputParser, self).__init__(
            pydantic_object=Script
        )

    def parse(self, text: str) -> Script:
        # remove trailing commas before ) or ] from text
        text = re.sub(r",\s*(\)|\])", r"\1", text)
        return cast(Script, super().parse(text))

    def get_format_instructions(self) -> str:
        format_instruction = super().get_format_instructions()
        return (
            format_instruction
            + "conversation is a list of tuples, where the first element is the speaker id (1 or 2) and the second element is the message. Don't leave trailing commas."
        )


@beartype
def obtain_chain(
    model_name: LLM_Name, template: str, input_variables: list[str]
) -> LLMChain:
    """
    Using langchain to sample profiles for participants
    """
    match model_name:
        case "gpt-3.5-turbo" | "gpt-4":
            human_message_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=template,
                    input_variables=input_variables,
                )
            )
            chat_prompt_template = ChatPromptTemplate.from_messages(
                [human_message_prompt]
            )
            chat = ChatOpenAI(model_name=model_name)  # type: ignore[call-arg]
            chain = LLMChain(llm=chat, prompt=chat_prompt_template)
            return chain
        case "text-davinci-003":
            # Warning: no interactive mode for 003
            llm = OpenAI(model_name=model_name)  # type: ignore[call-arg]
            prompt = PromptTemplate(
                input_variables=input_variables,
                template=template,
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain
        case _:
            raise ValueError(f"Invalid model name: {model_name}")


@beartype
def format_bad_output(
    ill_formed_output: str,
    format_instructions: str,
    model_name: LLM_Name = "gpt-3.5-turbo",
) -> str:
    template = """
    Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
    Original string: {ill_formed_output}

    Format instructions: {format_instructions}

    Please only generate the JSON:
    """
    chain = obtain_chain(
        model_name=model_name,
        template=template,
        input_variables=re.findall(r"{(.*?)}", template),
    )
    input_values = {
        "ill_formed_output": ill_formed_output,
        "format_instructions": format_instructions,
    }
    reformat = chain.predict(template=template, **input_values)
    return reformat


@beartype
def generate(
    model_name: LLM_Name,
    template: str,
    input_values: dict[str, str],
    output_struct: Type[OutputType],
    output_parser: Type[PydanticOutputParser] = PydanticOutputParser,
) -> OutputType:
    input_variables = re.findall(r"{(.*?)}", template)
    assert set(input_variables) == set(
        list(input_values.keys()) + ["format_instructions"]
    ), f"The variables in the template must match input_values except for format_instructions. Got {sorted(input_values.keys())}, expect {sorted(input_variables)}"
    chain = obtain_chain(
        model_name=model_name,
        template=template,
        input_variables=input_variables,
    )
    parser = output_parser(pydantic_object=output_struct)
    if "format_instructions" not in input_values:
        input_values["format_instructions"] = parser.get_format_instructions()
    result = chain.predict(template=template, **input_values)
    try:
        parsed_result = cast(OutputType, parser.parse(result))
    except Exception as e:
        log.warning(
            f"[red] Failed to parse result: {result}\nEncounter Exception {e}\nstart to reparse",
            extra={"markup": True},
        )
        reformat_parsed_result = format_bad_output(
            result, format_instructions=parser.get_format_instructions()
        )
        parsed_result = cast(OutputType, parser.parse(reformat_parsed_result))
    return parsed_result


@beartype
def generate_episode(
    model_name: LLM_Name,
    participants: str = "Jack (a greedy person), Rose",
    topic: str = "lawsuit",
    extra_info: str = "",
) -> Script:
    """
    Using langchain to generate an example episode
    """
    return generate(
        model_name=model_name,
        template="""
            Please generate a episode for the interaction between {participants} regarding {topic}.
            You should generate the personal backgrounds and goals in this interaction.
            Use the following extra info if given: {extra_info}
            Please use the following format:
            {format_instructions}
        """,
        input_values=dict(
            participants=participants,
            topic=topic,
            extra_info=extra_info,
        ),
        output_struct=Script,
        output_parser=ScriptPydanticOutputParser,
    )


@beartype
def generate_background(
    model_name: LLM_Name,
    participants: str = "Jack, Rose",
    topic: str = "borrow money",
    extra_info: str = "Jack speaks first, Rose speaks second",
) -> ScriptBackground:
    """
    Using langchain to generate the background
    """
    return generate(
        model_name=model_name,
        template="""
            Please generate the background for the interaction between {participants} regarding {topic}.
            You should generate the personal backgrounds and goals in this interaction.
            Use the following extra info if given: {extra_info}
            Please use the following format:
            {format_instructions}
        """,
        input_values=dict(
            participants=participants,
            topic=topic,
            extra_info=extra_info,
        ),
        output_struct=ScriptBackground,
    )


@beartype
def generate_environment_response(
    model_name: LLM_Name, history: str, action_str: str
) -> ScriptEnvironmentResponse:
    """
    Using langchain to generate the environment response
    """
    try:
        response = generate(
            model_name=model_name,
            template="""
                {history},
                {action_str},
                Is the conversation finished? Please consider the following questions:
                1. Is the conversation too long? (more than 30 turns)
                2. Is any of the agents leaving the conversation?
                3. Is the conversation stale for too long, i.e. no body is doing anythig for more than 5 turns?
                Only when one of the above questions is answered with yes, the conversation is finished.
                How well does participants finish their goals? Please use the following format:
                {format_instructions}
            """,
            input_values=dict(
                history=history,
                action_str=action_str,
            ),
            output_struct=ScriptEnvironmentResponse,
        )
        response.terminated = (
            response.conversation_too_long
            or response.p1_leaving
            or response.p2_leaving
            or response.stale_too_long
        )
        if response.terminated:
            log.warning(f"[green] The conversation is terminated. {response}")
        return response
    except Exception as e:
        log.warning(f"[red] Failed to generate environment response. {e}")
        return ScriptEnvironmentResponse(
            conversation_too_long=False,
            p1_leaving=False,
            p2_leaving=False,
            stale_too_long=False,
            terminated=False,
            p1_rate=None,
            p2_rate=None,
        )


@beartype
def generate_action(
    model_name: LLM_Name,
    history: str,
    turn_number: int,
    action_types: list[str],
    agent: str,
) -> AgentAction:
    """
    Using langchain to generate an example episode
    """
    try:
        return generate(
            model_name=model_name,
            template="""
                You are {agent}.
                {history}


                You are at Turn #{turn_number}. Your available action types are
                {action_list}. Please only generate a JSON string including the action type and the argument.
                Your action should follow the given format:
                {format_instructions}
            """,
            input_values=dict(
                agent=agent,
                turn_number=str(turn_number),
                history=history,
                action_list=" ".join(action_types),
            ),
            output_struct=AgentAction,
        )
    except:
        return AgentAction(action_type="none", argument="")


@beartype
def process_history(
    script: ScriptBackground | Script | dict[str, AgentAction]
) -> str:
    """
    Format the script background
    """
    result = ""
    if isinstance(script, ScriptBackground | Script):
        script = script.dict()
        result = "The initial observation\n\n"
    for key, value in script.items():
        if value:
            result += f"{key}: {value} \n"
    return result
