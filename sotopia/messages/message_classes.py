from typing import Literal

from pydantic import BaseModel, Field

from sotopia.utils import format_docstring

ActionType = Literal[
    "none", "speak", "non-verbal communication", "action", "leave"
]


class Message(BaseModel):
    """
    An interface for messages.
    There is only one required method: to_natural_language
    """

    def to_natural_language(self) -> str:
        raise NotImplementedError


class SimpleMessage(Message):
    """
    A simple message with a single string field.
    """

    message: str = Field(description="the message")

    def to_natural_language(self) -> str:
        return self.message


class Observation(Message):
    last_turn: str = Field(description="the last turn of the conversation")
    turn_number: int = Field(description="the turn number of the conversation")
    available_actions: list[ActionType] = Field(
        description="the available actions"
    )

    def to_natural_language(self) -> str:
        if self.turn_number == 0:
            return f"\n{self.last_turn}\nConversation Starts:\n"
        else:
            return f"Turn #{self.turn_number-1}: {self.last_turn}\n"


class ScriptBackground(Message):
    scenario: str = Field(description="scenario of the episode")
    p1_name: str = Field(description="name of participant 1")
    p2_name: str = Field(description="name of participant 2")
    p1_background: str = Field(description="background of participant 1")
    p2_background: str = Field(description="background of participant 2")
    p1_goal: str = Field(description="goal of participant 1")
    p2_goal: str = Field(description="goal of participant 2")

    def to_natural_language(self) -> str:
        return format_docstring(
            f"""Here is the context of this interaction:
        Scenario: {self.scenario}
        Participants: {self.p1_name} and {self.p2_name}
        {self.p1_name}'s background: {self.p1_background}
        {self.p2_name}'s background: {self.p2_background}
        {self.p1_name}'s goal: {self.p1_goal}
        {self.p2_name}'s goal: {self.p2_goal}
        """
        )


class ScriptEnvironmentResponse(Message):
    terminated: bool = Field(
        description="whether the conversation is terminated",
        default_factory=lambda: False,
    )
    p1_rate: float | tuple[float, dict[str, float]] | None = Field(
        description="rating of participant 1, on the scale of 1 to 10"
    )
    p2_rate: float | tuple[float, dict[str, float]] | None = Field(
        description="rating of participant 2, on the scale of 1 to 10"
    )
    comments: str | None = Field(
        description="All of the comments supporting the termination and rating"
    )

    def to_natural_language(self) -> str:
        reason_to_stop = format_docstring(
            f"""Environment response:
        {"The conversation is terminated." if self.terminated else ""}
        {"Rating of participant 1" + str(self.p1_rate) if self.p1_rate is not None else ""}
        {"Rating of participant 2" + str(self.p2_rate) if self.p2_rate is not None else ""}
        {self.comments if self.comments is not None else ""}
        """
        )
        clean_text = ""
        for line in reason_to_stop.split("\n"):
            if line.strip():
                clean_text += line + "\n"
        return clean_text


class AgentAction(Message):
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
                return f'said: "{self.argument}"'
            case "non-verbal communication":
                return f"[{self.action_type}] {self.argument}"
            case "action":
                return f"[{self.action_type}] {self.argument}"
            case "leave":
                return "left the conversation"
