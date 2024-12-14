from sotopia.messages import ActionType
from aact.messages import DataModel, DataModelFactory
from pydantic import Field


@DataModelFactory.register("observation")
class Observation(DataModel):
    agent_name: str = Field(description="the name of the agent")
    last_turn: str = Field(description="the last turn of the conversation")
    turn_number: int = Field(description="the turn number of the conversation")
    available_actions: list[ActionType] = Field(description="the available actions")

    def to_natural_language(self) -> str:
        if self.turn_number == 0:
            return f"\n{self.last_turn}\nConversation Starts:\n"
        else:
            return f"Turn #{self.turn_number-1}: {self.last_turn}\n"


@DataModelFactory.register("agent_action")
class AgentAction(DataModel):
    agent_name: str = Field(description="the name of the agent")
    output_channel: str = Field(description="the name of the output channel")
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
