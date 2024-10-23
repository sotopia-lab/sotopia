import asyncio
from typing import AsyncIterator
from aact import Message, NodeFactory, Node
from aact.messages import Text, Tick, DataModel, DataModelFactory, Zero
from sotopia.agents.llm_agent import ainput
from sotopia.experimental.agents import BaseAgent
from aact.nodes import PrintNode

from sotopia.generation_utils import agenerate
from sotopia.generation_utils.generate import StrOutputParser
from sotopia.messages import ActionType

from pydantic import Field
import logging
from rich.logging import RichHandler
from aiofiles import stdout
from aiofiles.threadpool.text import AsyncTextIndirectIOWrapper
import json
   
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from typing import Any, AsyncIterator

FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.WARNING,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

@DataModelFactory.register("agent_action")
class AgentAction(DataModel):
    agent_name: str = Field(description="the name of the agent")
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


def _format_message_history(message_history: list[tuple[str, str]]) -> str:
    return "\n".join(
        (f"{speaker} said {message}") for speaker, message in message_history
    )
            
            
@NodeFactory.register("llm_agent")
class LLMAgent(BaseAgent[AgentAction | Tick | Text, AgentAction]):
    def __init__(
        self,
        input_text_channels: list[str],
        input_tick_channel: str,
        input_env_channel: str,
        output_channel: str,
        query_interval: int,
        agent_name: str,
        goal: str,
        model_name: str,
        redis_url: str,
    ):
        super().__init__(
            [
                (input_text_channel, AgentAction)
                for input_text_channel in input_text_channels
            ]
            + [
                (input_tick_channel, Tick),
            ]
            + [
                (input_env_channel, Text)
            ],
            [(output_channel, AgentAction)],
            redis_url,
        )
        self.output_channel = output_channel
        self.query_interval = query_interval
        self.count_ticks = 0
        self.message_history: list[tuple[str, str]] = []
        self.name = agent_name
        self.model_name = model_name
        self.goal = goal

    async def send(self, message: AgentAction) -> None:
        if message.action_type == "speak":
            await self.r.publish(
                self.output_channel,
                Message[AgentAction](data=message).model_dump_json(),
            )

    async def aact(self, message: AgentAction | Tick | Text) -> AgentAction:
        match message:
            case Text(text=text):
                self.message_history.append((self.name, text))
                return AgentAction(
                    agent_name=self.name, action_type="none", argument=""
                )
            case Tick():
                self.count_ticks += 1
                if self.count_ticks % self.query_interval == 0:
                    agent_action: str = await agenerate(
                        model_name=self.model_name,
                        template="Imagine that you are a friend of the other persons. Here is the "
                        "conversation between you and them.\n"
                        "You are {agent_name} in the conversation.\n"
                        "{message_history}\n"
                        "and you plan to {goal}.\n"
                        "You can choose to interrupt the other person "
                        "by saying something or not to interrupt by outputting nothing. What would you say? "
                        "Please only output a sentence or not outputting anything."
                        "{format_instructions}",
                        input_values={
                            "message_history": _format_message_history(
                                self.message_history
                            ),
                            "goal": self.goal,
                            "agent_name": self.name,
                        },
                        temperature=0.7,
                        output_parser=StrOutputParser(),
                    )
                    if agent_action != "none" and agent_action != "":
                        self.message_history.append((self.name, agent_action))
                        return AgentAction(
                            agent_name=self.name,
                            action_type="speak",
                            argument=agent_action,
                        )
                    else:
                        return AgentAction(
                            agent_name=self.name, action_type="none", argument=""
                        )
                else:
                    return AgentAction(
                        agent_name=self.name, action_type="none", argument=""
                    )
            case AgentAction(
                agent_name=agent_name, action_type=action_type, argument=text
            ):
                if action_type == "speak":
                    self.message_history.append((agent_name, text))
                return AgentAction(
                    agent_name=self.name, action_type="none", argument=""
                )
            case _:
                raise ValueError(f"Unexpected message type: {type(message)}")



@NodeFactory.register("human_agent")
class HumanAgent(BaseAgent[AgentAction | Tick, AgentAction]):
    def __init__(
        self,
        input_text_channels: list[str],
        input_tick_channel: str,
        output_channel: str,
        query_interval: int,
        agent_name: str,
        goal: str,
        model_name: str,
        redis_url: str,
    ):
        super().__init__(
            [
                (input_text_channel, AgentAction)
                for input_text_channel in input_text_channels
            ]
            + [
                (input_tick_channel, Tick),
            ],
            [(output_channel, AgentAction)],
            redis_url,
        )
        self.output_channel = output_channel
        self.query_interval = query_interval
        self.count_ticks = 0
        self.message_history: list[tuple[str, str]] = []
        self.name = agent_name
        self.model_name = model_name
        self.goal = goal

    async def send(self, message: AgentAction) -> None:
        if message.action_type == "speak":
            await self.r.publish(
                self.output_channel,
                Message[AgentAction](data=message).model_dump_json(),
            )

    async def aact(self, message: AgentAction) -> AgentAction:
        match message:
            case Tick():
                self.count_ticks += 1
                if self.count_ticks % self.query_interval == 0:
                    argument = await ainput("Argument: ")

                    return AgentAction(agent_name=self.name, action_type="speak", argument=argument)
  
            case AgentAction(
                agent_name=agent_name, action_type=action_type, argument=text
            ):
                if action_type == "speak":
                    self.message_history.append((agent_name, text))
                return AgentAction(
                    agent_name=self.name, action_type="none", argument=""
                )
            case _:
                raise ValueError(f"Unexpected message type: {type(message)}")
            

@NodeFactory.register("scenario_context")
class ScenarioContext(Node[DataModel, Text]):
    def __init__(
        self, 
        input_tick_channel: str, 
        output_channels: list[str], 
        env_scenario: str,
        redis_url: str = "redis://localhost:6379/0"
    ):
        super().__init__(
            input_channel_types = [(input_tick_channel, Tick)],
            output_channel_types=[(output_channel, Text) for output_channel in output_channels],
            redis_url=redis_url,
        )
        self.env_scenario = env_scenario
        self.output_channels = output_channels

    async def send_env_scenario(self) -> None:
        # Send env_scenario to all output channels
        for output_channel in self.output_channels:
            await self.r.publish(
                output_channel, Message[Text](data=Text(text=self.env_scenario)).model_dump_json()
            )

    async def event_loop(self) -> None:
        # Directly call send_env_scenario without any interval
        await self.send_env_scenario()

    async def __aenter__(self) -> Self:
        return await super().__aenter__()

    async def event_handler(
        self, _: str, __: Message[Zero]
    ) -> AsyncIterator[tuple[str, Message[Tick]]]:
        raise NotImplementedError("ScenarioContext does not have an event handler.")
        yield "", Message[Text](data=Text(text==self.env_scenario))
        
@NodeFactory.register("chat_print")     
class ChatPrint(PrintNode):
    async def write_to_screen(self) -> None:
        while self.output:
            data_entry = await self.write_queue.get()
            
            # Parse the JSON data
            data = json.loads(data_entry.model_dump_json())
            
            # Extract agent_name and argument
            agent_name = data['data']['agent_name']
            argument = data['data']['argument']
            
            # Format the output
            output = f"{agent_name} says: {argument}"
            
            # Write to output
            await self.output.write(output + "\n")
            await self.output.flush()
            self.write_queue.task_done()

    
