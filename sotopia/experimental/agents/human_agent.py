from aact import Message, NodeFactory
from aact.messages import Tick
from sotopia.agents.llm_agent import ainput
from sotopia.experimental.agents import BaseAgent, AgentAction, ActionType

import logging
from rich.logging import RichHandler

# Configure logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.WARNING,  # Set to WARNING for production
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)


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
        """
        Initializes a HumanAgent instance.

        Args:
            input_text_channels (list[str]): List of input text channels.
            input_tick_channel (str): Input channel for ticks.
            output_channel (str): Output channel for publishing messages.
            query_interval (int): Interval for querying input.
            agent_name (str): Name of the agent.
            goal (str): Goal of the agent.
            model_name (str): Model name used by the agent.
            redis_url (str): URL for Redis connection.
        """
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
        """
        Sends a message to the output channel if the action type is 'speak'.

        Args:
            message (AgentAction): The message to be sent.
        """
        if message.action_type == ActionType.SPEAK:
            await self.r.publish(
                self.output_channel,
                Message[AgentAction](data=message).model_dump_json(),
            )

    async def aact(self, message: AgentAction) -> AgentAction:
        """
        Processes incoming messages and performs actions based on the message type.
        """

        match message:
            case Tick():
                self.count_ticks += 1
                if self.count_ticks % self.query_interval == 0:
                    argument = await ainput("Argument: ")
                    return AgentAction(
                        agent_name=self.name,
                        action_type=ActionType.SPEAK,
                        argument=argument,
                    )

            case AgentAction(
                agent_name=agent_name, action_type=action_type, argument=text
            ):
                if action_type == ActionType.SPEAK:
                    self.message_history.append((agent_name, text))
                return AgentAction(
                    agent_name=self.name, action_type=ActionType.NONE, argument=""
                )
            case _:
                raise ValueError(f"Unexpected message type: {type(message)}")
