from argparse import Action
import logging
import sys
from enum import Enum
from rich.logging import RichHandler
from pydantic import Field

from typing import Optional

from aact import Message, NodeFactory
from aact.messages import Text, Tick, DataModel
from aact.messages.registry import DataModelFactory

from sotopia.experimental.agents.base_agent import BaseAgent

from sotopia.generation_utils import agenerate, StrOutputParser

import json

# Check Python version
if sys.version_info >= (3, 11):
    pass
else:
    pass

# Configure logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.WARNING,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)


class ActionType(Enum):
    NONE = "none"
    SPEAK = "speak"
    NON_VERBAL = "non-verbal"
    LEAVE = "leave"
    THOUGHT = "thought"
    CHOOSE_TOOL = "choose_tool_action"
    TOOL_ACTION = "tool_action"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ActionType):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        else:
            return NotImplemented


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
        action_descriptions = {
            ActionType.NONE: "did nothing",
            ActionType.SPEAK: f'said: "{self.argument}"',
            ActionType.THOUGHT: f'thought: "{self.argument}"',
            ActionType.CHOOSE_TOOL: f'chose tool: "{self.argument}"',
            ActionType.TOOL_ACTION: f'used tool: "{self.argument}"',
            ActionType.NON_VERBAL: f"[{self.action_type.value}] {self.argument}",
            ActionType.LEAVE: "left the conversation",
        }

        return action_descriptions.get(self.action_type, "performed an unknown action")


@NodeFactory.register("llm_agent")
class LLMAgent(BaseAgent[AgentAction | Tick | Text, AgentAction]):
    def __init__(
        self,
        input_text_channels: list[str],
        input_tick_channel: str,
        input_env_channels: list[str],
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
            + [(input_env_channel, Text) for input_env_channel in input_env_channels],
            [(output_channel, AgentAction)],
            redis_url,
        )
        self.output_channel = output_channel
        self.query_interval = query_interval
        self.count_ticks = 0
        self.message_history: list[tuple[str, str, str]] = []
        self.name = agent_name
        self.model_name = model_name
        self.goal = goal

    async def send(self, message: AgentAction) -> None:
        if message.action_type in ("speak", "thought"):
            await self.r.publish(
                self.output_channel,
                Message[AgentAction](data=message).model_dump_json(),
            )

        elif message.action_type in ("choose_tool_action", "tool_action"):
            await self.r.publish(
                "Agent:Runtime",
                Message[AgentAction](data=message).model_dump_json(),
            )

    def _format_message_history(
        self, message_history: list[tuple[str, str, str]]
    ) -> str:
        ## TODO: akhatua Fix the mapping of action to be gramatically correct
        return "\n".join(
            (f"{speaker} {action} {message}")
            for speaker, action, message in message_history
        )

    def get_action_template(self, selected_actions: list[ActionType]) -> str:
        """
        Returns the action template string with selected actions.

        Args:
            selected_actions (list[ActionType]): List of ActionType enum members to include in the template.

        Returns:
            str: The action template with the selected actions.
        """
        base_template = """ You are talking to another agent.
        You are {agent_name}.\n
        {message_history}\nand you plan to {goal}.
        ## Action
        What is your next thought or action? Your response must be in JSON format.

        It must be an object, and it must contain two fields:
        * `action`, which is one of the actions below
        * `args`, which is a map of key-value pairs, specifying the arguments for that action
        """

        action_descriptions = {
            str(
                ActionType.SPEAK
            ): """`speak` - you can talk to the other agents to share information or ask them something. Arguments:
                * `content` - the message to send to the other agents (should be short)""",
            str(
                ActionType.THOUGHT
            ): """`thought` - only use this rarely to make a plan, set a goal, record your thoughts. Arguments:
                * `content` - the message you send yourself to organize your thoughts (should be short). You cannot think more than 2 turns.""",
            str(
                ActionType.NONE
            ): """`none` - you can choose not to take an action if you are waiting for some data""",
            str(
                ActionType.NON_VERBAL
            ): """`non-verbal` - you can choose to do a non verbal action
                * `content` - the non veral action you want to send to other agents. eg: smile, shrug, thumbs up""",
            str(
                ActionType.LEAVE
            ): """`leave` - if your goals have been completed or abandoned, and you're absolutely certain that you've completed your task and have tested your work, use the leave action to stop working.""",
            # TODO: Add action template
            str(ActionType.CHOOSE_TOOL): """""",
            str(ActionType.TOOL_ACTION): """""",
        }

        selected_action_descriptions = "\n\n".join(
            f"[{i+1}] {action_descriptions[str(action)]}"
            for i, action in enumerate(selected_actions)
            if str(action) in action_descriptions
        )

        return (
            base_template
            + selected_action_descriptions
            + """
                You must prioritize actions that move you closer to your goal. Communicate briefly when necessary and focus on executing tasks effectively. Always consider the next actionable step to avoid unnecessary delays.
                Again, you must reply with JSON, and only with JSON.
            """
        )

    async def aact(self, message: AgentAction | Tick | Text) -> AgentAction:
        match message:
            case Text(text=text):
                self.message_history.append((self.name, "observation data", text))
                return AgentAction(
                    agent_name=self.name, action_type="none", argument="", path=""
                )
            case Tick():
                self.count_ticks += 1
                if self.count_ticks % self.query_interval == 0:
                    try:
                        template = self.get_action_template(
                            [action for action in ActionType]
                        )

                        agent_action = await agenerate(
                            model_name=self.model_name,
                            template=template,
                            input_values={
                                "message_history": self._format_message_history(
                                    self.message_history
                                ),
                                "goal": self.goal,
                                "agent_name": self.name,
                            },
                            temperature=0.7,
                            output_parser=StrOutputParser(),
                        )
                    except Exception as e:
                        print(f"Error during agenerate: {e}")

                    agent_action = (
                        agent_action.replace("```", "")
                        .replace("json", "")
                        .strip('"')
                        .strip()
                    )

                    try:
                        data = json.loads(agent_action)
                        action = data["action"]
                        if action == "thought":
                            content = data["args"]["content"]
                            self.message_history.append((self.name, action, content))
                            return AgentAction(
                                agent_name=self.name,
                                action_type="thought",
                                argument=content,
                                path="",
                            )

                        elif action == "speak":
                            content = data["args"]["content"]
                            self.message_history.append((self.name, action, content))
                            return AgentAction(
                                agent_name=self.name,
                                action_type=action,
                                argument=content,
                                path="",
                            )

                        elif action == "non-verbal":
                            content = data["args"]["content"]
                            self.message_history.append((self.name, action, content))
                            return AgentAction(
                                agent_name=self.name,
                                action_type=action,
                                argument=content,
                                path="",
                            )

                        elif action == "choose_tool_action":
                            return AgentAction(
                                agent_name=self.name,
                                action_type=action,
                                argument=content,
                                path="",
                            )

                        elif action == "tool_action":
                            return AgentAction(
                                agent_name=self.name,
                                action_type=action,
                                argument=content,
                                path="",
                            )

                        elif action == "none":
                            return AgentAction(
                                agent_name=self.name,
                                action_type="none",
                                argument="",
                                path="",
                            )
                        else:
                            print(f"Unknown action: {action}")
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                else:
                    return AgentAction(
                        agent_name=self.name, action_type="none", argument="", path=""
                    )
            case AgentAction(
                agent_name=agent_name, action_type=action_type, argument=text
            ):
                if action_type == "speak":
                    self.message_history.append((agent_name, str(action_type), text))
                return AgentAction(
                    agent_name=self.name, action_type="none", argument="", path=""
                )
        raise ValueError(f"Unexpected message type: {type(message)}")
