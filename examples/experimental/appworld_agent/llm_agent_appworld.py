from argparse import Action
import logging
import sys
import json
from rich.logging import RichHandler
from enum import Enum
from pydantic import Field
from typing import Any

from aact import NodeFactory
from aact.messages.registry import DataModelFactory
from aact.messages import DataModel

from sotopia.experimental.agents.base_agent import BaseAgent
from sotopia.messages.message_classes import Observation
from sotopia.database.persistent_profile import AgentProfile

from sotopia.generation_utils import agenerate, StrOutputParser


# Configure logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.WARNING,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)


### Define all the space of possible actions an agent can do
class ActionType(Enum):
    NONE = "none"
    SPEAK = "speak"
    NON_VERBAL = "non-verbal"
    CHOOSE_TOOL = "choose_tool"
    USE_TOOL = "use_tool"
    LEAVE = "leave"

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
    # fmt: off
    agent_name: str = Field(description="the name of the agent")
    output_channel: str = Field(description="the name of the output channel")
    action_type: ActionType = Field(description="whether to speak at this turn or choose to not do anything")
    argument: str = Field(description="the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action")
    # fmt: on

    def to_natural_language(self) -> str:
        action_descriptions = {
            ActionType.NONE: "did nothing",
            ActionType.SPEAK: f'said: "{self.argument}"',
            ActionType.CHOOSE_TOOL: f"[{self.action_type.value}] {self.argument}",
            ActionType.USE_TOOL: f"[{self.action_type.value}] {self.argument}",
            ActionType.NON_VERBAL: f"[{self.action_type.value}] {self.argument}",
            ActionType.LEAVE: "left the conversation",
        }
        return action_descriptions.get(self.action_type, "performed an unknown action")


### Define how the agent responds given an action
@NodeFactory.register("llm_agent")
class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        input_channels: list[str],
        output_channel: str,
        node_name: str,
        model_name: str,
        goal: str,
        agent_name: str = "",
        background: dict[str, Any] | None = None,
        agent_pk: str = "",
        redis_url: str = "redis://localhost:6379/0",
    ):
        """Initialize an LLMAgent that can use real-world tools via AppWorld"""
        super().__init__(
            input_channel_types=[(channel, Observation) for channel in input_channels],
            output_channel_types=[(output_channel, AgentAction)],
            redis_url=redis_url,
            node_name=node_name,
        )

        self.output_channel = output_channel
        self.count_ticks: int = 0
        self.message_history: list[Observation] = []
        self.goal: str = goal
        self.model_name: str = model_name
        self.agent_profile_pk: str | None = agent_pk
        self.name: str = agent_name
        self.background: dict[str, Any] | None = background
        self.awake: bool = False

    async def aact(self, obs: Observation) -> AgentAction:
        """Main action function for an agent"""
        # Do nothing if it's not yet your turn.
        if obs.turn_number == -1:
            if self.awake:
                return AgentAction(
                    agent_name=self.name,
                    output_channel=self.output_channel,
                    action_type="none",
                    argument="",
                )

            args = json.loads(obs.last_turn)
            self.set_profile(use_pk_value=args["use_pk_value"])
            self.awake = True
            return AgentAction(
                agent_name=self.name,
                output_channel=self.output_channel,
                action_type="none",
                argument=json.dumps(
                    {"pk": self.agent_profile_pk, "model_name": self.model_name}
                ),
            )
        self.message_history.append(obs)

        # Handle logic for special-case actions, such as when performing no action or when leaving the conversation.
        if self._is_current_action(ActionType.NONE.value, obs):
            return AgentAction(
                agent_name=self.name,
                output_channel=self.output_channel,
                action_type=ActionType.NONE.value,
                argument="",
            )
        elif self._is_current_action(ActionType.LEAVE.value, obs):
            self.shutdown_event.set()
            return AgentAction(
                agent_name=self.name,
                output_channel=self.output_channel,
                action_type=ActionType.LEAVE.value,
                argument="",
            )
        else:
            # Handle logic for other actions.
            history: str = self._format_message_history(self.message_history)
            template = self.get_action_template([action for action in ActionType])
            agent_action: str = await agenerate(
                model_name=self.model_name,
                template=template,
                input_values={
                    "message_history": history,
                    "goal": self.goal,
                    "agent_name": self.name,
                },
                temperature=0.7,
                output_parser=StrOutputParser(),
            )
            self.message_history.append(obs)
            if self._is_current_action(ActionType.SPEAK.value, obs):
                return AgentAction(
                    agent_name=self.name,
                    action_type=ActionType.SPEAK.value,
                    argument=agent_action,
                    path="",
                )
            elif self._is_current_action(ActionType.NON_VERBAL.value, obs):
                return AgentAction(
                    agent_name=self.name,
                    action_type=ActionType.NON_VERBAL.value,
                    argument=agent_action,
                    path="",
                )
            elif self._is_current_action(ActionType.CHOOSE_TOOL.value, obs):
                return AgentAction(
                    agent_name=self.name,
                    action_type=ActionType.CHOOSE_TOOL.value,
                    argument=agent_action,
                    path="",
                )
            elif self._is_current_action(ActionType.USE_TOOL.value, obs):
                return AgentAction(
                    agent_name=self.name,
                    action_type=ActionType.USE_TOOL.value,
                    argument=agent_action,
                    path="",
                )
            else:
                error_msg = f"Unknown action space: {obs}"
                logging.error(error_msg)
                raise ValueError(error_msg)

    def set_profile(self, use_pk_value: bool) -> None:
        if not use_pk_value:
            assert (
                self.background is not None and self.name is not None
            ), "Background and name must be provided"
            if " " in self.name:
                first_name, last_name = self.name.split(" ", 1)
            else:
                first_name = self.name
                last_name = ""
            profile = AgentProfile(
                first_name=first_name, last_name=last_name, **self.background
            )
        else:
            assert not self.agent_profile_pk == "", "Agent profile pk must be provided"
            profile = AgentProfile.get(pk=self.agent_profile_pk)

        self.agent_profile_pk = profile.pk
        self.name = " ".join([profile.first_name, profile.last_name]).strip()
        self.background = profile.model_dump()

    def _is_current_action(self, action: str, obs: Observation):
        return (
            True
            if len(obs.available_actions) == 1 and action in obs.available_actions
            else False
        )

    def get_action_template(self, selected_actions: list[ActionType]) -> str:
        """Utility function to create an action template based on available actions to pass to an LLM

        For our case, we will prompt the LLM to return a structured output (naively, without using any structured-calling techniques),
        then we will parse this output and use that output to determine what action to perform.
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
                ActionType.NONE
            ): """`none` - you can choose not to take an action if you are waiting for some data""",
            str(
                ActionType.NON_VERBAL
            ): """`non-verbal` - you can choose to do a non verbal action
                * `content` - the non veral action you want to send to other agents. eg: smile, shrug, thumbs up""",
            str(
                ActionType.LEAVE
            ): """`leave` - if your goals have been completed or abandoned, and you're absolutely certain that you've completed your task and have tested your work, use the leave action to stop working.""",
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
                You must prioritize actions that move you closer to your goal.
                Communicate briefly when necessary and focus on executing tasks
                effectively. Always consider the next actionable step to avoid
                unnecessary delays.  Again, you must reply with JSON, and only
                with JSON.
            """
        )
