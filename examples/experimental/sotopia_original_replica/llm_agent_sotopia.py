import logging
import sys
import json
from rich.logging import RichHandler

from aact import NodeFactory

from sotopia.experimental.agents.base_agent import BaseAgent
from sotopia.experimental.agents.datamodels import Observation, AgentAction
from sotopia.database.persistent_profile import AgentProfile
from typing import Any

from sotopia.generation_utils import agenerate, StrOutputParser

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


@NodeFactory.register("llm_agent")
class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        input_channels: list[str],
        output_channel: str,
        query_interval: int,
        node_name: str,
        model_name: str,
        goal: str,
        agent_name: str = "",
        background: dict[str, Any] | None = None,
        agent_pk: str | None = None,
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(
            [(input_channel, Observation) for input_channel in input_channels],
            [(output_channel, AgentAction)],
            redis_url,
            node_name,
        )
        self.output_channel = output_channel
        self.query_interval = query_interval
        self.count_ticks: int = 0
        self.message_history: list[Observation] = []
        self.goal: str = goal
        self.model_name: str = model_name
        self.agent_profile_pk: str | None = agent_pk
        self.name: str = agent_name
        self.background: dict[str, Any] | None = background
        self.awake: bool = False

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
            profile.save()
        else:
            profile = AgentProfile.get(pk=self.agent_profile_pk)

        self.agent_profile_pk = profile.pk
        self.name = " ".join([profile.first_name, profile.last_name]).strip()
        self.background = profile.model_dump()

    def _format_message_history(self, message_history: list[Observation]) -> str:
        ## TODO: akhatua Fix the mapping of action to be gramatically correct
        return "\n".join(message.to_natural_language() for message in message_history)

    async def aact(self, obs: Observation) -> AgentAction:
        if obs.turn_number == -1:
            if self.awake:
                return AgentAction(
                    agent_name=self.name,
                    output_channel=self.output_channel,
                    action_type="none",
                    argument="",
                )
            args = json.loads(obs.last_turn)
            self.set_profile(args["use_pk_value"])
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

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(
                agent_name=self.name,
                output_channel=self.output_channel,
                action_type="none",
                argument="",
            )
        elif len(obs.available_actions) == 1 and "leave" in obs.available_actions:
            self.shutdown_event.set()
            return AgentAction(
                agent_name=self.name,
                output_channel=self.output_channel,
                action_type="leave",
                argument="",
            )
        else:
            history = self._format_message_history(self.message_history)
            action: str = await agenerate(
                model_name=self.model_name,
                template="Imagine that you are a friend of the other persons. Here is the "
                "conversation between you and them.\n"
                "You are {agent_name} in the conversation.\n"
                "{message_history}\n"
                "and you plan to {goal}.\n"
                "You can choose to interrupt the other person "
                "by saying something or not to interrupt by outputting notiong. What would you say? "
                "Please only output a sentence or not outputting anything."
                "{format_instructions}",
                input_values={
                    "message_history": history,
                    "goal": self.goal,
                    "agent_name": self.name,
                },
                temperature=0.7,
                output_parser=StrOutputParser(),
            )

            return AgentAction(
                agent_name=self.name,
                output_channel=self.output_channel,
                action_type="speak",
                argument=action,
            )
