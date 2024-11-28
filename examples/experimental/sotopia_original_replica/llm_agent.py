import logging
import sys
from rich.logging import RichHandler


from aact import NodeFactory

from sotopia.experimental.agents.base_agent import BaseAgent
from sotopia.experimental.agents.datamodels import Observation, AgentAction

from sotopia.generation_utils import agenerate_action


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
        agent_name: str,
        goal: str,
        model_name: str,
        redis_url: str,
    ):
        super().__init__(
            [(input_channel, Observation) for input_channel in input_channels],
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

    def _format_message_history(
        self, message_history: list[tuple[str, str, str]]
    ) -> str:
        ## TODO: akhatua Fix the mapping of action to be gramatically correct
        return "\n".join(
            (f"{speaker} {action} {message}")
            for speaker, action, message in message_history
        )

    async def aact(self, obs: Observation) -> AgentAction:
        self.message_history.append(
            (obs.agent_name, self.name, obs.to_natural_language())
        )

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(
                output_channel=self.output_channel, action_type="none", argument=""
            )
        else:
            action = await agenerate_action(
                self.model_name,
                history=self._format_message_history(self.message_history),
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=self.name,
                goal=self.goal,
            )

            return AgentAction(
                output_channel=self.output_channel,
                action_type=action.action_type,
                argument=action.argument,
            )
