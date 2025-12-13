import json

from aact import NodeFactory
from .base_agent import BaseAgent
from .logs import EpisodeLog
from .datamodels import AgentAction, Observation
from sotopia.database.persistent_profile import AgentProfile

from typing import Generic, TypeVar, Type, Any
from pydantic import BaseModel, Field

from sotopia.database import GoalDimension
from sotopia.generation_utils.generate import agenerate
from sotopia.generation_utils.output_parsers import PydanticOutputParser


default_reward_prompt = """
{history}
Based on previous interactions, evaluate how well each of the participants achieve their goals.
Here are the list of agents:
{agent_list}
Please output your evaluation following the format:
{format_instructions}
"""

T_eval_dim = TypeVar("T_eval_dim", bound=BaseModel)


class EvaluationForMutiAgents(BaseModel, Generic[T_eval_dim]):
    agents_evaluation: dict[str, T_eval_dim] = Field(
        description="the evaluation for each agent, the key is the agent name,be sure to include every agent in the agent list, the value should follow the evaluation dimension format"
    )


@NodeFactory.register("evaluator")
class Evaluator(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        node_name: str,
        model_name: str,
        input_channels: list[str],
        output_channels: list[str],
        redis_url: str,
        reward_prompt: str = default_reward_prompt,
        eval_dim_class: str = "GoalDimension",
        temperature: float = 0.0,
    ):
        super().__init__(
            input_channel_types=[
                (input_channel, Observation) for input_channel in input_channels
            ],
            output_channel_types=[
                (output_channel, AgentAction) for output_channel in output_channels
            ],
            node_name=node_name,
            redis_url=redis_url,
        )
        self.output_channels = output_channels
        self.model_name = model_name
        self.reward_prompt = reward_prompt
        self.temperature = temperature
        if eval_dim_class == "GoalDimension":
            self.response_format_class: Type[BaseModel] = EvaluationForMutiAgents[
                GoalDimension
            ]
        else:
            raise ValueError(
                f"the eval_dim_class : {eval_dim_class} is not implemented"
            )
        # TODO: need a registry for the evaluation dimension class, so dimension can be initialized with a str

    async def aact(self, content: Observation) -> AgentAction:
        epilog = EpisodeLog(**json.loads(content.last_turn))

        result = await self.aevaluate(epilog)
        return AgentAction(
            agent_name="evaluator",
            output_channel=f"evaluator:{content.agent_name}",
            action_type="speak",
            argument=json.dumps(
                {"reward": json.dumps(result), "reward_prompt": self.reward_prompt}
            ),
        )

    async def aevaluate(self, episode: EpisodeLog) -> Any:
        # TODO: below is a temporary implementation, need to replaced by using render_for_humans in EpisodeLog
        history = "\n".join(
            f"{msg[0][0]} said: {msg[0][2]}" for msg in episode.messages
        )
        agent_list = []
        for pk in episode.agents:
            agent = AgentProfile.get(pk)
            name = agent.first_name + " " + agent.last_name
            name = name.strip()
            agent_list.append(name)

        res: BaseModel = await agenerate(
            model_name=self.model_name,
            template=self.reward_prompt,
            input_values=dict(history=history, agent_list=str(agent_list)),
            output_parser=PydanticOutputParser[self.response_format_class](  # type: ignore[name-defined]
                pydantic_object=self.response_format_class
            ),
        )
        return res.model_dump()["agents_evaluation"]
