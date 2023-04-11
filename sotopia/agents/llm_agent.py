from sotopia.agents import BaseAgent
from sotopia.envs import Observation
from sotopia.generation_utils.generate import (
    AgentAction,
    LLM_Name,
    generate_action,
)


class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self, agent_name: str, model_name: LLM_Name = "gpt-3.5-turbo"
    ) -> None:
        super().__init__(agent_name=agent_name)
        self.model_name = model_name
        self.history: list[str] = []

    def reset(self) -> None:
        self.history = []

    def _prompt_generate(self) -> str:
        return " ".join(self.history)

    def act(self, obs: Observation) -> AgentAction:
        if obs["turn_number"] == 0:
            self.history.append(
                f"Here is the background for the conversation:\n{obs['history']}"
            )
            self.history.append("Conversation Start:\n")
        else:
            self.history.append(f"Turn #{obs['turn_number']-1}:")
            self.history.append(obs["history"])
        if (
            len(obs["available_actions"]) == 1
            and "none" in obs["available_actions"]
        ):
            return AgentAction(action_type="none", argument="")
        else:
            action = generate_action(
                self.model_name,
                history="\n".join(self.history),
                turn_number=obs["turn_number"],
                action_types=obs["available_actions"],
                agent=self.agent_name,
            )
            return action


class Agents(dict[str, LLMAgent]):
    def reset(self) -> None:
        for agent in self.values():
            agent.reset()

    def act(self, obs: dict[str, Observation]) -> dict[str, AgentAction]:
        return {
            agent_name: agent.act(obs[agent_name])
            for agent_name, agent in self.items()
        }
