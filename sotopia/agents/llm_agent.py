from sotopia.agents import BaseAgent
from sotopia.generation_utils.generate import LLM_Name, generate_action
from sotopia.messages import AgentAction, Message, Observation


class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self, agent_name: str, model_name: LLM_Name = "gpt-3.5-turbo"
    ) -> None:
        super().__init__(agent_name=agent_name)
        self.model_name = model_name
        self.inbox: list[tuple[str, Message]] = []

    def reset(self) -> None:
        self.inbox = []

    def recv_message(self, source: str, message: Message) -> None:
        self.inbox.append((source, message))

    def act(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="")
        else:
            action = generate_action(
                self.model_name,
                history="\n".join(
                    f"{x}: {y.to_natural_language()}" for x, y in self.inbox
                ),
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
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
