from rich import print

from sotopia.agents.llm_agent import Agents, LLMAgent
from sotopia.envs import ParallelSotopiaEnv
from sotopia.generation_utils.generate import process_history

env = ParallelSotopiaEnv(model_name="gpt-4")
obs = env.reset()
for agent_name in env.agents:
    print(f"Here's {agent_name}'s initial observation\n{obs[agent_name]}")
agents = Agents()
for agent_name in env.agents:
    agents[agent_name] = LLMAgent(agent_name)

agents.reset()
done = False
while not done:
    actions = agents.act(obs)
    print(actions)
    obs, _, terminated, ___, ____ = env.step(actions)
    done = all(terminated.values())
