from sotopia.agents.llm_agent import Agents, LLMAgent
from sotopia.envs import ParallelSotopiaEnv

env = ParallelSotopiaEnv()
obs = env.reset()
print(obs)
agents = Agents()
for agent_name in env.agent_names:
    agents[agent_name] = LLMAgent(agent_name)

agents.reset()
done = False

while not done:
    actions = agents.act(obs)
    print(actions)
    obs, _, terminated, ___, ____ = env.step(actions)
    done = all(terminated.values())
