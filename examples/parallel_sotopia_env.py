from sotopia.agents.llm_agent import Agents, LLMAgent
from sotopia.envs import ParallelSotopiaEnv
from sotopia.generation_utils.generate import process_history

env = ParallelSotopiaEnv()
obs = env.reset()
for agent_name in env.agent_names:
    print(f"Here's {agent_name}'s initial observation\n{obs[agent_name]}")
agents = Agents()
for agent_name in env.agent_names:
    agents[agent_name] = LLMAgent(agent_name)

agents.reset()
done = False
while not done:
    actions = agents.act(obs)
    print(process_history(actions))
    obs, _, terminated, ___, ____ = env.step(actions)
    done = all(terminated.values())
