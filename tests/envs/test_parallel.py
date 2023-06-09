from sotopia.agents import Agents, LLMAgent
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator
from sotopia.messages import AgentAction, Observation, ScriptBackground
from sotopia.samplers import UniformSampler


def test_parallel_sotopia_env() -> None:
    env = ParallelSotopiaEnv(model_name="gpt-3.5-turbo")
    env.reset()
    max_steps = 5
    while env.agents:
        max_steps -= 1
        actions = {
            agent: env.action_space(agent).sample() for agent in env.agents
        }  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = env.step(
            actions
        )
        if not max_steps:
            break
