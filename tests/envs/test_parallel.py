from sotopia.envs import ParallelSotopiaEnv
from sotopia.messages import ScriptBackground


def test_parallel_sotopia_env() -> None:
    env = ParallelSotopiaEnv()
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


def test_fillin_the_background() -> None:
    env = ParallelSotopiaEnv(model_name="gpt-4")
    env.reset(
        options={"partial_background_file": "tests/envs/test_background.json"}
    )
    background = env.background
    assert isinstance(background, ScriptBackground)
    assert background.scenario != "<missing_info>"
