from sotopia.generation_utils import (
    Script,
    generate_episode,
    generate_episode_single_round,
)


def test_generate_episode() -> None:
    """
    Test that the scenario generator works
    """
    scenario = generate_episode("gpt-3.5-turbo")
    assert isinstance(scenario, str)
    assert len(scenario) > 0


def test_generate_episode_single_round() -> None:
    """
    Test that the scenario generator works
    """
    scenario = generate_episode_single_round("gpt-3.5-turbo")
    assert isinstance(scenario, Script)
