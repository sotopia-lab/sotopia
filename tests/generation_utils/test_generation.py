from sotopia.generation_utils import Script, generate_episode


def test_generate_episode_interactive() -> None:
    """
    Test that the scenario generator works
    """
    scenario = generate_episode("gpt-3.5-turbo", "interactive")
    assert isinstance(scenario, str)
    assert len(scenario) > 0


def test_generate_episode_direct() -> None:
    """
    Test that the scenario generator works
    """
    scenario = generate_episode("gpt-3.5-turbo", "direct")
    assert isinstance(scenario, Script)
