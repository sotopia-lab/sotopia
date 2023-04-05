from sotopia.generation_utils import Script, generate_episode


def test_generate_episode() -> None:
    """
    Test that the scenario generator works
    """
    scenario = generate_episode("gpt-3.5-turbo")
    assert isinstance(scenario, Script)
