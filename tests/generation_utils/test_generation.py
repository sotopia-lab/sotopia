from sotopia.generation_utils import generate_scenario


def test_generate_scenario() -> None:
    """
    Test that the scenario generator works
    """
    scenario = generate_scenario("gpt-3.5-turbo")
    assert isinstance(scenario, str)
    assert len(scenario) > 0
