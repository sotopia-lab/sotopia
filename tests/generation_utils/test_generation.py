from sotopia.envs.utils import RuleBasedResponse
from sotopia.generation_utils import (
    Script,
    generate_episode,
    produce_environment_response,
)
from sotopia.messages import ScriptEnvironmentResponse


def test_generate_episode() -> None:
    """
    Test that the scenario generator works
    """
    scenario = generate_episode("gpt-3.5-turbo")
    assert isinstance(scenario, Script)


def test_produce_environment_response() -> None:
    """
    Test that the environment response generator works
    """
    stop_criteria = RuleBasedResponse()
    response = produce_environment_response(
        "gpt-3.5-turbo", stop_criteria, turn_number=0, message_box=[]
    )
    assert isinstance(response, ScriptEnvironmentResponse)
    assert response.conversation_too_long is False
    assert response.p1_leaving is False
    assert response.p2_leaving is False
    assert response.stale_too_long is False
