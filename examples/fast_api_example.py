# Example curl command to call the simulate endpoint:
import requests
import time

BASE_URL = "http://localhost:8080"


def _create_mock_agent_profile() -> None:
    agent1_data = {
        "first_name": "John",
        "last_name": "Doe",
        "occupation": "test_occupation",
        "gender": "test_gender",
        "pk": "tmppk_agent1",
        "tag": "test_tag",
    }
    response = requests.post(
        f"{BASE_URL}/agents/",
        headers={"Content-Type": "application/json"},
        json=agent1_data,
    )
    assert response.status_code == 200

    agent2_data = {
        "first_name": "Jane",
        "last_name": "Doe",
        "occupation": "test_occupation",
        "gender": "test_gender",
        "pk": "tmppk_agent2",
        "tag": "test_tag",
    }
    response = requests.post(
        f"{BASE_URL}/agents/",
        headers={"Content-Type": "application/json"},
        json=agent2_data,
    )
    assert response.status_code == 200


def _create_mock_env_profile() -> None:
    env_data = {
        "codename": "test_codename",
        "scenario": "A",
        "agent_goals": [
            "B",
            "C",
        ],
        "pk": "tmppk_env_profile",
        "tag": "test_tag",
    }
    response = requests.post(
        f"{BASE_URL}/scenarios/",
        headers={"Content-Type": "application/json"},
        json=env_data,
    )
    assert response.status_code == 200


_create_mock_agent_profile()
_create_mock_env_profile()


data = {
    "env_id": "tmppk_env_profile",
    "agent_ids": ["tmppk_agent1", "tmppk_agent2"],
    "models": ["custom/structured-llama3.2:1b@http://localhost:8000/v1"] * 3,
    "max_turns": 10,
    "tag": "test_tag",
}
try:
    response = requests.post(
        f"{BASE_URL}/simulate/", headers={"Content-Type": "application/json"}, json=data
    )
    print(response)
    assert response.status_code == 202
    assert isinstance(response.content.decode(), str)
    episode_pk = response.content.decode()
    print(episode_pk)
    max_retries = 200
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(f"{BASE_URL}/simulation_status/{episode_pk}")
            assert response.status_code == 200
            status = response.content.decode()
            print(status)
            if status == "Error":
                raise Exception("Error running simulation")
            elif status == "Completed":
                break
            # Status is "Started", keep polling
            time.sleep(1)
            retry_count += 1
        except Exception as e:
            print(f"Error checking simulation status: {e}")
            time.sleep(1)
            retry_count += 1
    else:
        raise TimeoutError("Simulation timed out after 10 retries")

finally:
    try:
        response = requests.delete(f"{BASE_URL}/agents/tmppk_agent1")
        assert response.status_code == 200
    except Exception as e:
        print(e)
    try:
        response = requests.delete(f"{BASE_URL}/agents/tmppk_agent2")
        assert response.status_code == 200
    except Exception as e:
        print(e)
    try:
        response = requests.delete(f"{BASE_URL}/scenarios/tmppk_env_profile")
        assert response.status_code == 200
    except Exception as e:
        print(e)

    try:
        response = requests.delete(f"{BASE_URL}/episodes/{episode_pk}")
        assert response.status_code == 200
    except Exception as e:
        print(e)
