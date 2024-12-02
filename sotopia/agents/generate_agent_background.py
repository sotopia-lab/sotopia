import json
from typing import Callable

from sotopia.messages import Message, ScriptBackground


def generate_background_conversation(
    seed: dict[str, str],
    basic_info: dict[str, str],
    initial_profile: str,
    profile: str,
    background_json_file: str,
    run_sync_server: Callable[..., list[tuple[str, str, Message]]],
) -> tuple[list[tuple[str, str, Message]], ScriptBackground]:
    scenario, _topic, role, q_goal, a_goal = (
        seed["scenario"],
        seed["topic"],
        seed["role"],
        seed["q_goal"],
        seed["a_goal"],
    )
    background = ScriptBackground(
        scenario=scenario,
        p1_name=role,
        p2_name=basic_info["name"],
        p1_background=role,
        p2_background=initial_profile + "\n" + profile,
        p1_goal=q_goal,
        p2_goal=a_goal,
    )
    with open(background_json_file, "w") as f:
        background_dict = json.loads(background.json())
        json.dump(background_dict, f, indent=4)

    model_names: dict[str, str] = {
        "env": "gpt-4o-mini",
        "agent2": "gpt-4o-mini",
        "agent1": "gpt-4",
    }

    agents_info: dict[str, dict[str, str]] = {
        "env": {"mode": "all"},
        basic_info["name"]: {
            "mode": "speak",
            "goal": background.p2_goal,
        },
        role: {
            "mode": "speak",
            "goal": background.p1_goal,
        },
    }

    messages = run_sync_server(
        model_name_dict=model_names,
        agents_info=agents_info,
        action_order="round-robin",
        full_background_file=background_json_file,
        mode="speak",
    )
    return messages, background
