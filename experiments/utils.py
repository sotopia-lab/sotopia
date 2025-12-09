import json
import os
import importlib.util
import sys
from typing import Any, cast


def get_game_module(game_name: str) -> Any:
    """
    Dynamically load the main module for a given game.
    Assumes standard path: examples/experimental/games/{game_name}/main.py
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    module_path = os.path.join(
        project_root, f"examples/experimental/games/{game_name}/main.py"
    )

    if not os.path.exists(module_path):
        raise ValueError(f"Game module not found at {module_path}")

    spec = importlib.util.spec_from_file_location(f"{game_name}_main", module_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{game_name}_main"] = module
        spec.loader.exec_module(module)
        return module
    else:
        raise ImportError(f"Could not load module from {module_path}")


def load_roster_template(game_name: str) -> dict[str, Any]:
    """Load the base roster.json for a game."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    roster_path = os.path.join(
        project_root, f"examples/experimental/games/{game_name}/roster.json"
    )

    if not os.path.exists(roster_path):
        raise FileNotFoundError(f"Roster file not found at {roster_path}")

    with open(roster_path, "r") as f:
        return cast(dict[str, Any], json.load(f))


def load_game_config(game_name: str) -> dict[str, Any]:
    """Load the base config.json for a game."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(
        project_root, f"examples/experimental/games/{game_name}/config.json"
    )

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return cast(dict[str, Any], json.load(f))
    return {}


def generate_roster(
    game_name: str,
    model_a: str,
    model_b: str,
    output_dir: str,
    pair_id: str,
    swap: bool = False,
) -> str:
    """
    Generate a roster.json for a specific matchup.

    Args:
        game_name: Name of the game (e.g., 'werewolves')
        model_a: Name of Model A
        model_b: Name of Model B
        output_dir: Directory to save the generated roster
        pair_id: Identifier for the pair (e.g., 'match0')
        swap: If True, swap the assignment (Model B takes Team 1/Slot 1, Model A takes Team 2/Slot 2)

    Returns:
        Path to the generated roster file.
    """
    # Load base config (rules, states, etc.)
    base_config = load_game_config(game_name)

    # Load roster template (agents list)
    base_roster = load_roster_template(game_name)
    agents = base_roster.get("agents", [])

    if not agents:
        raise ValueError("Roster must contain 'agents' list")

    # Determine assignment logic
    # Check if teams are present
    teams = {a.get("team") for a in agents if a.get("team")}
    # Filter out None if present
    unique_teams = sorted([t for t in teams if t])

    m1 = model_b if swap else model_a
    m2 = model_a if swap else model_b

    if len(unique_teams) == 2:
        # Asymmetric / Team-based game (Werewolf, Spyfall)
        team_1 = unique_teams[0]  # e.g. "Civilians" or "Non-Spy" (alphabetical)
        team_2 = unique_teams[1]  # e.g. "Undercover" or "Spy"

        for agent in agents:
            team = agent.get("team")
            if team == team_1:
                agent["agent_model"] = m1
            elif team == team_2:
                agent["agent_model"] = m2
            else:
                # Fallback or neutral role?
                agent["agent_model"] = m1

    else:
        # Symmetric / No-team game (PD, RPS)
        for i, agent in enumerate(agents):
            if i % 2 == 0:
                agent["agent_model"] = m1
            else:
                agent["agent_model"] = m2

    # Merge: Update agents in base_config
    base_config["agents"] = agents

    # If base_config was empty (no config.json), fall back to just roster structure
    final_output = base_config if base_config else base_roster

    # Save
    os.makedirs(output_dir, exist_ok=True)
    # Sanitize model names for filename
    sanitized_m1 = m1.replace("/", "_").replace("@", "_").split("v1")[0][-10:]
    sanitized_m2 = m2.replace("/", "_").replace("@", "_").split("v1")[0][-10:]

    filename = f"roster_{game_name}_{pair_id}_{'swapped' if swap else 'normal'}_{sanitized_m1}_vs_{sanitized_m2}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)

    return output_path
