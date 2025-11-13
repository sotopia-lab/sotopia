"""Launcher for the Duskmire Werewolves social game scenario."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import logging
from typing import Any, Dict, List, cast, Tuple

import redis

from sotopia.agents import LLMAgent
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipType,
)
from sotopia.envs import SocialGameEnv
from sotopia.envs.evaluators import (
    EpisodeLLMEvaluator,
    EvaluationForAgents,
    RuleBasedTerminatedEvaluator,
)
from sotopia.server import arun_one_episode
from sotopia.database import SotopiaDimensions

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

# Configure debug file logging for generation traces
LOG_FILE = BASE_DIR / "werewolves_game_debug.log"
# _fh is the file handler, which is used to log the >=DEBUG levels to a .log file.
_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_fh.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-7s %(name)s - %(message)s")
)
_gen_logger = logging.getLogger("sotopia.generation")
_gen_logger.setLevel(logging.DEBUG)
_gen_logger.addHandler(_fh)


def load_json(path: Path) -> Dict[str, Any]:
    return cast(Dict[str, Any], json.loads(path.read_text()))


def split_name(full_name: str) -> Tuple[str, str]:
    first_name, last_name = (
        full_name.split(" ", 1) if " " in full_name else (full_name, "")
    )
    return first_name, last_name


# for god roles (seer, witch), their roles can be revealed to other players


def ensure_agent(player: Dict[str, Any]) -> AgentProfile:
    try:
        existing = AgentProfile.find(
            (AgentProfile.first_name == player["first_name"])
            & (AgentProfile.last_name == player["last_name"])  # combine predicates
        ).all()
    except Exception:
        existing = []
    if existing:
        prof = AgentProfile.get(existing[0].pk)
        return prof

    profile = AgentProfile(
        first_name=player["first_name"],
        last_name=player["last_name"],
        age=player.get("age", ""),
        occupation=player.get("occupation", ""),
        gender=player.get("gender", ""),
        gender_pronoun=player.get("pronouns", ""),
        public_info=player.get("public_info", ""),
        personality_and_values=player.get("personality_and_values", ""),
        decision_making_style=player.get("decision_making_style", ""),
        secret=player.get("secret", ""),
    )
    profile.save()
    return profile


def prepare_scenario() -> tuple[EnvironmentProfile, List[AgentProfile], Dict[str, str]]:
    assert CONFIG_PATH.exists(), f"config.json not found at {CONFIG_PATH}"
    cfg = load_json(CONFIG_PATH)
    agents: List[AgentProfile] = []
    role_assignments: Dict[str, str] = {}

    for entry in cfg.get("agents", []):
        full_name = cast(str, entry.get("name", "Unknown Name"))
        role = cast(str, entry.get("role", "Unknown Role"))
        first_name, last_name = split_name(full_name)
        role_goal = cfg.get("role_goals", {}).get(role, "")
        role_secret = cfg.get("role_secrets", {}).get(role, "")
        # Build a complete player payload for profile creation/update
        player: Dict[str, Any] = {
            "first_name": first_name,
            "last_name": last_name,
            "pronouns": entry.get("pronouns", ""),
            "age": entry.get("age", ""),
            "gender": entry.get("gender", ""),
            "occupation": entry.get("occupation", ""),
            "public_info": entry.get("public_info", ""),
            "personality_and_values": entry.get("personality_and_values", ""),
            "decision_making_style": entry.get("decision_making_style", ""),
            "goal": role_goal,
            "secret": role_secret,
        }
        profile = ensure_agent(player)
        agents.append(profile)
        role_assignments[full_name] = role

    scenario_text = cast(
        str, cfg.get("description") or cfg.get("scenario") or "Werewolves game"
    )
    env_profile = EnvironmentProfile(
        scenario=scenario_text,
        relationship=RelationshipType.acquaintance,
        game_metadata={
            "mode": "social_game",
            "config_path": str(CONFIG_PATH),
        },
        tag="werewolves",
    )
    env_profile.save()
    return env_profile, agents, role_assignments


def build_environment(
    env_profile: EnvironmentProfile,
    role_assignments: Dict[str, str],
    model_name: str,
) -> SocialGameEnv:
    return SocialGameEnv(
        env_profile=env_profile,
        config_path=str(CONFIG_PATH),
        model_name=model_name,
        action_order="round-robin",
        evaluators=[RuleBasedTerminatedEvaluator(max_turn_number=40, max_stale_turn=2)],
        terminal_evaluators=[
            EpisodeLLMEvaluator(
                model_name,
                EvaluationForAgents[SotopiaDimensions],
            )
        ],
    )


def create_agents(
    agent_profiles: List[AgentProfile],
    env_profile: EnvironmentProfile,
    model_names: List[str],
    default_model: str,
) -> List[LLMAgent]:
    cfg = load_json(CONFIG_PATH)
    cfg_agents = cfg.get("agents", [])
    agents: List[LLMAgent] = []
    for idx, profile in enumerate(agent_profiles):
        # priority: explicit model_names list > per-agent config override > default_model
        if idx < len(model_names) and model_names[idx]:
            model_name = model_names[idx]
        elif idx < len(cfg_agents) and cfg_agents[idx].get("model"):
            model_name = cast(str, cfg_agents[idx]["model"])
        else:
            model_name = default_model
        agent = LLMAgent(agent_profile=profile, model_name=model_name)
        agent.goal = env_profile.agent_goals[idx]
        agents.append(agent)
    return agents


def print_roster(role_assignments: Dict[str, str]) -> None:
    print("Participants & roles:")
    for name, role in role_assignments.items():
        print(f" - {name}: {role}")


async def main() -> None:
    env_profile, agent_profiles, role_assignments = prepare_scenario()
    env_model = "gpt-5"
    agent_model_list = [
        "gpt-5",
        "gpt-5",
        "gpt-5",
        "gpt-5",
        "gpt-5",
        "gpt-5",
    ]

    env = build_environment(env_profile, role_assignments, env_model)
    agents = create_agents(agent_profiles, env_profile, agent_model_list, env_model)

    print("🌕 Duskmire Werewolves")
    print("=" * 60)
    print_roster(role_assignments)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag=None,
        push_to_db=False,
    )


if __name__ == "__main__":
    asyncio.run(main())
