"""Launcher for the Duskmire Werewolves social game scenario."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, cast

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
ROLE_ACTIONS_PATH = BASE_DIR / "role_actions.json"
RULEBOOK_PATH = BASE_DIR / "game_rules.json"
ROSTER_PATH = BASE_DIR / "roster.json"

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

COMMON_GUIDANCE = (
    "During your turn you must respond. If 'action' is available, use commands like 'kill NAME', "
    "'inspect NAME', 'save NAME', 'poison NAME', or 'vote NAME'. "
    "Day discussion is public. Voting requires an 'action' beginning with 'vote'."
)


def load_json(path: Path) -> Dict[str, Any]:
    return cast(Dict[str, Any], json.loads(path.read_text()))


def ensure_agent(player: Dict[str, Any]) -> AgentProfile:
    try:
        profile = AgentProfile.find(
            AgentProfile.first_name == player["first_name"],
            AgentProfile.last_name == player["last_name"],
        ).all()[0]
        return profile  # type: ignore[return-value]
    except IndexError:
        profile = AgentProfile(
            first_name=player["first_name"],
            last_name=player["last_name"],
            age=player.get("age", 30),
            occupation="",
            gender="",
            gender_pronoun=player.get("pronouns", "they/them"),
            public_info="",
            personality_and_values="",
            decision_making_style="",
            secret=player.get("secret", ""),
        )
        profile.save()
        return profile


def build_agent_goal(player: Dict[str, Any], role_name: str, role_prompt: str) -> str:
    # Build role description based on actual role
    if role_name == "Villager":
        role_desc = f"You are {player['first_name']} {player['last_name']}, a Villager."
    else:
        role_desc = f"You are {player['first_name']} {player['last_name']}. Your true role is {role_name}. Other players see you as a villager."

    return (
        f"{role_desc}"
        f"{player['goal']}"
        f"Role guidance: {role_prompt}\n"
        f"System constraints: {COMMON_GUIDANCE}"
    )


def prepare_scenario() -> tuple[EnvironmentProfile, List[AgentProfile], Dict[str, str]]:
    role_actions = load_json(ROLE_ACTIONS_PATH)
    roster = load_json(ROSTER_PATH)

    agents: List[AgentProfile] = []
    agent_goals: List[str] = []
    role_assignments: Dict[str, str] = {}

    for player in roster["players"]:
        profile = ensure_agent(player)
        agents.append(profile)
        full_name = f"{player['first_name']} {player['last_name']}"
        role = player["role"]
        role_prompt = role_actions["roles"][role]["goal_prompt"]
        agent_goals.append(build_agent_goal(player, role, role_prompt))
        role_assignments[full_name] = role

    scenario_text = roster["scenario"]

    env_profile = EnvironmentProfile(
        scenario=scenario_text,
        agent_goals=agent_goals,
        relationship=RelationshipType.acquaintance,
        game_metadata={
            "mode": "social_game",
            "rulebook_path": str(RULEBOOK_PATH),
            "actions_path": str(ROLE_ACTIONS_PATH),
            "role_assignments": role_assignments,
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
        rulebook_path=str(RULEBOOK_PATH),
        actions_path=str(ROLE_ACTIONS_PATH),
        role_assignments=role_assignments,
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
) -> List[LLMAgent]:
    agents: List[LLMAgent] = []
    for profile, model_name, goal in zip(
        agent_profiles,
        model_names,
        env_profile.agent_goals,
        strict=True,
    ):
        agent = LLMAgent(agent_profile=profile, model_name=model_name)
        agent.goal = goal
        agents.append(agent)
    return agents


def summarize_phase_log(phase_log: List[Dict[str, Any]]) -> None:
    if not phase_log:
        print("\nNo structured events recorded.")
        return

    print("\nTimeline by Phase")
    print("=" * 60)

    last_label: str | None = None
    for entry in phase_log:
        phase_name = entry["phase"]
        meta = entry.get("meta", {})
        group = meta.get("group")
        cycle = meta.get("group_cycle")
        stage = meta.get("group_stage")
        title = phase_name.replace("_", " ").title()
        if group:
            group_label = group.replace("_", " ").title()
            if cycle and stage:
                label = f"{group_label} {cycle}.{stage} – {title}"
            elif cycle:
                label = f"{group_label} {cycle} – {title}"
            else:
                label = f"{group_label}: {title}"
        else:
            label = title

        if label != last_label:
            print(f"\n[{label}]")
            last_label = label
            instructions = entry.get("instructions", [])
            for info_line in instructions:
                print(f"  Info: {info_line}")
            role_instr = entry.get("role_instructions", {})
            for role, lines in role_instr.items():
                for line in lines:
                    print(f"  Role {role}: {line}")

        for msg in entry.get("public", []):
            print(f"  Public: {msg}")
        for team, messages in entry.get("team", {}).items():
            for msg in messages:
                print(f"  Team ({team}) private: {msg}")
        for agent, messages in entry.get("private", {}).items():
            for msg in messages:
                print(f"  Private to {agent}: {msg}")
        for actor, action in entry.get("actions", {}).items():
            print(
                f"  Action logged: {actor} -> {action['action_type']} {action['argument']}"
            )


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
    agents = create_agents(agent_profiles, env_profile, agent_model_list)

    print("🌕 Duskmire Werewolves — Structured Social Game")
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

    summarize_phase_log(env.phase_log)

    if env._winner_payload:  # noqa: SLF001 (internal inspection for demo)
        print("\nGame Result:")
        print(f"Winner: {env._winner_payload['winner']}")
        print(f"Reason: {env._winner_payload['message']}")


if __name__ == "__main__":
    asyncio.run(main())
