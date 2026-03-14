"""Dedicated server for multiplayer Werewolf games with Redis state management (new SocialDeductionGame backend)."""

from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import typer

# Ensure repository root is on sys.path so we can import examples.*
REPO_ROOT = Path(__file__).resolve().parents[3]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from examples.experimental.werewolves.main import (
    create_environment,
    ensure_agent_profile,
    load_config,
)
from sotopia.agents import Agents, LLMAgent
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipType,
)
from sotopia.envs.social_game import SocialDeductionGame
from sotopia.messages import AgentAction, Observation

from .werewolf_state import WerewolfStateStore

# Config file lives under repo_root/examples/experimental/werewolves/config.json
CONFIG_PATH = (
    REPO_ROOT
    / "examples"
    / "experimental"
    / "werewolves"
    / "config.json"
)

app = typer.Typer()


class RedisHumanAgent:
    """Agent that reads actions from Redis (submitted via API)."""

    def __init__(
        self,
        session_id: str,
        participant_id: str,
        agent_name: str,
        state_store: WerewolfStateStore,
        agent_profile: AgentProfile,
    ):
        self.session_id = session_id
        self.participant_id = participant_id
        self.agent_name = agent_name
        self.goal = ""
        self.model_name = "human"
        self._state_store = state_store
        self.profile = agent_profile
        self.agent_profile = agent_profile

    async def aact(self, obs: Observation) -> AgentAction:
        """Poll Redis for human action."""
        available = list(getattr(obs, "available_actions", []))
        if available and all(action == "none" for action in available):
            return AgentAction(action_type="none", argument="")

        while True:
            payload = await self._state_store.pop_action(
                self.session_id,
                self.participant_id,
            )
            if payload:
                return AgentAction(
                    action_type=payload.get("action_type", "none"),
                    argument=payload.get("argument", ""),
                )
            await asyncio.sleep(0.5)

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """No-op for compatibility."""
        pass


def _build_agents_for_session(
    session_id: str,
    human_id: str,
    state_store: WerewolfStateStore,
    *,
    env_model_name: str = "gpt-4o-mini",
    agent_model_name: str = "gpt-4o-mini",
) -> tuple[SocialDeductionGame, Agents, Dict[str, str], str]:
    """Create env and agent roster using the SocialDeductionGame werewolf."""
    config = load_config(CONFIG_PATH)

    agent_profiles: List[AgentProfile] = []
    agent_goals: List[str] = []
    role_assignments: Dict[str, str] = {}

    for entry in config.get("agents", []):
        profile = ensure_agent_profile(entry)
        agent_profiles.append(profile)
        role = entry.get("role", "")
        goal = config.get("role_goals", {}).get(role, "")
        agent_goals.append(goal)
        full_name = entry.get("name", "")
        role_assignments[full_name] = role

    env_profile = EnvironmentProfile(
        scenario=config.get("description", ""),
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="werewolves",
    )
    env_profile.save()

    env = create_environment(
        env_profile=env_profile,
        model_name=env_model_name,
        config=config,
    )

    # Pick a random human slot from roster
    human_idx = random.randrange(len(agent_profiles))
    human_name = config.get("agents", [])[human_idx].get("name", f"Player {human_idx}")

    # Precompute werewolf names for secret prompts
    werewolf_role = "Werewolf"
    werewolf_names = [
        entry.get("name", "")
        for entry in config.get("agents", [])
        if entry.get("role") == werewolf_role
    ]

    def _build_llm_agent(name: str, profile: AgentProfile, goal: str) -> LLMAgent:
        secrets = ""
        if name in werewolf_names:
            partners = [n for n in werewolf_names if n != name]
            secrets = (
                f"Your secret: You are a werewolf. "
                + (f"Your partner(s) are: {', '.join(partners)}." if partners else "You have no partners.")
            )

        # Reuse the env's prompt template rendering if available; otherwise, fall back to goal+secret.
        try:
            custom_template = env.render_agent_prompt(name, goal, secrets)  # type: ignore[attr-defined]
        except Exception:
            custom_template = f"{goal}\n{secrets}"

        agent = LLMAgent(
            agent_name=name,
            agent_profile=profile,
            model_name=agent_model_name,
            strict_action_constraint=True,
            custom_template=custom_template,
        )
        agent.goal = goal
        return agent

    agents: Agents = Agents()
    for idx, profile in enumerate(agent_profiles):
        name = config.get("agents", [])[idx].get("name", profile.first_name)
        goal = agent_goals[idx]
        if idx == human_idx:
            agent = RedisHumanAgent(
                session_id=session_id,
                participant_id=human_id,
                agent_name=name,
                state_store=state_store,
                agent_profile=profile,
            )
            agent.goal = goal
        else:
            agent = _build_llm_agent(name, profile, goal)
        agents[name] = agent

    return env, agents, role_assignments, human_name


def _build_log(env: SocialDeductionGame) -> list[dict[str, Any]]:
    """Convert env inbox into a simple log payload."""
    log_entries: list[dict[str, Any]] = []
    phase = env.current_state
    public_msgs = [
        msg.to_natural_language() if hasattr(msg, "to_natural_language") else str(msg)
        for sender, msg in env.inbox
        if sender == "Environment"
    ]
    if public_msgs:
        log_entries.append({"phase": phase, "public": public_msgs, "turn": env.turn_number})
    return log_entries


async def publish_game_state(
    session_id: str,
    env: SocialDeductionGame,
    role_assignments: Dict[str, str],
    human_name: str,
    host_id: str,
    state_store: WerewolfStateStore,
    *,
    status: str = "active",
    available_actions: list[str] | None = None,
    active_player_id: str | None = None,
    waiting_for_action: bool = False,
    last_actions: dict[str, AgentAction] | None = None,
) -> None:
    """Publish current game state to Redis for frontend polling."""
    state_props = env._config.get("state_properties", {}).get(env.current_state, {})
    action_list = state_props.get("actions", [])
    allow_chat = "speak" in action_list
    allow_actions = "action" in action_list

    log_entries = _build_log(env)
    if last_actions:
        log_entries.append(
            {
                "phase": env.current_state,
                "turn": env.turn_number,
                "actions": {
                    name: {
                        "action_type": act.action_type,
                        "argument": act.argument,
                    }
                    for name, act in last_actions.items()
                },
            }
        )

    players_payload = []
    for name in env.agents:
        players_payload.append(
            {
                "id": name,
                "display_name": name,
                "role": role_assignments.get(name, "unknown")
                if name == human_name or not env.agent_alive.get(name, True)
                else "unknown",
                "team": env.get_agent_team(name),
                "is_alive": env.agent_alive.get(name, True),
                "is_host": name == human_name,
            }
        )

    human_role = env.agent_to_role.get(human_name, "unknown")
    human_is_wolf = human_role.lower() == "werewolf"

    pack_members: list[dict[str, Any]] = []
    if human_is_wolf:
        for name in env.agents:
            if env.agent_to_role.get(name, "").lower() == "werewolf":
                pack_members.append(
                    {
                        "id": name,
                        "display_name": name,
                        "is_alive": env.agent_alive.get(name, True),
                        "is_human": name == human_name,
                    }
                )

    witch_options = None
    if human_role.lower() == "witch":
        witch_options = {
            "can_save": bool(env.internal_state.get("witch_have_save", True)),
            "can_poison": bool(env.internal_state.get("witch_have_poison", True)),
            "pending_target": env.internal_state.get("kill_target"),
        }

    action_mask = getattr(env, "action_mask", [])
    if action_mask:
        try:
            active_idx = action_mask.index(True)
            active_player_id = env.agents[active_idx]
        except ValueError:
            pass

    payload = {
        "session_id": session_id,
        "players": players_payload,
        "me": next((p for p in players_payload if p["id"] == human_name), None),
        "host_id": host_id,
        "phase": {
            "phase": env.current_state,
            "description": state_props.get("description", ""),
            "allow_chat": allow_chat,
            "allow_actions": allow_actions,
        },
        "available_actions": available_actions or [],
        "active_player_id": active_player_id,
        "waiting_for_action": waiting_for_action,
        "last_updated": time.time(),
        "game_over": False,
        "winner": None,
        "winner_message": None,
        "status": status,
        "log": log_entries,
        "pack_members": pack_members,
        "team_chat": [],
        "witch_options": witch_options,
    }

    await state_store.write_state(session_id, payload, ttl=600)


async def async_run_werewolf_game(
    session_id: str,
    human_id: str,
    num_ai_players: int = 5,
) -> None:
    """Main game loop for werewolf session."""
    typer.echo(f"Starting Werewolf game {session_id} with human {human_id}")

    state_store = WerewolfStateStore(os.environ.get("REDIS_OM_URL", "redis://localhost:6379"))
    env: SocialDeductionGame | None = None
    role_assignments: Dict[str, str] = {}
    human_full_name = human_id

    try:
        env, agents, role_assignments, human_full_name = _build_agents_for_session(
            session_id,
            human_id,
            state_store,
        )

        observations = env.reset(agents=agents, omniscient=False)

        await publish_game_state(
            session_id,
            env,
            role_assignments,
            human_full_name,
            human_id,
            state_store,
            status="active",
            available_actions=observations[human_full_name].available_actions
            if human_full_name in observations
            else [],
            waiting_for_action=True,
        )

        terminated = {name: False for name in env.agents}
        while not all(terminated.values()):
            # Collect actions from all agents
            action_tasks = [
                agent.aact(observations[name]) for name, agent in agents.items()
            ]
            actions_list = await asyncio.gather(*action_tasks)
            actions = {name: act for name, act in zip(agents.keys(), actions_list)}

            observations, _, terminated, _, _ = await env.astep(actions)

            await publish_game_state(
                session_id,
                env,
                role_assignments,
                human_full_name,
                human_id,
                state_store,
                status="active" if not all(terminated.values()) else "completed",
                available_actions=observations[human_full_name].available_actions
                if human_full_name in observations
                else [],
                waiting_for_action=not all(terminated.values()),
                last_actions=actions,
            )

    except Exception as exc:
        if env:
            await publish_game_state(
                session_id,
                env,
                role_assignments,
                human_full_name,
                human_id,
                state_store,
                status="error",
                available_actions=[],
                active_player_id=None,
                waiting_for_action=False,
            )
        raise exc


@app.command()
def run_werewolf_game(
    session_id: str,
    human_id: str,
    num_ai_players: int = typer.Option(
        5, "--num-ai-players", help="Number of AI players (excluding human)"
    ),
) -> None:
    """CLI entry point for starting a werewolf game."""
    asyncio.run(async_run_werewolf_game(session_id, human_id, num_ai_players))
