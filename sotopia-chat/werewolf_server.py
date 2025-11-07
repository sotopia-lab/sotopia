"""Dedicated server for multiplayer Werewolf games with Redis state management."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import typer

if Path(__file__).resolve().parents[1].as_posix() not in sys.path:
    sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())

from examples.experimental.werewolves.main_human import (  # type: ignore[import]
    build_environment,
    create_agents,
    prepare_scenario,
)
from sotopia.agents import Agents, LLMAgent
from sotopia.database.persistent_profile import AgentProfile
from sotopia.database import SotopiaDimensions
from sotopia.envs import SocialGameEnv
from sotopia.messages import AgentAction, Observation

try:
    from .werewolf_state import WerewolfStateStore
except ImportError:  # pragma: no cover
    from werewolf_state import WerewolfStateStore  # type: ignore

BASE_DIR = Path(__file__).resolve().parents[1] / "examples" / "experimental" / "werewolves"
ROLE_ACTIONS_PATH = BASE_DIR / "role_actions.json"
RULEBOOK_PATH = BASE_DIR / "game_rules.json"
ROSTER_PATH = BASE_DIR / "roster.json"

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


async def publish_game_state(
    session_id: str,
    env: SocialGameEnv,
    role_assignments: Dict[str, str],
    human_name: str,
    host_id: str,
    state_store: WerewolfStateStore,
    *,
    status: str = "active",
    available_actions: list[str] | None = None,
    active_player_id: str | None = None,
    waiting_for_action: bool = False,
) -> None:
    """Publish current game state to Redis for frontend polling."""
    rulebook = env.game_rulebook
    if not rulebook:
        return

    players_payload = []
    for name, state in rulebook.agent_states.items():
        if not state.alive:
            revealed_role = role_assignments.get(name, "unknown")
        elif name == human_name:
            revealed_role = role_assignments.get(name, "unknown")
        else:
            revealed_role = "unknown"
        players_payload.append(
            {
                "id": name,
                "display_name": name,
                "role": revealed_role,
                "team": state.team,
                "is_alive": state.alive,
                "is_host": name == human_name,
            }
        )

    me_entry = next((player for player in players_payload if player["id"] == human_name), None)

    phase_def = rulebook.phase_lookup.get(rulebook.current_phase)
    human_role = role_assignments.get(human_name, "unknown")
    human_state = rulebook.agent_states.get(human_name)
    human_team = human_state.team if human_state else None

    pack_members: list[dict[str, Any]] = [
        {
            "id": name,
            "display_name": name,
            "is_alive": state.alive,
            "is_human": name == human_name,
        }
        for name, state in rulebook.agent_states.items()
        if state.role.lower() == "werewolf"
    ]

    pack_chat: list[dict[str, Any]] = []
    if human_state and human_state.role.lower() == "werewolf" and human_team:
        for entry in env.phase_log:
            team_msgs = entry.get("team", {}).get(human_team, [])
            for msg in team_msgs:
                pack_chat.append(
                    {
                        "phase": entry.get("phase"),
                        "message": msg,
                        "turn": entry.get("turn"),
                        "recorded_at": entry.get("recorded_at"),
                    }
                )

    witch_options = None
    if human_state and human_state.role.lower() == "witch":
        state_flags = getattr(env.game_rulebook, "state_flags", {})
        pending_target = state_flags.get("night_target_display") or state_flags.get("night_target")
        witch_options = {
            "can_save": bool(human_state.attributes.get("save_available", True)),
            "can_poison": bool(human_state.attributes.get("poison_available", True)),
            "pending_target": pending_target,
        }

    payload = {
        "session_id": session_id,
        "players": players_payload,
        "me": me_entry,
        "host_id": host_id,
        "phase": {
            "phase": rulebook.current_phase,
            "description": phase_def.description if phase_def else "",
            "allow_chat": True,
            "allow_actions": True,
        },
        "available_actions": available_actions or [],
        "active_player_id": active_player_id,
        "waiting_for_action": waiting_for_action,
        "last_updated": time.time(),
        "game_over": bool(env._winner_payload),
        "winner": env._winner_payload.get("winner") if env._winner_payload else None,
        "winner_message": env._winner_payload.get("message") if env._winner_payload else None,
        "status": status,
        "log": env.phase_log,
        "pack_members": pack_members,
        "team_chat": pack_chat,
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

    # Pick random slot for human player
    import random

    state_store = WerewolfStateStore(os.environ.get("REDIS_OM_URL", "redis://localhost:6379"))
    env: SocialGameEnv | None = None
    role_assignments: Dict[str, str] = {}
    human_full_name = human_id

    try:
        # Prepare scenario / environment using existing helpers
        env_profile, agent_profiles, role_assignments = prepare_scenario()
        player_names = list(role_assignments.keys())
        if not player_names:
            raise RuntimeError("Roster returned no players.")

        human_idx = random.randrange(len(player_names))
        human_full_name = player_names[human_idx]

        env = build_environment(
            env_profile=env_profile,
            role_assignments=role_assignments,
            model_name="gpt-4o-mini",
        )

        agent_objects = create_agents(
            agent_profiles,
            env_profile,
            model_names=["gpt-4o-mini"] * len(agent_profiles),
        )
        for idx, agent in enumerate(agent_objects):
            agent.agent_name = player_names[idx]

        # Replace chosen agent with RedisHumanAgent
        original_agent = agent_objects[human_idx]
        agent_objects[human_idx] = RedisHumanAgent(
            session_id=session_id,
            participant_id=human_id,
            agent_name=player_names[human_idx],
            state_store=state_store,
            agent_profile=original_agent.profile,
        )
        agent_objects[human_idx].goal = env_profile.agent_goals[human_idx]

        agent_mapping = Agents({agent.agent_name: agent for agent in agent_objects})
        environment_messages = env.reset(
            agents=agent_mapping,
            omniscient=False,
        )
        agent_mapping.reset()

        # Publish initial state after reset
        await publish_game_state(
            session_id,
            env,
            role_assignments,
            human_full_name,
            human_id,
            state_store,
            status="active",
            available_actions=[],
            active_player_id=None,
            waiting_for_action=False,
        )

        # Assign goals
        for index, agent_name in enumerate(env.agents):
            agent_mapping[agent_name].goal = env.profile.agent_goals[index]

        done = False
        turn_counter = 0
        while not done:
            active_agents = set(env.game_rulebook.active_agents_for_phase())
            agent_messages: dict[str, AgentAction] = {}
            for agent_name in env.agents:
                agent = agent_mapping[agent_name]
                observation = environment_messages[agent_name]
                available_actions = list(getattr(observation, "available_actions", []))
                meaningful_actions = [
                    action for action in available_actions if action != "none"
                ]

                if agent_name == human_full_name and agent_name in active_agents and not meaningful_actions:
                    agent_messages[agent_name] = AgentAction(action_type="none", argument="")
                    continue

                if agent_name == human_full_name and agent_name in active_agents:
                    await publish_game_state(
                        session_id,
                        env,
                        role_assignments,
                        human_full_name,
                        human_id,
                        state_store,
                        status="awaiting_human",
                        available_actions=meaningful_actions or available_actions,
                        active_player_id=agent_name,
                        waiting_for_action=True,
                    )

                action = await agent.aact(observation)

                if agent_name == human_full_name and agent_name in active_agents:
                    await publish_game_state(
                        session_id,
                        env,
                        role_assignments,
                        human_full_name,
                        human_id,
                        state_store,
                        status="active",
                        available_actions=[],
                        active_player_id=None,
                        waiting_for_action=False,
                    )

                agent_messages[agent_name] = action

            (
                environment_messages,
                _rewards,
                terminated,
                truncated,
                _info,
            ) = await env.astep(agent_messages)

            turn_counter += 1
            if any(terminated.values()) or any(truncated.values()):
                done = True

            await publish_game_state(
                session_id,
                env,
                role_assignments,
                human_full_name,
                human_id,
                state_store,
                status="active",
                available_actions=[],
                active_player_id=None,
                waiting_for_action=False,
            )

        await publish_game_state(
            session_id,
            env,
            role_assignments,
            human_full_name,
            human_id,
            state_store,
            status="completed",
        )
        typer.echo(f"Game {session_id} completed after {turn_counter} turns.")
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
            )
        typer.echo(f"Game {session_id} failed: {exc}", err=True)
        raise


@app.command()
def run_werewolf_game(
    session_id: str,
    human_id: str,
    num_ai_players: int = 5,
) -> None:
    """CLI entry point for starting a werewolf game."""
    asyncio.run(async_run_werewolf_game(session_id, human_id, num_ai_players))


if __name__ == "__main__":
    app()
