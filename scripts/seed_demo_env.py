"""
Seed a minimal set of Sotopia demo data so the chat server can start episodes.

This utility creates:
  * Two demo agent profiles (left / right participants)
  * One demo environment profile with simple goals
  * A corresponding EnvAgentComboStorage entry that links the environment with
    the two agents

Once this script is run you can execute
    python sotopia-chat/chat_server.py add-env-agent-combo-to-redis-queue
and start the FastAPI server. The waiting-room flow will then have at least one
agent/environment pairing to dequeue.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sotopia.database import EnvAgentComboStorage
    from sotopia.database.persistent_profile import (
        AgentProfile,
        EnvironmentProfile,
    )  # noqa:E402
except ModuleNotFoundError as exc:  # pragma: no cover - helpful message for setup
    raise SystemExit(
        "Missing dependencies; ensure you run from the repository root with\n"
        "the virtual environment active and `redis` installed "
        "(e.g. `pip install redis redis-om`)."
    ) from exc

REDIS_URL = os.environ.get("REDIS_OM_URL", "redis://localhost:6379")


def ensure_agents() -> Sequence[str]:
    """Create two demo agent profiles if they do not already exist."""
    agent_names = [
        ("Demo", "Left"),
        ("Demo", "Right"),
    ]

    agent_ids: list[str] = []

    for first_name, last_name in agent_names:
        existing = list(
            AgentProfile.find(
                (AgentProfile.first_name == first_name)
                & (AgentProfile.last_name == last_name)
            ).all()
        )
        if existing:
            agent_ids.append(existing[0].pk)  # type: ignore[arg-type]
            continue

        profile = AgentProfile(
            first_name=first_name,
            last_name=last_name,
            public_info="Demo agent used for local testing.",
            personality_and_values="Friendly and cooperative.",
            decision_making_style="Collaborative",
        )
        profile.save()
        if not profile.pk:
            raise RuntimeError("Failed to create AgentProfile")
        agent_ids.append(profile.pk)

    return agent_ids


def ensure_environment() -> str:
    existing = list(
        EnvironmentProfile.find(EnvironmentProfile.codename == "demo_environment").all()
    )
    if existing:
        return existing[0].pk  # type: ignore[arg-type]

    env = EnvironmentProfile(
        codename="demo_environment",
        scenario="Two participants meet in a virtual village to discuss the day.",
        agent_goals=[
            "Establish rapport and learn one fact about your partner.",
            "Share your plans for the evening village gathering.",
        ],
        relationship=2,
    )
    env.save()
    if not env.pk:
        raise RuntimeError("Failed to create EnvironmentProfile")
    return env.pk


def ensure_combo(env_id: str, agent_ids: Sequence[str]) -> str:
    existing = list(
        EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env_id).all()
    )
    if existing:
        combo = existing[0]
        if combo.agent_ids != list(agent_ids):
            combo.agent_ids = list(agent_ids)
            combo.save()
        return combo.pk  # type: ignore[return-value]

    combo = EnvAgentComboStorage(
        env_id=env_id,
        agent_ids=list(agent_ids),
    )
    combo.save()
    if not combo.pk:
        raise RuntimeError("Failed to create EnvAgentComboStorage")
    return combo.pk


def main() -> int:
    print(f"Using Redis at {REDIS_URL}")
    agent_ids = ensure_agents()
    env_id = ensure_environment()
    combo_id = ensure_combo(env_id, agent_ids)
    print("Demo data seeded successfully:")
    print(f"  Agents: {agent_ids}")
    print(f"  Environment: {env_id}")
    print(f"  EnvAgentComboStorage: {combo_id}")
    print(
        "\nRun `python sotopia-chat/chat_server.py add-env-agent-combo-to-redis-queue`\n"
        "to push these combos onto the chat server queues."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
