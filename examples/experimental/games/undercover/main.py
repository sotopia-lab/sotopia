"""Launcher for the Undercover social game scenario."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import redis
from rich.logging import RichHandler

from sotopia.agents import LLMAgent
from sotopia.agents.llm_agent import Agents
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipType,
)
from sotopia.envs import SocialDeductionGame
from sotopia.envs.evaluators import SocialGameEndEvaluator
from sotopia.envs.social_game import (
    SOCIAL_GAME_PROMPT_TEMPLATE,
    ActionHandler,
    load_config,
)
from sotopia.messages import AgentAction, Message, Observation, SimpleMessage
from sotopia.server import arun_one_episode

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

# Loggers (configured in main if running standalone)
_gen_logger = logging.getLogger("sotopia.generation")
_env_logger = logging.getLogger("sotopia.envs.social_game")


# ============================================================================
# Undercover game-end evaluator
# ============================================================================


class UndercoverGameEndEvaluator(SocialGameEndEvaluator):
    """Evaluator that checks undercover win conditions."""

    def _check_win_conditions(  # type: ignore[override]
        self, env: Any, turn_number: int, messages: List[tuple[str, Message]]
    ) -> tuple[bool, str, Dict[str, float]]:
        """Check if game has ended based on undercover win conditions."""

        # Count current alive
        civilians_count = 0
        undercover_count = 0

        for agent_name, alive in env.agent_alive.items():
            if alive:
                role = env.agent_to_role.get(agent_name, "")
                if role == "Undercover":
                    undercover_count += 1
                else:
                    civilians_count += 1

        # 1. Civilians win if all Undercovers are eliminated
        if undercover_count == 0:
            msg = "[Game] All Undercovers have been eliminated! Civilians win."
            env.recv_message("Environment", SimpleMessage(message=msg))

            rewards = {}
            for agent_name in env.agents:
                role = env.agent_to_role.get(agent_name, "")
                team_name = env.role_to_team.get(role, "")
                if team_name == "Civilians":
                    rewards[agent_name] = 1.0
                else:
                    rewards[agent_name] = -1.0
            return True, msg, rewards

        # 2. Undercovers win if all Civilians are eliminated
        if civilians_count == 0:
            msg = "[Game] All Civilians have been eliminated! Undercovers win."
            env.recv_message("Environment", SimpleMessage(message=msg))

            rewards = {}
            for agent_name in env.agents:
                role = env.agent_to_role.get(agent_name, "")
                team_name = env.role_to_team.get(role, "")
                if team_name == "Undercover":
                    rewards[agent_name] = 1.0
                else:
                    rewards[agent_name] = -1.0
            return True, msg, rewards

        # 3. Undercovers win if exactly 1 Undercover and 1 Civilian remain
        if undercover_count == 1 and civilians_count == 1:
            msg = "[Game] Only 1 Civilian and 1 Undercover remain! Undercover wins."
            env.recv_message("Environment", SimpleMessage(message=msg))

            rewards = {}
            for agent_name in env.agents:
                role = env.agent_to_role.get(agent_name, "")
                team_name = env.role_to_team.get(role, "")
                if team_name == "Undercover":
                    rewards[agent_name] = 1.0
                else:
                    rewards[agent_name] = -1.0
            return True, msg, rewards

        # Other cases (e.g. 2v2, 2v1, 1v2): Game continues
        # Note: 2v2 continues because undercovers don't know each other.

        return False, "", {}

    def __call__(
        self, turn_number: int, messages: List[tuple[str, Message]], **kwargs: Any
    ) -> List[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # Check turn limit
        if turn_number >= self.max_turn_number:
            return [("environment", (("terminated", True), "Max turns reached"))]

        env = kwargs.get("env")
        if not env:
            return [("environment", (("terminated", False), ""))]

        terminated, reason, rewards = self._check_win_conditions(
            env, turn_number, messages
        )

        response: List[tuple[str, tuple[tuple[str, int | float | bool], str]]] = [
            ("environment", (("terminated", terminated), reason))
        ]

        if terminated and rewards:
            agent_names = list(env.agents)
            for agent_name, reward in rewards.items():
                try:
                    idx = agent_names.index(agent_name)
                    generic_key = f"agent_{idx+1}"
                    response.append((generic_key, (("complete_rating", reward), "")))
                except ValueError:
                    continue

        return response


# ============================================================================
# Undercover-specific game logic
# ============================================================================


class UndercoverActionHandler(ActionHandler):
    """Handles actions for the Undercover game."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        """Handle a single action from an agent based on current state."""

        if env.current_state == "Round_vote":
            # Collect votes for elimination
            if "votes" not in env.internal_state:
                env.internal_state["votes"] = {}

            if action.action_type == "action" and "vote" in action.argument.lower():
                # Parse target from "vote Aurora" or "I vote for Aurora"
                words = action.argument.split()
                # Try to find a name (capitalized word)
                target = next(
                    (w for w in words if w[0].isupper() and w in env.agents), None
                )
                if target:
                    env.internal_state["votes"][agent_name] = target

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        """Get specific action instructions for an agent based on current state."""

        if env.current_state == "Round_vote":
            return "It is voting time. You MUST use the command 'vote NAME' to vote for the player you suspect is the Undercover. e.g. 'vote Alice'"

        return ""


class UndercoverEnv(SocialDeductionGame):
    """Undercover game with description and voting."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=UndercoverActionHandler(), **kwargs)

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, str] | None = None,
        agents: Agents | None = None,
        omniscient: bool = False,
        lite: bool = False,
        include_background_observations: bool = True,
    ) -> Dict[str, Observation]:
        return super().reset(
            seed=seed,
            options=options,
            agents=agents,
            omniscient=omniscient,
            lite=lite,
            include_background_observations=include_background_observations,
        )

    def _check_eliminations(self) -> None:
        """Apply eliminations based on collected actions."""
        # Only apply eliminations if we are about to transition state
        if not self._should_transition_state():
            return

        if self.current_state == "Round_vote":
            # Tally votes and eliminate most voted player
            votes = self.internal_state.get("votes", {})
            if votes:
                vote_counts: Dict[str, int] = {}
                for target in votes.values():
                    vote_counts[target] = vote_counts.get(target, 0) + 1

                if vote_counts:
                    # Find player with most votes
                    eliminated = max(vote_counts, key=vote_counts.get)  # type: ignore
                    self.agent_alive[eliminated] = False
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {eliminated} was voted out! They were a {self.agent_to_role[eliminated]}."
                        ),
                    )
                    _gen_logger.info(
                        f"{eliminated} was voted out! They were a {self.agent_to_role[eliminated]}."
                    )
                # Clear votes
                self.internal_state["votes"] = {}


# ============================================================================
# Setup helpers
# ============================================================================


def ensure_agent_profile(config: Dict[str, Any]) -> AgentProfile:
    """Create or retrieve agent profile."""
    name = config.get("name", "")
    role = config.get("role", "")

    first_name, _, last_name = name.partition(" ")
    if not last_name:
        last_name = ""

    # Try to find existing
    try:
        existing = AgentProfile.find(
            (AgentProfile.first_name == first_name)
            & (AgentProfile.last_name == last_name)
        ).all()
        if existing:
            return AgentProfile.get(existing[0].pk)
    except Exception:
        pass

    # Create new
    role_secret = config.get("role_secrets", {}).get(role, "")
    profile = AgentProfile(
        first_name=first_name,
        last_name=last_name,
        secret=role_secret,
    )
    profile.save()
    return profile


def create_environment(
    env_profile: EnvironmentProfile, model_name: str, config: Dict[str, Any]
) -> UndercoverEnv:
    """Create undercover game environment."""
    return UndercoverEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        action_order="round-robin",
        evaluators=[UndercoverGameEndEvaluator(max_turn_number=100)],
        terminal_evaluators=[],
        hide_unknown=True,
    )


def create_agents(
    agent_profiles: List[AgentProfile],
    env_profile: EnvironmentProfile,
    model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any],
) -> List[LLMAgent]:
    """Create LLM agents."""
    agents = []
    for idx, profile in enumerate(agent_profiles):
        # Calculate secrets
        agent_name = f"{profile.first_name}{' ' + profile.last_name if profile.last_name else ''}"
        role_goal = env_profile.agent_goals[idx]

        # Get secret based on role
        role = config.get("agents", [])[idx].get("role", "")
        # role_secrets is dictionary in config
        secrets = config.get("role_secrets", {}).get(role, "")

        # Fill template
        # SOCIAL_GAME_PROMPT_TEMPLATE comes with {secret} placeholder.
        filled_template = (
            SOCIAL_GAME_PROMPT_TEMPLATE.replace("{description}", env_profile.scenario)
            .replace("{secret}", f"Your secret info: {secrets}")
            .replace(
                "{goal}",
                role_goal,
            )
        )

        # Determine model
        if isinstance(model_name, dict):
            this_agent_model = model_name.get(
                agent_name, model_name.get("default", "gpt-4")
            )
        elif isinstance(model_name, list):
            this_agent_model = model_name[idx]
        else:
            this_agent_model = model_name

        agent = LLMAgent(
            agent_name=agent_name,
            agent_profile=profile,
            model_name=this_agent_model,
            strict_action_constraint=True,
            custom_template=filled_template,
        )
        agent.goal = env_profile.agent_goals[idx]
        agents.append(agent)
    return agents


def prepare_scenario(
    env_model_name: str,
    agent_model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any] | None = None,
) -> tuple[SocialDeductionGame, List[LLMAgent]]:
    """Load config and create profiles."""
    if config is None:
        config = load_config(CONFIG_PATH)

    # Create agent profiles
    agent_profiles = []
    agent_goals = []
    for entry in config.get("agents", []):
        profile = ensure_agent_profile(entry)
        agent_profiles.append(profile)

        role_goal = config.get("role_goals", {}).get(entry.get("role", ""), "")
        agent_goals.append(role_goal)

    # Create environment profile
    agent_names = [entry.get("name", "") for entry in config.get("agents", [])]
    scenario = config.get("description", "Undercover game").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="undercover",
    )
    env_profile.save()

    env = create_environment(env_profile, env_model_name, config)
    agents = create_agents(agent_profiles, env_profile, agent_model_name, config)
    return env, agents


def print_roster(config: Dict[str, Any]) -> None:
    """Print game roster."""
    print("Participants & roles:")
    for entry in config.get("agents", []):
        name = entry.get("name", "Unknown")
        role = entry.get("role", "Unknown")
        print(f" - {name}: {role}")


# ============================================================================
# Main
# ============================================================================


def get_model_names(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract model names from config. Enforces strict requirement."""
    model_map = {}
    for entry in config.get("agents", []):
        name = entry.get("name")
        model = entry.get("agent_model")
        if not name:
            continue
        if not model:
            raise ValueError(
                f"Agent '{name}' missing 'agent_model' in config configuration."
            )
        model_map[name] = model
    return model_map


async def main() -> None:
    """Run undercover game."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Undercover game.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--roster",
        type=str,
        default=str(BASE_DIR / "roster.json"),
        help="Path to roster file",
    )
    args = parser.parse_args()

    config_path = args.config
    roster_path = args.roster

    config = load_config(config_path)
    roster = load_config(roster_path)

    # Merge roster into config
    config["agents"] = roster.get("agents", [])

    env_model_name = "gpt-4o"
    agent_model_name = get_model_names(config)

    # Setup
    env, agents = prepare_scenario(env_model_name, agent_model_name, config)

    # Display roster
    # Config already loaded above
    print("🕵️ Undercover")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    # Run game
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_undercover",
        push_to_db=True,
    )


if __name__ == "__main__":
    # Configure logging for standalone execution
    LOG_FILE = BASE_DIR / "undercover_game_debug.log"
    _fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    _fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s - %(message)s")
    )

    _gen_logger.setLevel(logging.DEBUG)
    _gen_logger.addHandler(_fh)

    _env_logger.setLevel(logging.INFO)
    _env_logger.addHandler(_fh)
    _env_logger.addHandler(RichHandler())

    asyncio.run(main())
