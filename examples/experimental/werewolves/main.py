"""Launcher for the Duskmire Werewolves social game scenario."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import logging
from typing import Any, Dict, List, cast

import redis

from sotopia.agents import LLMAgent
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipType,
)
from sotopia.envs import SocialDeductionGame, ActionHandler
from sotopia.envs.evaluators import SocialGameEndEvaluator
from sotopia.server import arun_one_episode
from sotopia.messages import AgentAction, SimpleMessage, Message

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

# Configure debug file logging
LOG_FILE = BASE_DIR / "werewolves_game_debug.log"
_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_fh.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-7s %(name)s - %(message)s")
)
_gen_logger = logging.getLogger("sotopia.generation")
_gen_logger.setLevel(logging.DEBUG)
_gen_logger.addHandler(_fh)


# ============================================================================
# Werewolf game-end evaluator
# ============================================================================


class WerewolfGameEndEvaluator(SocialGameEndEvaluator):
    """Evaluator that checks werewolf win conditions."""

    def _check_win_conditions(
        self, env: Any, turn_number: int, messages: List[tuple[str, Message]]
    ) -> tuple[bool, str]:
        """Check if game has ended based on werewolf win conditions."""
        # Count alive players by team
        team_counts: Dict[str, int] = {}
        for agent_name, alive in env.agent_alive.items():
            if alive:
                role = env.agent_to_role.get(agent_name, "")
                team = env.role_to_team.get(role, "")
                team_counts[team] = team_counts.get(team, 0) + 1

        # Check end conditions from config
        end_conditions = env._config.get("end_conditions", [])
        for condition in end_conditions:
            cond_type = condition.get("type")

            if cond_type == "team_eliminated":
                team = condition.get("team", "")
                if team_counts.get(team, 0) == 0:
                    winner = condition.get("winner", "")
                    msg = condition.get("message", f"{winner} wins!")
                    env.recv_message("Environment", SimpleMessage(message=msg))
                    return True, msg

            elif cond_type == "parity":
                team1 = condition.get("team", "")
                team2 = condition.get("other", "")
                if team_counts.get(team1, 0) >= team_counts.get(team2, 0):
                    winner = condition.get("winner", "")
                    msg = condition.get("message", f"{winner} wins!")
                    env.recv_message("Environment", SimpleMessage(message=msg))
                    return True, msg

        return False, ""


# ============================================================================
# Werewolf-specific game logic
# ============================================================================


class WerewolfActionHandler(ActionHandler):
    """Handles actions for the Werewolf game."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        """Handle a single action from an agent based on current state."""

        if env.current_state == "Day_vote":
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

        elif env.current_state == "Night_werewolf":
            # Werewolves choose kill target
            role = env.agent_to_role.get(agent_name, "")
            if role == "Werewolf" and action.action_type == "action":
                if "kill" in action.argument.lower():
                    words = action.argument.split()
                    target = next(
                        (w for w in words if w[0].isupper() and w in env.agents),
                        None,
                    )
                    if target:
                        env.internal_state["kill_target"] = target

        elif env.current_state == "Night_seer":
            # Seer inspects someone
            role = env.agent_to_role.get(agent_name, "")
            if role == "Seer" and action.action_type == "action":
                if "inspect" in action.argument.lower():
                    words = action.argument.split()
                    target = next(
                        (w for w in words if w[0].isupper() and w in env.agents),
                        None,
                    )
                    if target:
                        # Reveal target's role to seer
                        target_role = env.agent_to_role.get(target, "Unknown")
                        target_team = env.role_to_team.get(target_role, "Unknown")
                        env.recv_message(
                            "Environment",
                            SimpleMessage(
                                message=f"[Private to {agent_name}] {target} is on team: {target_team}"
                            ),
                        )

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        """Get specific action instructions for an agent based on current state."""
        role = env.agent_to_role.get(agent_name, "")

        if env.current_state == "Day_vote":
            return "It is voting time. You MUST use the command 'vote NAME' to vote for a player to eliminate. e.g. 'vote Alice'"

        elif env.current_state == "Night_werewolf":
            if role == "Werewolf":
                return "It is Night. You are a Werewolf. You MUST use the command 'kill NAME' to choose a target. e.g. 'kill Bob'"
            else:
                return "It is Night. You are sleeping."

        elif env.current_state == "Night_seer":
            if role == "Seer":
                return "It is Night. You are the Seer. You MUST use the command 'inspect NAME' to check a player's team. e.g. 'inspect Charlie'"
            else:
                return "It is Night. You are sleeping."

        elif env.current_state == "Night_witch":
            if role == "Witch":
                return "It is Night. You are the Witch. You can use 'save NAME' or 'poison NAME'. If you don't want to use potions, you can 'action none'."
            else:
                return "It is Night. You are sleeping."

        return ""

    def enrich_backgrounds(self, env: SocialDeductionGame) -> None:
        """Enrich agent backgrounds with game-specific information."""
        # Find all werewolves
        werewolves = [
            name for name, role in env.agent_to_role.items() if role == "Werewolf"
        ]

        # Update backgrounds for werewolves
        for werewolf in werewolves:
            partners = [w for w in werewolves if w != werewolf]
            if partners:
                partner_str = ", ".join(partners)
                # Find index of this agent in env.agents
                try:
                    idx = env.agents.index(werewolf)
                    # Append to background
                    # Note: env.background.agent_backgrounds is a list of strings
                    current_bg = env.background.agent_backgrounds[idx]
                    env.background.agent_backgrounds[idx] = (
                        f"{current_bg} Your partner(s) are: {partner_str}."
                    )
                except ValueError:
                    continue


class WerewolfEnv(SocialDeductionGame):
    """Werewolf game with voting, kills, and special roles."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=WerewolfActionHandler(), **kwargs)

    def _check_eliminations(self) -> None:
        """Apply eliminations based on collected actions."""

        if self.current_state == "Day_vote":
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
                # Clear votes
                self.internal_state["votes"] = {}

        elif self.current_state == "Night_werewolf":
            # Apply werewolf kill
            target = self.internal_state.get("kill_target")
            if target and self.agent_alive.get(target, False):
                # Check if witch saves them (would be in Night_witch state)
                saved = self.internal_state.get("saved_target")
                if target != saved:
                    self.agent_alive[target] = False
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {target} was killed by werewolves!"
                        ),
                    )
                else:
                    self.recv_message(
                        "Environment",
                        SimpleMessage(message="[Game] An attack was prevented!"),
                    )
            # Clear kill target
            self.internal_state.pop("kill_target", None)


# ============================================================================
# Setup helpers
# ============================================================================


def load_config() -> Dict[str, Any]:
    """Load game configuration."""
    return cast(Dict[str, Any], json.loads(CONFIG_PATH.read_text()))


def ensure_agent_profile(name: str, role: str, config: Dict[str, Any]) -> AgentProfile:
    """Create or retrieve agent profile."""
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
        age=30,
        secret=role_secret,
    )
    profile.save()
    return profile


def create_environment(env_profile: EnvironmentProfile, model_name: str) -> WerewolfEnv:
    """Create werewolf game environment."""
    return WerewolfEnv(
        env_profile=env_profile,
        config_path=str(CONFIG_PATH),
        model_name=model_name,
        action_order="round-robin",
        evaluators=[WerewolfGameEndEvaluator(max_turn_number=40)],
        terminal_evaluators=[],
        hide_unknown=True,
    )


def create_agents(
    agent_profiles: List[AgentProfile],
    env_profile: EnvironmentProfile,
    model_name: str,
) -> List[LLMAgent]:
    """Create LLM agents."""
    agents = []
    for idx, profile in enumerate(agent_profiles):
        agent = LLMAgent(
            agent_profile=profile,
            model_name=model_name,
            strict_action_constraint=True,
        )
        agent.goal = env_profile.agent_goals[idx]
        agents.append(agent)
    return agents


def prepare_scenario(
    env_model_name: str, agent_model_name: str
) -> tuple[SocialDeductionGame, List[LLMAgent]]:
    """Load config and create profiles."""
    config = load_config()

    # Create agent profiles
    agent_profiles = []
    agent_goals = []
    for entry in config.get("agents", []):
        name = entry.get("name", "Unknown")
        role = entry.get("role", "Villager")

        profile = ensure_agent_profile(name, role, config)
        agent_profiles.append(profile)

        role_goal = config.get("role_goals", {}).get(role, "")
        agent_goals.append(role_goal)

    # Create environment profile
    scenario = config.get("description", "Werewolves game")
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="werewolves",
    )
    env_profile.save()

    env = create_environment(env_profile, env_model_name)
    agents = create_agents(agent_profiles, env_profile, agent_model_name)
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


async def main() -> None:
    """Run werewolf game."""
    # Configuration
    # env_model_name = "custom/google/gemma-3n-e4b@http://127.0.0.1:1234/v1"
    # agent_model_name = "custom/google/gemma-3n-e4b@http://127.0.0.1:1234/v1"
    env_model_name = "gpt-4o-mini"
    agent_model_name = "gpt-4o-mini"

    # Setup
    env, agents = prepare_scenario(env_model_name, agent_model_name)

    # Display roster
    config = load_config()
    print("🌕 Duskmire Werewolves")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    # Run game
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=True,  # Required for action_mask to work
        json_in_script=False,
        tag=None,
        push_to_db=False,
    )


if __name__ == "__main__":
    asyncio.run(main())
