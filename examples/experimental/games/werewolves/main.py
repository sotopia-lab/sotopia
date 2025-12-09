"""Launcher for the Duskmire Werewolves social game scenario."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import logging
from typing import Any, Dict, List
import random
from collections import Counter

from rich.logging import RichHandler
import redis

from sotopia.agents import LLMAgent
from sotopia.agents.llm_agent import Agents
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipType,
)
from sotopia.envs import SocialDeductionGame
from sotopia.envs.social_game import (
    ActionHandler,
    load_config,
    SOCIAL_GAME_PROMPT_TEMPLATE,
)
from sotopia.envs.evaluators import SocialGameEndEvaluator
from sotopia.server import arun_one_episode
from sotopia.messages import AgentAction, SimpleMessage, Message, Observation

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

# Loggers (configured in main if running standalone)
_gen_logger = logging.getLogger("sotopia.generation")
_env_logger = logging.getLogger("sotopia.envs.social_game")


# ============================================================================
# Werewolf game-end evaluator
# ============================================================================


class WerewolfGameEndEvaluator(SocialGameEndEvaluator):
    """Evaluator that checks werewolf win conditions."""

    def _check_win_conditions(  # type: ignore[override]
        self, env: Any, turn_number: int, messages: List[tuple[str, Message]]
    ) -> tuple[bool, str, Dict[str, float]]:
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

                    # Calculate rewards
                    rewards = {}
                    for agent_name in env.agents:
                        role = env.agent_to_role.get(agent_name, "")
                        team = env.role_to_team.get(role, "")
                        if team == winner:
                            rewards[agent_name] = 1.0
                        else:
                            rewards[agent_name] = -1.0

                    return True, msg, rewards

            elif cond_type == "parity":
                team1 = condition.get("team", "")
                team2 = condition.get("other", "")
                if team_counts.get(team1, 0) >= team_counts.get(team2, 0):
                    winner = condition.get("winner", "")
                    msg = condition.get("message", f"{winner} wins!")
                    env.recv_message("Environment", SimpleMessage(message=msg))

                    # Calculate rewards
                    rewards = {}
                    for agent_name in env.agents:
                        role = env.agent_to_role.get(agent_name, "")
                        team = env.role_to_team.get(role, "")
                        if team == winner:
                            rewards[agent_name] = 1.0
                        else:
                            rewards[agent_name] = -1.0

                    return True, msg, rewards

        return False, "", {}

    def __call__(
        self, turn_number: int, messages: List[tuple[str, Message]], **kwargs: Any
    ) -> List[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # Check turn limit
        if turn_number >= self.max_turn_number:
            return [("environment", (("terminated", True), "Max turns reached"))]

        # Extract environment from kwargs
        env = kwargs.get("env")
        if not env:
            return [("environment", (("terminated", False), ""))]

        # Check game-specific win conditions
        terminated, reason, rewards = self._check_win_conditions(
            env, turn_number, messages
        )

        response: List[tuple[str, tuple[tuple[str, int | float | bool], str]]] = [
            ("environment", (("terminated", terminated), reason))
        ]

        if terminated and rewards:
            agent_names = list(env.agents)  # Ensure order matches env.agents
            for agent_name, reward in rewards.items():
                # Find index of agent in env.agents
                try:
                    idx = agent_names.index(agent_name)
                    generic_key = f"agent_{idx+1}"  # e.g. agent_1, agent_2
                    response.append((generic_key, (("complete_rating", reward), "")))
                except ValueError:
                    # Should not happen if agent_name is in env.agents
                    continue

        return response


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
                        if "kill_target_proposals" not in env.internal_state:
                            env.internal_state["kill_target_proposals"] = {}
                        env.internal_state["kill_target_proposals"][agent_name] = target
                        # Update the werewolf kill result
                        kill_votes = env.internal_state.get("kill_target_proposals", {})
                        if kill_votes:
                            # Count votes
                            vote_counts = Counter(kill_votes.values())
                            if vote_counts:
                                # Find max votes
                                max_votes = max(vote_counts.values())
                                # Get all targets with max votes
                                candidates = [
                                    t for t, c in vote_counts.items() if c == max_votes
                                ]
                                # Break tie randomly
                                env.internal_state["kill_target"] = random.choice(
                                    candidates
                                )

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
                            receivers=[agent_name],
                        )

        elif env.current_state == "Night_witch":
            # Witch uses potions
            role = env.agent_to_role.get(agent_name, "")
            if role == "Witch" and action.action_type == "action":
                if "save" in action.argument.lower():
                    env.internal_state["witch_have_save"] = False
                    words = action.argument.split()
                    target = next(
                        (w for w in words if w[0].isupper() and w in env.agents),
                        None,
                    )
                    if target:
                        env.internal_state["saved_target"] = target
                elif "poison" in action.argument.lower():
                    env.internal_state["witch_have_posion"] = False
                    words = action.argument.split()
                    target = next(
                        (w for w in words if w[0].isupper() and w in env.agents),
                        None,
                    )
                    if target:
                        env.internal_state["poison_target"] = target

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
                if env.internal_state.get(
                    "witch_have_posion", True
                ) and env.internal_state.get("witch_have_save", True):
                    use_potion_guide = "You can use 'save NAME' or 'poison NAME'. If you don't want to use potions, you can put 'skip' in the argument of action."
                elif env.internal_state.get("witch_have_posion", True):
                    use_potion_guide = "You can use 'poison NAME'. If you don't want to use potions, you can put 'skip' in the argument of action."
                elif env.internal_state.get("witch_have_save", True):
                    use_potion_guide = "You can use 'save NAME'. If you don't want to use potions, you can put 'skip' in the argument of action."
                else:
                    use_potion_guide = (
                        "You can't use any potions as you don't have any left."
                    )
                killed_message = ""
                if kill_target := env.internal_state.get("kill_target", None):
                    killed_message = f"{kill_target} is killed by werewolves."
                return f"It is Night. You are the Witch. {use_potion_guide} {killed_message}"
            else:
                return "It is Night. You are sleeping."

        return ""


class WerewolfEnv(SocialDeductionGame):
    """Werewolf game with voting, kills, and special roles."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=WerewolfActionHandler(), **kwargs)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
        agents: Agents | None = None,
        omniscient: bool = False,
        lite: bool = False,
        include_background_observations: bool = True,
    ) -> Dict[str, Observation]:
        obs = super().reset(
            seed=seed,
            options=options,
            agents=agents,
            omniscient=omniscient,
            lite=lite,
            include_background_observations=include_background_observations,
        )
        # Witch has potions
        self.internal_state["witch_have_posion"] = True
        self.internal_state["witch_have_save"] = True
        # Werewolves have kill targets
        self.internal_state["kill_target_proposals"] = {}
        return obs

    def _check_eliminations(self) -> None:
        """Apply eliminations based on collected actions."""
        # Only apply eliminations if we are about to transition state
        if not self._should_transition_state():
            return

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
                        SimpleMessage(message=f"[Game] {eliminated} was voted out!"),
                    )
                # Clear votes
                self.internal_state["votes"] = {}
                # log elimination
                _gen_logger.info(
                    f"{eliminated} was voted out! They were a {self.agent_to_role[eliminated]}."
                )
                _gen_logger.info(f"Remaining players: {self.agent_alive}")

        elif self.current_state == "Night_witch":
            # Resolve Night actions (Werewolf kill + Witch save/poison)

            kill_target = self.internal_state.get("kill_target")
            saved_target = self.internal_state.get("saved_target")
            poison_target = self.internal_state.get("poison_target")

            # Check kill
            if kill_target and self.agent_alive.get(kill_target, False):
                if kill_target != saved_target:
                    self.agent_alive[kill_target] = False
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {kill_target} was killed by werewolves!"
                        ),
                    )
                    _gen_logger.info(f"{kill_target} was killed by werewolves!")
                    _gen_logger.info(f"Remaining players: {self.agent_alive}")
                else:
                    self.recv_message(
                        "Environment",
                        SimpleMessage(message="[Game] An attack was prevented!"),
                    )
                    _gen_logger.info(f"An attack to {kill_target} was prevented!")
                    _gen_logger.info(f"Remaining players: {self.agent_alive}")

            # 2. Witch Poison
            if poison_target and self.agent_alive.get(poison_target, False):
                self.agent_alive[poison_target] = False
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {poison_target} died by witch's poison!"
                    ),
                )
                _gen_logger.info(f"{poison_target} died by witch's poison!")
                _gen_logger.info(f"Remaining players: {self.agent_alive}")

            # Clear night states
            self.internal_state.pop("kill_target_proposals", None)
            self.internal_state.pop("kill_target", None)
            self.internal_state.pop("saved_target", None)
            self.internal_state.pop("poison_target", None)


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
) -> WerewolfEnv:
    """Create werewolf game environment."""
    return WerewolfEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        action_order="round-robin",
        evaluators=[WerewolfGameEndEvaluator(max_turn_number=40)],
        terminal_evaluators=[],
        hide_unknown=True,
    )


def create_agents(
    agent_profiles: List[AgentProfile],
    env_profile: EnvironmentProfile,
    model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any],  # Added config here
) -> List[LLMAgent]:
    # Identify werewolves for partner info
    werewolf_goal_str = config.get("role_goals", {}).get("Werewolf", "")
    werewolves = [
        p.first_name + (" " + p.last_name if p.last_name else "")
        for p in agent_profiles
        if env_profile.agent_goals[agent_profiles.index(p)] == werewolf_goal_str
    ]

    agents = []
    for idx, profile in enumerate(agent_profiles):
        # Calculate secrets
        agent_name = f"{profile.first_name}{' ' + profile.last_name if profile.last_name else ''}"
        role_goal = env_profile.agent_goals[idx]
        secrets = ""

        # Check if agent is a werewolf
        is_werewolf = env_profile.agent_goals[idx] == "Werewolf"

        if is_werewolf:
            partners = [w for w in werewolves if w != agent_name]
            if partners:
                secrets = f"Your secret: You are a werewolf. Your partner(s) are: {', '.join(partners)}."
            else:
                secrets = "Your secret: You are a werewolf. You have no partners."
        filled_template = (
            SOCIAL_GAME_PROMPT_TEMPLATE.replace("{description}", env_profile.scenario)
            .replace("{secret}", secrets)
            .replace(
                "{goal}",
                role_goal,  # Also replace the goal here
            )
        )
        # Determine model for this agent
        if isinstance(model_name, dict):
            # model_dict mapping "agentX" keys or names?
            # Tournament runner will likely pass a list or dict keyed by index/name.
            # Let's support a list or dict lookup.
            # Assuming 'model_name' might be a dict like {"Aurora": "gpt-4", ...}
            this_agent_model = model_name.get(
                agent_name, model_name.get("default", "gpt-4")
            )
        elif isinstance(model_name, list):
            this_agent_model = model_name[idx]
        else:
            this_agent_model = model_name

        agent = LLMAgent(
            agent_name=f"{profile.first_name}{' ' + profile.last_name if profile.last_name else ''}",
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
    scenario = config.get("description", "Werewolves game").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="werewolves",
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
    """Run werewolf game."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Werewolf game.")
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
    print("🌕 Duskmire Werewolves")
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
        tag="test_werewolves",
        push_to_db=True,
    )


if __name__ == "__main__":
    # Configure logging for standalone execution
    LOG_FILE = BASE_DIR / "werewolves_game_debug.log"
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
