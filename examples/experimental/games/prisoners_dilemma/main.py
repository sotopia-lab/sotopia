"""Launcher for the Prisoner's Dilemma game."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import redis
from rich.logging import RichHandler

from sotopia.agents import LLMAgent
from sotopia.agents.llm_agent import Agents
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipType,
)
from sotopia.envs.evaluators import SocialGameEndEvaluator
from sotopia.envs.social_game import (
    ActionHandler,
    SocialDeductionGame,
    SOCIAL_GAME_PROMPT_TEMPLATE,
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
# Evaluator
# ============================================================================


class PrisonersDilemmaEvaluator(SocialGameEndEvaluator):
    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        if turn_number >= self.max_turn_number:
            env = kwargs.get("env")
            if env:
                scores = env.internal_state.get("scores", {})
                response: List[
                    Tuple[str, Tuple[Tuple[str, int | float | bool], str]]
                ] = [("environment", (("terminated", True), "Max (5) turns reached"))]

                agent_names = list(env.agents)
                scores_dict = {name: scores.get(name, 0) for name in agent_names}

                # Logic:
                # 1. If both < 10 -> Draw (0)
                # 2. Otherwise -> Highest score wins (1), Loser (-1)
                # 3. Tie >= 10 -> Draw (0)

                values = list(scores_dict.values())
                if len(values) == 2:
                    s1, s2 = values[0], values[1]
                    n1, n2 = agent_names[0], agent_names[1]

                    rewards = {n1: 0.0, n2: 0.0}

                    if s1 < 10 and s2 < 10:
                        # Draw (both failed threshold)
                        pass
                    elif s1 > s2:
                        rewards[n1] = 1.0
                        rewards[n2] = -1.0
                    elif s2 > s1:
                        rewards[n1] = -1.0
                        rewards[n2] = 1.0
                    else:
                        # Tie and at least one >= 10 (which implies both >= 10)
                        pass

                    for agent_name in agent_names:
                        try:
                            idx = agent_names.index(agent_name)
                            key = f"agent_{idx+1}"
                            raw_score = scores_dict[agent_name]
                            reward = rewards[agent_name]
                            response.append(
                                (
                                    key,
                                    (
                                        ("complete_rating", reward),
                                        f"Final Score: {raw_score}",
                                    ),
                                )
                            )
                        except ValueError:
                            continue
                    return response

                # Fallback for != 2 agents
                for agent_name in agent_names:
                    score = scores.get(agent_name, 0)
                    idx = agent_names.index(agent_name)
                    key = f"agent_{idx+1}"
                    response.append(
                        (key, (("complete_rating", score), f"Final Score: {score}"))
                    )
                return response
            return [("environment", (("terminated", True), "Max turns reached"))]

        return [("environment", (("terminated", False), ""))]


# ============================================================================
# Action Handler
# ============================================================================


class PrisonersDilemmaActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if isinstance(env, PrisonersDilemmaEnv):
            if action.action_type in ["action", "speak"]:
                move = action.argument.lower()
                current_move = None
                if "defect" in move:
                    current_move = "Defect"
                elif "cooperate" in move:
                    current_move = "Cooperate"

                if current_move:
                    env.internal_state["current_moves"][agent_name] = current_move

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        return "You are playing Prisoner's Dilemma. Choose to 'action: cooperate' or 'action: defect'. You cannot speak."


# ============================================================================
# Environment
# ============================================================================


class PrisonersDilemmaEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=PrisonersDilemmaActionHandler(), **kwargs)
        self.internal_state: Dict[str, Any] = {
            "round": 0,
            "scores": {},
            "current_moves": {},
        }
        self.payoff_matrix = self._config.get("payoff_matrix", {})

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, str] | None = None,
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
        self.internal_state = {"round": 0, "scores": {}, "current_moves": {}}
        for agent in self.agents:
            self.internal_state["scores"][agent] = 0
        return obs

    async def astep(
        self, actions: Dict[str, AgentAction] | Dict[str, Dict[str, int | str]]
    ) -> Tuple[
        Dict[str, Observation],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[Any, Any]],
    ]:
        # Filter for AgentAction
        valid_actions: Dict[str, AgentAction] = {
            k: v for k, v in actions.items() if isinstance(v, AgentAction)
        }

        # Process actions
        if self.action_handler:
            for agent_name, action in valid_actions.items():
                self.action_handler.handle_action(self, agent_name, action)

        # Check if everyone moved
        # We need to access internal_state safely
        moves = self.internal_state.get("current_moves", {})
        scores = self.internal_state.get("scores", {})

        if len(moves) == len(self.agents) and len(self.agents) == 2:
            agents = list(self.agents)
            a1, a2 = agents[0], agents[1]
            m1 = moves.get(a1, "Cooperate")
            m2 = moves.get(a2, "Cooperate")

            # Use loaded payoff matrix
            try:
                payoffs = self.payoff_matrix[m1][m2]
                r1, r2 = payoffs[0], payoffs[1]
            except KeyError:
                # Fallback to standard PD if matrix missing or key error
                r1, r2 = 0, 0

            # Update scores
            scores[a1] = scores.get(a1, 0) + r1
            scores[a2] = scores.get(a2, 0) + r2
            self.internal_state["scores"] = scores

            msg = f"Round {self.internal_state.get('round', 0)+1} Results:\n{a1} chose {m1}, gets {r1}.\n{a2} chose {m2}, gets {r2}.\nTotal Scores: {scores}"
            self.recv_message("Environment", SimpleMessage(message=msg))

            self.internal_state["round"] = self.internal_state.get("round", 0) + 1
            self.internal_state["current_moves"] = {}

        return await super().astep(actions)


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
) -> PrisonersDilemmaEnv:
    """Create PD game environment."""
    return PrisonersDilemmaEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[PrisonersDilemmaEvaluator(max_turn_number=5)],
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
        agent_name = f"{profile.first_name}{' ' + profile.last_name if profile.last_name else ''}"
        role_goal = env_profile.agent_goals[idx]

        # Get secret based on role from config, matching profile index
        # Assumption: agents list in config matches profile order
        role = config.get("agents", [])[idx].get("role", "")
        secrets = config.get("role_secrets", {}).get(role, "")

        filled_template = (
            SOCIAL_GAME_PROMPT_TEMPLATE.replace("{description}", env_profile.scenario)
            .replace("{secret}", f"Your secret info: {secrets}")
            .replace("{goal}", role_goal)
        )

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
        agent.goal = role_goal
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
    scenario = config.get("description", "Prisoner's Dilemma").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="pd",
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
    """Run Prisoner's Dilemma game."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Prisoner's Dilemma game.")
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

    agent_model_name = get_model_names(config)
    env_model_name = "gpt-4o"

    # We pass config explicitly to prepare_scenario
    env, agents = prepare_scenario(env_model_name, agent_model_name, config)

    print("⛓️ Prisoner's Dilemma")
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
        tag="test_pd",
        push_to_db=True,
    )


if __name__ == "__main__":
    # Configure logging for standalone execution
    LOG_FILE = BASE_DIR / "pd_game_debug.log"
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
