"""Launcher for the Rock, Paper, Scissors game."""

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


class RPSEvaluator(SocialGameEndEvaluator):
    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        if turn_number >= self.max_turn_number:
            env = kwargs.get("env")
            if env:
                scores = env.internal_state.get("scores", {})
                response: List[
                    Tuple[str, Tuple[Tuple[str, int | float | bool], str]]
                ] = [("environment", (("terminated", True), "Max (10) turns reached"))]

                agent_names = list(env.agents)
                scores_dict = {name: scores.get(name, 0) for name in agent_names}

                # Logic:
                # Highest score wins (1), Loser (-1). Tie -> 0.

                values = list(scores_dict.values())
                rewards = {name: 0.0 for name in agent_names}

                if len(values) == 2:
                    s1, s2 = values[0], values[1]
                    n1, n2 = agent_names[0], agent_names[1]

                    if s1 > s2:
                        rewards[n1] = 1.0
                        rewards[n2] = -1.0
                    elif s2 > s1:
                        rewards[n1] = -1.0
                        rewards[n2] = 1.0

                for agent_name in agent_names:
                    idx = agent_names.index(agent_name)
                    key = f"agent_{idx+1}"
                    raw_score = scores_dict.get(agent_name, 0)
                    reward = rewards.get(agent_name, 0.0)
                    response.append(
                        (
                            key,
                            (("complete_rating", reward), f"Final Score: {raw_score}"),
                        )
                    )
                return response
            return [("environment", (("terminated", True), "Max turns reached"))]
        return [("environment", (("terminated", False), ""))]


# ============================================================================
# Action Handler
# ============================================================================


class RPSActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if isinstance(env, RPSEnv) and action.action_type in ["action", "speak"]:
            move_str = action.argument.lower()
            current_move = None
            if "rock" in move_str:
                current_move = "Rock"
            elif "paper" in move_str:
                current_move = "Paper"
            elif "scissors" in move_str or "scissor" in move_str:
                current_move = "Scissors"

            if current_move:
                env.internal_state["current_moves"][agent_name] = current_move

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        return "You are playing Rock-Paper-Scissors. Choose 'action: rock', 'action: paper', or 'action: scissors'. You cannot speak."


# ============================================================================
# Environment
# ============================================================================


class RPSEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=RPSActionHandler(), **kwargs)
        self.internal_state = {"round": 0, "scores": {}, "current_moves": {}}

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
        self.internal_state = {"round": 0, "scores": {}, "current_moves": {}}
        for agent in self.agents:
            self.internal_state["scores"][agent] = 0
        return obs

    async def astep(
        self, actions: Dict[str, AgentAction] | Dict[str, Dict[str, int | str]]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        if self.action_handler:
            for agent_name, action in actions.items():
                if isinstance(action, AgentAction):
                    self.action_handler.handle_action(self, agent_name, action)

        moves = self.internal_state["current_moves"]
        if len(moves) == len(self.agents) and len(self.agents) == 2:
            agents = list(self.agents)
            a1, a2 = agents[0], agents[1]
            m1, m2 = moves[a1], moves[a2]

            result, winner = "Draw", None
            if m1 == m2:
                result = "Draw"
            elif (
                (m1 == "Rock" and m2 == "Scissors")
                or (m1 == "Scissors" and m2 == "Paper")
                or (m1 == "Paper" and m2 == "Rock")
            ):
                winner = a1
            else:
                winner = a2

            r1, r2 = (1, -1) if winner == a1 else (-1, 1) if winner == a2 else (0, 0)
            result = f"{winner} wins!" if winner else "Draw"

            self.internal_state["scores"][a1] += r1
            self.internal_state["scores"][a2] += r2

            msg = f"Round {self.internal_state['round']+1}: {a1} ({m1}) vs {a2} ({m2}) -> {result}"
            self.recv_message("Environment", SimpleMessage(message=msg))

            self.internal_state["round"] += 1
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

    try:
        existing = AgentProfile.find(
            (AgentProfile.first_name == first_name)
            & (AgentProfile.last_name == last_name)
        ).all()
        if existing:
            return AgentProfile.get(existing[0].pk)
    except Exception:
        pass

    role_secret = config.get("role_secrets", {}).get(role, "")
    profile = AgentProfile(
        first_name=first_name, last_name=last_name, secret=role_secret
    )
    profile.save()
    return profile


def create_environment(
    env_profile: EnvironmentProfile, model_name: str, config: Dict[str, Any]
) -> RPSEnv:
    return RPSEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        # action_order is handled by config state_properties
        evaluators=[RPSEvaluator(max_turn_number=10)],
        terminal_evaluators=[],
        hide_unknown=True,
    )


def create_agents(
    agent_profiles: List[AgentProfile],
    env_profile: EnvironmentProfile,
    model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any],
) -> List[LLMAgent]:
    agents = []
    for idx, profile in enumerate(agent_profiles):
        agent_name = f"{profile.first_name}{' ' + profile.last_name if profile.last_name else ''}"
        role_goal = env_profile.agent_goals[idx]
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
    if config is None:
        config = load_config(CONFIG_PATH)
    agent_profiles = [ensure_agent_profile(entry) for entry in config.get("agents", [])]
    agent_goals = [
        config.get("role_goals", {}).get(entry.get("role", ""), "")
        for entry in config.get("agents", [])
    ]
    agent_names = [entry.get("name", "") for entry in config.get("agents", [])]
    scenario = config.get("description", "RPS").format(
        agent_names=", ".join(agent_names)
    )

    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="rps",
    )
    env_profile.save()

    env = create_environment(env_profile, env_model_name, config)
    agents = create_agents(agent_profiles, env_profile, agent_model_name, config)
    return env, agents


def print_roster(config: Dict[str, Any]) -> None:
    print("Participants & roles:")
    for entry in config.get("agents", []):
        print(f" - {entry.get('name')}: {entry.get('role')}")


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
    import argparse

    parser = argparse.ArgumentParser(description="Run Rock-Paper-Scissors game.")
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

    # Merge roster into config for consistent access
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env_model_name = "gpt-4o"

    # We pass config explicitly to prepare_scenario
    env, agents = prepare_scenario(env_model_name, agent_model_name, config)

    print("✊✋✌️ Rock Paper Scissors")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_rps",
        push_to_db=True,
    )


if __name__ == "__main__":
    # Configure logging for standalone execution
    LOG_FILE = BASE_DIR / "rps_game_debug.log"
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
