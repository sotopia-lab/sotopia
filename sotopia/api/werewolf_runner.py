"""
Werewolf WebSocket Game Runner

This module provides the WebSocket-compatible runner for Werewolf games,
allowing a human player to participate via the frontend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from sotopia.agents import LLMAgent
from sotopia.database import AgentProfile, EnvironmentProfile, RelationshipType, GameResult
from sotopia.api.elo import process_game_elo, get_rank_tier
from sotopia.envs.social_game import (
    SocialDeductionGame,
    ActionHandler,
    SOCIAL_GAME_PROMPT_TEMPLATE,
)
from sotopia.envs.evaluators import SocialGameEndEvaluator
from sotopia.messages import AgentAction, Observation, SimpleMessage, Message
from sotopia.api.websocket_utils import WebSocketHumanAgent

logger = logging.getLogger(__name__)

# Default config path (can be overridden)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "examples" / "experimental" / "werewolves" / "config.json"



# Werewolf Game End Evaluator


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
                    return True, msg

            elif cond_type == "parity":
                team1 = condition.get("team", "")
                team2 = condition.get("other", "")
                if team_counts.get(team1, 0) >= team_counts.get(team2, 0):
                    winner = condition.get("winner", "")
                    msg = condition.get("message", f"{winner} wins!")
                    return True, msg

        return False, ""


# Werewolf Action Handler


class WerewolfActionHandler(ActionHandler):
    """Handles actions for the Werewolf game."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        """Handle a single action from an agent based on current state."""

        if env.current_state == "Day_vote":
            if "votes" not in env.internal_state:
                env.internal_state["votes"] = {}

            if action.action_type == "action" and "vote" in action.argument.lower():
                words = action.argument.split()
                target = next(
                    (w for w in words if w[0].isupper() and w in env.agents), None
                )
                if target:
                    env.internal_state["votes"][agent_name] = target

        elif env.current_state == "Night_werewolf":
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
                        # Record this werewolf's proposal; do not set kill_target here.
                        env.internal_state["kill_target_proposals"][agent_name] = target

        elif env.current_state == "Night_seer":
            role = env.agent_to_role.get(agent_name, "")
            if role == "Seer" and action.action_type == "action":
                if "inspect" in action.argument.lower():
                    words = action.argument.split()
                    target = next(
                        (w for w in words if w[0].isupper() and w in env.agents),
                        None,
                    )
                    if target:
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
                    env.internal_state["witch_have_poison"] = False
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
        
        # Get list of alive players (excluding self)
        alive_players = [
            name for name, alive in env.agent_alive.items() 
            if alive and name != agent_name
        ]
        players_str = ", ".join(alive_players)

        if env.current_state == "Day_vote":
            return f"It is voting time. You MUST vote for one player to eliminate. Available targets: {players_str}. Use 'vote [NAME]' where NAME is one of the players listed."

        elif env.current_state == "Night_werewolf":
            if role == "Werewolf":
                return f"It is Night. You are a Werewolf. Choose a target to kill. Available targets: {players_str}. Use 'kill [NAME]'."
            else:
                return "It is Night. You are sleeping."

        elif env.current_state == "Night_seer":
            if role == "Seer":
                return f"It is Night. You are the Seer. Choose a player to inspect. Available targets: {players_str}. Use 'inspect [NAME]'."
            else:
                return "It is Night. You are sleeping."

        elif env.current_state == "Night_witch":
            if role == "Witch":
                have_poison = env.internal_state.get("witch_have_poison", True)
                have_save = env.internal_state.get("witch_have_save", True)
                if have_poison and have_save:
                    guide = f"You can use 'save [NAME]' or 'poison [NAME]', or 'skip'. Available targets: {players_str}."
                elif have_poison:
                    guide = f"You can use 'poison [NAME]' or 'skip'. Available targets: {players_str}."
                elif have_save:
                    guide = f"You can use 'save [NAME]' or 'skip'. Available targets: {players_str}."
                else:
                    guide = "You have no potions left. Use 'skip'."
                
                kill_target = env.internal_state.get("kill_target")
                if kill_target:
                    guide += f" {kill_target} was targeted by werewolves tonight."
                return f"It is Night. You are the Witch. {guide}"
            else:
                return "It is Night. You are sleeping."

        return ""


# Werewolf Environment


class WerewolfEnv(SocialDeductionGame):
    """Werewolf game with voting, kills, and special roles."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=WerewolfActionHandler(), **kwargs)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
        agents: Any = None,
        omniscient: bool = False,
        lite: bool = False,
        include_background_observations: bool | None = False,
    ) -> Dict[str, Observation]:
        obs = super().reset(
            seed=seed,
            options=options,
            agents=agents,
            omniscient=omniscient,
            lite=lite,
            include_background_observations=include_background_observations,
        )
        self.internal_state["witch_have_poison"] = True
        self.internal_state["witch_have_save"] = True
        self.internal_state["kill_target_proposals"] = {}
        return obs

    _WEREWOLF_MAX_VOTE_ROUNDS = 2  # total rounds before forcing a decision

    def _should_transition_state(self) -> bool:
        """Override transition logic for werewolf night kill target.

        Round 1: if all werewolves agree → use that target immediately.
                 if they disagree → clear proposals and retry (round 2).
        Round 2: if they agree → great.
                 if still split → tally ALL proposals from both rounds
                 and pick the most-voted target.  True ties are broken
                 randomly so the game always progresses fairly.
        """
        if self.current_state == "Night_werewolf":
            proposals = self.internal_state.get("kill_target_proposals", {}) or {}
            alive_werewolves = [
                name
                for name, role in self.agent_to_role.items()
                if role == "Werewolf" and self.agent_alive.get(name, False)
            ]
            if not alive_werewolves:
                return True

            # Wait until every alive werewolf has submitted a proposal
            if len(proposals) < len(alive_werewolves):
                return False

            # Check for unanimous agreement (works on any round)
            targets = set(proposals.values())
            if len(targets) == 1:
                self.internal_state["kill_target"] = next(iter(targets))
                self.internal_state.pop("_werewolf_vote_round", None)
                self.internal_state.pop("_werewolf_vote_history", None)
                return True

            # --- Not unanimous ---
            vote_round = self.internal_state.get("_werewolf_vote_round", 1)
            history: list[str] = self.internal_state.get("_werewolf_vote_history", [])
            history.extend(proposals.values())
            self.internal_state["_werewolf_vote_history"] = history

            if vote_round < self._WEREWOLF_MAX_VOTE_ROUNDS:
                # Allow another round: clear proposals so agents vote again
                self.internal_state["kill_target_proposals"] = {}
                self.internal_state["_werewolf_vote_round"] = vote_round + 1
                return False

            # Max rounds exhausted — majority across all rounds
            vote_counts = Counter(history)
            max_votes = max(vote_counts.values())
            top_targets = [t for t, c in vote_counts.items() if c == max_votes]
            chosen = random.choice(top_targets) if len(top_targets) > 1 else top_targets[0]
            self.internal_state["kill_target"] = chosen
            # Cleanup retry state
            self.internal_state.pop("_werewolf_vote_round", None)
            self.internal_state.pop("_werewolf_vote_history", None)
            return True

        # Fallback to default behaviour
        return super()._should_transition_state()

    def _check_eliminations(self) -> None:
        """Apply eliminations based on collected actions."""
        if not self._should_transition_state():
            return

        if self.current_state == "Day_vote":
            votes = self.internal_state.get("votes", {})
            if votes:
                vote_counts: Dict[str, int] = {}
                for target in votes.values():
                    vote_counts[target] = vote_counts.get(target, 0) + 1

                if vote_counts:
                    eliminated = max(vote_counts, key=vote_counts.get)  # type: ignore
                    self.agent_alive[eliminated] = False
                    self.recv_message(
                        "Environment",
                        SimpleMessage(message=f"[Game] {eliminated} was voted out!"),
                    )
                self.internal_state["votes"] = {}

        elif self.current_state == "Night_witch":
            kill_target = self.internal_state.get("kill_target")
            saved_target = self.internal_state.get("saved_target")
            poison_target = self.internal_state.get("poison_target")

            if kill_target and self.agent_alive.get(kill_target, False):
                if kill_target != saved_target:
                    self.agent_alive[kill_target] = False
                    self.recv_message(
                        "Environment",
                        SimpleMessage(message=f"[Game] {kill_target} was killed by werewolves!"),
                    )
                else:
                    self.recv_message(
                        "Environment",
                        SimpleMessage(message="[Game] An attack was prevented!"),
                    )

            if poison_target and self.agent_alive.get(poison_target, False):
                self.agent_alive[poison_target] = False
                self.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Game] {poison_target} died by witch's poison!"),
                )

            self.internal_state.pop("kill_target_proposals", None)
            self.internal_state.pop("kill_target", None)
            self.internal_state.pop("saved_target", None)
            self.internal_state.pop("poison_target", None)


# WebSocket Werewolf Runner


class WebSocketWerewolfRunner:
    """
    Runs a Werewolf game with WebSocket streaming support.
    
    Supports one human player via WebSocketHumanAgent.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        human_agent_index: Optional[int] = None,
        human_agent_name: Optional[str] = None,
        action_queue_getter: Optional[Callable[[], asyncio.Queue[Dict[str, Any]]]] = None,
        pack_chat_getter: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        ai_model: str = "gpt-4o-mini",
        user_id: Optional[str] = None,
    ) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.human_agent_index = human_agent_index
        self.human_agent_name = human_agent_name
        self.action_queue_getter = action_queue_getter
        self.ai_model = ai_model
        self.user_id = user_id  # For linking game history
        
        self.config: Dict[str, Any] = {}
        self.env: Optional[WerewolfEnv] = None
        self.agents: List[Any] = []
        self.game_over = False
        self.winner: Optional[str] = None
        self.turn_count: int = 0
        self.game_start_time: Optional[float] = None
        self.pack_chat_getter = pack_chat_getter

    def _load_config(self) -> Dict[str, Any]:
        """Load game configuration."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return json.load(f)
        else:
            # Default config
            return {
                "scenario": "Werewolves Game",
                "description": "A social deduction game with werewolves and villagers.",
                "role_goals": {
                    "Villager": "Act openly to identify werewolves.",
                    "Werewolf": "Deceive others and eliminate villagers.",
                    "Seer": "Inspect players to uncover werewolves.",
                    "Witch": "Use potions strategically.",
                },
                "agents": [
                    {"name": "Aurora", "role": "Villager", "team": "Villagers"},
                    {"name": "Bram", "role": "Werewolf", "team": "Werewolves"},
                    {"name": "Celeste", "role": "Seer", "team": "Villagers"},
                    {"name": "Dorian", "role": "Werewolf", "team": "Werewolves"},
                    {"name": "Elise", "role": "Witch", "team": "Villagers"},
                    {"name": "Finn", "role": "Villager", "team": "Villagers"},
                ],
                "initial_state": "Night_werewolf",
                "state_transition": {
                    "Night_werewolf": "Night_seer",
                    "Night_seer": "Night_witch",
                    "Night_witch": "Day_discussion",
                    "Day_discussion": "Day_vote",
                    "Day_vote": "Night_werewolf",
                },
                "end_conditions": [
                    {"type": "team_eliminated", "team": "Werewolves", "winner": "Villagers"},
                    {"type": "parity", "team": "Werewolves", "other": "Villagers", "winner": "Werewolves"},
                ],
            }

    def _create_agent_profiles(self) -> List[AgentProfile]:
        """Create agent profiles from config."""
        profiles = []
        for entry in self.config.get("agents", []):
            profile = AgentProfile(
                first_name=entry.get("name", "Agent"),
                last_name="",
                occupation=entry.get("role", "Villager"),
            )
            profiles.append(profile)
        return profiles

    def _create_environment(self) -> WerewolfEnv:
        """Create the Werewolf environment."""
        agent_names = [a.get("name") for a in self.config.get("agents", [])]
        scenario = self.config.get("description", "Werewolves game").format(
            agent_names=", ".join(agent_names)
        )
        
        agent_goals = []
        for entry in self.config.get("agents", []):
            role = entry.get("role", "Villager")
            goal = self.config.get("role_goals", {}).get(role, "Play the game.")
            agent_goals.append(goal)

        env_profile = EnvironmentProfile(
            scenario=scenario,
            relationship=RelationshipType.acquaintance,
            agent_goals=agent_goals,
            tag="werewolves",
        )

        env = WerewolfEnv(
            env_profile=env_profile,
            config=self.config,  # Pass config to constructor
            evaluators=[WerewolfGameEndEvaluator(max_turn_number=50)],
            terminal_evaluators=[],
            action_order="round-robin",
        )
        
        return env

    def _create_agents(self, agent_profiles: List[AgentProfile]) -> List[Any]:
        """Create agents, substituting human at specified index."""
        agents = []
        agent_entries = self.config.get("agents", [])
        
        # Find werewolf names for partner info
        werewolves = [
            e.get("name") for e in agent_entries if e.get("role") == "Werewolf"
        ]

        for idx, profile in enumerate(agent_profiles):
            agent_name = profile.first_name
            role = agent_entries[idx].get("role", "Villager")
            goal = self.config.get("role_goals", {}).get(role, "Play the game.")
            
            # Build secrets for werewolves
            secrets = ""
            if role == "Werewolf":
                partners = [w for w in werewolves if w != agent_name]
                if partners:
                    secrets = f"Your secret: You are a werewolf. Your partner(s): {', '.join(partners)}."
                else:
                    secrets = "Your secret: You are a werewolf."

            # Check if this is the human player
            is_human = (
                (self.human_agent_index is not None and idx == self.human_agent_index) or
                (self.human_agent_name is not None and agent_name == self.human_agent_name)
            )

            if is_human and self.action_queue_getter is not None:
                # Create human agent
                agent = WebSocketHumanAgent(
                    agent_name=agent_name,
                    agent_profile=profile,
                    action_queue_getter=self.action_queue_getter,
                    goal=goal,
                )
                logger.info(f"Created HUMAN agent (WebSocketHumanAgent): {agent_name} (role: {role}, index: {idx})")
            else:
                # Create LLM agent
                template = SOCIAL_GAME_PROMPT_TEMPLATE.replace("{goal}", f"{{goal}}\n{secrets}")
                template = template.replace(
                    "{description}", self.config.get("description", "")
                )
                
                agent = LLMAgent(
                    agent_name=agent_name,
                    agent_profile=profile,
                    model_name=self.ai_model,
                    strict_action_constraint=True,
                )
                agent.goal = goal
                if is_human:
                    logger.warning(f"Created LLM agent for human player {agent_name} - action_queue_getter is None!")
                else:
                    logger.info(f"Created LLM agent: {agent_name} (role: {role}, index: {idx})")

            agents.append(agent)

        return agents

    def setup(self) -> None:
        """Initialize the game environment and agents."""
        self.config = self._load_config()
        logger.info(f"Loaded config with {len(self.config.get('agents', []))} agents")
        logger.info(f"Config agents: {self.config.get('agents', [])}")
        agent_profiles = self._create_agent_profiles()
        self.env = self._create_environment()
        self.agents = self._create_agents(agent_profiles)
        
        # Create agents dict for env
        from sotopia.agents import Agents
        agents_dict = Agents({agent.agent_name: agent for agent in self.agents})
        
        # Reset environment
        self.env.reset(agents=agents_dict, omniscient=False)

    async def run(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the game loop, yielding messages for WebSocket streaming.
        
        Yields dictionaries with game state updates to send to frontend.
        """
        if self.env is None:
            self.setup()
        
        assert self.env is not None
        
        # Check if playing as guest
        is_guest = not self.user_id
        
        # Sanitize initial phase name
        initial_phase = self.env.current_state
        display_initial_phase = "Night" if initial_phase.startswith("Night_") else initial_phase.replace("_", " ").title()

        # Debug: log human agent info
        logger.info(f"[Game Start] human_agent_index: {self.human_agent_index}, human_agent_name: {self.human_agent_name}")
        logger.info(f"[Game Start] agents from config: {[a.get('name') for a in self.config.get('agents', [])]}")
        
        # Build player list with human player marked
        players_list = []
        for i, a in enumerate(self.config.get("agents", [])):
            is_human = i == self.human_agent_index
            players_list.append({
                "name": a.get("name"),
                "role": a.get("role") if is_human else "Hidden",
                "team": a.get("team") if is_human else "Hidden",
                "is_alive": True,
                "is_human": is_human,
            })
            if is_human:
                logger.info(f"[Game Start] Human player: {a.get('name')} at index {i} with role {a.get('role')}")
        
        # Send initial game state
        yield {
            "type": "game_start",
            "phase": display_initial_phase,
            "players": players_list,
            "message": "Game started!",
            "is_guest": is_guest,
        }
        
        # Send guest warning if not logged in
        if is_guest:
            yield {
                "type": "guest_warning",
                "title": "Playing as Guest",
                "message": "Your game won't be saved to the leaderboard. Sign in to track your progress and compete for rankings!",
                "action": "sign_in",
            }

        turn = 0
        max_turns = 50
        self.game_start_time = time.time()

        while turn < max_turns and not self.game_over:
            turn += 1
            self.turn_count = turn
            current_phase = self.env.current_state
            
            # Sanitize phase name to not reveal role information
            display_phase = "Night" if current_phase.startswith("Night_") else current_phase.replace("_", " ").title()
            
            # Notify phase change
            yield {
                "type": "phase_change",
                "phase": display_phase,
                "internal_phase": current_phase,  # Keep for internal logic
                "turn": turn,
                "alive_players": [
                    name for name, alive in self.env.agent_alive.items() if alive
                ],
            }

            # Inject any pending pack chat into the environment so AI werewolves see it
            try:
                if hasattr(self, "pack_chat_getter") and callable(self.pack_chat_getter):
                    pending = self.pack_chat_getter() or []
                    for chat in pending:
                        sender = chat.get("from") or "Someone"
                        message = chat.get("message") or ""
                        recorded_at = chat.get("recordedAt") or int(time.time())
                        # Deliver pack chat only to werewolves
                        werewolves = [
                            name for name, role in self.env.agent_to_role.items()
                            if role == "Werewolf" and self.env.agent_alive.get(name, False)
                        ]
                        if werewolves:
                            try:
                                self.env.recv_message(
                                    "PackChat",
                                    SimpleMessage(message=f"[Pack] {sender}: {message}"),
                                    receivers=werewolves,
                                )
                            except Exception:
                                logger.exception("Failed to inject pack chat into env")
                        # Also yield pack_chat event to frontend so UI shows it
                        yield {
                            "type": "pack_chat",
                            "from": sender,
                            "message": message,
                            "recordedAt": recorded_at,
                        }
            except Exception:
                logger.exception("Error processing pack chat getter")

            # Collect actions from all agents
            actions: Dict[str, AgentAction] = {}
            
            # Debug: log agent iteration
            logger.info(f"[Phase {current_phase}] Iterating through {len(self.agents)} agents")
            logger.info(f"[Phase {current_phase}] agent_alive: {self.env.agent_alive}")
            
            for agent in self.agents:
                agent_name = agent.agent_name
                is_human = isinstance(agent, WebSocketHumanAgent)
                logger.info(f"[Phase {current_phase}] Processing agent: {agent_name}, is_human: {is_human}")
                
                # Skip dead agents
                if not self.env.agent_alive.get(agent_name, False):
                    logger.info(f"[Phase {current_phase}] Skipping {agent_name} - dead")
                    continue
                
                # Check if agent can act in this phase
                role = self.env.agent_to_role.get(agent_name, "")
                can_act = self._can_agent_act(current_phase, role)
                
                if not can_act:
                    # Agent skips this phase
                    logger.info(f"[Phase {current_phase}] Skipping {agent_name} - cannot act (role: {role})")
                    actions[agent_name] = AgentAction(action_type="none", argument="", to=[])
                    continue
                
                logger.info(f"[Phase {current_phase}] Agent {agent_name} can act (role: {role}, is_human: {is_human})")

                # Get observation for agent
                obs = self.env._get_observation(agent_name)
                
                # Get action instruction
                action_instruction = ""
                if self.env.action_handler:
                    action_instruction = self.env.action_handler.get_action_instruction(
                        self.env, agent_name
                    )
                
                # Sanitize phase for display
                display_phase = "Night" if current_phase.startswith("Night_") else current_phase.replace("_", " ").title()
                
                if is_human:
                    # Notify frontend that we're waiting for human action
                    yield {
                        "type": "waiting_for_action",
                        "agent_name": agent_name,
                        "phase": display_phase,
                        "role": role,
                        "instruction": action_instruction,
                        "available_actions": self._get_available_actions(current_phase, role),
                        "alive_players": [
                            name for name, alive in self.env.agent_alive.items() 
                            if alive and name != agent_name
                        ],
                    }

                # Get action from agent (human will wait on queue)
                try:
                    action = await agent.aact(obs)
                    actions[agent_name] = action

                    # Decide whether to send this action to the frontend.
                    # Public phases: always show.
                    # Night_werewolf: show to the human if they are also a werewolf
                    #   (pack members see each other's picks).
                    # Other private phases: only show the human's own action.
                    is_private = self._is_private_phase(current_phase)
                    human_role = self.env.agent_to_role.get(self.human_agent_name or "", "")
                    agent_role = self.env.agent_to_role.get(agent_name, "")
                    is_same_pack = (
                        current_phase == "Night_werewolf"
                        and human_role == "Werewolf"
                        and agent_role == "Werewolf"
                    )
                    should_show = (
                        (not is_private)
                        or is_human
                        or is_same_pack
                    )
                    if should_show:
                        yield {
                            "type": "action_taken",
                            "agent_name": agent_name,
                            "action_type": action.action_type,
                            "argument": action.argument,
                            "is_human": is_human,
                            "phase": display_phase,
                        }

                except Exception as e:
                    logger.error(f"Error getting action from {agent_name}: {e}")
                    actions[agent_name] = AgentAction(action_type="none", argument="", to=[])

            # Step the environment
            try:
                observations, rewards, terminated, truncated, info = await self.env.astep(actions)
                
                # Check for game end
                if any(terminated.values()):
                    self.game_over = True
                    self.winner = self._determine_winner()
                    self._save_game_result(self.winner)
                    
                    game_end_msg = {
                        "type": "game_end",
                        "winner": self.winner,
                        "final_state": {
                            "alive_players": [
                                name for name, alive in self.env.agent_alive.items() if alive
                            ],
                            "roles_revealed": [
                                {
                                    "name": a.get("name"),
                                    "role": a.get("role"),
                                    "team": a.get("team"),
                                    "survived": self.env.agent_alive.get(a.get("name"), False),
                                }
                                for a in self.config.get("agents", [])
                            ],
                        },
                        "message": f"{self.winner} wins!",
                        "is_guest": is_guest,
                    }
                    
                    # Add guest prompt if not logged in
                    if is_guest:
                        game_end_msg["guest_prompt"] = {
                            "title": "Great game!",
                            "message": "Sign in to save your progress and climb the leaderboard!",
                            "action": "sign_in",
                        }
                    
                    yield game_end_msg
                    break
                    
            except Exception as e:
                logger.error(f"Error in environment step: {e}")
                yield {
                    "type": "error",
                    "message": str(e),
                }
                break

        if not self.game_over:
            self._save_game_result("Draw")
            draw_msg = {
                "type": "game_end",
                "winner": "Draw",
                "message": "Game ended due to turn limit.",
                "is_guest": is_guest,
            }
            if is_guest:
                draw_msg["guest_prompt"] = {
                    "title": "Great game!",
                    "message": "Sign in to save your progress and climb the leaderboard!",
                    "action": "sign_in",
                }
            yield draw_msg

    def _save_game_result(self, winner: str) -> None:
        """Save game result to database for user history tracking."""
        try:
            duration = time.time() - self.game_start_time if self.game_start_time else 0
            
            # Build player info
            player_ids = []
            player_names = []
            for agent_cfg in self.config.get("agents", []):
                player_names.append(agent_cfg.get("name", "Unknown"))
                # If this is the human player and we have a user_id, use it
                if agent_cfg.get("name") == self.human_agent_name and self.user_id:
                    player_ids.append(self.user_id)
                else:
                    player_ids.append(f"ai_{agent_cfg.get('name', 'unknown')}")
            
            game_result = GameResult(
                game_type="werewolf",
                player_ids=player_ids,
                player_names=player_names,
                winner_team=winner,
                turns_played=self.turn_count,
                duration_seconds=int(duration),
                metadata={
                    "human_player": self.human_agent_name,
                    "user_id": self.user_id,
                    "ai_model": self.ai_model,
                    "roles": [
                        {"name": a.get("name"), "role": a.get("role"), "team": a.get("team")}
                        for a in self.config.get("agents", [])
                    ],
                    "alive_at_end": [
                        name for name, alive in (self.env.agent_alive if self.env else {}).items() if alive
                    ],
                },
            )
            game_result.save()
            logger.info(f"Saved game result: {game_result.pk}")
            
            # Process ELO changes for human players
            if self.user_id and winner not in ("Draw", "Unknown"):
                try:
                    elo_results = process_game_elo(game_result, self.ai_model)
                    if self.user_id in elo_results:
                        elo_before, elo_after, elo_change = elo_results[self.user_id]
                        rank_name, rank_emoji = get_rank_tier(elo_after)
                        logger.info(
                            f"ELO updated for user {self.user_id}: "
                            f"{elo_before} -> {elo_after} ({elo_change:+d}) [{rank_emoji} {rank_name}]"
                        )
                        # Store ELO change in metadata for frontend display
                        game_result.metadata["elo_changes"] = {
                            self.user_id: {
                                "before": elo_before,
                                "after": elo_after,
                                "change": elo_change,
                                "rank": rank_name,
                                "rank_emoji": rank_emoji,
                            }
                        }
                        game_result.save()  # Save updated metadata
                except Exception as elo_error:
                    logger.error(f"Failed to process ELO: {elo_error}")
        except Exception as e:
            logger.error(f"Failed to save game result: {e}")

    def _can_agent_act(self, phase: str, role: str) -> bool:
        """Check if an agent can act in the current phase."""
        state_props = self.config.get("state_properties", {}).get(phase, {})
        acting_roles = state_props.get("acting_roles", [])
        
        # If no specific roles, everyone can act
        if not acting_roles:
            return True
        
        return role in acting_roles

    def _get_available_actions(self, phase: str, role: str) -> List[str]:
        """Get available actions for an agent in the current phase."""
        if phase == "Day_vote":
            return ["vote"]
        elif phase == "Day_discussion":
            return ["speak"]
        elif phase == "Night_werewolf" and role == "Werewolf":
            return ["kill"]
        elif phase == "Night_seer" and role == "Seer":
            return ["inspect"]
        elif phase == "Night_witch" and role == "Witch":
            return ["save", "poison", "skip"]
        return ["none"]

    def _is_private_phase(self, phase: str) -> bool:
        """Check if actions in this phase are private."""
        return phase.startswith("Night_")

    def _determine_winner(self) -> str:
        """Determine the winning team."""
        if self.env is None:
            return "Unknown"
            
        team_counts: Dict[str, int] = {}
        for agent_name, alive in self.env.agent_alive.items():
            if alive:
                role = self.env.agent_to_role.get(agent_name, "")
                team = self.env.role_to_team.get(role, "")
                team_counts[team] = team_counts.get(team, 0) + 1

        if team_counts.get("Werewolves", 0) == 0:
            return "Villagers"
        elif team_counts.get("Werewolves", 0) >= team_counts.get("Villagers", 0):
            return "Werewolves"
        
        return "Unknown"


async def run_werewolf_simulation(
    human_agent_index: Optional[int] = None,
    human_agent_name: Optional[str] = None,
    action_queue_getter: Optional[Callable[[], asyncio.Queue[Dict[str, Any]]]] = None,
    pack_chat_getter: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    ai_model: str = "gpt-4o-mini",
    config_path: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Convenience function to run a werewolf simulation.
    
    Args:
        human_agent_index: Index of the human player (0-5)
        human_agent_name: Name of the human player's character
        action_queue_getter: Function that returns the action queue for human input
        ai_model: Model to use for AI agents
        config_path: Path to game config file
        user_id: Optional user ID for game history tracking
        
    Yields:
        Game state updates as dictionaries
    """
    runner = WebSocketWerewolfRunner(
        config_path=config_path,
        human_agent_index=human_agent_index,
        human_agent_name=human_agent_name,
        action_queue_getter=action_queue_getter,
        pack_chat_getter=pack_chat_getter,
        ai_model=ai_model,
        user_id=user_id,
    )
    
    async for message in runner.run():
        yield message
