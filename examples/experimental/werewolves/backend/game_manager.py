"""Game manager that wraps the existing werewolf game logic."""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import logging

# Add parent directory to path to import existing game code
sys.path.insert(0, str(Path(__file__).parent.parent))

from main_human import (
    prepare_scenario,
    build_environment,
    create_agents,
    PlayerViewHumanAgent,
)
from sotopia.server import arun_one_episode

logger = logging.getLogger(__name__)


class GameSession:
    """Manages a single game session."""

    def __init__(
        self,
        game_id: str,
        player_name: str,
        broadcast_callback: Callable[[dict], None],
    ):
        self.game_id = game_id
        self.player_name = player_name
        self.broadcast_callback = broadcast_callback
        self.status = "initializing"  # initializing, playing, finished
        self.events = []
        self.player_role = None
        self.all_player_names = []
        self.current_phase = None
        self.input_queue = asyncio.Queue()
        self.waiting_for_input = False
        self.available_actions = []

    async def broadcast_event(self, event: dict):
        """Broadcast an event to all connected clients."""
        event["timestamp"] = datetime.now().isoformat()
        event["game_id"] = self.game_id
        self.events.append(event)

        # Call the callback (will be the WebSocket broadcast)
        if self.broadcast_callback:
            await self.broadcast_callback(event)

    async def start_game(self):
        """Start the game loop in background."""
        try:
            self.status = "playing"
            await self.broadcast_event(
                {
                    "type": "phase_change",
                    "content": "Game starting...",
                    "event_class": "phase",
                }
            )

            # Run the game
            await self._run_game()

        except Exception as e:
            logger.error(f"Game {self.game_id} crashed: {e}", exc_info=True)
            self.status = "error"
            await self.broadcast_event(
                {
                    "type": "error",
                    "content": f"Game error: {str(e)}",
                    "event_class": "phase",
                }
            )

    async def _run_game(self):
        """Run the game loop (adapted from main_human.py)."""
        try:
            # Prepare scenario
            logger.info(f"Game {self.game_id}: Preparing scenario...")
            env_profile, agent_profiles, role_assignments = prepare_scenario()
            logger.info(f"Game {self.game_id}: Scenario prepared successfully")
        except Exception as e:
            logger.error(
                f"Game {self.game_id}: Failed to prepare scenario: {e}", exc_info=True
            )
            raise

        # Store player info
        self.all_player_names = [
            f"{p.first_name} {p.last_name}" for p in agent_profiles
        ]
        self.player_role = list(role_assignments.values())[0]

        await self.broadcast_event(
            {
                "type": "game_info",
                "content": f"You are playing as {self.player_name}. Role: {self.player_role}",
                "event_class": "phase",
            }
        )

        # Build environment
        env_model = "gpt-4o-mini"
        agent_model_list = ["gpt-4o-mini"] * len(agent_profiles)

        env = build_environment(env_profile, role_assignments, env_model)
        agents = create_agents(agent_profiles, env_profile, agent_model_list)

        # Create custom PlayerView that broadcasts via WebSocket
        class WebSocketPlayerView:
            """Player view that broadcasts to WebSocket instead of writing HTML."""

            def __init__(self, game_session):
                self.game_session = game_session
                self.events = []

            async def add_event(self, event_type: str, content: str, speaker: str = ""):
                event = {
                    "type": event_type,
                    "content": content,
                    "speaker": speaker,
                    "event_class": event_type,
                }
                self.events.append(event)
                await self.game_session.broadcast_event(event)

            def enable_input(self, available_actions):
                self.game_session.waiting_for_input = True
                self.game_session.available_actions = available_actions
                asyncio.create_task(
                    self.game_session.broadcast_event(
                        {
                            "type": "input_request",
                            "content": "Your turn!",
                            "available_actions": available_actions,
                            "event_class": "phase",
                        }
                    )
                )

            async def wait_for_input(self):
                """Wait for player input from the queue."""
                data = await self.game_session.input_queue.get()
                self.game_session.waiting_for_input = False
                self.game_session.available_actions = []
                return data

        player_view = WebSocketPlayerView(self)

        # Replace first agent with human player
        class WebSocketHumanAgent(PlayerViewHumanAgent):
            """Human agent that uses WebSocket player view with full observation parsing."""

            async def aact(self, obs):
                from sotopia.messages import AgentAction

                self.recv_message("Environment", obs)

                # Parse observation to extract player-visible information
                if hasattr(obs, "to_natural_language"):
                    obs_text = obs.to_natural_language()

                    # Parse line by line to properly categorize events
                    lines = obs_text.split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        # Skip system prompts and metadata
                        if (
                            line.startswith("Scenario:")
                            or " goal:" in line
                            or line.startswith("GAME RULES:")
                            or line.startswith("You are ")
                            or line.startswith("Primary directives:")
                            or line.startswith("Role guidance:")
                            or line.startswith("System constraints:")
                        ):
                            continue

                        # Check for game over
                        if (
                            "GAME OVER" in line.upper()
                            or (
                                "Winner:" in line
                                and ("Werewolves" in line or "Villagers" in line)
                            )
                            or ("[God] Werewolves win;" in line)
                            or ("[God] Villagers win;" in line)
                        ):
                            clean_line = line.replace("[God]", "").strip()
                            await player_view.add_event("phase", f"🎮 {clean_line}")

                        # Check for death announcements
                        elif (
                            "was found dead" in line
                            or " died" in line
                            or "was killed" in line
                        ) and (" said:" not in line and " says:" not in line):
                            clean_death = line.replace("[God]", "").strip()
                            if clean_death:  # Only add if not empty
                                await player_view.add_event(
                                    "death", f"💀 {clean_death}"
                                )

                        # Check for phase announcements
                        elif (
                            "Night phase begins" in line
                            or "Phase: 'night'" in line.lower()
                        ):
                            await player_view.add_event(
                                "phase", "🌙 Night phase begins"
                            )

                        elif (
                            "Day discussion starts" in line
                            or "Phase: 'day_discussion' begins" in line
                        ):
                            await player_view.add_event(
                                "phase", "☀️ Day breaks. Time to discuss!"
                            )

                        elif (
                            "Voting phase" in line
                            or "Phase: 'voting' begins" in line
                            or "Phase 'day_vote' begins" in line
                        ):
                            await player_view.add_event("phase", "🗳️ Voting phase")

                        elif (
                            "twilight_execution" in line or "Execution results" in line
                        ):
                            await player_view.add_event("phase", "⚖️ Execution results")

                        elif "Night returns" in line:
                            await player_view.add_event("phase", "🌙 Night returns")

                        # Check for player speech
                        elif (
                            " said:" in line or " says:" in line
                        ) and "[God]" not in line:
                            parts = line.split(
                                " said:" if " said:" in line else " says:"
                            )
                            if len(parts) == 2:
                                speaker = parts[0].strip()
                                message = parts[1].strip().strip('"')
                                await player_view.add_event("speak", message, speaker)

                        # Check for voting and eliminations
                        elif (
                            "voted for" in line
                            or "was executed" in line
                            or "was eliminated" in line
                            or "Votes are tallied" in line
                            or "Majority condemns" in line
                        ):
                            clean_action = line.replace("[God]", "").strip()
                            if "was executed" in clean_action:
                                await player_view.add_event(
                                    "death", f"⚖️ {clean_action}"
                                )
                            elif "Majority condemns" in clean_action:
                                await player_view.add_event(
                                    "action", f"🗳️ {clean_action}"
                                )
                            elif clean_action:
                                await player_view.add_event("action", clean_action)

                # Get available actions
                available_actions = getattr(obs, "available_actions", ["none"])

                if available_actions != ["none"]:
                    player_view.enable_input(available_actions)
                    logger.info(f"Waiting for {self.agent_name}'s input...")

                    # Wait for input from queue
                    input_data = await player_view.wait_for_input()
                    action_type = input_data.get("action_type", "none")
                    argument = input_data.get("argument", "")

                    return AgentAction(action_type=action_type, argument=argument)
                else:
                    return AgentAction(action_type="none", argument="")

        human_agent = WebSocketHumanAgent(
            agent_profile=agent_profiles[0],
            available_agent_names=self.all_player_names,
            player_view=player_view,
        )
        human_agent.goal = env_profile.agent_goals[0]
        agents[0] = human_agent

        # Run the episode
        await arun_one_episode(
            env=env,
            agent_list=agents,
            omniscient=False,
            script_like=False,
            json_in_script=False,
            tag=None,
            push_to_db=False,
        )

        # Game finished - check for winner
        self.status = "finished"

        # Check if there's a winner payload
        if hasattr(env, "_winner_payload") and env._winner_payload:
            winner = env._winner_payload.get("winner", "Unknown")
            reason = env._winner_payload.get("message", "")

            logger.info(f"Game {self.game_id}: Winner={winner}, Reason={reason}")

            await self.broadcast_event(
                {
                    "type": "game_over",
                    "content": f"🎮 Game Over! Winner: {winner}. {reason}",
                    "event_class": "phase",
                }
            )
        else:
            logger.warning(f"Game {self.game_id}: No winner payload found")
            await self.broadcast_event(
                {
                    "type": "game_over",
                    "content": "Game finished!",
                    "event_class": "phase",
                }
            )

    async def submit_action(self, action: Dict[str, Any]):
        """Submit a player action."""
        if not self.waiting_for_input:
            return {"error": "Not waiting for input"}

        # Put action in queue for game loop to process
        await self.input_queue.put(action)
        return {"status": "received"}


class GameManager:
    """Manages all active game sessions."""

    def __init__(self):
        self.games: Dict[str, GameSession] = {}

    async def create_game(
        self,
        game_id: str,
        player_name: str,
        broadcast_callback: Callable[[dict], None],
    ) -> GameSession:
        """Create and start a new game."""
        game = GameSession(game_id, player_name, broadcast_callback)
        self.games[game_id] = game

        # Start game in background
        asyncio.create_task(game.start_game())

        return game

    def get_game(self, game_id: str) -> Optional[GameSession]:
        """Get a game session by ID."""
        return self.games.get(game_id)

    def get_game_state(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get current game state."""
        game = self.get_game(game_id)
        if not game:
            return None

        return {
            "game_id": game_id,
            "status": game.status,
            "phase": game.current_phase,
            "players": game.all_player_names,
            "events": game.events[-50:],  # Last 50 events
            "your_turn": game.waiting_for_input,
            "available_actions": game.available_actions,
        }
