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
        # Prepare scenario
        env_profile, agent_profiles, role_assignments = prepare_scenario()

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
            """Human agent that uses WebSocket player view."""

            async def aact(self, obs):
                # Custom implementation that uses our WebSocket player view
                # (simplified version - you can copy full logic from main_human.py)
                from sotopia.messages import AgentAction

                self.recv_message("Environment", obs)

                # Parse observation and broadcast events
                if hasattr(obs, "to_natural_language"):
                    obs_text = obs.to_natural_language()
                    # Add parsing logic here similar to main_human.py
                    # For now, just broadcast the raw observation
                    await player_view.add_event("speak", obs_text[:200], "System")

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

        # Game finished
        self.status = "finished"
        await self.broadcast_event(
            {"type": "game_over", "content": "Game finished!", "event_class": "phase"}
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
