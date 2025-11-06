"""Enhanced game manager combining multi-game support with action visibility."""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import logging

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main_human import prepare_scenario, build_environment, create_agents
from sotopia.server import arun_one_episode
from sotopia.messages import AgentAction
from sotopia.agents.llm_agent import LLMAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GameSession:
    """Manages a single game session with full action visibility."""

    def __init__(
        self,
        game_id: str,
        human_players: list[str],
        broadcast_callback: Callable[[dict], Any],
    ):
        self.game_id = game_id
        self.human_players = set(human_players)
        self.broadcast_callback = broadcast_callback
        self.status = "initializing"
        self.events = []
        self.all_player_names = []
        self.role_assignments = {}
        self.input_queues: Dict[str, asyncio.Queue] = {}
        self.waiting_for_input: Dict[str, bool] = {}
        self.env = None
        self.agents = {}

    async def broadcast_event(self, event: dict):
        """Broadcast event to all connected clients."""
        event["timestamp"] = datetime.now().isoformat()
        event["game_id"] = self.game_id
        self.events.append(event)
        
        logger.info(f"📣 Broadcasting: {event.get('type')} - {event.get('content', '')[:50]}")
        
        if self.broadcast_callback:
            try:
                result = self.broadcast_callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

    async def start_game(self):
        """Start game with auto-play and action visibility."""
        try:
            self.status = "playing"
            await self.broadcast_event({
                "type": "phase",
                "content": "🎮 Game starting...",
            })
            
            logger.info(f"🎮 Starting game {self.game_id}")
            await self._run_game()
            
        except Exception as e:
            logger.error(f"Game {self.game_id} error: {e}", exc_info=True)
            self.status = "error"
            await self.broadcast_event({
                "type": "error",
                "content": f"Game error: {str(e)}",
            })

    async def _run_game(self):
        """Run game with enhanced action visibility."""
        # Prepare scenario
        logger.info("Preparing scenario...")
        env_profile, agent_profiles, role_assignments = prepare_scenario()
        self.role_assignments = role_assignments
        self.all_player_names = [f"{p.first_name} {p.last_name}" for p in agent_profiles]
        
        logger.info(f"✅ Players: {self.all_player_names}")
        
        # Build environment
        env = build_environment(env_profile, role_assignments, "gpt-4o-mini")
        self.env = env
        
        # Create agents
        agent_model_list = ["gpt-4o-mini"] * len(agent_profiles)
        agents = create_agents(agent_profiles, env_profile, agent_model_list)
        self.agents = {name: agent for name, agent in zip(self.all_player_names, agents)}
        
        # Setup input queues for human players
        for player in self.human_players:
            self.input_queues[player] = asyncio.Queue()
            self.waiting_for_input[player] = False
        
        # Wrap agents with action visibility
        for i, (name, agent) in enumerate(zip(self.all_player_names, agents)):
            if name in self.human_players:
                agents[i] = self._wrap_human_agent(agent, name)
            else:
                agents[i] = self._wrap_ai_agent(agent, name)
        
        logger.info("🚀 Starting game episode...")
        
        # Run episode
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
        logger.info("🏁 Game finished!")
        
        if hasattr(env, '_winner_payload') and env._winner_payload:
            await self.broadcast_event({
                "type": "game_over",
                "content": f"🏁 {env._winner_payload.get('message', 'Game finished!')}",
            })

    def _wrap_ai_agent(self, agent: LLMAgent, name: str):
        """Wrap AI agent to capture and broadcast actions."""
        original_aact = agent.aact
        
        async def enhanced_aact(obs):
            action = await original_aact(obs)
            await self._broadcast_action(name, action)
            return action
        
        agent.aact = enhanced_aact
        return agent

    def _wrap_human_agent(self, agent: LLMAgent, name: str):
        """Wrap human agent to use queue-based input."""
        original_aact = agent.aact
        
        async def human_aact(obs):
            await self._parse_and_broadcast_obs(obs, name)
            
            available_actions = getattr(obs, "available_actions", ["none"])
            
            if available_actions != ["none"]:
                self.waiting_for_input[name] = True
                await self.broadcast_event({
                    "type": "input_request",
                    "content": f"⏰ {name}'s turn!",
                    "player": name,
                    "available_actions": available_actions,
                })
                
                logger.info(f"⏳ Waiting for {name} input...")
                input_data = await self.input_queues[name].get()
                self.waiting_for_input[name] = False
                
                action = AgentAction(
                    action_type=input_data.get("action_type", "none"),
                    argument=input_data.get("argument", ""),
                )
                
                await self._broadcast_action(name, action)
                return action
            else:
                return AgentAction(action_type="none", argument="")
        
        agent.aact = human_aact
        return agent

    async def _broadcast_action(self, player: str, action: AgentAction):
        """Broadcast player action with emojis and context."""
        if action.action_type == "speak" and action.argument.strip():
            await self.broadcast_event({
                "type": "speech",
                "speaker": player,
                "content": action.argument,
            })
        
        elif action.action_type == "action" and action.argument.strip():
            arg_lower = action.argument.lower()
            
            if "kill" in arg_lower:
                target = self._extract_name(action.argument)
                await self.broadcast_event({
                    "type": "action",
                    "content": f"🗡️ {player} (Werewolf) targeted {target} for elimination",
                })
            elif "inspect" in arg_lower:
                target = self._extract_name(action.argument)
                await self.broadcast_event({
                    "type": "action",
                    "content": f"🔮 {player} (Seer) inspected {target}",
                })
            elif "save" in arg_lower:
                target = self._extract_name(action.argument)
                await self.broadcast_event({
                    "type": "action",
                    "content": f"💊 {player} (Witch) used save potion on {target}",
                })
            elif "poison" in arg_lower:
                target = self._extract_name(action.argument)
                await self.broadcast_event({
                    "type": "action",
                    "content": f"☠️ {player} (Witch) poisoned {target}",
                })
            elif "vote" in arg_lower:
                target = self._extract_name(action.argument)
                await self.broadcast_event({
                    "type": "vote",
                    "content": f"🗳️ {player} voted for {target}",
                })

    async def _parse_and_broadcast_obs(self, obs, player: str):
        """Parse observation and broadcast relevant events."""
        if not hasattr(obs, "to_natural_language"):
            return
        
        obs_text = obs.to_natural_language()
        
        for line in obs_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("Scenario:") or line.startswith("You are"):
                continue
            
            if "Phase" in line and "begins:" in line:
                phase_name = line.split("'")[1] if "'" in line else "unknown"
                context = self._get_phase_context(phase_name)
                await self.broadcast_event({
                    "type": "phase",
                    "content": f"{line}\n{context}" if context else line,
                })
            
            elif ("died" in line or "dead" in line or "executed" in line):
                await self.broadcast_event({
                    "type": "death",
                    "content": line.replace("[God]", "").strip(),
                })
            
            elif " said: " in line and player not in line:
                parts = line.split(" said: ")
                if len(parts) == 2:
                    await self.broadcast_event({
                        "type": "speech",
                        "speaker": parts[0].strip(),
                        "content": parts[1].strip('"'),
                    })

    def _get_phase_context(self, phase_name: str) -> str:
        contexts = {
            "night_werewolves": "🌙 Werewolves secretly choose their victim",
            "night_seer": "🔮 Seer investigates one player",
            "night_witch": "🧪 Witch can save or poison someone",
            "dawn_report": "☀️ Results of the night revealed",
            "day_discussion": "💬 Everyone discusses who might be a werewolf",
            "day_vote": "🗳️ Time to vote someone out",
        }
        return contexts.get(phase_name, "")

    def _extract_name(self, text: str) -> str:
        text_lower = text.lower()
        for name in self.all_player_names:
            if name.lower() in text_lower:
                return name
            first_name = name.split()[0].lower()
            if first_name in text_lower:
                return name
        return "unknown"

    async def submit_action(self, player: str, action: Dict[str, Any]):
        """Submit action for a human player."""
        if player not in self.input_queues:
            return {"error": "Not a human player"}
        
        if not self.waiting_for_input.get(player, False):
            return {"error": "Not waiting for input"}
        
        await self.input_queues[player].put(action)
        return {"status": "received"}


class GameManager:
    """Manages multiple concurrent game sessions."""

    def __init__(self):
        self.games: Dict[str, GameSession] = {}

    async def create_game(
        self,
        game_id: str,
        human_players: list[str],
        broadcast_callback: Callable[[dict], Any],
    ) -> GameSession:
        """Create and start a new game."""
        game = GameSession(game_id, human_players, broadcast_callback)
        self.games[game_id] = game
        
        # Start game in background
        asyncio.create_task(game.start_game())
        
        return game

    def get_game(self, game_id: str) -> Optional[GameSession]:
        return self.games.get(game_id)

    def get_game_state(self, game_id: str) -> Optional[Dict[str, Any]]:
        game = self.get_game(game_id)
        if not game:
            return None
        
        return {
            "game_id": game_id,
            "status": game.status,
            "players": game.all_player_names,
            "events": game.events[-50:],
            "waiting_for": [p for p, waiting in game.waiting_for_input.items() if waiting],
        }