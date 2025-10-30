"""
Werewolf Game FastAPI Server
Production-ready backend for multi-agent werewolf game
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Game imports
from sotopia.envs.social_game import SocialGameEnv
from sotopia.messages import AgentAction, Observation
from sotopia.database import EnvironmentProfile, AgentProfile
from sotopia.database.persistent_profile import RelationshipType
from sotopia.agents.llm_agent import LLMAgent

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE = Path(__file__).parent
RULEBOOK_PATH = BASE / "game_rules.json"
ROLEACTIONS_PATH = BASE / "role_actions.json"
ROSTER_PATH = BASE / "roster.json"

app = FastAPI(title="Werewolf Game Server", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GAME STATE MANAGEMENT
# ============================================================================

class GameState:
    """Singleton game state manager"""
    def __init__(self):
        self.env: SocialGameEnv | None = None
        self.agents: Dict[str, LLMAgent] = {}
        self.role_assignments: Dict[str, str] = {}
        self.human_players: set[str] = set()
        self.game_active = False
        self.pending_actions: Dict[str, AgentAction] = {}
        self.websocket_clients: List[WebSocket] = []
        self.all_events: List[str] = []
        self.game_loop_task = None
        
    def initialize_game(self, human_player_names: List[str] = None):
        """Setup new game with optional human players"""
        # Load roster
        roster_data = json.loads(ROSTER_PATH.read_text())
        self.role_assignments = {
            f"{p['first_name']} {p['last_name']}": p["role"]
            for p in roster_data["players"]
        }
        
        # Track human players
        self.human_players = set(human_player_names or [])
        
        # Create environment profile
        agent_goals = []
        for p in roster_data["players"]:
            goal = p.get("goal", "Survive and win the game")
            agent_goals.append(goal)
        
        env_profile = EnvironmentProfile(
            scenario=roster_data["scenario"],
            relationship=RelationshipType.acquaintance,
            agent_goals=agent_goals,
        )
        
        self.env = SocialGameEnv(
            env_profile=env_profile,
            rulebook_path=str(RULEBOOK_PATH),
            actions_path=str(ROLEACTIONS_PATH),
            role_assignments=self.role_assignments,
        )
        
        # Create agents
        self.agents = {}
        for player in roster_data["players"]:
            name = f"{player['first_name']} {player['last_name']}"
            profile = AgentProfile(
                first_name=player["first_name"],
                last_name=player["last_name"],
                age=player.get("age", 30),
                occupation=self.role_assignments[name],
                public_info=f"Playing as {self.role_assignments[name]}",
                gender="",
                gender_pronoun=player.get("pronouns", "they/them"),
                personality_and_values="",
                decision_making_style="",
                secret=player.get("secret", ""),
            )
            self.agents[name] = LLMAgent(agent_profile=profile, model_name="gpt-4o-mini")
        
        # Reset environment with ALL agents
        self.env.reset(agents=self.agents)
        
        self.game_active = True
        self.pending_actions.clear()
        self.all_events = []
        
        # Capture initial phase message
        if hasattr(self.env, '_last_events'):
            self.all_events.extend(self.env._last_events.public)
        
        print(f"✅ Game initialized with {len(self.agents)} players")
        print(f"Human players: {self.human_players or 'None (all AI)'}")
        print(f"Phase: {self.env.game_rulebook.current_phase}")
        
        return {
            "status": "initialized",
            "players": list(self.role_assignments.keys()),
            "phase": self.env.game_rulebook.current_phase,
            "human_players": list(self.human_players),
        }
    
    async def broadcast_update(self, update_data: dict):
        """Send updates to all connected websocket clients"""
        disconnected = []
        for ws in self.websocket_clients:
            try:
                await ws.send_json(update_data)
            except:
                disconnected.append(ws)
        
        for ws in disconnected:
            self.websocket_clients.remove(ws)
    
    async def auto_play_loop(self):
        """Automatically play AI turns"""
        while self.game_active and self.env:
            try:
                rb = self.env.game_rulebook
                active_players = rb.active_agents_for_phase()
                
                # Check if any active player is human and waiting for input
                human_waiting = any(p in self.human_players for p in active_players)
                
                if not human_waiting:
                    # All active players are AI - auto-play
                    print(f"🤖 Auto-playing for AI players: {active_players}")
                    await self.execute_turn_with_ai()
                    await asyncio.sleep(3)  # Delay between turns for readability
                else:
                    # Human player's turn - wait for input
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"❌ Auto-play error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(2)
    
    async def execute_turn_with_ai(self):
        """Execute turn with AI agents acting automatically"""
        if not self.env:
            return
        
        rb = self.env.game_rulebook
        active_players = rb.active_agents_for_phase()
        current_phase = rb.current_phase
        
        # Get AI actions
        actions = {}
        for player in active_players:
            if player in self.agents and player not in self.human_players:
                try:
                    obs_dict = self.env._create_blank_observations()
                    obs = obs_dict.get(player)
                    if obs:
                        action = await self.agents[player].aact(obs)
                        actions[player] = action
                        print(f"  🤖 {player}: {action.action_type} '{action.argument}'")
                        
                        # SHOW ALL ACTIONS TO SPECTATORS
                        if action.action_type == "speak" and action.argument.strip():
                            speech_event = f'{player} said: "{action.argument}"'
                            self.all_events.append(speech_event)
                        elif action.action_type == "action" and action.argument.strip():
                            # Show action attempts with context
                            action_text = action.argument.lower()
                            
                            if "kill" in action_text:
                                target = self._extract_name(action.argument, active_players)
                                event = f"🗡️ {player} (Werewolf) targeted {target or 'someone'} for elimination"
                                self.all_events.append(event)
                            elif "inspect" in action_text:
                                target = self._extract_name(action.argument, rb.agent_states.keys())
                                event = f"🔮 {player} (Seer) inspected {target or 'someone'}"
                                self.all_events.append(event)
                            elif "save" in action_text:
                                target = self._extract_name(action.argument, rb.agent_states.keys())
                                event = f"💊 {player} (Witch) used save potion on {target or 'the target'}"
                                self.all_events.append(event)
                            elif "poison" in action_text:
                                target = self._extract_name(action.argument, rb.agent_states.keys())
                                event = f"☠️ {player} (Witch) poisoned {target or 'someone'}"
                                self.all_events.append(event)
                            elif "vote" in action_text:
                                target = self._extract_name(action.argument, rb.agent_states.keys())
                                event = f"🗳️ {player} voted for {target or 'no one'}"
                                self.all_events.append(event)
                            else:
                                # Generic action display
                                event = f"⚡ {player} took action: {action.argument}"
                                self.all_events.append(event)
                                
                except Exception as e:
                    print(f"  ❌ {player} action failed: {e}")
                    actions[player] = AgentAction(action_type="none", argument="")
        
        if actions:
            # Execute step
            obs, rewards, terminated, truncations, info = await self.env.astep(actions)
            
            # Capture system events
            if hasattr(self.env, '_last_events'):
                new_events = self.env._last_events.public
                # Filter out speech (already captured above) to avoid duplicates
                system_events = [e for e in new_events if ' said: ' not in e]
                self.all_events.extend(system_events)
                print(f"📝 New system events: {len(system_events)}")
            
            # Check for game end
            if any(terminated.values()):
                self.game_active = False
                print("🏁 Game ended!")
            
            # Broadcast update
            await self.broadcast_update({
                "event": "turn_executed",
                "phase": rb.current_phase,
            })
    
    def _extract_name(self, text: str, possible_names: list) -> str:
        """Extract player name from action text"""
        text_lower = text.lower()
        for name in possible_names:
            if name.lower() in text_lower:
                return name
        # Try first names
        for name in possible_names:
            first_name = name.split()[0].lower()
            if first_name in text_lower:
                return name
        return "unknown"

game_state = GameState()

# ============================================================================
# REQUEST MODELS
# ============================================================================

class InitGameRequest(BaseModel):
    human_players: List[str] = []
    auto_play: bool = True

class PlayerActionRequest(BaseModel):
    player_name: str
    action_type: str
    argument: str = ""

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Health check"""
    return {
        "status": "online",
        "game_active": game_state.game_active,
        "phase": game_state.env.game_rulebook.current_phase if game_state.env else None,
    }

@app.post("/game/init")
async def init_game(request: InitGameRequest, background_tasks: BackgroundTasks):
    """Initialize a new game"""
    try:
        result = game_state.initialize_game(request.human_players)
        
        # Start auto-play loop if enabled
        if request.auto_play:
            game_state.game_loop_task = asyncio.create_task(game_state.auto_play_loop())
            print("🎮 Auto-play loop started")
        
        await game_state.broadcast_update({"event": "game_initialized", "data": result})
        return result
    except Exception as e:
        print(f"❌ Init error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/game/state")
def get_game_state():
    """Get current game state"""
    if not game_state.env:
        return {"status": "no_game"}
    
    rb = game_state.env.game_rulebook
    
    return {
        "status": "active" if game_state.game_active else "ended",
        "phase": rb.current_phase,
        "phase_meta": rb.current_phase_metadata(),
        "alive_players": rb.alive_agents(),
        "active_players": rb.active_agents_for_phase(),
        "recent_events": {
            "public": game_state.all_events[-20:],  # Last 20 events
        },
        "winner": game_state.env._winner_payload if hasattr(game_state.env, '_winner_payload') else None,
    }

@app.get("/game/state/{player_name}")
def get_player_state(player_name: str):
    """Get player-specific state"""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="No active game")
    
    rb = game_state.env.game_rulebook
    
    if player_name not in rb.agent_states:
        available_players = list(rb.agent_states.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Player '{player_name}' not found. Available: {available_players}"
        )
    
    agent_state = rb.agent_states[player_name]
    is_active = player_name in rb.active_agents_for_phase()
    available_actions = rb.available_actions(player_name) if is_active else ["none"]
    
    return {
        "player_name": player_name,
        "role": agent_state.role,
        "team": agent_state.team,
        "alive": agent_state.alive,
        "is_active": is_active,
        "available_actions": available_actions,
        "phase": rb.current_phase,
        "phase_meta": rb.current_phase_metadata(),
    }

@app.post("/game/action")
async def submit_action(action: PlayerActionRequest):
    """Submit a human player action"""
    if not game_state.env or not game_state.game_active:
        raise HTTPException(status_code=400, detail="No active game")
    
    rb = game_state.env.game_rulebook
    
    if action.player_name not in rb.active_agents_for_phase():
        raise HTTPException(status_code=403, detail="Not your turn")
    
    print(f"📝 Human action: {action.player_name} -> {action.action_type} '{action.argument}'")
    
    # Capture human speech immediately
    if action.action_type == "speak" and action.argument.strip():
        speech_event = f'{action.player_name} said: "{action.argument}"'
        game_state.all_events.append(speech_event)
    
    # Execute turn with human action
    agent_action = AgentAction(action_type=action.action_type, argument=action.argument)
    
    # Get AI actions for other active players
    active_players = rb.active_agents_for_phase()
    actions = {action.player_name: agent_action}
    
    for player in active_players:
        if player != action.player_name and player in game_state.agents:
            try:
                obs_dict = game_state.env._create_blank_observations()
                obs = obs_dict.get(player)
                if obs:
                    ai_action = await game_state.agents[player].aact(obs)
                    actions[player] = ai_action
                    
                    # Capture AI speech in this turn
                    if ai_action.action_type == "speak" and ai_action.argument.strip():
                        speech_event = f'{player} said: "{ai_action.argument}"'
                        game_state.all_events.append(speech_event)
            except Exception as e:
                print(f"❌ AI action failed for {player}: {e}")
                actions[player] = AgentAction(action_type="none", argument="")
    
    # Execute step
    obs, rewards, terminated, truncations, info = await game_state.env.astep(actions)
    
    # Capture system events (filter out speech to avoid duplicates)
    if hasattr(game_state.env, '_last_events'):
        system_events = [e for e in game_state.env._last_events.public if ' said: ' not in e]
        game_state.all_events.extend(system_events)
    
    if any(terminated.values()):
        game_state.game_active = False
    
    await game_state.broadcast_update({
        "event": "turn_executed",
        "phase": rb.current_phase,
    })
    
    return {"status": "action_received"}

@app.get("/game/roster")
def get_roster():
    """Get available players from roster"""
    roster_data = json.loads(ROSTER_PATH.read_text())
    return {
        "players": [
            {
                "name": f"{p['first_name']} {p['last_name']}",
                "role": p["role"],
            }
            for p in roster_data["players"]
        ]
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time game updates"""
    await websocket.accept()
    game_state.websocket_clients.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        game_state.websocket_clients.remove(websocket)

@app.post("/game/reset")
async def reset_game():
    """Reset game state"""
    if game_state.game_loop_task:
        game_state.game_loop_task.cancel()
    
    game_state.env = None
    game_state.agents.clear()
    game_state.role_assignments.clear()
    game_state.human_players.clear()
    game_state.game_active = False
    game_state.pending_actions.clear()
    game_state.all_events.clear()
    
    await game_state.broadcast_update({"event": "game_reset"})
    
    return {"status": "reset"}

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)