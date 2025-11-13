"""
Integrated Werewolf Game Server
Combines multi-game support with full action visibility
"""

import uuid
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path

from game_manager import GameManager

# ============================================================================
# CONFIGURATION
# ============================================================================

app = FastAPI(title="Werewolf Game Server", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Game manager
game_manager = GameManager()

# WebSocket connections per game
game_websockets: dict[str, list[WebSocket]] = {}

# ============================================================================
# REQUEST MODELS
# ============================================================================


class StartGameRequest(BaseModel):
    human_players: List[str] = []
    game_mode: str = "spectate"  # 'spectate' or 'play'


class SubmitActionRequest(BaseModel):
    game_id: str
    player_name: str
    action_type: str
    argument: str = ""


# ============================================================================
# WEBSOCKET BROADCAST
# ============================================================================


async def broadcast_to_game(game_id: str, event: dict):
    """Broadcast event to all clients watching this game."""
    if game_id not in game_websockets:
        return

    disconnected = []
    for ws in game_websockets[game_id]:
        try:
            await ws.send_json(event)
        except:
            disconnected.append(ws)

    # Remove disconnected clients
    for ws in disconnected:
        game_websockets[game_id].remove(ws)


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/")
def root():
    """Health check"""
    return {
        "status": "online",
        "active_games": len(game_manager.games),
    }


@app.post("/game/start")
async def start_game(request: StartGameRequest):
    """Start a new game session."""
    game_id = str(uuid.uuid4())
    game_websockets[game_id] = []

    human_players = request.human_players if request.game_mode == "play" else []

    # Create broadcast callback for this game
    async def broadcast_callback(event: dict):
        await broadcast_to_game(game_id, event)

    game = await game_manager.create_game(
        game_id=game_id,
        human_players=human_players,
        broadcast_callback=broadcast_callback,
    )

    return {
        "game_id": game_id,
        "status": "started",
        "human_players": human_players,
    }


@app.get("/game/{game_id}/state")
def get_game_state(game_id: str):
    """Get current game state."""
    state = game_manager.get_game_state(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")
    return state


@app.post("/game/action")
async def submit_action(request: SubmitActionRequest):
    """Submit a player action."""
    game = game_manager.get_game(request.game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    result = await game.submit_action(
        player=request.player_name,
        action={
            "action_type": request.action_type,
            "argument": request.argument,
        },
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/game/roster")
def get_roster():
    """Get available players from roster."""
    roster_path = Path(__file__).parent.parent / "roster.json"
    roster_data = json.loads(roster_path.read_text())
    return {
        "players": [
            {
                "name": f"{p['first_name']} {p['last_name']}",
                "role": p["role"],
            }
            for p in roster_data["players"]
        ]
    }


@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket for real-time game updates."""
    await websocket.accept()

    # Add to game's websocket list
    if game_id not in game_websockets:
        game_websockets[game_id] = []
    game_websockets[game_id].append(websocket)

    try:
        # Send current game state on connect
        state = game_manager.get_game_state(game_id)
        if state:
            await websocket.send_json(
                {
                    "type": "state_sync",
                    "data": state,
                }
            )

        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Echo for heartbeat
            await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        game_websockets[game_id].remove(websocket)


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
