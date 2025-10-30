"""FastAPI backend for Werewolf game."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import uuid
import logging
from dotenv import load_dotenv

from models import (
    CreateGameRequest,
    CreateGameResponse,
    PlayerAction,
    GameStateResponse,
)
from ws_manager import ConnectionManager
from game_manager import GameManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Werewolf Game API",
    description="Backend for Duskmire Werewolves social deduction game",
    version="1.0.0",
)

# CORS configuration
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,https://*.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
ws_manager = ConnectionManager()
game_manager = GameManager()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "werewolf-game-api", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "active_games": len(game_manager.games),
        "websocket_connections": sum(
            ws_manager.get_connection_count(gid) for gid in game_manager.games.keys()
        ),
    }


@app.post("/api/game/create", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest):
    """Create a new game session."""
    try:
        game_id = str(uuid.uuid4())
        logger.info(f"Creating game {game_id} for player {request.player_name}")

        # Define broadcast callback
        async def broadcast(event: dict):
            await ws_manager.broadcast(event, game_id)

        # Create game
        game = await game_manager.create_game(
            game_id=game_id,
            player_name=request.player_name,
            broadcast_callback=broadcast,
        )

        return CreateGameResponse(
            game_id=game_id,
            player_role=game.player_role or "Unknown",
            all_players=game.all_player_names,
        )

    except Exception as e:
        logger.error(f"Failed to create game: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/game/{game_id}/state", response_model=GameStateResponse)
async def get_game_state(game_id: str):
    """Get current game state."""
    state = game_manager.get_game_state(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")

    return GameStateResponse(**state)


@app.post("/api/game/{game_id}/action")
async def submit_action(game_id: str, action: PlayerAction):
    """Submit a player action."""
    game = game_manager.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    try:
        result = await game.submit_action(action.dict())

        # Broadcast action acknowledgment
        await ws_manager.broadcast(
            {
                "type": "action_received",
                "content": f"Action received: {action.action_type}",
                "timestamp": action.timestamp,
            },
            game_id,
        )

        return result

    except Exception as e:
        logger.error(f"Failed to submit action: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for real-time game events."""
    await ws_manager.connect(websocket, game_id)

    try:
        # Send initial connection confirmation
        await ws_manager.send_personal_message(
            {
                "type": "connected",
                "content": f"Connected to game {game_id}",
                "game_id": game_id,
            },
            websocket,
        )

        # Keep connection alive and listen for messages
        while True:
            # Client can send messages (e.g., ping/pong)
            data = await websocket.receive_text()
            logger.debug(f"Received WebSocket message: {data}")

            # Echo back (optional)
            # await ws_manager.send_personal_message(
            #     {"type": "echo", "content": data},
            #     websocket
            # )

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, game_id)
        logger.info(f"WebSocket disconnected from game {game_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        ws_manager.disconnect(websocket, game_id)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")
