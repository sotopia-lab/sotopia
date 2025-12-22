"""WebSocket connection manager for real-time game events."""

from typing import Dict, Set
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for game rooms."""

    def __init__(self):
        # game_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, game_id: str):
        """Accept a new WebSocket connection for a game."""
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = set()
        self.active_connections[game_id].add(websocket)
        logger.info(
            f"Client connected to game {game_id}. Total: {len(self.active_connections[game_id])}"
        )

    def disconnect(self, websocket: WebSocket, game_id: str):
        """Remove a WebSocket connection."""
        if game_id in self.active_connections:
            self.active_connections[game_id].discard(websocket)
            logger.info(
                f"Client disconnected from game {game_id}. Remaining: {len(self.active_connections[game_id])}"
            )

            # Clean up empty game rooms
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")

    async def broadcast(self, message: dict, game_id: str):
        """Broadcast a message to all clients in a game room."""
        if game_id not in self.active_connections:
            logger.warning(f"No connections for game {game_id}")
            return

        dead_connections = []
        for connection in list(self.active_connections[game_id]):
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send to connection: {e}")
                dead_connections.append(connection)

        # Clean up dead connections
        for connection in dead_connections:
            self.disconnect(connection, game_id)

    def get_connection_count(self, game_id: str) -> int:
        """Get the number of active connections for a game."""
        return len(self.active_connections.get(game_id, set()))
