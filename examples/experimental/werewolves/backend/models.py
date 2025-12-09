"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class CreateGameRequest(BaseModel):
    """Request to create a new game."""

    player_name: str = Field(..., min_length=1, max_length=50)


class CreateGameResponse(BaseModel):
    """Response with game ID."""

    game_id: str
    player_role: str
    all_players: List[str]


class PlayerAction(BaseModel):
    """Player action submitted during game."""

    action_type: str = Field(..., description="Type of action: speak, action, none")
    argument: str = Field(
        default="", description="Action argument (e.g., message text, vote target)"
    )
    timestamp: Optional[str] = None


class GameStateResponse(BaseModel):
    """Current game state snapshot."""

    game_id: str
    status: str  # "lobby", "playing", "finished"
    phase: Optional[str] = None
    players: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    your_turn: bool = False
    available_actions: List[str] = []


class GameEvent(BaseModel):
    """Real-time game event pushed via WebSocket."""

    type: str = Field(
        ..., description="Event type: phase_change, speak, action, death, game_over"
    )
    timestamp: str
    speaker: Optional[str] = None
    content: str
    event_class: Optional[str] = Field(
        None, description="CSS class: phase, speak, action, death"
    )


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
