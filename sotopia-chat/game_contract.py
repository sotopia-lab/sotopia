"""Shared schemas for social-game adapters.

These dataclasses define the minimum REST contract expected by the arena UI so
that future games can implement consistent endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence


@dataclass
class ActionMessage:
    actor: str
    action_type: str
    argument: str
    recorded_at: float


@dataclass
class PhaseLog:
    phase: str
    public: List[str] = field(default_factory=list)
    private: dict[str, Sequence[str]] = field(default_factory=dict)
    actions: List[ActionMessage] = field(default_factory=list)


@dataclass
class PlayerSummary:
    id: str
    display_name: str
    role: str
    team: str
    is_alive: bool


@dataclass
class GameSessionEnvelope:
    session_id: str
    players: List[PlayerSummary]
    phase: str
    available_actions: Sequence[str]
    log: Sequence[PhaseLog]
    status: str = "active"
    metadata: dict[str, Any] | None = None


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to dictionaries."""
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {key: to_dict(getattr(obj, key)) for key in obj.__dataclass_fields__}
    return obj
