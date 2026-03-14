"""User and game history models for persistent storage."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel, Field
from redis_om import JsonModel

from .storage_backend import is_local_backend
from .base_models import add_local_storage_methods


class BaseUserProfile(BaseModel):
    """User profile for authentication and game tracking."""
    
    pk: str = Field(default="")
    username: str = Field(..., index=True, description="Unique username")
    email: str = Field(..., index=True, description="User email address")
    password_hash: str = Field(default="", description="Bcrypt password hash (empty for OAuth-only users)")
    auth_provider: str = Field(default="local", description="Primary auth provider: local, google, github, discord")
    elo_rating: int = Field(default=1000, description="ELO rating for matchmaking")
    games_played: int = Field(default=0, description="Total games played")
    games_won: int = Field(default=0, description="Total games won")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Account creation timestamp"
    )
    last_login: str = Field(
        default="",
        description="Last login timestamp"
    )
    is_active: bool = Field(default=True, description="Whether account is active")
    avatar_url: str = Field(default="", description="Profile picture URL from OAuth provider")


# Define UserProfile conditionally based on backend
if TYPE_CHECKING:
    class UserProfile(BaseUserProfile, JsonModel):
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)
elif is_local_backend():
    class UserProfile(BaseUserProfile):
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)
    # Add local storage methods
    add_local_storage_methods(UserProfile)
else:
    class UserProfile(BaseUserProfile, JsonModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)


class BaseGameResult(BaseModel):
    """Record of a completed game."""
    
    pk: str = Field(default="")
    game_type: str = Field(default="werewolf", index=True, description="Type of game played")
    player_ids: List[str] = Field(default_factory=list, description="List of user PKs who played")
    player_names: List[str] = Field(default_factory=list, description="Display names in game")
    player_roles: List[str] = Field(default_factory=list, description="Roles assigned to players")
    winner_team: str = Field(default="", index=True, description="Winning team (e.g., 'Werewolves', 'Villagers')")
    winner_player_ids: List[str] = Field(default_factory=list, description="PKs of winning players")
    turns_played: int = Field(default=0, description="Number of turns in the game")
    duration_seconds: int = Field(default=0, description="Game duration in seconds")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="When the game was played"
    )
    # Optional: store game log or key events
    summary: str = Field(default="", description="Brief summary of key events")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional game data")


# Define GameResult conditionally based on backend
if TYPE_CHECKING:
    class GameResult(BaseGameResult, JsonModel):
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)
elif is_local_backend():
    class GameResult(BaseGameResult):
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)
    # Add local storage methods
    add_local_storage_methods(GameResult)
else:
    class GameResult(BaseGameResult, JsonModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)


# OAuth Account Model - Links OAuth provider accounts to users

class BaseOAuthAccount(BaseModel):
    """OAuth account linking a provider identity to a user."""
    
    pk: str = Field(default="")
    user_pk: str = Field(..., index=True, description="FK to UserProfile.pk")
    provider: str = Field(..., index=True, description="OAuth provider: google, github, discord")
    provider_user_id: str = Field(..., index=True, description="User ID from the OAuth provider")
    provider_email: str = Field(default="", description="Email from OAuth provider")
    access_token: str = Field(default="", description="OAuth access token (encrypted in production)")
    refresh_token: str = Field(default="", description="OAuth refresh token (encrypted in production)")
    token_expires_at: str = Field(default="", description="Token expiration timestamp")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="When this OAuth link was created"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Last token refresh timestamp"
    )


# Define OAuthAccount conditionally based on backend
if TYPE_CHECKING:
    class OAuthAccount(BaseOAuthAccount, JsonModel):
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)
elif is_local_backend():
    class OAuthAccount(BaseOAuthAccount):
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)
    # Add local storage methods
    add_local_storage_methods(OAuthAccount)
else:
    class OAuthAccount(BaseOAuthAccount, JsonModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)


# ELO History Model - Tracks rating changes over time

class BaseEloHistory(BaseModel):
    """Record of an ELO rating change."""
    
    pk: str = Field(default="")
    user_pk: str = Field(..., index=True, description="FK to UserProfile.pk")
    game_pk: str = Field(default="", index=True, description="FK to GameResult.pk that caused this change")
    elo_before: int = Field(..., description="ELO before the game")
    elo_after: int = Field(..., description="ELO after the game")
    elo_change: int = Field(..., description="ELO change amount (can be negative)")
    game_type: str = Field(default="werewolf", description="Type of game")
    opponent_rating: int = Field(default=0, description="Average opponent rating")
    won: bool = Field(default=False, description="Whether the user won")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="When this change occurred"
    )


# Define EloHistory conditionally based on backend
if TYPE_CHECKING:
    class EloHistory(BaseEloHistory, JsonModel):
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)
elif is_local_backend():
    class EloHistory(BaseEloHistory):
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)
    # Add local storage methods
    add_local_storage_methods(EloHistory)
else:
    class EloHistory(BaseEloHistory, JsonModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)
