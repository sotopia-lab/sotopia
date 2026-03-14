"""Profile API endpoints for user profiles, ELO history, and match history."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from sotopia.database import UserProfile, GameResult, EloHistory, OAuthAccount
from sotopia.database.storage_backend import get_storage_backend
from sotopia.api.elo import get_rank_progress
from sotopia.api.auth import require_auth, user_to_response, UserResponse


router = APIRouter(prefix="/profile", tags=["profile"])


# Response Models

class EloHistoryEntry(BaseModel):
    """Single ELO history entry."""
    game_pk: str
    elo_before: int
    elo_after: int
    elo_change: int
    game_type: str
    opponent_rating: int
    won: bool
    created_at: str


class EloHistoryResponse(BaseModel):
    """ELO history for a user."""
    entries: List[EloHistoryEntry]
    total: int
    current_elo: int
    peak_elo: int
    lowest_elo: int


class MatchHistoryEntry(BaseModel):
    """Single match history entry."""
    pk: str
    game_type: str
    player_role: str
    player_team: str
    winner_team: str
    won: bool
    elo_change: Optional[int] = None
    turns_played: int
    duration_seconds: int
    created_at: str
    players: List[Dict[str, Any]]


class MatchHistoryResponse(BaseModel):
    """Match history for a user."""
    matches: List[MatchHistoryEntry]
    total: int
    limit: int
    offset: int


class ProfileStatsResponse(BaseModel):
    """Detailed stats for a user profile."""
    user: UserResponse
    rank: Optional[int]
    total_players: int
    percentile: Optional[float]
    rank_tier: Dict[str, Any]
    win_streak: int
    best_win_streak: int
    recent_form: List[str]  # e.g., ["W", "W", "L", "W", "L"]
    role_stats: Dict[str, Dict[str, int]]  # e.g., {"Werewolf": {"played": 5, "won": 3}}
    linked_providers: List[str]


class UpdateProfileRequest(BaseModel):
    """Request for updating profile fields."""
    username: Optional[str] = Field(None, min_length=3, max_length=30)
    email: Optional[str] = None
    avatar_url: Optional[str] = None


# Helper Functions

def _calculate_win_streaks(elo_history: List[Dict[str, Any]]) -> tuple[int, int]:
    """Calculate current and best win streaks from ELO history.
    
    Args:
        elo_history: List of ELO history entries, sorted by created_at ascending
        
    Returns:
        Tuple of (current_streak, best_streak)
    """
    if not elo_history:
        return (0, 0)
    
    current_streak = 0
    best_streak = 0
    streak = 0
    
    for entry in elo_history:
        if entry.get("won"):
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0
    
    # Current streak is from the end
    current_streak = 0
    for entry in reversed(elo_history):
        if entry.get("won"):
            current_streak += 1
        else:
            break
    
    return (current_streak, best_streak)


def _calculate_role_stats(
    user_pk: str,
    games: List[Dict[str, Any]]
) -> Dict[str, Dict[str, int]]:
    """Calculate win/loss stats per role.

    Args:
        user_pk: The user's PK
        games: List of game results

    Returns:
        Dict mapping role names to stats
    """
    role_stats: Dict[str, Dict[str, int]] = {}

    for game in games:
        metadata = game.get("metadata", {})
        roles = metadata.get("roles", [])
        player_ids = game.get("player_ids", [])
        winner_team = game.get("winner_team", "")

        user_role = None
        user_team = None
        for i, pid in enumerate(player_ids):
            if pid == user_pk and i < len(roles):
                user_role = roles[i].get("role")
                user_team = roles[i].get("team")
                break

        if user_role:
            if user_role not in role_stats:
                role_stats[user_role] = {"played": 0, "won": 0}
            role_stats[user_role]["played"] += 1
            if user_team == winner_team:
                role_stats[user_role]["won"] += 1

    return role_stats


def _build_profile_stats(
    user: UserProfile,
    backend: Any,
    include_linked_providers: bool = True,
) -> Dict[str, Any]:
    """Build profile stats (rank, streaks, role_stats, etc.) for a user.

    Shared by get_my_profile and get_user_profile. Does not change response
    shape or game behavior; only centralizes duplicated logic.
    """
    all_users = backend.all(UserProfile)
    active_users = [u for u in all_users if u.get("games_played", 0) > 0]
    active_users.sort(key=lambda u: u.get("elo_rating", 0), reverse=True)

    user_rank = None
    for i, u in enumerate(active_users):
        if u.get("pk") == user.pk:
            user_rank = i + 1
            break

    total_players = len(active_users)
    percentile = None
    if user_rank and total_players > 0:
        percentile = round((1 - (user_rank / total_players)) * 100, 1)

    all_elo_history = backend.all(EloHistory)
    user_elo_history = [
        e for e in all_elo_history
        if e.get("user_pk") == user.pk
    ]
    user_elo_history.sort(key=lambda e: e.get("created_at", ""))

    current_streak, best_streak = _calculate_win_streaks(user_elo_history)
    recent_form = ["W" if entry.get("won") else "L" for entry in user_elo_history[-5:]]

    all_games = backend.all(GameResult)
    user_games = [g for g in all_games if user.pk in g.get("player_ids", [])]
    role_stats = _calculate_role_stats(user.pk, user_games)

    linked_providers: List[str] = []
    if include_linked_providers:
        all_oauth = backend.all(OAuthAccount)
        linked_providers = [
            oa.get("provider") for oa in all_oauth
            if oa.get("user_pk") == user.pk
        ]

    rank_tier = get_rank_progress(user.elo_rating)

    return {
        "user": user_to_response(user),
        "rank": user_rank,
        "total_players": total_players,
        "percentile": percentile,
        "rank_tier": rank_tier,
        "win_streak": current_streak,
        "best_win_streak": best_streak,
        "recent_form": recent_form,
        "role_stats": role_stats,
        "linked_providers": linked_providers,
    }


# Endpoints

@router.get("/me", response_model=ProfileStatsResponse)
async def get_my_profile(
    user: UserProfile = Depends(require_auth)
) -> ProfileStatsResponse:
    """Get the current user's full profile with stats."""
    backend = get_storage_backend()
    stats = _build_profile_stats(user, backend, include_linked_providers=True)
    return ProfileStatsResponse(**stats)


@router.get("/me/elo-history", response_model=EloHistoryResponse)
async def get_my_elo_history(
    user: UserProfile = Depends(require_auth),
    limit: int = Query(50, ge=1, le=200),
) -> EloHistoryResponse:
    """Get the current user's ELO rating history."""
    backend = get_storage_backend()
    
    # Get all ELO history for user
    all_elo_history = backend.all(EloHistory)
    user_elo_history = [
        e for e in all_elo_history
        if e.get("user_pk") == user.pk
    ]
    
    # Sort by created_at ascending
    user_elo_history.sort(key=lambda e: e.get("created_at", ""))
    
    # Calculate peak and lowest
    peak_elo = user.elo_rating
    lowest_elo = user.elo_rating
    
    for entry in user_elo_history:
        elo_after = entry.get("elo_after", 1000)
        peak_elo = max(peak_elo, elo_after)
        lowest_elo = min(lowest_elo, entry.get("elo_before", 1000))
    
    # Limit entries (newest first for response)
    limited_entries = user_elo_history[-limit:]
    limited_entries.reverse()
    
    entries = [
        EloHistoryEntry(
            game_pk=e.get("game_pk", ""),
            elo_before=e.get("elo_before", 1000),
            elo_after=e.get("elo_after", 1000),
            elo_change=e.get("elo_change", 0),
            game_type=e.get("game_type", "werewolf"),
            opponent_rating=e.get("opponent_rating", 0),
            won=e.get("won", False),
            created_at=e.get("created_at", ""),
        )
        for e in limited_entries
    ]
    
    return EloHistoryResponse(
        entries=entries,
        total=len(user_elo_history),
        current_elo=user.elo_rating,
        peak_elo=peak_elo,
        lowest_elo=lowest_elo,
    )


@router.get("/me/matches", response_model=MatchHistoryResponse)
async def get_my_match_history(
    user: UserProfile = Depends(require_auth),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    game_type: Optional[str] = Query(None),
) -> MatchHistoryResponse:
    """Get the current user's match history with details."""
    backend = get_storage_backend()
    
    # Get all games
    all_games = backend.all(GameResult)
    
    # Filter to user's games
    user_games = [
        g for g in all_games
        if user.pk in g.get("player_ids", [])
    ]
    
    # Filter by game type if specified
    if game_type:
        user_games = [g for g in user_games if g.get("game_type") == game_type]
    
    # Sort by created_at descending (newest first)
    user_games.sort(key=lambda g: g.get("created_at", ""), reverse=True)
    
    total = len(user_games)
    paginated = user_games[offset:offset + limit]
    
    # Get ELO history for this user to add elo_change to matches
    all_elo_history = backend.all(EloHistory)
    elo_by_game: Dict[str, int] = {}
    for e in all_elo_history:
        if e.get("user_pk") == user.pk:
            elo_by_game[e.get("game_pk", "")] = e.get("elo_change", 0)
    
    matches = []
    for game in paginated:
        metadata = game.get("metadata", {})
        roles = metadata.get("roles", [])
        player_ids = game.get("player_ids", [])
        player_names = game.get("player_names", [])
        winner_team = game.get("winner_team", "")
        
        # Find user's role and team
        user_role = "Unknown"
        user_team = "Unknown"
        for i, pid in enumerate(player_ids):
            if pid == user.pk and i < len(roles):
                user_role = roles[i].get("role", "Unknown")
                user_team = roles[i].get("team", "Unknown")
                break
        
        won = user_team == winner_team
        
        # Build player list
        players = []
        for i, name in enumerate(player_names):
            player_info = {"name": name}
            if i < len(roles):
                player_info["role"] = roles[i].get("role", "")
                player_info["team"] = roles[i].get("team", "")
            if i < len(player_ids):
                player_info["is_ai"] = player_ids[i].startswith("ai_")
                player_info["is_user"] = player_ids[i] == user.pk
            players.append(player_info)
        
        matches.append(MatchHistoryEntry(
            pk=game.get("pk", ""),
            game_type=game.get("game_type", "werewolf"),
            player_role=user_role,
            player_team=user_team,
            winner_team=winner_team,
            won=won,
            elo_change=elo_by_game.get(game.get("pk", "")),
            turns_played=game.get("turns_played", 0),
            duration_seconds=game.get("duration_seconds", 0),
            created_at=game.get("created_at", ""),
            players=players,
        ))
    
    return MatchHistoryResponse(
        matches=matches,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.put("/me", response_model=UserResponse)
async def update_my_profile(
    request: UpdateProfileRequest,
    user: UserProfile = Depends(require_auth),
) -> UserResponse:
    """Update the current user's profile."""
    backend = get_storage_backend()
    
    if request.username and request.username != user.username:
        # Check if username is taken
        existing = backend.find(UserProfile, {"username": request.username})
        if existing and existing[0].get("pk") != user.pk:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        user.username = request.username
    
    if request.email and request.email != user.email:
        # Check if email is taken
        existing = backend.find(UserProfile, {"email": request.email})
        if existing and existing[0].get("pk") != user.pk:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
        user.email = request.email
    
    if request.avatar_url is not None:
        user.avatar_url = request.avatar_url
    
    backend.save(UserProfile, user.pk, user.model_dump())
    
    return user_to_response(user)


@router.get("/{user_pk}", response_model=ProfileStatsResponse)
async def get_user_profile(user_pk: str) -> ProfileStatsResponse:
    """Get a public user profile by PK."""
    backend = get_storage_backend()
    try:
        user_data = backend.get(UserProfile, user_pk)
        user = UserProfile(**user_data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    stats = _build_profile_stats(user, backend, include_linked_providers=False)
    return ProfileStatsResponse(**stats)


@router.get("/{user_pk}/matches", response_model=MatchHistoryResponse)
async def get_user_match_history(
    user_pk: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> MatchHistoryResponse:
    """Get a user's public match history."""
    backend = get_storage_backend()
    
    # Verify user exists
    try:
        backend.get(UserProfile, user_pk)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get all games
    all_games = backend.all(GameResult)
    
    # Filter to user's games
    user_games = [
        g for g in all_games
        if user_pk in g.get("player_ids", [])
    ]
    
    # Sort by created_at descending
    user_games.sort(key=lambda g: g.get("created_at", ""), reverse=True)
    
    total = len(user_games)
    paginated = user_games[offset:offset + limit]
    
    matches = []
    for game in paginated:
        metadata = game.get("metadata", {})
        roles = metadata.get("roles", [])
        player_ids = game.get("player_ids", [])
        player_names = game.get("player_names", [])
        winner_team = game.get("winner_team", "")
        
        # Find user's role and team
        user_role = "Unknown"
        user_team = "Unknown"
        for i, pid in enumerate(player_ids):
            if pid == user_pk and i < len(roles):
                user_role = roles[i].get("role", "Unknown")
                user_team = roles[i].get("team", "Unknown")
                break
        
        won = user_team == winner_team
        
        # Build player list
        players = []
        for i, name in enumerate(player_names):
            player_info = {"name": name}
            if i < len(roles):
                player_info["role"] = roles[i].get("role", "")
                player_info["team"] = roles[i].get("team", "")
            players.append(player_info)
        
        matches.append(MatchHistoryEntry(
            pk=game.get("pk", ""),
            game_type=game.get("game_type", "werewolf"),
            player_role=user_role,
            player_team=user_team,
            winner_team=winner_team,
            won=won,
            elo_change=None,  # Don't expose ELO changes for other users
            turns_played=game.get("turns_played", 0),
            duration_seconds=game.get("duration_seconds", 0),
            created_at=game.get("created_at", ""),
            players=players,
        ))
    
    return MatchHistoryResponse(
        matches=matches,
        total=total,
        limit=limit,
        offset=offset,
    )
