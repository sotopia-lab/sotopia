"""Leaderboard API endpoints for Sotopia games."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

from sotopia.database import UserProfile, GameResult
from sotopia.database.storage_backend import get_storage_backend
from sotopia.api.elo import get_rank_tier, get_rank_progress, RANK_TIERS


router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


# Response Models

class LeaderboardEntry(BaseModel):
    """A single entry in the leaderboard."""
    rank: int
    username: str
    user_pk: str
    elo_rating: int
    games_played: int
    games_won: int
    win_rate: float
    rank_tier: str
    rank_emoji: str
    is_current_user: bool = False


class LeaderboardResponse(BaseModel):
    """Full leaderboard response."""
    entries: List[LeaderboardEntry]
    total_players: int
    game_type: str
    current_user_rank: Optional[int] = None
    current_user_entry: Optional[LeaderboardEntry] = None


class RankTierInfo(BaseModel):
    """Information about a rank tier."""
    name: str
    emoji: str
    min_elo: int
    max_elo: int
    player_count: int


class RankDistributionResponse(BaseModel):
    """Distribution of players across rank tiers."""
    tiers: List[RankTierInfo]
    total_players: int


# Endpoints

@router.get("", response_model=LeaderboardResponse)
@router.get("/", response_model=LeaderboardResponse)
async def get_leaderboard(
    game_type: str = Query("werewolf", description="Game type to show leaderboard for"),
    limit: int = Query(50, ge=1, le=100, description="Number of entries to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user_pk: Optional[str] = Query(None, description="Current user PK to highlight"),
) -> LeaderboardResponse:
    """Get the leaderboard for a specific game type.
    
    Returns users ranked by ELO rating, with their stats and rank tier.
    """
    backend = get_storage_backend()
    
    # Get all users
    all_users = backend.all(UserProfile)
    
    # Filter users who have played at least one game
    # and sort by ELO descending
    active_users = [
        UserProfile(**u) for u in all_users
        if u.get("games_played", 0) > 0
    ]
    active_users.sort(key=lambda u: u.elo_rating, reverse=True)
    
    total_players = len(active_users)
    
    # Find current user's rank if provided
    current_user_rank: Optional[int] = None
    current_user_entry: Optional[LeaderboardEntry] = None
    
    if current_user_pk:
        for i, user in enumerate(active_users):
            if user.pk == current_user_pk:
                current_user_rank = i + 1
                rank_name, rank_emoji = get_rank_tier(user.elo_rating)
                win_rate = (user.games_won / user.games_played * 100) if user.games_played > 0 else 0
                current_user_entry = LeaderboardEntry(
                    rank=current_user_rank,
                    username=user.username,
                    user_pk=user.pk,
                    elo_rating=user.elo_rating,
                    games_played=user.games_played,
                    games_won=user.games_won,
                    win_rate=round(win_rate, 1),
                    rank_tier=rank_name,
                    rank_emoji=rank_emoji,
                    is_current_user=True,
                )
                break
    
    # Build leaderboard entries
    entries: List[LeaderboardEntry] = []
    paginated_users = active_users[offset:offset + limit]
    
    for i, user in enumerate(paginated_users):
        rank = offset + i + 1
        rank_name, rank_emoji = get_rank_tier(user.elo_rating)
        win_rate = (user.games_won / user.games_played * 100) if user.games_played > 0 else 0
        
        entries.append(LeaderboardEntry(
            rank=rank,
            username=user.username,
            user_pk=user.pk,
            elo_rating=user.elo_rating,
            games_played=user.games_played,
            games_won=user.games_won,
            win_rate=round(win_rate, 1),
            rank_tier=rank_name,
            rank_emoji=rank_emoji,
            is_current_user=(user.pk == current_user_pk),
        ))
    
    return LeaderboardResponse(
        entries=entries,
        total_players=total_players,
        game_type=game_type,
        current_user_rank=current_user_rank,
        current_user_entry=current_user_entry,
    )


@router.get("/ranks", response_model=RankDistributionResponse)
async def get_rank_distribution() -> RankDistributionResponse:
    """Get distribution of players across rank tiers."""
    backend = get_storage_backend()
    
    # Get all users with games
    all_users = backend.all(UserProfile)
    active_users = [u for u in all_users if u.get("games_played", 0) > 0]
    
    # Count players in each tier
    tier_counts: Dict[str, int] = {tier[2]: 0 for tier in RANK_TIERS}
    
    for user in active_users:
        elo = user.get("elo_rating", 1000)
        tier_name, _ = get_rank_tier(elo)
        if tier_name in tier_counts:
            tier_counts[tier_name] += 1
    
    # Build response
    tiers: List[RankTierInfo] = []
    for min_elo, max_elo, name, emoji in RANK_TIERS:
        tiers.append(RankTierInfo(
            name=name,
            emoji=emoji,
            min_elo=min_elo,
            max_elo=max_elo,
            player_count=tier_counts.get(name, 0),
        ))
    
    return RankDistributionResponse(
        tiers=tiers,
        total_players=len(active_users),
    )


@router.get("/user/{user_pk}")
async def get_user_ranking(user_pk: str) -> Dict[str, Any]:
    """Get detailed ranking info for a specific user."""
    backend = get_storage_backend()
    
    try:
        user_data = backend.get(UserProfile, user_pk)
        user = UserProfile(**user_data)
    except Exception:
        return {"error": "User not found"}
    
    # Get all active users to calculate rank
    all_users = backend.all(UserProfile)
    active_users = [
        UserProfile(**u) for u in all_users
        if u.get("games_played", 0) > 0
    ]
    active_users.sort(key=lambda u: u.elo_rating, reverse=True)
    
    # Find user's rank
    user_rank = None
    for i, u in enumerate(active_users):
        if u.pk == user_pk:
            user_rank = i + 1
            break
    
    total_players = len(active_users)
    rank_progress = get_rank_progress(user.elo_rating)
    win_rate = (user.games_won / user.games_played * 100) if user.games_played > 0 else 0
    
    # Get recent games
    all_games = backend.all(GameResult)
    user_games = [
        g for g in all_games
        if user_pk in g.get("player_ids", [])
    ]
    user_games.sort(key=lambda g: g.get("created_at", ""), reverse=True)
    recent_games = user_games[:5]
    
    # Format recent games with ELO changes
    recent_game_summaries = []
    for game in recent_games:
        meta = game.get("metadata", {})
        elo_changes = meta.get("elo_changes", {})
        user_elo_change = elo_changes.get(user_pk, {})
        
        recent_game_summaries.append({
            "pk": game.get("pk"),
            "winner_team": game.get("winner_team"),
            "created_at": game.get("created_at"),
            "elo_before": user_elo_change.get("before"),
            "elo_after": user_elo_change.get("after"),
            "elo_change": user_elo_change.get("change"),
        })
    
    return {
        "user_pk": user_pk,
        "username": user.username,
        "elo_rating": user.elo_rating,
        "rank": user_rank,
        "total_players": total_players,
        "percentile": round((1 - (user_rank / total_players)) * 100, 1) if user_rank and total_players else None,
        "games_played": user.games_played,
        "games_won": user.games_won,
        "win_rate": round(win_rate, 1),
        "rank_tier": rank_progress,
        "recent_games": recent_game_summaries,
    }


@router.get("/top/{count}")
async def get_top_players(
    count: int = 10,
    game_type: str = "werewolf",
) -> Dict[str, Any]:
    """Get the top N players for a game type."""
    result = await get_leaderboard(
        game_type=game_type,
        limit=count,
        offset=0,
        current_user_pk=None,
    )
    return {
        "top_players": [e.model_dump() for e in result.entries],
        "total_players": result.total_players,
        "game_type": game_type,
    }
