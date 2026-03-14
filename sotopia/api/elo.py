"""ELO Rating System for Sotopia Games.

This module implements the ELO rating system for tracking player skill levels
in social deduction games like Werewolf.
"""

from typing import Any, Dict, List, Tuple
import logging

from sotopia.database import UserProfile, GameResult, EloHistory
from sotopia.database.storage_backend import get_storage_backend

logger = logging.getLogger(__name__)


# AI Model Base Ratings

AI_MODEL_RATINGS: Dict[str, int] = {
    # OpenAI models
    "gpt-4": 1600,
    "gpt-4-turbo": 1550,
    "gpt-4o": 1500,
    "gpt-4o-mini": 1300,
    "gpt-3.5-turbo": 1200,
    
    # Anthropic models
    "claude-3-opus": 1600,
    "claude-3-sonnet": 1500,
    "claude-3-haiku": 1350,
    "claude-3.5-sonnet": 1550,
    
    # Meta models
    "llama-3.1-70b": 1400,
    "llama-3.1-8b": 1250,
    "llama-3-70b": 1350,
    
    # Google models
    "gemini-pro": 1400,
    "gemini-1.5-pro": 1500,
    
    # Default for unknown models
    "default": 1200,
}


def get_ai_rating(model_name: str) -> int:
    """Get the base ELO rating for an AI model.
    
    Args:
        model_name: The name of the AI model
        
    Returns:
        The base ELO rating for the model
    """
    # Try exact match first
    if model_name in AI_MODEL_RATINGS:
        return AI_MODEL_RATINGS[model_name]
    
    # Try partial match (e.g., "gpt-4o-mini-2024-07-18" should match "gpt-4o-mini")
    model_lower = model_name.lower()
    for key, rating in AI_MODEL_RATINGS.items():
        if key in model_lower:
            return rating
    
    return AI_MODEL_RATINGS["default"]



# ELO Calculation

# ELO Constants
K_FACTOR_NEW = 40      # K-factor for players with < 30 games
K_FACTOR_NORMAL = 20   # K-factor for established players
K_FACTOR_THRESHOLD = 30  # Games before switching to normal K-factor


def calculate_expected_score(rating_a: int, rating_b: int) -> float:
    """Calculate expected score for player A against player B.
    
    Uses the standard ELO expected score formula:
    E_a = 1 / (1 + 10^((R_b - R_a) / 400))
    
    Args:
        rating_a: Player A's rating
        rating_b: Player B's rating
        
    Returns:
        Expected score (0-1) for player A
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def calculate_elo_change(
    player_rating: int,
    opponent_rating: int,
    won: bool,
    games_played: int = 30,
) -> int:
    """Calculate the ELO rating change for a player.
    
    Args:
        player_rating: Current rating of the player
        opponent_rating: Rating of the opponent (or average opponent rating)
        won: Whether the player won
        games_played: Number of games the player has played (affects K-factor)
        
    Returns:
        The rating change (can be positive or negative)
    """
    # Determine K-factor based on experience
    k_factor = K_FACTOR_NEW if games_played < K_FACTOR_THRESHOLD else K_FACTOR_NORMAL
    
    # Calculate expected score
    expected = calculate_expected_score(player_rating, opponent_rating)
    
    # Actual score: 1 for win, 0 for loss, 0.5 for draw
    actual = 1.0 if won else 0.0
    
    # Calculate rating change
    change = round(k_factor * (actual - expected))
    
    return change


def calculate_team_average_rating(
    team_players: List[str],
    user_ratings: Dict[str, int],
    ai_model: str = "gpt-4o-mini",
) -> int:
    """Calculate the average rating of a team.
    
    Args:
        team_players: List of player IDs in the team
        user_ratings: Dict mapping user IDs to their ratings
        ai_model: AI model name for AI players
        
    Returns:
        Average rating of the team
    """
    if not team_players:
        return AI_MODEL_RATINGS["default"]
    
    ai_rating = get_ai_rating(ai_model)
    
    ratings = []
    for player_id in team_players:
        if player_id.startswith("ai_"):
            ratings.append(ai_rating)
        elif player_id in user_ratings:
            ratings.append(user_ratings[player_id])
        else:
            ratings.append(AI_MODEL_RATINGS["default"])
    
    return int(sum(ratings) / len(ratings))


# ELO Update Functions

def update_user_elo_after_game(
    user_pk: str,
    won: bool,
    opponent_rating: int,
    game_pk: str = "",
    game_type: str = "werewolf",
) -> Tuple[int, int, int]:
    """Update a user's ELO rating after a game.
    
    Args:
        user_pk: The user's primary key
        won: Whether the user won
        opponent_rating: The average rating of the opposing team
        game_pk: The game result PK (for history tracking)
        game_type: Type of game played
        
    Returns:
        Tuple of (elo_before, elo_after, elo_change)
    """
    backend = get_storage_backend()
    
    try:
        user_data = backend.get(UserProfile, user_pk)
        user = UserProfile(**user_data)
    except Exception as e:
        logger.error(f"Failed to get user {user_pk} for ELO update: {e}")
        return (1000, 1000, 0)
    
    elo_before = user.elo_rating
    games_played = user.games_played
    
    # Calculate change
    elo_change = calculate_elo_change(
        player_rating=elo_before,
        opponent_rating=opponent_rating,
        won=won,
        games_played=games_played,
    )
    
    elo_after = max(0, elo_before + elo_change)  # Don't go below 0
    
    # Update user record
    user.elo_rating = elo_after
    user.games_played = games_played + 1
    if won:
        user.games_won = user.games_won + 1
    
    backend.save(UserProfile, user.pk, user.model_dump())
    
    # Record ELO history
    try:
        history_pk = backend.generate_pk()
        elo_history = EloHistory(
            pk=history_pk,
            user_pk=user_pk,
            game_pk=game_pk,
            elo_before=elo_before,
            elo_after=elo_after,
            elo_change=elo_change,
            game_type=game_type,
            opponent_rating=opponent_rating,
            won=won,
        )
        backend.save(EloHistory, history_pk, elo_history.model_dump())
    except Exception as e:
        logger.error(f"Failed to save ELO history: {e}")
    
    logger.info(
        f"Updated ELO for {user.username}: {elo_before} -> {elo_after} "
        f"({'won' if won else 'lost'}, change: {elo_change:+d})"
    )
    
    return (elo_before, elo_after, elo_change)


def process_game_elo(
    game_result: GameResult,
    ai_model: str = "gpt-4o-mini",
) -> Dict[str, Tuple[int, int, int]]:
    """Process ELO changes for all human players in a game.
    
    Args:
        game_result: The GameResult record
        ai_model: AI model used in the game
        
    Returns:
        Dict mapping user_pk to (elo_before, elo_after, elo_change)
    """
    backend = get_storage_backend()
    
    # Get team info from metadata
    metadata = game_result.metadata or {}
    roles = metadata.get("roles", [])
    winner_team = game_result.winner_team
    
    if not winner_team or winner_team in ("Draw", "Unknown"):
        logger.info("Game ended in draw/unknown - no ELO changes")
        return {}
    
    # Build player -> team mapping
    player_teams: Dict[str, str] = {}
    for i, player_id in enumerate(game_result.player_ids):
        if i < len(roles):
            player_teams[player_id] = roles[i].get("team", "")
    
    # Get ratings for all players
    ai_rating = get_ai_rating(ai_model)
    user_ratings: Dict[str, int] = {}
    
    for player_id in game_result.player_ids:
        if player_id.startswith("ai_"):
            user_ratings[player_id] = ai_rating
        else:
            try:
                user_data = backend.get(UserProfile, player_id)
                user_ratings[player_id] = user_data.get("elo_rating", 1000)
            except Exception:
                user_ratings[player_id] = 1000
    
    # Calculate team ratings
    winning_team_players = [
        pid for pid, team in player_teams.items()
        if team == winner_team
    ]
    losing_team_players = [
        pid for pid, team in player_teams.items()
        if team and team != winner_team
    ]
    
    winning_avg = calculate_team_average_rating(winning_team_players, user_ratings, ai_model)
    losing_avg = calculate_team_average_rating(losing_team_players, user_ratings, ai_model)
    
    # Update ELO for human players
    elo_results: Dict[str, Tuple[int, int, int]] = {}
    
    for player_id in game_result.player_ids:
        if player_id.startswith("ai_"):
            continue  # Don't update AI ratings
        
        player_team = player_teams.get(player_id, "")
        if not player_team:
            continue
        
        won = player_team == winner_team
        opponent_rating = losing_avg if won else winning_avg
        
        elo_before, elo_after, elo_change = update_user_elo_after_game(
            user_pk=player_id,
            won=won,
            opponent_rating=opponent_rating,
            game_pk=game_result.pk,
            game_type=game_result.game_type,
        )
        
        elo_results[player_id] = (elo_before, elo_after, elo_change)
    
    return elo_results


# Rank Tiers

RANK_TIERS = [
    (0, 1199, "Bronze", "🥉"),
    (1200, 1399, "Silver", "🥈"),
    (1400, 1599, "Gold", "🥇"),
    (1600, 1799, "Platinum", "💎"),
    (1800, 1999, "Diamond", "💠"),
    (2000, 9999, "Master", "👑"),
]


def get_rank_tier(elo: int) -> Tuple[str, str]:
    """Get the rank tier for a given ELO rating.
    
    Args:
        elo: The ELO rating
        
    Returns:
        Tuple of (tier_name, tier_emoji)
    """
    for min_elo, max_elo, name, emoji in RANK_TIERS:
        if min_elo <= elo <= max_elo:
            return (name, emoji)
    return ("Unranked", "❓")


def get_rank_progress(elo: int) -> Dict[str, Any]:
    """Get progress within current rank tier.
    
    Args:
        elo: The ELO rating
        
    Returns:
        Dict with tier info and progress
    """
    for min_elo, max_elo, name, emoji in RANK_TIERS:
        if min_elo <= elo <= max_elo:
            tier_range = max_elo - min_elo + 1
            progress = (elo - min_elo) / tier_range * 100
            
            # Find next tier
            next_tier = None
            next_tier_emoji = None
            elo_to_next = 0
            for next_min, next_max, next_name, next_emoji in RANK_TIERS:
                if next_min > max_elo:
                    next_tier = next_name
                    next_tier_emoji = next_emoji
                    elo_to_next = next_min - elo
                    break
            
            return {
                "tier": name,
                "emoji": emoji,
                "elo": elo,
                "tier_min": min_elo,
                "tier_max": max_elo,
                "progress_percent": round(progress, 1),
                "next_tier": next_tier,
                "next_tier_emoji": next_tier_emoji,
                "elo_to_next": elo_to_next,
            }
    
    return {
        "tier": "Unranked",
        "emoji": "❓",
        "elo": elo,
        "tier_min": 0,
        "tier_max": 0,
        "progress_percent": 0,
        "next_tier": "Bronze",
        "next_tier_emoji": "🥉",
        "elo_to_next": max(0, 1000 - elo),
    }
