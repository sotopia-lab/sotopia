"""Authentication endpoints for user registration, login, and session management."""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field

from sotopia.database import UserProfile, GameResult
from sotopia.database.storage_backend import get_storage_backend

# Router for auth endpoints
router = APIRouter(prefix="/auth", tags=["authentication"])

# Security scheme for JWT
security = HTTPBearer(auto_error=False)

# JWT configuration
JWT_SECRET = os.environ.get("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


# Request/Response Models

class RegisterRequest(BaseModel):
    """Request body for user registration."""
    username: str = Field(..., min_length=3, max_length=30, pattern="^[a-zA-Z0-9_]+$")
    email: EmailStr
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    """Request body for user login."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Response containing JWT token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: Dict[str, Any]


class UserResponse(BaseModel):
    """Public user information (no password hash)."""
    pk: str
    username: str
    email: str
    elo_rating: int
    games_played: int
    games_won: int
    created_at: str
    auth_provider: str = "local"
    avatar_url: str = ""
    has_password: bool = True  # True if user can login with password


class UpdateProfileRequest(BaseModel):
    """Request body for updating user profile."""
    email: Optional[EmailStr] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = Field(None, min_length=6)


# Helper Functions


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_access_token(user_pk: str, username: str) -> str:
    """Create a JWT access token."""
    expires = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "sub": user_pk,
        "username": username,
        "exp": expires,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[UserProfile]:
    """Dependency to get the current authenticated user."""
    if credentials is None:
        return None
    
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if payload is None:
        return None
    
    user_pk = payload.get("sub")
    if not user_pk:
        return None
    
    # Retrieve user from storage
    backend = get_storage_backend()
    try:
        user_data = backend.get(UserProfile, user_pk)
        return UserProfile(**user_data)
    except Exception:
        return None


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> UserProfile:
    """Dependency that requires authentication."""
    user = await get_current_user(credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def user_to_response(user: UserProfile) -> UserResponse:
    """Convert UserProfile to public response (excluding password)."""
    return UserResponse(
        pk=user.pk,
        username=user.username,
        email=user.email,
        elo_rating=user.elo_rating,
        games_played=user.games_played,
        games_won=user.games_won,
        created_at=user.created_at,
        auth_provider=getattr(user, 'auth_provider', 'local'),
        avatar_url=getattr(user, 'avatar_url', ''),
        has_password=bool(user.password_hash),
    )


# Auth Endpoints

@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest) -> TokenResponse:
    """Register a new user account."""
    backend = get_storage_backend()
    
    # Check if username already exists
    existing_users = backend.find(UserProfile, {"username": request.username})
    if existing_users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Check if email already exists
    existing_emails = backend.find(UserProfile, {"email": request.email})
    if existing_emails:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user_pk = backend.generate_pk()
    password_hash = hash_password(request.password)
    
    user = UserProfile(
        pk=user_pk,
        username=request.username,
        email=request.email,
        password_hash=password_hash,
        elo_rating=1000,
        games_played=0,
        games_won=0,
        created_at=datetime.utcnow().isoformat(),
        last_login=datetime.utcnow().isoformat(),
        is_active=True,
    )
    
    # Save to storage
    backend.save(UserProfile, user_pk, user.model_dump())
    
    # Generate token
    access_token = create_access_token(user_pk, request.username)
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user=user_to_response(user).model_dump(),
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    """Login with username and password."""
    backend = get_storage_backend()
    
    # Find user by username
    users = backend.find(UserProfile, {"username": request.username})
    if not users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    user_data = users[0]
    user = UserProfile(**user_data)
    
    # Verify password
    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Check if account is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )
    
    # Update last login
    user.last_login = datetime.utcnow().isoformat()
    backend.save(UserProfile, user.pk, user.model_dump())
    
    # Generate token
    access_token = create_access_token(user.pk, user.username)
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user=user_to_response(user).model_dump(),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: UserProfile = Depends(require_auth)) -> UserResponse:
    """Get the current authenticated user's profile."""
    return user_to_response(user)


@router.put("/me", response_model=UserResponse)
async def update_me(
    request: UpdateProfileRequest,
    user: UserProfile = Depends(require_auth)
) -> UserResponse:
    """Update the current user's profile."""
    backend = get_storage_backend()
    
    # Update email if provided
    if request.email and request.email != user.email:
        # Check if email is taken
        existing = backend.find(UserProfile, {"email": request.email})
        if existing and existing[0].get("pk") != user.pk:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
        user.email = request.email
    
    # Update password if provided
    if request.new_password:
        if not request.current_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password required to change password"
            )
        if not verify_password(request.current_password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
        user.password_hash = hash_password(request.new_password)
    
    # Save updates
    backend.save(UserProfile, user.pk, user.model_dump())
    
    return user_to_response(user)


@router.post("/logout")
async def logout(user: UserProfile = Depends(require_auth)) -> Dict[str, str]:
    """Logout the current user."""
    # For JWT, we don't need to do anything server-side
    # The client should discard the token
    return {"message": "Successfully logged out"}


# Game History Endpoints

@router.get("/me/games")
async def get_my_games(
    user: UserProfile = Depends(require_auth),
    limit: int = 20,
    offset: int = 0
) -> Dict[str, Any]:
    """Get the current user's game history."""
    backend = get_storage_backend()
    
    # Get all games (we'll filter by player_ids containing user.pk)
    all_games = backend.all(GameResult)
    
    # Filter games where user participated
    user_games = [
        GameResult(**g) for g in all_games
        if user.pk in g.get("player_ids", [])
    ]
    
    # Sort by created_at descending (newest first)
    user_games.sort(key=lambda g: g.created_at, reverse=True)
    
    # Paginate
    total = len(user_games)
    games = user_games[offset:offset + limit]
    
    return {
        "games": [g.model_dump() for g in games],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/me/stats")
async def get_my_stats(user: UserProfile = Depends(require_auth)) -> Dict[str, Any]:
    """Get the current user's game statistics, calculated from actual game history."""
    backend = get_storage_backend()
    
    # Count games from game history
    games_played = 0
    games_won = 0
    
    all_games = backend.all(GameResult)
    for game_data in all_games:
        # Backend returns dicts, convert to model if needed
        if isinstance(game_data, dict):
            game = GameResult(**game_data)
        else:
            game = game_data
        
        player_ids = game.player_ids or []
        if user.pk in player_ids:
            games_played += 1
            # Check if user's team won
            if game.winner_team:
                # Get user's role/team from metadata if available
                metadata = game.metadata or {}
                user_id = user.pk
                
                # If user was marked as winning or was on winning team
                if user_id in (game.winner_player_ids or []):
                    games_won += 1
                elif metadata.get('user_id') == user_id:
                    # Check roles to see if user won
                    roles = metadata.get('roles', [])
                    human_player = metadata.get('human_player')
                    for role_info in roles:
                        if role_info.get('name') == human_player:
                            if role_info.get('team') == game.winner_team:
                                games_won += 1
                            break
    
    win_rate = (games_won / games_played * 100) if games_played > 0 else 0
    
    return {
        "username": user.username,
        "elo_rating": user.elo_rating,
        "games_played": games_played,
        "games_won": games_won,
        "win_rate": round(win_rate, 1),
    }


# Legacy Identity Endpoints (for backward compatibility with existing frontend)

class IdentityRequest(BaseModel):
    """Request body for legacy identity registration."""
    participant_id: str
    display_name: Optional[str] = None


class IdentityRecord(BaseModel):
    """Legacy identity record."""
    token: str
    participantId: str
    displayName: Optional[str] = None
    createdAt: float


# Simple in-memory token storage for legacy identity (or use Redis)
_identity_tokens: Dict[str, Dict[str, Any]] = {}
_participant_to_token: Dict[str, str] = {}  # participantId -> token mapping


@router.post("/identity", response_model=IdentityRecord)
async def register_identity(request: IdentityRequest) -> IdentityRecord:
    """
    Register a participant identity (legacy endpoint for existing frontend).
    Creates a simple token without full user account.
    Idempotent: returns existing token if participantId already registered.
    """
    import time
    import secrets
    
    # Check if this participantId already has a token
    existing_token = _participant_to_token.get(request.participant_id)
    if existing_token and existing_token in _identity_tokens:
        record = _identity_tokens[existing_token]
        # Update display name if provided
        if request.display_name:
            record["display_name"] = request.display_name
        return IdentityRecord(
            token=existing_token,
            participantId=record["participant_id"],
            displayName=record.get("display_name"),
            createdAt=record["created_at"],
        )
    
    token = secrets.token_urlsafe(16)
    created_at = time.time()
    
    record = {
        "token": token,
        "participant_id": request.participant_id,
        "display_name": request.display_name or request.participant_id,
        "created_at": created_at,
    }
    
    _identity_tokens[token] = record
    _participant_to_token[request.participant_id] = token
    
    return IdentityRecord(
        token=token,
        participantId=request.participant_id,
        displayName=request.display_name,
        createdAt=created_at,
    )


@router.get("/identity/{token}", response_model=IdentityRecord)
async def get_identity(token: str) -> IdentityRecord:
    """Fetch identity by token (legacy endpoint)."""
    record = _identity_tokens.get(token)
    if not record:
        raise HTTPException(status_code=404, detail="Identity not found")
    
    return IdentityRecord(
        token=token,
        participantId=record["participant_id"],
        displayName=record.get("display_name"),
        createdAt=record["created_at"],
    )
    