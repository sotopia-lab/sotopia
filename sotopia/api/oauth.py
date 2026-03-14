"""OAuth authentication with Google, GitHub, and Discord providers."""

import os
import secrets
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from sotopia.database import OAuthAccount, UserProfile
from sotopia.database.storage_backend import get_storage_backend

from .auth import create_access_token


# OAuth Configuration

class OAuthProvider:
    """Configuration for an OAuth provider."""
    
    def __init__(
        self,
        name: str,
        client_id: str,
        client_secret: str,
        authorize_url: str,
        token_url: str,
        userinfo_url: str,
        scopes: list[str],
    ):
        self.name = name
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorize_url = authorize_url
        self.token_url = token_url
        self.userinfo_url = userinfo_url
        self.scopes = scopes


# Provider configurations - set via environment variables
OAUTH_PROVIDERS: Dict[str, OAuthProvider] = {}

# Base URL for callbacks (set via env var or default)
OAUTH_CALLBACK_BASE_URL = os.environ.get(
    "OAUTH_CALLBACK_BASE_URL",
    "http://localhost:8800"
)

# Frontend URL for redirect after auth
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")


def _init_providers() -> None:
    """Initialize OAuth providers from environment variables."""
    global OAUTH_PROVIDERS
    
    # Google OAuth
    google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    google_client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    if google_client_id and google_client_secret:
        OAUTH_PROVIDERS["google"] = OAuthProvider(
            name="google",
            client_id=google_client_id,
            client_secret=google_client_secret,
            authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
            scopes=["openid", "email", "profile"],
        )
    
    # GitHub OAuth
    github_client_id = os.environ.get("GITHUB_CLIENT_ID", "")
    github_client_secret = os.environ.get("GITHUB_CLIENT_SECRET", "")
    if github_client_id and github_client_secret:
        OAUTH_PROVIDERS["github"] = OAuthProvider(
            name="github",
            client_id=github_client_id,
            client_secret=github_client_secret,
            authorize_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            userinfo_url="https://api.github.com/user",
            scopes=["read:user", "user:email"],
        )
    
    # Discord OAuth
    discord_client_id = os.environ.get("DISCORD_CLIENT_ID", "")
    discord_client_secret = os.environ.get("DISCORD_CLIENT_SECRET", "")
    if discord_client_id and discord_client_secret:
        OAUTH_PROVIDERS["discord"] = OAuthProvider(
            name="discord",
            client_id=discord_client_id,
            client_secret=discord_client_secret,
            authorize_url="https://discord.com/api/oauth2/authorize",
            token_url="https://discord.com/api/oauth2/token",
            userinfo_url="https://discord.com/api/users/@me",
            scopes=["identify", "email"],
        )


# Initialize providers on module load
_init_providers()



# Router


router = APIRouter(prefix="/oauth", tags=["oauth"])


# Response Models

class OAuthTokenResponse(BaseModel):
    """Response after successful OAuth authentication."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]
    is_new_user: bool


class AvailableProvidersResponse(BaseModel):
    """List of available OAuth providers."""
    providers: list[str]


# Helper Functions

def _get_provider(provider_name: str) -> OAuthProvider:
    """Get OAuth provider by name, raising if not configured."""
    if provider_name not in OAUTH_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth provider '{provider_name}' is not configured. "
                   f"Available: {list(OAUTH_PROVIDERS.keys())}"
        )
    return OAUTH_PROVIDERS[provider_name]


async def _exchange_code_for_token(
    provider: OAuthProvider,
    code: str,
    redirect_uri: str,
) -> Dict[str, Any]:
    """Exchange authorization code for access token."""
    async with httpx.AsyncClient() as client:
        data = {
            "client_id": provider.client_id,
            "client_secret": provider.client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        
        headers = {"Accept": "application/json"}
        
        response = await client.post(
            provider.token_url,
            data=data,
            headers=headers,
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to exchange code for token: {response.text}"
            )
        
        return response.json()


async def _get_user_info(
    provider: OAuthProvider,
    access_token: str,
) -> Dict[str, Any]:
    """Fetch user info from OAuth provider."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}
        
        response = await client.get(
            provider.userinfo_url,
            headers=headers,
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to fetch user info: {response.text}"
            )
        
        user_info = response.json()
        
        # For GitHub, we need to fetch email separately if not public
        if provider.name == "github" and not user_info.get("email"):
            email_response = await client.get(
                "https://api.github.com/user/emails",
                headers=headers,
            )
            if email_response.status_code == 200:
                emails = email_response.json()
                # Get primary email
                for email in emails:
                    if email.get("primary"):
                        user_info["email"] = email.get("email")
                        break
        
        return user_info


def _normalize_user_info(provider_name: str, raw_info: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize user info from different providers to a common format."""
    if provider_name == "google":
        return {
            "provider_user_id": raw_info.get("id", ""),
            "email": raw_info.get("email", ""),
            "name": raw_info.get("name", ""),
            "avatar_url": raw_info.get("picture", ""),
        }
    elif provider_name == "github":
        return {
            "provider_user_id": str(raw_info.get("id", "")),
            "email": raw_info.get("email", ""),
            "name": raw_info.get("name") or raw_info.get("login", ""),
            "avatar_url": raw_info.get("avatar_url", ""),
            "username_hint": raw_info.get("login", ""),
        }
    elif provider_name == "discord":
        avatar_hash = raw_info.get("avatar")
        user_id = raw_info.get("id", "")
        avatar_url = ""
        if avatar_hash:
            avatar_url = f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.png"
        
        return {
            "provider_user_id": user_id,
            "email": raw_info.get("email", ""),
            "name": raw_info.get("global_name") or raw_info.get("username", ""),
            "avatar_url": avatar_url,
            "username_hint": raw_info.get("username", ""),
        }
    else:
        return {
            "provider_user_id": raw_info.get("id", ""),
            "email": raw_info.get("email", ""),
            "name": raw_info.get("name", ""),
            "avatar_url": "",
        }


def _generate_username(name: str, hint: Optional[str] = None) -> str:
    """Generate a unique username from name or hint."""
    backend = get_storage_backend()
    
    # Try hint first (e.g., GitHub username)
    base = hint or name
    # Clean: keep alphanumeric and underscore only
    base = "".join(c if c.isalnum() or c == "_" else "_" for c in base)
    base = base[:20]  # Limit length
    
    if not base:
        base = "user"
    
    # Check if exists
    existing = backend.find(UserProfile, {"username": base})
    if not existing:
        return base
    
    # Add random suffix
    for _ in range(10):
        suffix = secrets.token_hex(3)
        candidate = f"{base}_{suffix}"
        existing = backend.find(UserProfile, {"username": candidate})
        if not existing:
            return candidate
    
    # Fallback to fully random
    return f"user_{secrets.token_hex(8)}"


# Endpoints

@router.get("/providers", response_model=AvailableProvidersResponse)
async def get_providers() -> AvailableProvidersResponse:
    """Get list of configured OAuth providers."""
    # Re-init to pick up any runtime env changes
    _init_providers()
    return AvailableProvidersResponse(providers=list(OAUTH_PROVIDERS.keys()))


@router.get("/login/{provider}")
async def oauth_login(
    provider: str,
    redirect_url: Optional[str] = Query(None, description="Where to redirect after auth"),
) -> RedirectResponse:
    """Initiate OAuth login flow by redirecting to provider."""
    _init_providers()
    oauth_provider = _get_provider(provider)
    
    # Build callback URL
    callback_url = f"{OAUTH_CALLBACK_BASE_URL}/oauth/callback/{provider}"
    
    # Build state with optional redirect
    state = secrets.token_urlsafe(32)
    if redirect_url:
        # In production, encode redirect_url in state or use session
        state = f"{state}|{redirect_url}"
    
    # Build authorization URL
    params = {
        "client_id": oauth_provider.client_id,
        "redirect_uri": callback_url,
        "scope": " ".join(oauth_provider.scopes),
        "response_type": "code",
        "state": state,
    }
    
    # Provider-specific params
    if provider == "google":
        params["access_type"] = "offline"
        params["prompt"] = "consent"
    elif provider == "discord":
        params["prompt"] = "consent"
    
    auth_url = f"{oauth_provider.authorize_url}?{urlencode(params)}"
    
    return RedirectResponse(url=auth_url)


@router.get("/callback/{provider}")
async def oauth_callback(
    provider: str,
    code: str = Query(..., description="Authorization code from provider"),
    state: str = Query("", description="State parameter"),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
) -> RedirectResponse:
    """Handle OAuth callback from provider."""
    # Check for errors
    if error:
        error_msg = error_description or error
        return RedirectResponse(
            url=f"{FRONTEND_URL}/auth/error?message={error_msg}"
        )
    
    _init_providers()
    oauth_provider = _get_provider(provider)
    backend = get_storage_backend()
    
    # Build callback URL (must match what we sent)
    callback_url = f"{OAUTH_CALLBACK_BASE_URL}/oauth/callback/{provider}"
    
    # Exchange code for token
    token_data = await _exchange_code_for_token(
        oauth_provider, code, callback_url
    )
    
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token", "")
    expires_in = token_data.get("expires_in", 3600)
    
    if not access_token:
        return RedirectResponse(
            url=f"{FRONTEND_URL}/auth/error?message=No access token received"
        )
    
    # Get user info
    raw_user_info = await _get_user_info(oauth_provider, access_token)
    user_info = _normalize_user_info(provider, raw_user_info)
    
    provider_user_id = user_info["provider_user_id"]
    email = user_info.get("email", "")
    
    if not provider_user_id:
        return RedirectResponse(
            url=f"{FRONTEND_URL}/auth/error?message=Could not get user ID from provider"
        )
    
    # Check if OAuth account already exists
    existing_oauth = backend.find(OAuthAccount, {
        "provider": provider,
        "provider_user_id": provider_user_id,
    })
    
    is_new_user = False
    user: Optional[UserProfile] = None
    
    if existing_oauth:
        # User has logged in before with this provider
        oauth_account_data = existing_oauth[0]
        user_pk = oauth_account_data.get("user_pk")
        
        try:
            user_data = backend.get(UserProfile, user_pk)
            user = UserProfile(**user_data)
        except Exception:
            # User was deleted, treat as new
            pass
        
        if user:
            # Update OAuth tokens
            oauth_account = OAuthAccount(**oauth_account_data)
            oauth_account.access_token = access_token
            oauth_account.refresh_token = refresh_token
            oauth_account.token_expires_at = datetime.utcnow().isoformat()
            oauth_account.updated_at = datetime.utcnow().isoformat()
            backend.save(OAuthAccount, oauth_account.pk, oauth_account.model_dump())
            
            # Update last login
            user.last_login = datetime.utcnow().isoformat()
            if user_info.get("avatar_url") and not user.avatar_url:
                user.avatar_url = user_info["avatar_url"]
            backend.save(UserProfile, user.pk, user.model_dump())
    
    if not user:
        # Check if email is already registered (link accounts)
        if email:
            existing_users = backend.find(UserProfile, {"email": email})
            if existing_users:
                user_data = existing_users[0]
                user = UserProfile(**user_data)
                is_new_user = False
            
        if not user:
            # Create new user
            is_new_user = True
            user_pk = backend.generate_pk()
            
            username = _generate_username(
                user_info.get("name", ""),
                user_info.get("username_hint"),
            )
            
            user = UserProfile(
                pk=user_pk,
                username=username,
                email=email or f"{provider_user_id}@{provider}.oauth",
                password_hash="",  # OAuth users have no password
                auth_provider=provider,
                elo_rating=1000,
                games_played=0,
                games_won=0,
                created_at=datetime.utcnow().isoformat(),
                last_login=datetime.utcnow().isoformat(),
                is_active=True,
                avatar_url=user_info.get("avatar_url", ""),
            )
            backend.save(UserProfile, user.pk, user.model_dump())
        
        # Create OAuth account link
        oauth_pk = backend.generate_pk()
        oauth_account = OAuthAccount(
            pk=oauth_pk,
            user_pk=user.pk,
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=email,
            access_token=access_token,
            refresh_token=refresh_token,
            token_expires_at=datetime.utcnow().isoformat(),
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
        )
        backend.save(OAuthAccount, oauth_account.pk, oauth_account.model_dump())
    
    # Generate JWT for our app
    jwt_token = create_access_token(user.pk, user.username)
    
    # Parse redirect from state
    redirect_url = FRONTEND_URL
    if "|" in state:
        _, redirect_override = state.split("|", 1)
        if redirect_override:
            redirect_url = redirect_override
    
    # Redirect to frontend with token
    # In production, you might use a secure cookie instead
    return RedirectResponse(
        url=f"{redirect_url}?token={jwt_token}&new_user={is_new_user}"
    )


@router.post("/link/{provider}")
async def link_oauth_account(
    provider: str,
    code: str = Query(..., description="Authorization code from provider"),
    user_pk: str = Query(..., description="Current user's PK"),
) -> Dict[str, Any]:
    """Link an OAuth provider to an existing account."""
    _init_providers()
    oauth_provider = _get_provider(provider)
    backend = get_storage_backend()
    
    # Verify user exists
    try:
        user_data = backend.get(UserProfile, user_pk)
        user = UserProfile(**user_data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Build callback URL
    callback_url = f"{OAUTH_CALLBACK_BASE_URL}/oauth/callback/{provider}"
    
    # Exchange code for token
    token_data = await _exchange_code_for_token(
        oauth_provider, code, callback_url
    )
    
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token", "")
    
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No access token received"
        )
    
    # Get user info
    raw_user_info = await _get_user_info(oauth_provider, access_token)
    user_info = _normalize_user_info(provider, raw_user_info)
    
    provider_user_id = user_info["provider_user_id"]
    
    # Check if this OAuth is already linked to another user
    existing_oauth = backend.find(OAuthAccount, {
        "provider": provider,
        "provider_user_id": provider_user_id,
    })
    
    if existing_oauth:
        existing_user_pk = existing_oauth[0].get("user_pk")
        if existing_user_pk != user_pk:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"This {provider} account is already linked to another user"
            )
        # Already linked to this user, just update tokens
        oauth_account = OAuthAccount(**existing_oauth[0])
        oauth_account.access_token = access_token
        oauth_account.refresh_token = refresh_token
        oauth_account.updated_at = datetime.utcnow().isoformat()
        backend.save(OAuthAccount, oauth_account.pk, oauth_account.model_dump())
    else:
        # Create new link
        oauth_pk = backend.generate_pk()
        oauth_account = OAuthAccount(
            pk=oauth_pk,
            user_pk=user.pk,
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=user_info.get("email", ""),
            access_token=access_token,
            refresh_token=refresh_token,
            token_expires_at=datetime.utcnow().isoformat(),
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
        )
        backend.save(OAuthAccount, oauth_account.pk, oauth_account.model_dump())
    
    return {
        "message": f"Successfully linked {provider} account",
        "provider": provider,
        "provider_email": user_info.get("email", ""),
    }


@router.get("/accounts/{user_pk}")
async def get_linked_accounts(user_pk: str) -> Dict[str, Any]:
    """Get all OAuth providers linked to a user account."""
    backend = get_storage_backend()
    
    # Verify user exists
    try:
        backend.get(UserProfile, user_pk)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get all OAuth accounts for this user
    all_oauth = backend.all(OAuthAccount)
    user_oauth = [
        {
            "provider": oa.get("provider"),
            "provider_email": oa.get("provider_email", ""),
            "created_at": oa.get("created_at", ""),
        }
        for oa in all_oauth
        if oa.get("user_pk") == user_pk
    ]
    
    return {
        "user_pk": user_pk,
        "linked_providers": user_oauth,
    }


@router.delete("/unlink/{provider}")
async def unlink_oauth_account(
    provider: str,
    user_pk: str = Query(..., description="Current user's PK"),
) -> Dict[str, str]:
    """Unlink an OAuth provider from an account."""
    backend = get_storage_backend()
    
    # Verify user exists
    try:
        user_data = backend.get(UserProfile, user_pk)
        user = UserProfile(**user_data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if user has password (can't unlink if no other auth method)
    if not user.password_hash:
        # Count linked providers
        all_oauth = backend.all(OAuthAccount)
        user_providers = [
            oa for oa in all_oauth
            if oa.get("user_pk") == user_pk
        ]
        
        if len(user_providers) <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot unlink last authentication method. Set a password first."
            )
    
    # Find and delete the OAuth account
    all_oauth = backend.all(OAuthAccount)
    for oa in all_oauth:
        if oa.get("user_pk") == user_pk and oa.get("provider") == provider:
            backend.delete(OAuthAccount, oa["pk"])
            return {"message": f"Successfully unlinked {provider}"}
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No {provider} account linked"
    )
