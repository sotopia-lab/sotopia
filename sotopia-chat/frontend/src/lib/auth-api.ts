/**
 * Authentication API client
 * Handles JWT-based auth with the Sotopia backend
 */

import { API_BASE_URL } from "@/lib/config";

const AUTH_TOKEN_KEY = "sotopia_access_token";
const AUTH_USER_KEY = "sotopia_user";

export interface AuthUser {
    pk: string;
    username: string;
    email: string;
    elo_rating: number;
    games_played: number;
    games_won: number;
    created_at: string;
    auth_provider: string;
    avatar_url: string;
    has_password: boolean;
}

export interface LoginResponse {
    access_token: string;
    token_type: string;
    expires_in: number;
    user: AuthUser;
}

export interface RegisterRequest {
    username: string;
    email: string;
    password: string;
}

export interface LoginRequest {
    username: string;
    password: string;
}

export class AuthError extends Error {
    readonly status: number;
    readonly detail?: unknown;

    constructor(message: string, status: number, detail?: unknown) {
        super(message);
        this.status = status;
        this.detail = detail;
    }
}

// Get stored token
export function getStoredToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem(AUTH_TOKEN_KEY);
}

// Get stored user
export function getStoredUser(): AuthUser | null {
    if (typeof window === "undefined") return null;
    const stored = localStorage.getItem(AUTH_USER_KEY);
    if (!stored) return null;
    try {
        return JSON.parse(stored);
    } catch {
        return null;
    }
}

// Save auth data
function saveAuth(token: string, user: AuthUser): void {
    if (typeof window === "undefined") return;
    localStorage.setItem(AUTH_TOKEN_KEY, token);
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
}

// Clear auth data
export function clearAuth(): void {
    if (typeof window === "undefined") return;
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
}

// Make authenticated request
async function authRequest(
    path: string,
    init?: RequestInit
): Promise<Response> {
    const token = getStoredToken();
    const headers: Record<string, string> = {
        Accept: "application/json",
        ...(init?.headers as Record<string, string> ?? {}),
    };
    
    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }

    const response = await fetch(`${API_BASE_URL}${path}`, {
        cache: "no-store",
        ...init,
        headers,
    });

    if (!response.ok) {
        let detail: unknown;
        try {
            detail = await response.json();
        } catch {
            detail = await response.text();
        }
        throw new AuthError(
            `Request to ${path} failed with status ${response.status}`,
            response.status,
            detail
        );
    }

    return response;
}

// Register a new user
export async function register(data: RegisterRequest): Promise<LoginResponse> {
    const response = await authRequest("/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    const result: LoginResponse = await response.json();
    saveAuth(result.access_token, result.user);
    return result;
}

// Login with username/password
export async function login(data: LoginRequest): Promise<LoginResponse> {
    const response = await authRequest("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    const result: LoginResponse = await response.json();
    saveAuth(result.access_token, result.user);
    return result;
}

// Get current user (validates token)
export async function getCurrentUser(): Promise<AuthUser> {
    const response = await authRequest("/auth/me");
    const user: AuthUser = await response.json();
    // Update stored user data
    const token = getStoredToken();
    if (token) {
        saveAuth(token, user);
    }
    return user;
}

// Logout
export function logout(): void {
    clearAuth();
}

// Check if user is authenticated
export function isAuthenticated(): boolean {
    return getStoredToken() !== null;
}

// OAuth login URLs (backend route is GET /oauth/login/{provider})
export function getOAuthLoginUrl(provider: "google" | "github" | "discord"): string {
    return `${API_BASE_URL}/oauth/login/${provider}`;
}

// Handle OAuth callback (exchange code for token)
export async function handleOAuthCallback(
    provider: "google" | "github" | "discord",
    code: string
): Promise<LoginResponse> {
    const response = await authRequest(`/oauth/${provider}?code=${encodeURIComponent(code)}`, {
        method: "POST",
    });
    const result: LoginResponse = await response.json();
    saveAuth(result.access_token, result.user);
    return result;
}

// Profile API
export interface EloHistoryEntry {
    game_pk: string;
    elo_before: number;
    elo_after: number;
    elo_change: number;
    game_type: string;
    opponent_rating: number;
    won: boolean;
    created_at: string;
}

export interface MatchHistoryEntry {
    game_pk: string;
    game_type: string;
    opponent_type: string;
    opponent_name: string;
    won: boolean;
    elo_change: number;
    played_at: string;
}

export interface ProfileStats {
    pk: string;
    username: string;
    email: string;
    elo_rating: number;
    rank_tier: string;
    rank_emoji: string;
    games_played: number;
    games_won: number;
    win_rate: number;
    created_at: string;
    auth_provider: string;
    avatar_url: string;
}

/** Backend returns { user, rank_tier (object), ... }; we flatten to ProfileStats for the UI. */
export async function getProfile(): Promise<ProfileStats> {
    const response = await authRequest("/profile/me");
    const data = await response.json();
    const u = data.user ?? {};
    const rt = data.rank_tier ?? {};
    const gamesPlayed = u.games_played ?? 0;
    const gamesWon = u.games_won ?? 0;
    return {
        pk: u.pk ?? "",
        username: u.username ?? "",
        email: u.email ?? "",
        elo_rating: u.elo_rating ?? 1000,
        rank_tier: rt.tier ?? "Unranked",
        rank_emoji: rt.emoji ?? "❓",
        games_played: gamesPlayed,
        games_won: gamesWon,
        win_rate: gamesPlayed > 0 ? gamesWon / gamesPlayed : 0,
        created_at: u.created_at ?? "",
        auth_provider: u.auth_provider ?? "local",
        avatar_url: u.avatar_url ?? "",
    };
}

export async function getEloHistory(limit?: number): Promise<EloHistoryEntry[]> {
    const url = limit ? `/profile/me/elo-history?limit=${limit}` : "/profile/me/elo-history";
    const response = await authRequest(url);
    const data = await response.json();
    return data.entries ?? [];
}

/** Backend returns { matches } with pk, created_at, players[]; we map to UI shape (game_pk, played_at, opponent_*). */
export async function getMatchHistory(limit?: number): Promise<MatchHistoryEntry[]> {
    const url = limit ? `/profile/me/matches?limit=${limit}` : "/profile/me/matches";
    const response = await authRequest(url);
    const data = await response.json();
    const raw = (data.matches ?? []) as Array<{
        pk: string;
        game_type: string;
        won: boolean;
        elo_change?: number | null;
        created_at: string;
        players?: Array<{ name?: string; role?: string; is_user?: boolean }>;
    }>;
    return raw.map((m) => {
        const other = m.players?.find((p) => !p.is_user);
        return {
            game_pk: m.pk,
            game_type: m.game_type ?? "werewolf",
            opponent_name: other?.name ?? "AI",
            opponent_type: other?.role ?? "AI",
            won: m.won,
            elo_change: m.elo_change ?? 0,
            played_at: m.created_at ?? "",
        };
    });
}

// Leaderboard API (matches backend: limit, offset, game_type, current_user_pk)
export interface LeaderboardEntry {
    rank: number;
    username: string;
    user_pk: string;
    elo_rating: number;
    games_played: number;
    games_won: number;
    win_rate: number;
    rank_tier: string;
    rank_emoji: string;
    is_current_user: boolean;
}

export interface LeaderboardResponse {
    entries: LeaderboardEntry[];
    total_players: number;
    game_type: string;
    current_user_rank: number | null;
    current_user_entry: LeaderboardEntry | null;
}

export async function getLeaderboard(
    limit?: number,
    offset?: number,
    game_type?: string,
    current_user_pk?: string | null
): Promise<LeaderboardResponse> {
    const params = new URLSearchParams();
    if (limit != null) params.set("limit", limit.toString());
    if (offset != null) params.set("offset", offset.toString());
    if (game_type) params.set("game_type", game_type);
    if (current_user_pk) params.set("current_user_pk", current_user_pk);
    const url = params.toString() ? `/leaderboard?${params}` : "/leaderboard";
    const response = await authRequest(url);
    return response.json();
}

/**
 * Handle OAuth token from URL parameters.
 * Called after OAuth provider redirects back to frontend with token.
 * @returns user if token was valid, null if no token in URL
 */
export async function handleOAuthTokenFromUrl(): Promise<AuthUser | null> {
    if (typeof window === "undefined") return null;
    
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    
    if (!token) return null;
    
    // Store token temporarily to make authenticated request
    localStorage.setItem(AUTH_TOKEN_KEY, token);
    
    try {
        // Fetch user profile with the token
        const user = await getCurrentUser();
        
        // Clean URL by removing token params
        const url = new URL(window.location.href);
        url.searchParams.delete("token");
        url.searchParams.delete("new_user");
        window.history.replaceState({}, "", url.pathname + url.search);
        
        return user;
    } catch (error) {
        // Token invalid, clear it
        clearAuth();
        throw error;
    }
}
