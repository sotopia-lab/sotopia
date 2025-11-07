/**
 * API client functions for Werewolf game endpoints
 */

import type {
    WerewolfSessionState,
    PlayerState,
    PhaseState,
    PackMember,
    PackChatMessage,
    WitchOptions,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface CreateGameResponse {
    session_id: string;
    status: string;
    message: string;
}

export interface WerewolfAction {
    action_type: string;
    argument: string;
}

/**
 * Create a new werewolf game with AI players
 */
export async function createWerewolfGame(
    hostId: string,
    numAiPlayers: number = 5
): Promise<CreateGameResponse> {
    const response = await fetch(`${API_BASE}/games/werewolf/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            host_id: hostId,
            num_ai_players: numAiPlayers,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Failed to create game");
    }

    const raw = await response.json();
    return {
        session_id: raw.session_id,
        status: raw.status,
        message: raw.message,
    };
}

/**
 * Get current game state (called by useWerewolfSession hook)
 */
function resolveProp<T = unknown>(
    raw: Record<string, any>,
    ...keys: string[]
): T | undefined {
    for (const key of keys) {
        if (raw?.[key] !== undefined) {
            return raw[key] as T;
        }
    }
    return undefined;
}

function mapPlayer(raw: any): PlayerState {
    const displayName =
        resolveProp<string>(raw, "displayName", "display_name") ?? raw.id;
    const isAliveRaw = resolveProp(raw, "isAlive", "is_alive");
    const isHostRaw = resolveProp(raw, "isHost", "is_host");

    return {
        id: raw.id,
        displayName,
        role: resolveProp(raw, "role") ?? "unknown",
        isAlive:
            isAliveRaw === undefined ? true : Boolean(isAliveRaw as boolean),
        isHost: Boolean(isHostRaw),
        team: resolveProp<string>(raw, "team"),
    };
}

function mapPhase(raw: any): PhaseState {
    return {
        phase: raw.phase ?? "waiting",
        countdownSeconds: resolveProp<number>(
            raw,
            "countdownSeconds",
            "countdown_seconds"
        ),
        description: resolveProp<string>(raw, "description"),
        allowChat: Boolean(
            resolveProp(raw, "allowChat", "allow_chat") ?? false
        ),
        allowActions: Boolean(
            resolveProp(raw, "allowActions", "allow_actions") ?? false
        ),
    };
}

function mapPackMember(raw: any): PackMember {
    return {
        id: raw.id,
        displayName: resolveProp<string>(raw, "displayName", "display_name") ?? raw.id,
        isAlive: Boolean(resolveProp(raw, "isAlive", "is_alive") ?? true),
        isHuman: Boolean(resolveProp(raw, "isHuman", "is_human") ?? false),
    };
}

function mapPackChat(raw: any): PackChatMessage {
    return {
        phase: raw.phase ?? undefined,
        message: resolveProp<string>(raw, "message") ?? "",
        turn: resolveProp<number>(raw, "turn"),
        recordedAt: resolveProp<number>(raw, "recordedAt", "recorded_at"),
    };
}

function mapWitchOptions(raw: any): WitchOptions {
    return {
        canSave: Boolean(resolveProp(raw, "canSave", "can_save") ?? false),
        canPoison: Boolean(resolveProp(raw, "canPoison", "can_poison") ?? false),
        pendingTarget:
            resolveProp<string | null>(raw, "pendingTarget", "pending_target") ??
            null,
    };
}

export async function getWerewolfSessionState(
    sessionId: string
): Promise<WerewolfSessionState> {
    const response = await fetch(
        `${API_BASE}/games/werewolf/sessions/${sessionId}`
    );

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Failed to fetch game state");
    }

    const raw = await response.json();

    return {
        sessionId: resolveProp<string>(raw, "sessionId", "session_id") ?? sessionId,
        players: Array.isArray(raw.players)
            ? raw.players.map(mapPlayer)
            : [],
        me: raw.me ? mapPlayer(raw.me) : undefined,
        phase: mapPhase(raw.phase ?? {}),
        availableActions:
            resolveProp<string[]>(raw, "availableActions", "available_actions") ??
            [],
        lastUpdated:
            resolveProp<number>(raw, "lastUpdated", "last_updated") ??
            Date.now() / 1000,
        status: resolveProp<string>(raw, "status") ?? "initializing",
        gameOver: Boolean(resolveProp(raw, "gameOver", "game_over")),
        winner: resolveProp<string | null>(raw, "winner") ?? null,
        winnerMessage:
            resolveProp<string | null>(
                raw,
                "winnerMessage",
                "winner_message"
            ) ?? null,
        log: resolveProp(raw, "log") ?? [],
        hostId: resolveProp<string>(raw, "hostId", "host_id"),
        activePlayerId: resolveProp<string | null>(
            raw,
            "activePlayerId",
            "active_player_id"
        ) ?? null,
        waitingForAction: Boolean(
            resolveProp(raw, "waitingForAction", "waiting_for_action")
        ),
        packMembers: Array.isArray(raw.pack_members)
            ? raw.pack_members.map(mapPackMember)
            : [],
        teamChat: Array.isArray(raw.team_chat)
            ? raw.team_chat.map(mapPackChat)
            : [],
        witchOptions: raw.witch_options ? mapWitchOptions(raw.witch_options) : null,
    };
}

/**
 * Submit an action (speak, vote, night ability)
 */
export async function submitWerewolfAction(
    sessionId: string,
    participantId: string,
    action: WerewolfAction
): Promise<{ status: string; message: string }> {
    const response = await fetch(
        `${API_BASE}/games/werewolf/sessions/${sessionId}/actions?participant_id=${participantId}`,
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(action),
        }
    );

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Failed to submit action");
    }

    return response.json();
}

/**
 * Leave game early (cleanup)
 */
export async function leaveWerewolfGame(
    sessionId: string,
    participantId: string
): Promise<void> {
    await fetch(
        `${API_BASE}/games/werewolf/sessions/${sessionId}?participant_id=${participantId}`,
        { method: "DELETE" }
    );
}
