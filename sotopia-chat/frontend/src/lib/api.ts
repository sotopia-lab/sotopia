import { API_BASE_URL, API_ROUTES } from "@/lib/config";
import type {
    AgentProfile,
    EpisodeLog,
    MessageTransaction
} from "@/lib/types/sotopia";
import type { WerewolfSessionState } from "@/games/werewolf/types";

export class ApiError extends Error {
    readonly status: number;
    readonly detail?: unknown;

    constructor(message: string, status: number, detail?: unknown) {
        super(message);
        this.status = status;
        this.detail = detail;
    }
}

async function request(
    path: string,
    init?: RequestInit
): Promise<Response> {
    const response = await fetch(`${API_BASE_URL}${path}`, {
        cache: "no-store",
        ...init,
        headers: {
            Accept: "application/json",
            ...(init?.headers ?? {})
        }
    });

    if (!response.ok) {
        let detail: unknown;
        try {
            detail = await response.json();
        } catch {
            detail = await response.text();
        }
        throw new ApiError(
            `Request to ${path} failed with status ${response.status}`,
            response.status,
            detail
        );
    }

    return response;
}

export async function enterWaitingRoom(
    senderId: string
): Promise<string> {
    const response = await request(
        API_ROUTES.enterWaitingRoom(senderId)
    );
    return response.json();
}

export async function connectSessionAsClient(
    sessionId: string,
    clientId: string,
    { retries = 20, delayMs = 500 } = {}
): Promise<MessageTransaction[]> {
    let attempt = 0;
    let lastError: unknown = null;

    while (attempt < retries) {
        try {
            const response = await request(
                API_ROUTES.connect(sessionId, "client", clientId),
                { method: "POST" }
            );
            return response.json();
        } catch (error) {
            lastError = error;
            await new Promise((resolve) => setTimeout(resolve, delayMs));
            attempt += 1;
        }
    }

    throw lastError instanceof Error
        ? lastError
        : new Error("Failed to join session after multiple attempts.");
}

export async function connectSessionAsServer(
    sessionId: string,
    serverId: string
): Promise<MessageTransaction[]> {
    const response = await request(
        API_ROUTES.connect(sessionId, "server", serverId),
        { method: "POST" }
    );
    return response.json();
}

export async function getSessionMessages(
    sessionId: string
): Promise<MessageTransaction[]> {
    try {
        const response = await request(
            API_ROUTES.getMessages(sessionId)
        );
        return response.json();
    } catch (error) {
        if (error instanceof ApiError && error.status == 404) {
            return [];
        }
        throw error;
    }
}

export async function sendMessage(
    sessionId: string,
    senderId: string,
    message: string
): Promise<MessageTransaction[]> {
    const response = await request(API_ROUTES.send(sessionId, senderId), {
        method: "POST",
        headers: {
            "Content-Type": "text/plain"
        },
        body: message
    });
    return response.json();
}

export async function setClientLock(
    sessionId: string,
    serverId: string,
    lock: "no action" | "action"
): Promise<string> {
    const response = await request(
        API_ROUTES.lock(sessionId, serverId, lock),
        { method: "PUT" }
    );
    return response.json();
}

export async function getClientLock(
    sessionId: string
): Promise<string> {
    try {
        const response = await request(API_ROUTES.getLock(sessionId));
        return response.json();
    } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
            return "no action";
        }
        throw error;
    }
}

export async function deleteSession(
    sessionId: string,
    serverId: string
): Promise<string> {
    const response = await request(
        API_ROUTES.delete(sessionId, serverId),
        { method: "DELETE" }
    );
    return response.json();
}

export async function getEpisode(
    episodeId: string
): Promise<EpisodeLog> {
    const response = await request(
        API_ROUTES.getEpisode(episodeId)
    );
    return response.json();
}

export async function getAgent(
    agentId: string
): Promise<AgentProfile> {
    const response = await request(API_ROUTES.getAgent(agentId));
    return response.json();
}

export async function getWerewolfSessionState(
    sessionId: string
): Promise<WerewolfSessionState> {
    try {
        const response = await request(
            API_ROUTES.werewolfSession(sessionId)
        );
        return response.json();
    } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
            return {
                sessionId,
                players: [],
                me: undefined,
                phase: {
                    phase: "initializing",
                    allowChat: false,
                    allowActions: false,
                },
                availableActions: [],
                lastUpdated: Date.now() / 1000,
                status: "initializing",
                gameOver: false,
                waitingForAction: false,
                log: [],
            };
        }
        throw error;
    }
}

export interface MatchmakingStatus {
    avgWaitSeconds: number;
    activeSessions: number;
    queueDepth: number;
    serverStatus: string;
    lastUpdated: number;
    issues: string[];
}

export interface GameQueueStatus {
    slug: string;
    title: string;
    queueDepth: number;
    avgWaitSeconds: number;
    runningSessions: number;
    lastMatch?: number | null;
    gamesPlayed: number;
    avgSessionSeconds?: number | null;
    enabled: boolean;
}

export interface QueueOverview {
    globalStats: MatchmakingStatus;
    games: GameQueueStatus[];
}

export interface MatchmakingQueuePayload {
    participantId: string;
    games: string[];
    priority?: string;
}

export interface MatchmakingQueueResponse {
    ticketId: string;
    position: number;
    estimatedWaitSeconds: number;
    status: string;
    message: string;
}

export async function fetchQueueOverview(): Promise<QueueOverview> {
    const response = await request(API_ROUTES.queueOverview());
    return response.json();
}

export async function enqueueMatchmaking(
    payload: MatchmakingQueuePayload
): Promise<MatchmakingQueueResponse> {
    const response = await request(API_ROUTES.matchmakingQueue(), {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            participant_id: payload.participantId,
            games: payload.games,
            priority: payload.priority,
        }),
    });
    return response.json();
}

export interface LeaderboardEntry {
    game: string;
    totalMatches: number;
    humanWins: number;
    aiWins: number;
    humanWinRate: number;
    avgDurationSeconds: number;
}

export interface LeaderboardResponse {
    entries: LeaderboardEntry[];
    lastUpdated: number;
}

export interface PersonalHistoryEntry {
    game: string;
    opponentModel: string;
    winner: string;
    durationSeconds: number;
    recordedAt: number;
}

export interface PersonalHistoryResponse {
    participantId: string;
    history: PersonalHistoryEntry[];
}

export interface TicketStatus {
    ticketId: string;
    participantId: string;
    games: string[];
    status: string;
    queuedAt: number;
    matchedAt?: number | null;
    matchedGame?: string | null;
}

export async function fetchLeaderboard(): Promise<LeaderboardResponse> {
    const response = await request(API_ROUTES.leaderboard());
    return response.json();
}

export async function fetchPersonalHistory(
    participantId: string
): Promise<PersonalHistoryResponse> {
    const response = await request(
        API_ROUTES.personalHistory(participantId)
    );
    return response.json();
}

export async function fetchTicketStatus(
    ticketId: string
): Promise<TicketStatus> {
    const response = await request(API_ROUTES.queueTicket(ticketId));
    return response.json();
}

export async function cancelTicket(ticketId: string): Promise<void> {
    await request(API_ROUTES.cancelTicket(ticketId), { method: "DELETE" });
}

export interface IdentityRecord {
    token: string;
    participantId: string;
    displayName?: string;
    createdAt: number;
}

export interface IdentityRequestPayload {
    participantId: string;
    displayName?: string;
}

export async function registerIdentity(
    payload: IdentityRequestPayload
): Promise<IdentityRecord> {
    const response = await request(API_ROUTES.identity(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            participant_id: payload.participantId,
            display_name: payload.displayName,
        }),
    });
    return response.json();
}

export async function fetchIdentity(token: string): Promise<IdentityRecord> {
    const response = await request(API_ROUTES.identity(token));
    return response.json();
}

export interface MemoryPayload {
    participantId: string;
    notes?: string;
    roleHistory?: string[];
    voteHistory?: string[];
    custom?: Record<string, unknown>;
}

export async function fetchMemory(
    participantId: string
): Promise<MemoryPayload> {
    const response = await request(API_ROUTES.memory(participantId));
    const data = await response.json();
    return {
        participantId: data.participant_id,
        notes: data.notes,
        roleHistory: data.role_history,
        voteHistory: data.vote_history,
        custom: data.custom,
    };
}

export async function updateMemory(
    participantId: string,
    payload: MemoryPayload
): Promise<MemoryPayload> {
    const response = await request(API_ROUTES.memory(participantId), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            participant_id: participantId,
            notes: payload.notes,
            role_history: payload.roleHistory,
            vote_history: payload.voteHistory,
            custom: payload.custom,
        }),
    });
    const data = await response.json();
    return {
        participantId: data.participant_id,
        notes: data.notes,
        roleHistory: data.role_history,
        voteHistory: data.vote_history,
        custom: data.custom,
    };
}

export interface AdminStatus {
    uptimeSeconds: number;
    redisAlive: boolean;
    globalStats: MatchmakingStatus;
    games: GameQueueStatus[];
}

export async function fetchAdminStatus(
    token: string
): Promise<AdminStatus> {
    const response = await request(API_ROUTES.adminStatus(), {
        headers: {
            "x-admin-token": token,
        },
    });
    const data = await response.json();
    return data;
}

export async function setGameEnabled(
    slug: string,
    enabled: boolean,
    token: string
): Promise<void> {
    await request(API_ROUTES.adminToggle(slug), {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "x-admin-token": token,
        },
        body: JSON.stringify({ enabled }),
    });
}
