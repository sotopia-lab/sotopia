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
