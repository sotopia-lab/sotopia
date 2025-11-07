const rawBaseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export const API_BASE_URL = rawBaseUrl.replace(/\/$/, "");

export const API_ROUTES = {
    connect: (sessionId: string, role: "server" | "client", id: string) =>
        `/connect/${encodeURIComponent(sessionId)}/${role}/${encodeURIComponent(id)}`,
    send: (sessionId: string, senderId: string) =>
        `/send/${encodeURIComponent(sessionId)}/${encodeURIComponent(senderId)}`,
    lock: (sessionId: string, serverId: string, lock: "action" | "no action") =>
        `/lock/${encodeURIComponent(sessionId)}/${encodeURIComponent(serverId)}/${lock}`,
    delete: (sessionId: string, serverId: string) =>
        `/delete/${encodeURIComponent(sessionId)}/${encodeURIComponent(serverId)}`,
    getMessages: (sessionId: string) =>
        `/get/${encodeURIComponent(sessionId)}`,
    getLock: (sessionId: string) => `/get_lock/${encodeURIComponent(sessionId)}`,
    enterWaitingRoom: (senderId: string) =>
        `/enter_waiting_room/${encodeURIComponent(senderId)}`,
    getEpisode: (episodeId: string) =>
        `/get_episode/${encodeURIComponent(episodeId)}`,
    getAgent: (agentId: string) => `/get_agent/${encodeURIComponent(agentId)}`,
    werewolfSession: (sessionId: string) =>
        `/games/werewolf/sessions/${encodeURIComponent(sessionId)}`
} as const;
