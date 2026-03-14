/**
 * API client for Werewolf game via WebSocket
 * 
 * Connects to the Sotopia backend FastAPI /ws/simulation WebSocket endpoint
 * and manages game state through message streams.
 */

import {
    SimulationWebSocket,
    getOrCreateSessionToken,
    WSMessageType,
} from "@/lib/websocket-utils";
import type { EpisodeLog } from "@/lib/types/sotopia";

const WS_BASE =
    process.env.NEXT_PUBLIC_WS_BASE || "ws://127.0.0.1:8800";

/**
 * Initialize a Werewolf game session
 */
export async function initializeWerewolfGame(): Promise<SimulationWebSocket> {
    const token = getOrCreateSessionToken();
    const ws = new SimulationWebSocket(WS_BASE, token);

    try {
        await ws.connect();
        return ws;
    } catch (err) {
        ws.disconnect();
        throw err;
    }
}

export interface GameStateCallbacks {
    onEpisodeLog: (episode: EpisodeLog) => void;
    onGameEnd?: () => void;
    onError?: (data: unknown) => void;
    onWerewolfMessage?: (data: WerewolfServerMessage) => void;
}

/**
 * Server message types from werewolf backend
 */
export interface WerewolfServerMessage {
    type: string;
    [key: string]: unknown;
}

export interface WerewolfGameStartMessage extends WerewolfServerMessage {
    type: "game_start";
    phase: string;
    players: Array<{
        name: string;
        role: string;
        team: string;
        is_alive: boolean;
        is_human: boolean;
    }>;
    message: string;
    is_guest: boolean;
}

export interface WerewolfPhaseChangeMessage extends WerewolfServerMessage {
    type: "phase_change";
    phase: string;
    turn: number;
    alive_players: string[];
}

export interface WerewolfActionDisplayMessage extends WerewolfServerMessage {
    type: "action_display";
    actor: string;
    action_type: string;
    content: string;
    phase: string;
    turn: number;
}

export interface WerewolfWaitingForActionMessage extends WerewolfServerMessage {
    type: "waiting_for_action";
    agent_name: string;
    available_actions: string[];
    phase: string;
}

export interface WerewolfGameEndMessage extends WerewolfServerMessage {
    type: "game_end";
    winner: string;
    message: string;
}

/**
 * Register WebSocket handlers and surface structured callbacks
 */
export function createGameStateUpdater(callbacks: GameStateCallbacks) {
    return (ws: SimulationWebSocket) => {
        ws.on(WSMessageType.SERVER_MSG, (data) => {
            const payload = data as WerewolfServerMessage;
            
            // Old format: { type: "messages", messages: EpisodeLog }
            if (payload?.type === "messages" && "messages" in payload) {
                callbacks.onEpisodeLog(payload.messages as EpisodeLog);
                return;
            }
            
            // New werewolf-specific format
            if (payload?.type && callbacks.onWerewolfMessage) {
                callbacks.onWerewolfMessage(payload);
            }
            
            // Check for game end
            if (payload?.type === "game_end") {
                callbacks.onGameEnd?.();
            }
        });

        ws.on(WSMessageType.END_SIM, () => {
            callbacks.onGameEnd?.();
        });

        ws.on(WSMessageType.ERROR, (data) => {
            console.error("[Game] Error received:", data);
            callbacks.onError?.(data);
        });
    };
}

/**
 * Send a game action via WebSocket
 */
export function sendGameAction(
    ws: SimulationWebSocket,
    actionType: string,
    actionData: Record<string, unknown>
): void {
    ws.sendClientMessage({
        type: "CLIENT_MSG",
        action_type: actionType,
        ...actionData,
    });
}
