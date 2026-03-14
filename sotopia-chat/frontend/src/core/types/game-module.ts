import type { ComponentType } from "react";

export interface GameActionsState {
    isSubmitting: boolean;
    lastError?: string;
}

export interface GameActionsControls {
    submitAction: (actionType: string, argument: string) => Promise<void>;
    clearError: () => void;
    /** Optional: Set the WebSocket for sending CLIENT_MSG (used by werewolf) */
    setWebSocket?: (ws: unknown) => void;
}

export interface SessionHookOptions {
    /** Called when WebSocket is ready (for connecting actions hook) */
    onWebSocketReady?: (ws: unknown) => void;
    /** Human player's agent name for multi-agent games */
    humanAgentName?: string;
    /** Human player's agent index for multi-agent games */
    humanAgentIndex?: number;
}

export interface GameSessionHookResult<TSession = unknown> {
    session?: TSession;
    isLoading: boolean;
    error?: unknown;
}

export interface GameBoardProps<TSession = unknown> {
    session: TSession | undefined;
    sessionId: string;
    participantId: string;
    isLoading: boolean;
    error?: unknown;
    actionsState: GameActionsState;
    actionsControls: GameActionsControls;
}

export interface ConsentComponentProps {
    onAccept: () => void;
}

export interface LobbyComponentProps {
    onGameCreated: (
        sessionId: string,
        participantId: string,
        humanAgentInfo?: {
            name: string;
            role: string;
            team: string;
            index: number;
        }
    ) => void;
}

export type GameStatus = "online" | "maintenance" | "coming-soon";

export interface GameSummary {
    slug: string;
    title: string;
    summary: string;
    tags?: string[];
    accentColor?: string;
    minPlayers?: number;
    maxPlayers?: number;
    estDurationMinutes?: number;
    status?: GameStatus;
    features?: {
        teamChat?: boolean;
        spectators?: boolean;
        hasLeaderboard?: boolean;
    };
}

export interface GameDefinition<TSession = unknown>
    extends GameSummary {
    components: {
        Consent: ComponentType<ConsentComponentProps>;
        Lobby: ComponentType<LobbyComponentProps>;
        GameBoard: ComponentType<GameBoardProps<TSession>>;
    };
    hooks: {
        useSession: (
            sessionId: string | null,
            participantId: string | null,
            options?: SessionHookOptions
        ) => GameSessionHookResult<TSession>;
        useActions: (
            sessionId: string | null,
            participantId: string | null
        ) => [GameActionsState, GameActionsControls];
    };
}
