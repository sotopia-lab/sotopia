import type { ComponentType } from "react";

export interface GameActionsState {
    isSubmitting: boolean;
    lastError?: string;
}

export interface GameActionsControls {
    submitAction: (actionType: string, argument: string) => Promise<void>;
    clearError: () => void;
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
    onGameCreated: (sessionId: string, participantId: string) => void;
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
}

export interface GameDefinition<TSession = unknown>
    extends GameSummary {
    features?: {
        teamChat?: boolean;
        spectators?: boolean;
        hasLeaderboard?: boolean;
    };
    components: {
        Consent: ComponentType<ConsentComponentProps>;
        Lobby: ComponentType<LobbyComponentProps>;
        GameBoard: ComponentType<GameBoardProps<TSession>>;
    };
    hooks: {
        useSession: (
            sessionId: string | null,
            participantId: string | null
        ) => GameSessionHookResult<TSession>;
        useActions: (
            sessionId: string | null,
            participantId: string | null
        ) => [GameActionsState, GameActionsControls];
    };
}
