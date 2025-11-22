export type PlayerRole =
    | "villager"
    | "werewolf"
    | "seer"
    | "medic"
    | "moderator"
    | "spectator"
    | string;

export type SessionPhase =
    | "waiting"
    | "intro"
    | "day-discussion"
    | "day-vote"
    | "night"
    | "resolution"
    | "summary"
    | "ended"
    | string;

export interface PlayerState {
    id: string;
    displayName: string;
    role: PlayerRole;
    isAlive: boolean;
    isHost?: boolean;
    team?: string;
}

export interface PackMember {
    id: string;
    displayName: string;
    isAlive: boolean;
    isHuman?: boolean;
}

export interface PackChatMessage {
    phase?: string;
    message: string;
    turn?: number;
    recordedAt?: number;
}

export interface PhaseState {
    phase: SessionPhase;
    countdownSeconds?: number;
    description?: string;
    allowChat?: boolean;
    allowActions?: boolean;
}

export interface WitchOptions {
    canSave: boolean;
    canPoison: boolean;
    pendingTarget?: string | null;
}

export interface WerewolfSessionState {
    sessionId: string;
    players: PlayerState[];
    me?: PlayerState | null;
    phase: PhaseState;
    availableActions: string[];
    lastUpdated: number;
    status: "initializing" | "active" | "completed" | "error" | string;
    gameOver?: boolean;
    winner?: string | null;
    winnerMessage?: string | null;
    log?: Array<Record<string, unknown>>;
    hostId?: string;
    activePlayerId?: string | null;
    waitingForAction?: boolean;
    packMembers?: PackMember[];
    teamChat?: PackChatMessage[];
    witchOptions?: WitchOptions | null;
}

export type WerewolfPlayer = PlayerState;
export type WerewolfPhase = PhaseState;
