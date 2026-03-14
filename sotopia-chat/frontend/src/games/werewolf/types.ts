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
    | "day_discussion"
    | "day-vote"
    | "day_vote"
    | "dawn_report"
    | "night"
    | "night_werewolves"
    | "night_seer"
    | "night_witch"
    | "twilight_execution"
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

export interface WerewolfActionLog {
    actor: string;
    action_type: string;
    argument: string;
    recorded_at: number;
}

export interface WerewolfPhaseLogEntry {
    phase: SessionPhase;
    turn: number;
    recorded_at: number;
    public: string[];
    private?: Record<string, string[]>;
    actions: WerewolfActionLog[];
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
    log?: WerewolfPhaseLogEntry[];
    hostId?: string;
    activePlayerId?: string | null;
    waitingForAction?: boolean;
    packMembers?: PackMember[];
    teamChat?: PackChatMessage[];
    witchOptions?: WitchOptions | null;
}

export type WerewolfPlayer = PlayerState;
export type WerewolfPhase = PhaseState;
