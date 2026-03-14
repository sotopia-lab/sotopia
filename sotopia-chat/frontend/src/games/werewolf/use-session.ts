"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { 
    initializeWerewolfGame, 
    createGameStateUpdater,
    type WerewolfServerMessage,
} from "@/games/werewolf/api";
import {
    type WerewolfSessionState,
    type PlayerState,
    type WerewolfPhaseLogEntry,
    type WerewolfActionLog,
    type PackChatMessage,
    type SessionPhase,
} from "@/games/werewolf/types";
import type { SimulationWebSocket } from "@/lib/websocket-utils";
import {
    werewolfAgentIds,
    werewolfAgentModels,
    werewolfAgentProfiles,
    werewolfEnvironmentProfile,
    werewolfEvaluatorModel,
    werewolfMaxTurns,
    werewolfAgentMetadata,
} from "@/games/werewolf/config";
import type { EpisodeLog } from "@/lib/types/sotopia";

export function useWerewolfSession(
    sessionId?: string | null,
    participantIdentifier?: string | null,
    options?: {
        onWebSocketReady?: (ws: SimulationWebSocket) => void;
        humanAgentName?: string;
        humanAgentIndex?: number;
    }
) {
    const [session, setSession] = useState<WerewolfSessionState>(() => createPlaceholderSession());
    const [connectionStatus, setConnectionStatus] = useState<"idle" | "connecting" | "running" | "ended" | "error">(
        "idle"
    );
    const [error, setError] = useState<string | null>(null);
    const wsRef = useRef<SimulationWebSocket | null>(null);
    const activeSessionRef = useRef<string | null>(null);
    const participantRef = useRef<string | null>(participantIdentifier ?? null);
    const basePlayers = useMemo(() => mapMetadataToPlayers(), []);

    // Store options in refs to avoid recreating callbacks when options change
    const onWebSocketReadyRef = useRef(options?.onWebSocketReady);
    const humanAgentNameRef = useRef(options?.humanAgentName);
    const humanAgentIndexRef = useRef(options?.humanAgentIndex);
    
    // Keep refs up to date
    useEffect(() => {
        onWebSocketReadyRef.current = options?.onWebSocketReady;
        humanAgentNameRef.current = options?.humanAgentName;
        humanAgentIndexRef.current = options?.humanAgentIndex;
    }, [options?.onWebSocketReady, options?.humanAgentName, options?.humanAgentIndex]);

    const teardownConnection = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.disconnect();
            wsRef.current = null;
        }
        activeSessionRef.current = null;
    }, []);

    const applyEpisode = useCallback(
        (episode: EpisodeLog) => {
            setSession((prev) => {
                const current =
                    prev ??
                    createInitialSessionState(
                        activeSessionRef.current ?? "werewolf-session",
                        participantRef.current
                    );
                const players = derivePlayersFromEpisode(episode, basePlayers);
                const logEntries = buildSessionLog(episode);
                const mergedLog = mergePhaseLogs(current.log ?? [], logEntries);
                return {
                    ...current,
                    status: "active",
                    phase: {
                        phase: "day_discussion",
                        description: "Simulation in progress",
                        allowChat: false,
                        allowActions: false,
                    },
                    players,
                    packMembers: derivePackMembers(players),
                    log: mergedLog,
                    lastUpdated: Date.now(),
                    waitingForAction: false,
                };
            });
            setConnectionStatus("running");
        },
        [basePlayers]
    );

    // Handle pack chat server messages (defined first - used in handleWerewolfMessage)
    const handlePackChat = useCallback((chatMsg: { from?: string; message?: string; recordedAt?: number }) => {
        setSession((prev) => {
            const current = prev ?? createInitialSessionState(activeSessionRef.current ?? "werewolf-session", participantRef.current);
            const entry = {
                phase: "night",
                turn: prev?.log?.length ?? 0,
                recorded_at: chatMsg.recordedAt ?? Math.floor(Date.now() / 1000),
                public: [],
                private: {},
                actions: [],
            } as unknown as WerewolfPhaseLogEntry;

            const chatEntry = {
                message: `${chatMsg.from ?? "Someone"}: ${chatMsg.message ?? ""}`,
                recordedAt: chatMsg.recordedAt ?? Math.floor(Date.now() / 1000),
            } as PackChatMessage;

            return {
                ...current,
                teamChat: [...(current.teamChat ?? []), chatEntry],
                log: [...(current.log ?? []), entry],
                lastUpdated: Date.now(),
            };
        });
    }, []);

    // Handle werewolf-specific messages from backend
    const handleWerewolfMessage = useCallback((msg: WerewolfServerMessage) => {
        switch (msg.type) {
            case "game_start": {
                const data = msg as {
                    type: string;
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
                };
                setSession((prev) => {
                    const humanPlayer = data.players.find((p) => p.is_human);
                    return {
                        ...prev,
                        status: "active",
                        phase: {
                            phase: data.phase as SessionPhase,
                            description: data.message,
                            allowChat: false,
                            allowActions: false,
                        },
                        players: data.players.map((p) => ({
                            id: p.name,
                            displayName: p.name,
                            role: p.role.toLowerCase(),
                            team: p.team,
                            isAlive: p.is_alive,
                        })),
                        me: humanPlayer
                            ? {
                                  id: humanPlayer.name,
                                  displayName: humanPlayer.name,
                                  role: humanPlayer.role.toLowerCase(),
                                  team: humanPlayer.team,
                                  isAlive: humanPlayer.is_alive,
                              }
                            : prev.me,
                        lastUpdated: Date.now(),
                    };
                });
                setConnectionStatus("running");
                break;
            }
            
            case "phase_change": {
                const data = msg as {
                    type: string;
                    phase: string;
                    turn: number;
                    alive_players: string[];
                };
                setSession((prev) => ({
                    ...prev,
                    phase: {
                        phase: data.phase as SessionPhase,
                        description: `Turn ${data.turn} - ${data.phase}`,
                    },
                    players: prev.players.map((p) => ({
                        ...p,
                        isAlive: data.alive_players.includes(p.displayName),
                    })),
                    lastUpdated: Date.now(),
                }));
                break;
            }
            
            case "action_display": {
                const data = msg as {
                    type: string;
                    actor: string;
                    action_type: string;
                    content: string;
                    phase: string;
                    turn: number;
                };
                setSession((prev) => {
                    const now = Date.now() / 1000;
                    const newLogEntry: WerewolfPhaseLogEntry = {
                        phase: data.phase,
                        turn: data.turn,
                        recorded_at: now,
                        public: [],
                        actions: [{
                            actor: data.actor,
                            action_type: data.action_type,
                            argument: data.content,
                            recorded_at: now,
                        }],
                    };
                    return {
                        ...prev,
                        log: [...(prev.log ?? []), newLogEntry],
                        lastUpdated: Date.now(),
                    };
                });
                break;
            }
            
            case "waiting_for_action": {
                const data = msg as {
                    type: string;
                    agent_name: string;
                    available_actions: string[];
                    phase: string;
                    role?: string;
                    instruction?: string;
                };
                setSession((prev) => {
                    const isMyAction = prev.me?.displayName === data.agent_name;
                    return {
                        ...prev,
                        waitingForAction: true,
                        activePlayerId: data.agent_name,
                        availableActions: data.available_actions,
                        phase: {
                            ...prev.phase,
                            phase: data.phase as SessionPhase,
                            allowActions: isMyAction,
                        },
                        lastUpdated: Date.now(),
                    };
                });
                break;
            }
            
            case "action_taken": {
                const data = msg as {
                    type: string;
                    agent_name: string;
                    action_type: string;
                    argument: string;
                    is_human: boolean;
                    phase?: string;
                };
                // Add to game log so it displays in the UI
                setSession((prev) => {
                    // Use sanitized phase from server, or fall back to current phase (sanitized)
                    const rawPhase = data.phase || prev.phase?.phase || "unknown";
                    const phase = rawPhase.startsWith("Night_") ? "Night" : rawPhase.replace(/_/g, " ");
                    const now = Date.now() / 1000;
                    const newLogEntry: WerewolfPhaseLogEntry = {
                        phase: String(phase),
                        turn: prev.log?.length || 0,
                        recorded_at: now,
                        public: [],
                        actions: [{
                            actor: data.agent_name,
                            action_type: data.action_type,
                            argument: data.argument,
                            recorded_at: now,
                        }],
                    };
                    return {
                        ...prev,
                        log: [...(prev.log ?? []), newLogEntry],
                        lastUpdated: Date.now(),
                    };
                });
                break;
            }
            
            case "game_end": {
                const data = msg as {
                    type: string;
                    winner: string;
                    message: string;
                    final_state?: {
                        alive_players: string[];
                        roles_revealed: Array<{
                            name: string;
                            role: string;
                            team: string;
                            survived: boolean;
                        }>;
                    };
                };
                setSession((prev) => ({
                    ...prev,
                    status: "completed",
                    gameOver: true,
                    winner: data.winner,
                    winnerMessage: data.message,
                    // Update players with revealed roles from final_state
                    players: data.final_state?.roles_revealed
                        ? data.final_state.roles_revealed.map((p) => ({
                              id: p.name,
                              displayName: p.name,
                              role: p.role.toLowerCase(),
                              team: p.team,
                              isAlive: p.survived,
                          }))
                        : prev.players,
                    phase: {
                        phase: "summary",
                        description: data.message,
                        allowActions: false,
                        allowChat: false,
                    },
                    lastUpdated: Date.now(),
                }));
                setConnectionStatus("ended");
                break;
            }
            
            case "guest_warning": {
                // Could show a toast or banner
                break;
            }
            case "pack_chat": {
                const data = msg as { type: string; from?: string; message?: string; recordedAt?: number };
                handlePackChat({ from: data.from, message: data.message, recordedAt: data.recordedAt });
                break;
            }
            
            default:
                break;
        }
    }, [handlePackChat]);

    const markGameEnd = useCallback(() => {
        setConnectionStatus("ended");
        setSession((prev) =>
            prev
                ? {
                      ...prev,
                      status: "completed",
                      gameOver: true,
                      phase: {
                          phase: "summary",
                          description: "Simulation completed",
                          allowActions: false,
                          allowChat: false,
                      },
                  }
                : prev
        );
    }, []);

    const handleError = useCallback((details: unknown) => {
        const message =
            (details as Record<string, unknown>)?.detail ||
            (details ? JSON.stringify(details) : "Simulation error");
        setError(String(message));
        setConnectionStatus("error");
        setSession((prev) =>
            prev
                ? {
                      ...prev,
                      status: "error",
                      phase: {
                          phase: "ended",
                          description: "Simulation error",
                      },
                  }
                : prev
        );
    }, []);

    const startGame = useCallback(
        async (targetSessionId: string, targetParticipantId: string) => {
            setError(null);
            setConnectionStatus("connecting");
            teardownConnection();
            const initial = createInitialSessionState(targetSessionId, targetParticipantId);
            setSession(initial);

            try {
                const ws = await initializeWerewolfGame();
                wsRef.current = ws;
                activeSessionRef.current = targetSessionId;

                // Notify caller that WebSocket is ready (for action submission)
                if (onWebSocketReadyRef.current) {
                    onWebSocketReadyRef.current(ws);
                }

                const updateHandler = createGameStateUpdater({
                    onEpisodeLog: applyEpisode,
                    onWerewolfMessage: handleWerewolfMessage,
                    onGameEnd: () => {
                        activeSessionRef.current = null;
                        markGameEnd();
                    },
                    onError: handleError,
                });
                updateHandler(ws);

                ws.startSimulation({
                    type: "START_SIM",
                    data: {
                        session_id: targetSessionId,
                        participant_id: targetParticipantId,
                        env_id: werewolfEnvironmentProfile.env_id,
                        agent_ids: werewolfAgentIds,
                        agent_models: werewolfAgentModels,
                        evaluator_model: werewolfEvaluatorModel,
                        evaluation_dimension_list_name: "sotopia",
                        max_turns: werewolfMaxTurns,
                        env_profile_dict: werewolfEnvironmentProfile,
                        agent_profile_dicts: werewolfAgentProfiles,
                        // Werewolf-specific fields for human player support
                        game_type: "werewolf",
                        human_agent_name: humanAgentNameRef.current,
                        human_agent_index: humanAgentIndexRef.current,
                    },
                });
            } catch (err) {
                console.error("Failed to start game:", err);
                teardownConnection();
                const message = err instanceof Error ? err.message : "Failed to start game";
                setError(message);
                setConnectionStatus("error");
                setSession((prev) =>
                    prev
                        ? {
                              ...prev,
                              status: "error",
                              phase: {
                                  phase: "ended",
                                  description: "Connection failed",
                              },
                          }
                        : prev
                );
            }
        },
        [applyEpisode, handleWerewolfMessage, handleError, markGameEnd, teardownConnection]
    );

    useEffect(() => {
        participantRef.current = participantIdentifier ?? null;
    }, [participantIdentifier]);

    useEffect(() => {
        if (!sessionId || !participantIdentifier) {
            teardownConnection();
            setConnectionStatus("idle");
            setSession(createPlaceholderSession());
            return;
        }

        // Wait for humanAgentIndex to be set (it comes from the session creation)
        // This ensures we don't start the game before the agent info is ready
        if (humanAgentIndexRef.current === undefined) {
            return;
        }

        if (activeSessionRef.current === sessionId) {
            return;
        }

        startGame(sessionId, participantIdentifier).catch((err) =>
            console.error("[Game] Failed to start:", err)
        );
    }, [sessionId, participantIdentifier, startGame, teardownConnection, options?.humanAgentIndex]);

    useEffect(
        () => () => {
            teardownConnection();
        },
        [teardownConnection]
    );

    return {
        session,
        isLoading: connectionStatus === "connecting",
        error,
    };
}

function createPlaceholderSession(): WerewolfSessionState {
    return {
        sessionId: "werewolf-session",
        players: [],
        phase: {
            phase: "waiting",
            description: "Waiting for lobby",
        },
        availableActions: [],
        lastUpdated: Date.now(),
        status: "initializing",
        log: [],
    };
}

function createInitialSessionState(sessionId: string, participantId: string | null): WerewolfSessionState {
    const players = mapMetadataToPlayers();
    return {
        sessionId,
        players,
        me: participantId
            ? {
                  id: participantId,
                  displayName: participantId,
                  role: "spectator",
                  isAlive: true,
              }
            : null,
        phase: {
            phase: "waiting",
            description: "Initializing simulation…",
        },
        availableActions: [],
        lastUpdated: Date.now(),
        status: "initializing",
        log: [],
        packMembers: derivePackMembers(players),
    };
}

function mapMetadataToPlayers(): PlayerState[] {
    return werewolfAgentMetadata.map((meta) => ({
        id: meta.id,
        displayName: meta.name,
        role: meta.role.toLowerCase(),
        team: meta.team,
        isAlive: true,
    }));
}

function derivePlayersFromEpisode(episode: EpisodeLog, blueprint: PlayerState[]): PlayerState[] {
    const fallen = new Set<string>();
    const lowerNameLookup = new Map(
        blueprint.map((player) => [player.displayName.toLowerCase(), player.id])
    );

    episode.messages.forEach((turn) => {
        turn.forEach(([, , raw]) => {
            if (typeof raw !== "string") return;
            lowerNameLookup.forEach((playerId, lowerName) => {
                if (raw.toLowerCase().includes(`${lowerName} left`)) {
                    fallen.add(playerId);
                }
            });
        });
    });

    return blueprint.map((player) => ({
        ...player,
        isAlive: !fallen.has(player.id),
    }));
}

function derivePackMembers(players: PlayerState[]) {
    return players
        .filter((player) => player.role.toLowerCase().includes("werewolf"))
        .map((player) => ({
            id: player.id,
            displayName: player.displayName,
            isAlive: player.isAlive,
        }));
}

function buildSessionLog(episode: EpisodeLog): WerewolfPhaseLogEntry[] {
    const now = Math.floor(Date.now() / 1000);
    return episode.messages.map((turnMessages, turnIndex) => {
        const recordedAt = now + turnIndex;
        const phase = resolvePhaseForTurn(turnIndex, turnMessages);
        const publicMessages = turnMessages
            .map(([actor, , content]) => formatLogLine(actor, content))
            .filter((value): value is string => Boolean(value));
        const actions = buildActionList(turnMessages, recordedAt);

        return {
            phase,
            turn: turnIndex,
            recorded_at: recordedAt,
            public: publicMessages,
            private: {},
            actions,
        };
    });
}

function mergePhaseLogs(
    existing: WerewolfPhaseLogEntry[],
    incoming: WerewolfPhaseLogEntry[]
): WerewolfPhaseLogEntry[] {
    if (!existing.length) {
        return [...incoming];
    }
    const keyed = new Map<string, WerewolfPhaseLogEntry>();
    existing.forEach((entry) => {
        keyed.set(`${entry.phase}:${entry.turn}`, entry);
    });
    incoming.forEach((entry) => {
        keyed.set(`${entry.phase}:${entry.turn}`, entry);
    });
    return Array.from(keyed.values()).sort(
        (a, b) => a.recorded_at - b.recorded_at
    );
}

function resolvePhaseForTurn(
    turnIndex: number,
    turnMessages: EpisodeLog["messages"][number]
): SessionPhase {
    if (turnIndex === 0) {
        return "intro";
    }

    const serialized = turnMessages
        .map(([, , content]) =>
            typeof content === "string" ? content.toLowerCase() : ""
        )
        .join(" ");

    if (serialized.includes("night") || serialized.includes("dawn")) {
        if (serialized.includes("dawn")) {
            return "dawn_report";
        }
        return "night";
    }

    if (serialized.includes("vote")) {
        return "day_vote";
    }

    if (serialized.includes("execution")) {
        return "twilight_execution";
    }

    return "day_discussion";
}

function formatLogLine(actor: string, content: unknown): string | null {
    if (typeof content !== "string") {
        return null;
    }
    const trimmed = content.trim();
    if (!trimmed.length) {
        return null;
    }
    if (actor === "Environment") {
        return trimmed;
    }
    return `[${actor}] ${trimmed}`;
}

function buildActionList(
    turnMessages: EpisodeLog["messages"][number],
    baseTimestamp: number
): WerewolfActionLog[] {
    const actions: WerewolfActionLog[] = [];
    turnMessages.forEach(([actor, , content], idx) => {
        if (!actor || actor === "Environment" || typeof content !== "string") {
            return;
        }
        const parsed = parseAction(content);
        actions.push({
            actor,
            recorded_at: baseTimestamp + idx,
            ...parsed,
        });
    });
    return actions;
}

function parseAction(content: string) {
    const normalized = content.trim();
    if (/left\./i.test(normalized)) {
        return {
            action_type: "leave",
            argument: "",
        };
    }
    if (/did nothing/i.test(normalized)) {
        return {
            action_type: "none",
            argument: "",
        };
    }
    const speakMatch = normalized.match(/said:\s*(.*)/i);
    if (speakMatch) {
        return {
            action_type: "speak",
            argument: speakMatch[1]?.replace(/^"+|"+$/g, "").trim() ?? "",
        };
    }
    return {
        action_type: "action",
        argument: normalized,
    };
}
