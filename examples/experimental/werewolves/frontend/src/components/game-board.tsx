"use client";

import { useEffect, useState } from "react";
import { useWerewolfSession } from "@/hooks/use-werewolf-session";
import { useWerewolfActions } from "@/hooks/use-werewolf-actions";
import {
    PlayerList,
    PhasePanel,
    ActionPanel,
    GameLog,
    GameOverScreen,
    PackPanel,
} from "./supporting-components";

interface WerewolfGameBoardProps {
    sessionId: string;
    participantId: string;
}

type HistoryEntry = {
    id: string;
    phase: string;
    action: string;
    timestamp: number;
};

export function WerewolfGameBoard({
    sessionId,
    participantId,
}: WerewolfGameBoardProps) {
    const { session, isLoading, error } = useWerewolfSession(
        sessionId,
        participantId
    );
    const [actionsState, actionsControls] = useWerewolfActions(
        sessionId,
        participantId
    );

    const [actionHistory, setActionHistory] = useState<HistoryEntry[]>([]);

    useEffect(() => {
        setActionHistory([]);
    }, [sessionId]);

    // Synchronise historical events from backend log
    useEffect(() => {
        if (!session?.log) {
            return;
        }

        const stripTag = (value: unknown): string | null => {
            if (typeof value !== "string") return null;
            const cleaned = value.replace(/^\[[^\]]+\]\s*/, "").trim();
            return cleaned.length ? cleaned : null;
        };

        const describeAction = (
            phase: string,
            actor: string,
            actionValue: Record<string, unknown>
        ): string | null => {
            const actionType = actionValue["action_type"];
            const argumentRaw = actionValue["argument"];
            const argument =
                typeof argumentRaw === "string" ? argumentRaw.trim() : "";
            const lowerArgument = argument.toLowerCase();

            if (phase === "night_werewolves") {
                return null;
            }

            if (actionType === "speak") {
                return null;
            }

            if (actionType === "action") {
                if (!argument) {
                    return `${actor} submits an action.`;
                }
                if (lowerArgument.startsWith("vote")) {
                    const target = argument.slice(4).trim() || "none";
                    return target.toLowerCase() === "none"
                        ? `${actor} votes for no one`
                        : `${actor} votes for ${target}`;
                }
                if (lowerArgument.startsWith("kill")) {
                    const target = argument.slice(4).trim() || "none";
                    return target.toLowerCase() === "none"
                        ? `${actor} decides not to kill anyone`
                        : `${actor} targets ${target}`;
                }
                if (lowerArgument.startsWith("inspect")) {
                    const target = argument.slice(7).trim();
                    return target
                        ? `${actor} inspects ${target}`
                        : `${actor} finishes their vision`;
                }
                if (lowerArgument.startsWith("save")) {
                    const target = argument.slice(4).trim();
                    return target
                        ? `${actor} uses a potion to save ${target}`
                        : `${actor} saves no one`;
                }
                if (lowerArgument.startsWith("poison")) {
                    const target = argument.slice(6).trim();
                    return target
                        ? `${actor} poisons ${target}`
                        : `${actor} withholds the poison`;
                }
                if (lowerArgument === "pass") {
                    return `${actor} passes`;
                }
                return `${actor} performs ${argument}`;
            }

            if (typeof actionType === "string" && actionType.length > 0) {
                return argument
                    ? `${actor} ${actionType}: ${argument}`
                    : `${actor} ${actionType}`;
            }

            return null;
        };

        const sessionLog = session.log;

        setActionHistory((prev) => {
            const existing = new Set(prev.map((entry) => entry.id));
            const next = [...prev];

            const pushMessage = (
                phase: string,
                action: string | null,
                timestamp: number,
                key: string
            ) => {
                if (!action) return;
                if (existing.has(key)) return;
                existing.add(key);
                next.push({
                    id: key,
                    phase,
                    action,
                    timestamp,
                });
            };

            sessionLog.forEach((entry: any, idx: number) => {
                if (!entry || typeof entry !== "object") return;
                const phase =
                    typeof entry.phase === "string" && entry.phase.length
                        ? entry.phase
                        : "unknown";
                const baseTimestamp =
                    typeof entry.recorded_at === "number"
                        ? entry.recorded_at * 1000
                        : Date.now() + idx;

                const formattedActions: string[] = [];
                if (entry.actions && typeof entry.actions === "object") {
                    Object.entries(entry.actions).forEach(
                        ([actor, actionValue]) => {
                            if (
                                !actionValue ||
                                typeof actionValue !== "object"
                            ) {
                                return;
                            }
                            const description = describeAction(
                                phase,
                                actor,
                                actionValue as Record<string, unknown>
                            );
                            if (description) {
                                formattedActions.push(description);
                            }
                        }
                    );
                }

                let actionsInserted = false;
                const maybeInsertActions = () => {
                    if (!actionsInserted && formattedActions.length) {
                        formattedActions.forEach((msg, actionIdx) =>
                            pushMessage(
                                phase,
                                msg,
                                baseTimestamp + actionIdx,
                                `${phase}:${entry.turn}:${msg}`
                            )
                        );
                        actionsInserted = true;
                    }
                };

                if (Array.isArray(entry.public)) {
                    entry.public.forEach((msg: unknown, msgIdx: number) => {
                        const cleaned = stripTag(msg);
                        if (
                            cleaned &&
                            /Votes are tallied|Majority condemns/i.test(cleaned)
                        ) {
                            maybeInsertActions();
                        }
                        pushMessage(
                            phase,
                            cleaned,
                            baseTimestamp + msgIdx + 1,
                            `${phase}:${entry.turn}:public:${msgIdx}:${cleaned}`
                        );
                    });
                }

                maybeInsertActions();

                if (
                    session.me?.id &&
                    entry.private &&
                    typeof entry.private === "object"
                ) {
                    const privateMessages = (
                        entry.private as Record<string, unknown>
                    )[session.me.id];
                    if (Array.isArray(privateMessages)) {
                        privateMessages.forEach(
                            (msg: unknown, privateIdx: number) => {
                                const cleaned = stripTag(msg);
                                if (!cleaned) {
                                    return;
                                }
                                pushMessage(
                                    phase,
                                    `[Private] ${cleaned}`,
                                    baseTimestamp + privateIdx + 3,
                                    `${phase}:${entry.turn}:private:${privateIdx}:${cleaned}`
                                );
                            }
                        );
                    }
                }
            });

            next.sort((a, b) => a.timestamp - b.timestamp);
            return next;
        });
    }, [session?.log]);

    if (isLoading) {
        return (
            <div className="flex h-screen items-center justify-center">
                <div className="space-y-4 text-center">
                    <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                    <p className="text-sm text-muted-foreground">
                        Loading game state...
                    </p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex h-screen items-center justify-center">
                <div className="max-w-md space-y-4 rounded-lg border border-destructive bg-destructive/10 p-6">
                    <h2 className="text-lg font-semibold text-destructive">
                        Connection Error
                    </h2>
                    <p className="text-sm text-muted-foreground">
                        {error instanceof Error
                            ? error.message
                            : "Failed to load game"}
                    </p>
                    <button
                        onClick={() => window.location.reload()}
                        className="w-full rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    if (!session) {
        return (
            <div className="flex h-screen items-center justify-center">
                <p className="text-sm text-muted-foreground">
                    Waiting for game to start...
                </p>
            </div>
        );
    }

    // Show game over screen if game ended
    if (session.gameOver) {
        return (
            <GameOverScreen
                winner={session.winner || "Unknown"}
                message={session.winnerMessage || "Game ended"}
                players={session.players}
                onReturnToLobby={() => window.location.reload()}
            />
        );
    }

    const me = session.me;
    const isMyTurn =
        session.waitingForAction &&
        (session.activePlayerId === me?.id ||
            session.activePlayerId === me?.displayName);

    const actionableActions = Array.isArray(session.availableActions)
        ? session.availableActions.filter(
              (action) =>
                  typeof action === "string" &&
                  action.trim().toLowerCase() !== "none"
          )
        : [];

    const canAct = isMyTurn && actionableActions.length > 0;

    if (session.status === "initializing") {
        return (
            <div className="flex h-screen items-center justify-center">
                <div className="space-y-4 text-center">
                    <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                    <p className="text-sm text-muted-foreground">
                        Preparing Werewolf game…
                    </p>
                </div>
            </div>
        );
    }

    if (session.status === "error") {
        return (
            <div className="flex h-screen items-center justify-center">
                <div className="max-w-md space-y-4 rounded-lg border border-destructive bg-destructive/10 p-6">
                    <h2 className="text-lg font-semibold text-destructive">
                        Game Error
                    </h2>
                    <p className="text-sm text-muted-foreground">
                        The werewolf game encountered an error. Please return to the lobby and try again.
                    </p>
                    <button
                        onClick={() => window.location.reload()}
                        className="w-full rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90"
                    >
                        Return to Lobby
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-screen flex-col">
            {/* Header */}
            <header className="border-b bg-card px-6 py-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold">🌕 Werewolf Game</h1>
                        <p className="text-sm text-muted-foreground">
                            Session: {sessionId.slice(0, 8)}...
                        </p>
                    </div>
                    <div className="text-right">
                        <p className="text-sm font-medium">
                            You are: {me?.displayName || "Observer"}
                        </p>
                        <p className="text-xs text-muted-foreground">
                            Role: {me?.role || "Unknown"}
                        </p>
                    </div>
                </div>
            </header>

            {/* Main Content - 3 Column Layout */}
            <div className="flex flex-1 overflow-hidden">
                {/* Left Sidebar - Players */}
                <aside className="w-64 overflow-y-auto border-r bg-muted/30 p-4">
                    <PlayerList
                        players={session.players}
                        activePlayerId={session.activePlayerId}
                        selfPlayerId={me?.id}
                    />
                </aside>

                {/* Center - Game Log + Action Input */}
                <main className="flex flex-1 flex-col">
                    <GameLog
                        history={actionHistory}
                        currentPhase={session.phase.phase}
                    />

                    {canAct && (
                        <ActionPanel
                            availableActions={actionableActions}
                            onSubmit={actionsControls.submitAction}
                            isSubmitting={actionsState.isSubmitting}
                            error={actionsState.lastError}
                            players={session.players}
                            witchOptions={session.witchOptions}
                            phaseName={session.phase.phase}
                            meRole={me?.role}
                            meName={me?.displayName}
                        />
                    )}

                    {!canAct && (
                        <div className="border-t bg-muted/20 px-4 py-3 text-center text-sm text-muted-foreground">
                            {isMyTurn && actionableActions.length === 0
                                ? "No action required this phase."
                                : isMyTurn
                                ? "Waiting for new instructions from the host..."
                                : session.activePlayerId &&
                                  session.activePlayerId !== me?.id
                                ? `Waiting for ${session.activePlayerId} to act...`
                                : "Waiting for the next turn..."}
                        </div>
                    )}
                </main>

                {/* Right Sidebar - Phase Info */}
                <aside className="w-80 overflow-y-auto border-l bg-muted/30 p-4 space-y-4">
                    <PhasePanel
                        phase={session.phase}
                        availableActions={actionableActions}
                        isMyTurn={isMyTurn}
                    />
                    {session.me?.role?.toLowerCase() === "werewolf" && (
                        <PackPanel
                            members={session.packMembers ?? []}
                            chat={session.teamChat ?? []}
                        />
                    )}
                </aside>
            </div>
        </div>
    );
}
