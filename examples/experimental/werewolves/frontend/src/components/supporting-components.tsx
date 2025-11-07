"use client";

import { useMemo, useState } from "react";
import type {
    WerewolfPlayer,
    WerewolfPhase,
    PackMember,
    PackChatMessage,
    WitchOptions,
} from "@/lib/types";
import { Button } from "@/components/ui/button";

// ============= Player List Component =============
interface PlayerListProps {
    players: WerewolfPlayer[];
    activePlayerId?: string | null;
    selfPlayerId?: string;
}

export function PlayerList({
    players,
    activePlayerId,
    selfPlayerId,
}: PlayerListProps) {
    return (
        <div className="space-y-3">
            <h2 className="text-lg font-semibold">👥 Players</h2>
            <ul className="space-y-2">
                {players.map((player) => {
                    const isActive =
                        player.id === activePlayerId ||
                        player.displayName === activePlayerId;
                    const isSelf = player.id === selfPlayerId;
                    return (
                        <li
                            key={player.id}
                            className={`rounded-lg border p-3 ${
                                isActive
                                    ? "border-primary bg-primary/5 shadow-sm"
                                    : "border-border"
                            } ${!player.isAlive ? "opacity-50" : ""}`}
                        >
                            <div className="flex items-start justify-between">
                                <div className="flex-1 space-y-1">
                                    <div className="flex flex-wrap items-center gap-2">
                                        <span className="font-medium">
                                            {player.displayName}
                                        </span>
                                        {player.isHost && (
                                            <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-800">
                                                Host
                                            </span>
                                        )}
                                        {isSelf && (
                                            <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-semibold text-sky-800">
                                                You
                                            </span>
                                        )}
                                        {isActive && (
                                            <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-700">
                                                Taking Turn
                                            </span>
                                        )}
                                    </div>
                                    <p className="text-xs text-muted-foreground">
                                        {player.isAlive
                                            ? player.role
                                            : `Revealed: ${player.role}`}
                                    </p>
                                </div>
                                <span
                                    className={`text-xs font-semibold uppercase ${
                                        player.isAlive
                                            ? "text-emerald-600"
                                            : "text-red-600"
                                    }`}
                                >
                                    {player.isAlive ? "Alive" : "Eliminated 💀"}
                                </span>
                            </div>
                        </li>
                    );
                })}
            </ul>
        </div>
    );
}

// ============= Phase Panel Component =============
interface PhasePanelProps {
    phase: WerewolfPhase;
    availableActions: string[];
    isMyTurn?: boolean;
}

export function PhasePanel({
    phase,
    availableActions,
    isMyTurn = false,
}: PhasePanelProps) {
    return (
        <div className="space-y-4">
            <div className="rounded-lg bg-gradient-to-br from-purple-500 to-indigo-600 p-4 text-white">
                <h2 className="text-sm font-medium uppercase opacity-90">
                    Current Phase
                </h2>
                <p className="text-2xl font-bold">{formatPhase(phase.phase)}</p>
                {phase.description && (
                    <p className="mt-2 text-sm opacity-90">{phase.description}</p>
                )}
            </div>

            <div
                className={`rounded-md border px-3 py-2 text-sm ${
                    isMyTurn
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700"
                        : "border-border bg-muted/40 text-muted-foreground"
                }`}
            >
                {isMyTurn
                    ? "Your turn! Choose an action while the phase is active."
                    : "Waiting for other players to finish their actions."}
            </div>

            <div className="space-y-2">
                <h3 className="text-sm font-semibold">Available Actions</h3>
                {availableActions.length > 0 ? (
                    <ul className="flex flex-wrap gap-2">
                        {availableActions.map((action) => (
                            <li
                                key={action}
                                className="rounded-full border border-border bg-background px-3 py-1 text-xs font-medium"
                            >
                                {action}
                            </li>
                        ))}
                    </ul>
                ) : (
                    <p className="text-sm text-muted-foreground">
                        {isMyTurn
                            ? "Submit an action using the panel below."
                            : "Waiting for your turn..."}
                    </p>
                )}
            </div>

            <div className="space-y-2 rounded-lg bg-muted p-4 text-sm">
                <h3 className="font-semibold">Phase Rules</h3>
                {getPhaseRules(phase.phase)}
            </div>
        </div>
    );
}

// ============= Action Panel Component =============
interface ActionPanelProps {
    availableActions: string[];
    onSubmit: (actionType: string, argument: string) => Promise<void>;
    isSubmitting: boolean;
    error?: string;
    players: WerewolfPlayer[];
    witchOptions?: WitchOptions | null;
    phaseName?: string;
    meRole?: string;
    meName?: string;
}

export function ActionPanel({
    availableActions,
    onSubmit,
    isSubmitting,
    error,
    players,
    witchOptions,
    phaseName,
    meRole,
    meName,
}: ActionPanelProps) {
    const [selectedAction, setSelectedAction] = useState("");
    const [argument, setArgument] = useState("");
    const [poisonTarget, setPoisonTarget] = useState("");
    const [witchError, setWitchError] = useState<string | undefined>(undefined);
    const [seerTarget, setSeerTarget] = useState("");
    const [seerError, setSeerError] = useState<string | undefined>(undefined);

    const handleSubmit = async () => {
        if (!selectedAction) return;
        await onSubmit(selectedAction, argument);
        setArgument("");
        setSelectedAction("");
    };

    const alivePlayers = players.filter((p) => p.isAlive);

    const isWitchTurn =
        phaseName === "night_witch" && meRole?.toLowerCase() === "witch";

    const isSeerTurn =
        phaseName === "night_seer" && meRole?.toLowerCase() === "seer";

    if (isWitchTurn) {
        const aliveTargets = players.filter((p) => p.isAlive);
        const pendingTarget = witchOptions?.pendingTarget || "";
        const canSave = witchOptions?.canSave ?? true;
        const canPoison = witchOptions?.canPoison ?? true;

        const handleWitchAction = async (mode: "save" | "poison" | "pass") => {
            if (isSubmitting) return;
            try {
                setWitchError(undefined);
                if (mode === "save") {
                    if (!pendingTarget) {
                        setWitchError("No player to save this night.");
                        return;
                    }
                    await onSubmit("action", `save ${pendingTarget}`);
                } else if (mode === "poison") {
                    if (!poisonTarget) {
                        setWitchError("Select a player to poison.");
                        return;
                    }
                    await onSubmit("action", `poison ${poisonTarget}`);
                    setPoisonTarget("");
                } else {
                    await onSubmit("action", "pass");
                }
            } catch (err) {
                if (err instanceof Error) {
                    setWitchError(err.message);
                } else {
                    setWitchError("Failed to submit action.");
                }
            }
        };

        return (
            <div className="border-t bg-card p-4">
                <div className="mx-auto max-w-3xl space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="font-semibold">🧪 Witch Potions</h3>
                        {isSubmitting && (
                            <span className="text-sm text-muted-foreground">
                                Submitting...
                            </span>
                        )}
                    </div>

                    <div className="rounded-md border border-border bg-muted/40 p-3 text-sm text-muted-foreground">
                        {pendingTarget
                            ? `Werewolves targeted ${pendingTarget}.`
                            : "No victim identified yet."}
                    </div>

                    <div className="space-y-2">
                        <button
                            className="w-full rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
                            onClick={() => handleWitchAction("save")}
                            disabled={!canSave || !pendingTarget || isSubmitting}
                        >
                            {canSave
                                ? pendingTarget
                                    ? `Save ${pendingTarget}`
                                    : "No one to save"
                                : "Save potion used"}
                        </button>

                        <div className="rounded-md border border-border p-3">
                            <label className="mb-1 block text-sm font-medium">
                                Poison a player
                            </label>
                            <select
                                value={poisonTarget}
                                onChange={(e) => setPoisonTarget(e.target.value)}
                                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            disabled={!canPoison || isSubmitting}
                            >
                                <option value="">-- Select player --</option>
                                {aliveTargets.map((player) => (
                                    <option key={player.id} value={player.displayName}>
                                        {player.displayName}
                                    </option>
                                ))}
                            </select>
                            <button
                                className="mt-2 w-full rounded-md bg-rose-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
                                onClick={() => handleWitchAction("poison")}
                                disabled={!canPoison || isSubmitting}
                            >
                                {canPoison
                                    ? "Use poison potion"
                                    : "Poison potion used"}
                            </button>
                        </div>

                        <button
                            className="w-full rounded-md border border-border px-4 py-2 text-sm font-medium text-foreground hover:bg-muted disabled:opacity-50"
                            onClick={() => handleWitchAction("pass")}
                            disabled={isSubmitting}
                        >
                            Pass / Do nothing
                        </button>
                    </div>

                    {(error || witchError) && (
                        <div className="rounded-md bg-destructive/10 px-4 py-2 text-sm text-destructive">
                            {witchError || error}
                        </div>
                    )}
                </div>
            </div>
        );
    }

    if (isSeerTurn) {
        const handleInspect = async () => {
            if (!seerTarget) {
                setSeerError("Select a player to inspect.");
                return;
            }
            try {
                setSeerError(undefined);
                await onSubmit("action", `inspect ${seerTarget}`);
                setSeerTarget("");
            } catch (err) {
                setSeerError(
                    err instanceof Error
                        ? err.message
                        : "Failed to submit inspect action."
                );
            }
        };

        return (
            <div className="border-t bg-card p-4">
                <div className="mx-auto max-w-3xl space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="font-semibold">🔮 Seer Vision</h3>
                        {isSubmitting && (
                            <span className="text-sm text-muted-foreground">
                                Inspecting...
                            </span>
                        )}
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-medium">
                            Choose a player to inspect
                        </label>
                        <select
                            value={seerTarget}
                            onChange={(e) => setSeerTarget(e.target.value)}
                            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            disabled={isSubmitting}
                        >
                            <option value="">-- Select a player --</option>
                            {players
                                .filter(
                                    (player) =>
                                        player.isAlive &&
                                        player.displayName !== meName
                                )
                                .map((player) => (
                                    <option key={player.id} value={player.displayName}>
                                        {player.displayName}
                                    </option>
                                ))}
                        </select>
                    </div>

                    {seerError && (
                        <div className="rounded-md bg-destructive/10 px-4 py-2 text-sm text-destructive">
                            {seerError}
                        </div>
                    )}

                    <Button
                        onClick={handleInspect}
                        disabled={!seerTarget || isSubmitting}
                        className="w-full"
                        size="lg"
                    >
                        Inspect Player
                    </Button>
                </div>
            </div>
        );
    }

    return (
        <div className="border-t bg-card p-4">
            <div className="mx-auto max-w-3xl space-y-4">
                <div className="flex items-center justify-between">
                    <h3 className="font-semibold">🎮 Your Turn</h3>
                    {isSubmitting && (
                        <span className="text-sm text-muted-foreground">
                            Submitting...
                        </span>
                    )}
                </div>

                {error && (
                    <div className="rounded-md bg-destructive/10 px-4 py-2 text-sm text-destructive">
                        {error}
                    </div>
                )}

                {/* Action Type Selection */}
                <div className="flex gap-2">
                    {availableActions.map((action) => (
                        <Button
                            key={action}
                            variant={selectedAction === action ? "default" : "outline"}
                            size="sm"
                            onClick={() => setSelectedAction(action)}
                            disabled={isSubmitting}
                        >
                            {action}
                        </Button>
                    ))}
                </div>

                {/* Argument Input */}
                {selectedAction && selectedAction !== "none" && (
                    <div className="space-y-2">
                        {selectedAction === "speak" ? (
                            <textarea
                                value={argument}
                                onChange={(e) => setArgument(e.target.value)}
                                placeholder="What do you want to say?"
                                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                rows={3}
                                disabled={isSubmitting}
                            />
                        ) : (
                            <div>
                                <label className="mb-2 block text-sm font-medium">
                                    Target Player
                                </label>
                                <select
                                    value={argument}
                                    onChange={(e) => setArgument(e.target.value)}
                                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                    disabled={isSubmitting}
                                >
                                    <option value="">-- Select a player --</option>
                                    {alivePlayers.map((player) => (
                                        <option
                                            key={player.id}
                                            value={player.displayName}
                                        >
                                            {player.displayName}
                                        </option>
                                    ))}
                                    <option value="none">None</option>
                                </select>
                            </div>
                        )}
                    </div>
                )}

                <Button
                    onClick={handleSubmit}
                    disabled={!selectedAction || isSubmitting}
                    className="w-full"
                    size="lg"
                >
                    Submit Action
                </Button>
            </div>
        </div>
    );
}

// ============= Game Log Component =============
interface GameLogProps {
    history: Array<{
        id?: string;
        phase: string;
        action: string;
        timestamp: number;
    }>;
    currentPhase: string;
}

export function GameLog({ history, currentPhase }: GameLogProps) {
    return (
        <div className="flex-1 overflow-y-auto p-6">
            <div className="mx-auto max-w-3xl space-y-4">
                <h2 className="text-lg font-semibold">📜 Game Log</h2>
                {history.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                        Game starting...
                    </p>
                ) : (
                    <ul className="space-y-3">
                        {history.map((entry, idx) => (
                            <li
                                key={entry.id ?? idx}
                                className="rounded-lg border border-border bg-card p-4"
                            >
                                <div className="flex items-start justify-between">
                                    <div>
                                        <p className="text-sm font-medium">
                                            {entry.action}
                                        </p>
                                        <p className="text-xs text-muted-foreground">
                                            {new Date(
                                                entry.timestamp
                                            ).toLocaleTimeString()}
                                        </p>
                                    </div>
                                    <span className="rounded-full bg-primary/10 px-2 py-1 text-xs font-medium text-primary">
                                        {formatPhase(entry.phase)}
                                    </span>
                                </div>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

// ============= Game Over Screen =============
interface GameOverScreenProps {
    winner: string;
    message: string;
    players: WerewolfPlayer[];
    onReturnToLobby: () => void;
}

export function GameOverScreen({
    winner,
    message,
    players,
    onReturnToLobby,
}: GameOverScreenProps) {
    return (
        <div className="flex h-screen items-center justify-center bg-gradient-to-br from-slate-900 to-slate-800 p-6">
            <div className="w-full max-w-2xl space-y-6 rounded-xl border border-border bg-card p-8 shadow-2xl">
                <div className="text-center">
                    <h1 className="mb-2 text-4xl font-bold">🎮 Game Over</h1>
                    <p className="text-2xl font-semibold text-primary">
                        {winner} Win!
                    </p>
                    <p className="mt-2 text-sm text-muted-foreground">{message}</p>
                </div>

                <div className="space-y-3">
                    <h2 className="text-lg font-semibold">Final Standings</h2>
                    <ul className="space-y-2">
                        {players.map((player) => (
                            <li
                                key={player.id}
                                className="flex items-center justify-between rounded-lg border border-border bg-muted/50 p-3"
                            >
                                <div>
                                    <p className="font-medium">
                                        {player.displayName}
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                        {player.role}
                                    </p>
                                </div>
                                <span
                                    className={`text-sm font-semibold ${
                                        player.isAlive
                                            ? "text-emerald-600"
                                            : "text-red-600"
                                    }`}
                                >
                                    {player.isAlive ? "Survived" : "Eliminated"}
                                </span>
                            </li>
                        ))}
                    </ul>
                </div>

                <Button onClick={onReturnToLobby} className="w-full" size="lg">
                    Return to Lobby
                </Button>
            </div>
        </div>
    );
}

// ============= Pack Panel =============
interface PackPanelProps {
    members: PackMember[];
    chat: PackChatMessage[];
}

export function PackPanel({ members, chat }: PackPanelProps) {
    const latestMessages = useMemo(
        () => chat.slice(-10).reverse(),
        [chat]
    );

    return (
        <div className="space-y-4 rounded-lg border border-border bg-card/70 p-4">
            <div>
                <h3 className="text-sm font-semibold text-muted-foreground">
                    Packmates
                </h3>
                <ul className="mt-2 space-y-1 text-sm">
                    {members.map((ally) => (
                        <li
                            key={ally.id}
                            className="flex items-center justify-between text-foreground"
                        >
                            <span>{ally.displayName}</span>
                            <span
                                className={`text-xs uppercase ${
                                    ally.isAlive
                                    ? "text-emerald-500"
                                    : "text-destructive"
                                }`}
                            >
                                {ally.isHuman
                                    ? "You"
                                    : ally.isAlive
                                    ? "Alive"
                                    : "Eliminated"}
                            </span>
                        </li>
                    ))}
                    {members.length === 0 && (
                        <li className="text-xs text-muted-foreground">
                            Waiting for another werewolf to join the pack…
                        </li>
                    )}
                </ul>
            </div>

            <div>
                <h3 className="text-sm font-semibold text-muted-foreground">
                    Pack Chat
                </h3>
                {latestMessages.length === 0 ? (
                    <p className="mt-2 text-xs text-muted-foreground">
                        Quiet night. Coordinate with your fellow wolves once they speak.
                    </p>
                ) : (
                    <ul className="mt-2 space-y-2 text-xs">
                        {latestMessages.map((entry, idx) => (
                            <li
                                key={`${entry.recordedAt ?? idx}-${idx}`}
                                className="rounded-md bg-muted/60 p-2 text-muted-foreground"
                            >
                                <p>{stripSystemTags(entry.message)}</p>
                                {entry.recordedAt && (
                                    <p className="mt-1 text-[10px] uppercase tracking-wide text-muted-foreground/70">
                                        {new Date(entry.recordedAt * 1000).toLocaleTimeString()}
                                    </p>
                                )}
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

// ============= Helper Functions =============
function formatPhase(phase: string): string {
    return phase
        .split("_")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
}

function getPhaseRules(phase: string): React.ReactNode {
    const rules: Record<string, React.ReactNode> = {
        night_werewolves: (
            <p>
                <strong>Werewolves:</strong> Secretly choose a victim using the
                "action" command with "kill [NAME]".
            </p>
        ),
        night_seer: (
            <p>
                <strong>Seer:</strong> Inspect a player using "inspect [NAME]" to
                learn their alignment.
            </p>
        ),
        night_witch: (
            <p>
                <strong>Witch:</strong> Use "save [NAME]" or "poison [NAME]". Each
                potion can only be used once.
            </p>
        ),
        dawn_report: (
            <p>
                <strong>Dawn:</strong> Results of night actions are announced. See
                who died.
            </p>
        ),
        day_discussion: (
            <p>
                <strong>Discussion:</strong> Openly debate suspicions using "speak".
                Share theories!
            </p>
        ),
        day_vote: (
            <p>
                <strong>Voting:</strong> Vote to execute a suspected werewolf using
                "action" with "vote [NAME]".
            </p>
        ),
        twilight_execution: (
            <p>
                <strong>Execution:</strong> The vote result is revealed. Night
                returns.
            </p>
        ),
    };

    return (
        rules[phase] || (
            <p className="text-muted-foreground">Phase in progress...</p>
        )
    );
}

function stripSystemTags(value?: string): string {
    if (!value) return "";
    return value.replace(/^\[[^\]]+\]\s*/g, "").replace(/^"+|"+$/g, "");
}
