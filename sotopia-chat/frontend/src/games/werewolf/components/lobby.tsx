"use client";

/* eslint-disable react/no-unescaped-entities */

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { createWerewolfGame } from "@/games/werewolf/api";

interface WerewolfLobbyProps {
    onGameCreated: (sessionId: string, playerId: string) => void;
}

export function WerewolfLobby({ onGameCreated }: WerewolfLobbyProps) {
    const [playerId, setPlayerId] = useState("");
    const [isCreating, setIsCreating] = useState(false);
    const [error, setError] = useState<string | undefined>();

    const handleCreateGame = async () => {
        if (!playerId.trim()) {
            setError("Please enter a player identifier");
            return;
        }

        try {
            setIsCreating(true);
            setError(undefined);

            const response = await createWerewolfGame(playerId.trim(), 5);
            onGameCreated(response.session_id, playerId.trim());
        } catch (err) {
            console.error("Failed to create game:", err);
            setError(
                err instanceof Error ? err.message : "Failed to create game"
            );
        } finally {
            setIsCreating(false);
        }
    };

    return (
        <div className="mx-auto flex w-full max-w-xl flex-col gap-6 rounded-lg border bg-card p-8 shadow-lg">
            <header className="space-y-2">
                <h1 className="text-3xl font-bold">🌕 Werewolf Game</h1>
                <p className="text-sm text-muted-foreground">
                    A social deduction game where villagers must identify and
                    eliminate werewolves before it's too late.
                </p>
            </header>

            <div className="space-y-4 rounded-lg bg-muted p-4">
                <h2 className="font-semibold">Game Setup</h2>
                <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• 6 players total: You + 5 AI agents</li>
                    <li>• Roles: 2 Villagers, 2 Werewolves, 1 Seer, 1 Witch</li>
                    <li>• Win condition: Eliminate all werewolves (or achieve parity)</li>
                    <li>• Phases: Night (secret actions) → Day (discussion + voting)</li>
                </ul>
            </div>

            <div className="space-y-2">
                <label className="text-sm font-medium">
                    Your Player Identifier
                </label>
                <input
                    type="text"
                    value={playerId}
                    onChange={(e) => setPlayerId(e.target.value)}
                    placeholder="e.g., alice@example.com or username"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    disabled={isCreating}
                />
                <p className="text-xs text-muted-foreground">
                    This will be used to track your session. Choose any unique
                    identifier.
                </p>
            </div>

            {error && (
                <div className="rounded-md bg-destructive/10 px-4 py-3 text-sm text-destructive">
                    {error}
                </div>
            )}

            <Button
                size="lg"
                onClick={handleCreateGame}
                disabled={isCreating || !playerId.trim()}
                className="w-full"
            >
                {isCreating ? "Creating Game..." : "Start New Game"}
            </Button>

            <div className="space-y-2 rounded-lg border-l-4 border-amber-500 bg-amber-50 p-4 dark:bg-amber-950">
                <h3 className="text-sm font-semibold text-amber-900 dark:text-amber-100">
                    📋 Quick Rules
                </h3>
                <div className="space-y-1 text-xs text-amber-800 dark:text-amber-200">
                    <p>
                        <strong>Night:</strong> Werewolves choose a victim, Seer inspects a player, Witch can save/poison
                    </p>
                    <p>
                        <strong>Day:</strong> Discuss suspicions, then vote to execute someone
                    </p>
                    <p>
                        <strong>Actions:</strong> Use commands like "vote Alice" or "inspect Bob"
                    </p>
                </div>
            </div>
        </div>
    );
}
