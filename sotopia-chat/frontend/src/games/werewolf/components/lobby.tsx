"use client";

/* eslint-disable react/no-unescaped-entities */

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/auth-context";

interface WerewolfLobbyProps {
    onGameCreated: (sessionId: string, participantId: string, humanAgentInfo?: {
        name: string;
        role: string;
        team: string;
        index: number;
    }) => void;
}

interface WerewolfSessionResponse {
    session_id: string;
    participant_id: string;
    human_agent: {
        name: string;
        role: string;
        team: string;
        index: number;
    };
    all_agents: Array<{
        name: string;
        role: string;
        team: string;
        is_human: boolean;
    }>;
    game_config: {
        total_players: number;
        werewolf_count: number;
        villager_count: number;
    };
}

export function WerewolfLobby({ onGameCreated }: WerewolfLobbyProps) {
    const { user, isAuthenticated } = useAuth();
    const [isCreating, setIsCreating] = useState(false);
    const [error, setError] = useState<string | undefined>();
    const [consentChecked, setConsentChecked] = useState(false);
    const [roleInfo, setRoleInfo] = useState<WerewolfSessionResponse | null>(null);

    // Generate a participant ID based on auth state
    const getParticipantId = (): string => {
        if (isAuthenticated && user?.username) {
            return user.username;
        }
        // For guests, generate a unique ID
        return `guest_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
    };

    const handleStartGame = async () => {
        if (!consentChecked) {
            setError("Please accept the research consent to continue");
            return;
        }

        setIsCreating(true);
        setError(undefined);

        try {
            const participantId = getParticipantId();
            const apiUrl =
                process.env.NEXT_PUBLIC_API_BASE_URL ||
                process.env.NEXT_PUBLIC_SOTOPIA_API_URL ||
                "http://localhost:8800";
            const response = await fetch(
                `${apiUrl.replace(/\/$/, "")}/games/werewolf/sessions/create?participant_id=${encodeURIComponent(participantId)}&random_role=true`,
                { method: "POST" }
            );

            if (!response.ok) {
                throw new Error(`Failed to create session: ${response.statusText}`);
            }

            const data: WerewolfSessionResponse = await response.json();
            setRoleInfo(data);
            
            // Pass the human agent info to the parent
            onGameCreated(data.session_id, data.participant_id, data.human_agent);
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
        <div className="mx-auto flex w-full max-w-2xl flex-col gap-6 rounded-lg border bg-card p-8 shadow-lg max-h-[90vh] overflow-y-auto">
            <header className="space-y-2">
                <h1 className="text-3xl font-bold">🌕 Werewolf Game</h1>
                <p className="text-sm text-muted-foreground">
                    A social deduction game where villagers must identify and
                    eliminate werewolves before it's too late.
                </p>
            </header>

            {/* Game Rules */}
            <div className="space-y-4 rounded-lg bg-muted p-4">
                <h2 className="font-semibold">📜 Game Rules</h2>
                <div className="space-y-3 text-sm text-muted-foreground">
                    <div>
                        <p className="font-medium text-foreground">Setup:</p>
                        <ul className="mt-1 space-y-1 ml-4">
                            <li>• 6 players: You + 5 AI agents</li>
                            <li>• Roles: 2 Villagers, 2 Werewolves, 1 Seer, 1 Witch</li>
                            <li>• Your role is randomly assigned</li>
                        </ul>
                    </div>
                    <div>
                        <p className="font-medium text-foreground">Night Phase:</p>
                        <ul className="mt-1 space-y-1 ml-4">
                            <li>• <strong>Werewolves:</strong> Choose a victim to kill</li>
                            <li>• <strong>Seer:</strong> Inspect one player to learn their role</li>
                            <li>• <strong>Witch:</strong> Use save potion or poison potion</li>
                            <li>• <strong>Villagers:</strong> Sleep (no action)</li>
                        </ul>
                    </div>
                    <div>
                        <p className="font-medium text-foreground">Day Phase:</p>
                        <ul className="mt-1 space-y-1 ml-4">
                            <li>• <strong>Discussion:</strong> All players debate who might be a werewolf</li>
                            <li>• <strong>Voting:</strong> Vote to eliminate a suspected werewolf</li>
                        </ul>
                    </div>
                    <div>
                        <p className="font-medium text-foreground">Win Conditions:</p>
                        <ul className="mt-1 space-y-1 ml-4">
                            <li>• <strong>Villagers win:</strong> All werewolves are eliminated</li>
                            <li>• <strong>Werewolves win:</strong> Werewolves equal or outnumber villagers</li>
                        </ul>
                    </div>
                </div>
            </div>

            {/* Role Assignment (shown after game creation) */}
            {roleInfo && (
                <div className="space-y-2 rounded-lg border-2 border-purple-500 bg-purple-50 p-4 dark:bg-purple-950">
                    <h3 className="text-sm font-semibold text-purple-900 dark:text-purple-100">
                        🎭 Your Role Assignment
                    </h3>
                    <div className="space-y-1 text-sm text-purple-800 dark:text-purple-200">
                        <p><strong>Character:</strong> {roleInfo.human_agent.name}</p>
                        <p><strong>Role:</strong> {roleInfo.human_agent.role}</p>
                        <p><strong>Team:</strong> {roleInfo.human_agent.team}</p>
                    </div>
                    <p className="text-xs text-purple-700 dark:text-purple-300 mt-2">
                        Game starting... The other players' roles are hidden.
                    </p>
                </div>
            )}

            {!roleInfo && (
                <>
                    {/* Research Consent */}
                    <div className="space-y-3 rounded-lg border border-border p-4">
                        <h2 className="font-semibold">📋 Research Participation Consent</h2>
                        <div className="text-xs text-muted-foreground space-y-2">
                            <p>
                                You are invited to participate in a research study on social
                                interactions between humans and AI agents. You will play as a
                                character in a social deduction game.
                            </p>
                            <p>
                                <strong>Confidentiality:</strong> Your responses may be used for
                                research analysis. Data is stored securely. You must be 18+ to
                                participate.
                            </p>
                            <p className="text-[10px]">
                                Questions? Contact CMU Office of Research Integrity
                                (irb-review@andrew.cmu.edu, 412-268-4721)
                            </p>
                        </div>
                        <label className="flex items-start gap-3 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={consentChecked}
                                onChange={(e) => setConsentChecked(e.target.checked)}
                                className="mt-1 h-4 w-4 rounded border-gray-300"
                            />
                            <span className="text-sm">
                                I am 18+ years old and agree to participate in this research study
                            </span>
                        </label>
                    </div>

                    {/* Player Info */}
                    <div className="rounded-lg bg-muted/50 p-3 text-sm">
                        <p className="text-muted-foreground">
                            {isAuthenticated && user ? (
                                <>Playing as: <strong>{user.username}</strong> (ELO: {user.elo_rating})</>
                            ) : (
                                <>Playing as: <strong>Guest</strong> (progress won't be saved)</>
                            )}
                        </p>
                    </div>

                    {error && (
                        <div className="rounded-md bg-destructive/10 px-4 py-3 text-sm text-destructive">
                            {error}
                        </div>
                    )}

                    <Button
                        size="lg"
                        onClick={handleStartGame}
                        disabled={isCreating || !consentChecked}
                        className="w-full"
                    >
                        {isCreating ? "Starting Game..." : "I Agree & Start Game"}
                    </Button>
                </>
            )}
        </div>
    );
}
