"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import useSWR from "swr";
import { getGameBySlug, games as registeredGames } from "@/core/config/games";
import type { GameDefinition } from "@/core/types/game-module";
import { Button } from "@/components/ui/button";
import {
    fetchQueueOverview,
    type QueueOverview,
} from "@/lib/api";
import { useIdentity } from "@/hooks/use-identity";
import { usePlayerMemory } from "@/hooks/use-memory";

interface GameExperienceProps {
    slug: string;
}

type GameModuleLoader = () => Promise<{
    default?: GameDefinition;
    werewolfGame?: GameDefinition;
}>;

const moduleLoaders: Record<string, GameModuleLoader> = {
    werewolf: async () => import("@/games/werewolf"),
};

export function GameExperience({ slug }: GameExperienceProps) {
    const [game, setGame] = useState<GameDefinition | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);

    const summary = useMemo(() => getGameBySlug(slug), [slug]);

    useEffect(() => {
        let cancelled = false;
        async function load() {
            setLoadError(null);
            setGame(null);
            const loader = moduleLoaders[slug];
            if (!loader) {
                if (summary?.status === "coming-soon") {
                    setLoadError("This game is coming soon. Check back shortly.");
                } else if (summary?.status === "maintenance") {
                    setLoadError("This game is temporarily under maintenance.");
                } else {
                    setLoadError(`Game "${slug}" is not registered.`);
                }
                return;
            }
            try {
                const mod = await loader();
                const definition = mod.werewolfGame ?? mod.default ?? null;
                if (!cancelled) {
                    setGame(definition);
                }
            } catch (err) {
                console.error(err);
                if (!cancelled) {
                    setLoadError(
                        err instanceof Error
                            ? err.message
                            : "Failed to load game module."
                    );
                }
            }
        }
        load();
        return () => {
            cancelled = true;
        };
    }, [slug, summary?.status]);

    if (!summary) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-background p-4">
                <div className="rounded-lg border border-border bg-card p-6 text-center">
                    <p className="text-lg font-semibold">
                        Game &ldquo;{slug}&rdquo; is not registered.
                    </p>
                </div>
            </div>
        );
    }

    if (loadError) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-background p-4">
                <div className="rounded-lg border border-border bg-card p-6 text-center">
                    <p className="text-lg font-semibold">{loadError}</p>
                </div>
            </div>
        );
    }

    if (!game) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-background p-4">
                <div className="rounded-lg border border-border bg-card p-6 text-center">
                    <p className="text-sm text-muted-foreground">
                        Loading {summary.title}…
                    </p>
                </div>
            </div>
        );
    }

    return <LoadedGameExperience game={game} slug={slug} />;
}

function LoadedGameExperience({ game, slug }: { game: GameDefinition; slug: string }) {
    const [consentAccepted, setConsentAccepted] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [participantId, setParticipantId] = useState<string | null>(null);
    const router = useRouter();
    const [notificationsOpen, setNotificationsOpen] = useState(false);
    const [dossierOpen, setDossierOpen] = useState(false);
    const identity = useIdentity();
    const playerId = identity.identity?.participantId || null;
    const memory = usePlayerMemory(playerId);

    const sessionHook = game.hooks.useSession(sessionId, participantId);
    const [actionsState, actionsControls] = game.hooks.useActions(
        sessionId,
        participantId
    );

    const handleGameCreated = (newSessionId: string, newParticipantId: string) => {
        setSessionId(newSessionId);
        setParticipantId(newParticipantId);
    };

    const showLobby = !sessionId || !participantId;

    const { data: queueOverview } = useSWR<QueueOverview>(
        "queue-overview-shell",
        fetchQueueOverview,
        { refreshInterval: 5000 }
    );
    const issues = queueOverview?.globalStats.issues ?? [];

    const modalMode = !consentAccepted ? "consent" : showLobby ? "lobby" : null;
    const overlayMode = modalMode ?? (dossierOpen ? "dossier" : null);

    return (
        <div className="flex min-h-screen flex-col bg-background">
            <SessionToolbar
                currentSlug={slug}
                gameTitle={game.title}
                issues={issues}
                notificationsOpen={notificationsOpen}
                onToggleNotifications={() => setNotificationsOpen((prev) => !prev)}
                onReturnHome={() => router.push("/")}
                onSwitchGame={(nextSlug) => router.push(`/games/${nextSlug}`)}
                dossierAvailable={Boolean(playerId)}
                dossierOpen={dossierOpen}
                onToggleDossier={() => setDossierOpen((prev) => !prev)}
            />

            <div className={`flex-1 overflow-y-auto bg-muted/10`}>
                <div
                    className={`mx-auto h-full max-w-6xl px-4 py-6 ${
                        modalMode ? "pointer-events-none blur-sm" : ""
                    }`}
                >
                    {sessionId && participantId ? (
                        <game.components.GameBoard
                            sessionId={sessionId}
                            participantId={participantId}
                            session={sessionHook.session}
                            isLoading={sessionHook.isLoading}
                            error={sessionHook.error}
                            actionsState={actionsState}
                            actionsControls={actionsControls}
                        />
                    ) : (
                        <div className="flex h-full min-h-[480px] items-center justify-center rounded-2xl border border-dashed border-border bg-card">
                            <div className="text-center text-sm text-muted-foreground">
                                Start the lobby to begin a new session.
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {overlayMode && (
                <ModalOverlay>
                    {overlayMode === "consent" ? (
                        <game.components.Consent onAccept={() => setConsentAccepted(true)} />
                    ) : overlayMode === "lobby" ? (
                        <game.components.Lobby onGameCreated={handleGameCreated} />
                    ) : playerId ? (
                        <PlayerDossierPanel
                            participantId={playerId}
                            playerName={identity.identity?.displayName}
                            memory={memory}
                            onClose={() => setDossierOpen(false)}
                        />
                    ) : null}
                </ModalOverlay>
            )}

            {notificationsOpen && issues.length > 0 && (
                <div className="fixed bottom-6 right-6 z-30 w-80 rounded-2xl border border-border bg-card p-4 shadow-lg">
                    <h3 className="text-sm font-semibold">Arena Notifications</h3>
                    <ul className="mt-2 space-y-2 text-xs text-muted-foreground">
                        {issues.map((issue) => (
                            <li key={issue} className="rounded-md bg-muted/60 px-2 py-1">
                                {issue}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}

function ModalOverlay({ children }: { children: React.ReactNode }) {
    return (
        <div className="fixed inset-0 z-20 flex items-center justify-center bg-background/80 px-4 py-6 backdrop-blur-sm">
            <div className="w-full max-w-2xl rounded-2xl border border-border bg-card p-6 shadow-2xl">
                {children}
            </div>
        </div>
    );
}

function SessionToolbar({
    currentSlug,
    gameTitle,
    issues,
    notificationsOpen,
    onToggleNotifications,
    onReturnHome,
    onSwitchGame,
    dossierAvailable,
    dossierOpen,
    onToggleDossier,
}: {
    currentSlug: string;
    gameTitle: string;
    issues: string[];
    notificationsOpen: boolean;
    onToggleNotifications: () => void;
    onReturnHome: () => void;
    onSwitchGame: (slug: string) => void;
    dossierAvailable: boolean;
    dossierOpen: boolean;
    onToggleDossier: () => void;
}) {
    const onlineGames = registeredGames.filter(
        (entry) => entry.status !== "coming-soon"
    );

    return (
        <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur">
            <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3">
                <div className="flex items-center gap-2">
                    <Button variant="ghost" size="sm" onClick={onReturnHome}>
                        ← Back to Arena
                    </Button>
                    <div>
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">
                            Now Playing
                        </p>
                        <p className="text-lg font-semibold">{gameTitle}</p>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <select
                        value={currentSlug}
                        onChange={(e) => onSwitchGame(e.target.value)}
                        className="rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                        {onlineGames.map((entry) => (
                            <option key={entry.slug} value={entry.slug}>
                                {entry.title}
                            </option>
                        ))}
                    </select>
                    {dossierAvailable && (
                        <Button
                            variant={dossierOpen ? "default" : "outline"}
                            size="sm"
                            onClick={onToggleDossier}
                        >
                            Player Dossier
                        </Button>
                    )}
                    <Button
                        variant={
                            notificationsOpen
                                ? "default"
                                : issues.length
                                ? "secondary"
                                : "ghost"
                        }
                        size="sm"
                        onClick={onToggleNotifications}
                        aria-pressed={notificationsOpen}
                    >
                        Notifications
                        {issues.length ? (
                            <span className="ml-1 rounded-full bg-primary px-2 text-xs text-primary-foreground">
                                {issues.length}
                            </span>
                        ) : null}
                    </Button>
                </div>
            </div>
        </header>
    );
}

function PlayerDossierPanel({
    participantId,
    playerName,
    memory,
    onClose,
}: {
    participantId: string;
    playerName?: string;
    memory: ReturnType<typeof usePlayerMemory>;
    onClose: () => void;
}) {
    const [notes, setNotes] = useState("");

    useEffect(() => {
        if (memory.memory?.notes !== undefined) {
            setNotes(memory.memory.notes);
        }
    }, [memory.memory?.notes]);

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold">Player Dossier</h2>
                    <p className="text-sm text-muted-foreground">
                        {playerName || participantId} — persistent notes & history
                    </p>
                </div>
                <Button variant="ghost" onClick={onClose}>
                    Close
                </Button>
            </div>

            <div className="space-y-2">
                <label className="text-sm font-medium">Personal Notes</label>
                <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    className="h-32 w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                />
                <div className="flex gap-2">
                    <Button
                        size="sm"
                        onClick={() => memory.save({ notes })}
                        disabled={memory.isSaving}
                    >
                        {memory.isSaving ? "Saving…" : "Save Notes"}
                    </Button>
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => memory.save({ notes: "" })}
                        disabled={memory.isSaving}
                    >
                        Clear
                    </Button>
                </div>
                {memory.saveError && (
                    <p className="text-xs text-destructive">{memory.saveError}</p>
                )}
            </div>

            <div className="grid gap-4 md:grid-cols-2">
                <div>
                    <h3 className="text-sm font-semibold text-muted-foreground">
                        Role History
                    </h3>
                    <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
                        {memory.memory?.roleHistory?.length ? (
                            memory.memory.roleHistory.map((role, idx) => (
                                <li key={`${role}-${idx}`} className="rounded bg-muted px-2 py-1">
                                    {role}
                                </li>
                            ))
                        ) : (
                            <li className="rounded bg-muted/40 px-2 py-1">
                                No roles recorded yet.
                            </li>
                        )}
                    </ul>
                </div>
                <div>
                    <h3 className="text-sm font-semibold text-muted-foreground">
                        Voting History
                    </h3>
                    <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
                        {memory.memory?.voteHistory?.length ? (
                            memory.memory.voteHistory.map((vote, idx) => (
                                <li key={`${vote}-${idx}`} className="rounded bg-muted px-2 py-1">
                                    {vote}
                                </li>
                            ))
                        ) : (
                            <li className="rounded bg-muted/40 px-2 py-1">
                                No votes recorded yet.
                            </li>
                        )}
                    </ul>
                </div>
            </div>
        </div>
    );
}
