"use client";

import { useEffect, useMemo, useState } from "react";
import { getGameBySlug } from "@/core/config/games";
import type { GameDefinition } from "@/core/types/game-module";

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
                setLoadError(`Game "${slug}" is not registered.`);
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
    }, [slug]);

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

    return <LoadedGameExperience game={game} />;
}

function LoadedGameExperience({ game }: { game: GameDefinition }) {
    const [consentAccepted, setConsentAccepted] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [participantId, setParticipantId] = useState<string | null>(null);

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

    if (!consentAccepted) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-background p-4">
                <game.components.Consent onAccept={() => setConsentAccepted(true)} />
            </div>
        );
    }

    if (showLobby) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-background p-4">
                <game.components.Lobby onGameCreated={handleGameCreated} />
            </div>
        );
    }

    return (
        <div className="flex min-h-screen items-center justify-center bg-background p-4">
            <game.components.GameBoard
                sessionId={sessionId!}
                participantId={participantId!}
                session={sessionHook.session}
                isLoading={sessionHook.isLoading}
                error={sessionHook.error}
                actionsState={actionsState}
                actionsControls={actionsControls}
            />
        </div>
    );
}
