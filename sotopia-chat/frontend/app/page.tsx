"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import useSWR from "swr";
import { games } from "@/core/config/games";
import { Button } from "@/components/ui/button";
import {
    enqueueMatchmaking,
    fetchQueueOverview,
    fetchTicketStatus,
    cancelTicket,
    type QueueOverview,
    type TicketStatus,
} from "@/lib/api";
import { useIdentity } from "@/hooks/use-identity";

export default function GamesLandingPage() {
    const router = useRouter();
    const identity = useIdentity();
    const [selected, setSelected] = useState<string[]>([]);
    const [queueMessage, setQueueMessage] = useState<string | null>(null);
    const [queueError, setQueueError] = useState<string | null>(null);
    const [queueLoading, setQueueLoading] = useState(false);
    const [queueParticipantId, setQueueParticipantId] = useState("");
    const [activeTicket, setActiveTicket] = useState<string | null>(null);

    const allSlugs = useMemo(() => games.map((game) => game.slug), []);
    const statusStyles = {
        online: "bg-emerald-100 text-emerald-800",
        maintenance: "bg-amber-100 text-amber-900",
        "coming-soon": "bg-slate-200 text-slate-700",
    } as const;

    const {
        data: queueOverview,
        error: matchmakingStatusError,
        isLoading: matchmakingStatusLoading,
    } = useSWR<QueueOverview>("matchmaking-status", fetchQueueOverview, {
        refreshInterval: 5000,
    });
    const matchmakingStatus = queueOverview?.globalStats;
    const {
        data: ticketStatus,
        error: ticketStatusError,
        isLoading: ticketStatusLoading,
    } = useSWR<TicketStatus>(
        activeTicket ? ["ticket-status", activeTicket] : null,
        () => fetchTicketStatus(activeTicket as string),
        { refreshInterval: 4000 }
    );
    useEffect(() => {
        if (
            identity.identity?.participantId &&
            !queueParticipantId &&
            identity.identity.participantId.length
        ) {
            setQueueParticipantId(identity.identity.participantId);
        }
    }, [identity.identity?.participantId, queueParticipantId]);

    const toggleGame = (slug: string) => {
        setSelected((prev) =>
            prev.includes(slug)
                ? prev.filter((item) => item !== slug)
                : [...prev, slug]
        );
        setQueueMessage(null);
        setQueueError(null);
    };

    const selectAll = () => {
        setSelected(allSlugs);
        setQueueMessage(null);
        setQueueError(null);
    };

    const clearSelection = () => {
        setSelected([]);
        setQueueMessage(null);
        setQueueError(null);
    };

    const handlePlaySelected = () => {
        if (!selected.length) return;
        router.push(`/games/${selected[0]}`);
    };

    const handleQueueSelected = async () => {
        const participantId =
            identity.identity?.participantId || queueParticipantId.trim();
        if (!participantId) {
            setQueueError("Set your participant identity first.");
            return;
        }
        if (!selected.length || queueLoading) {
            if (!selected.length) {
                setQueueError("Select at least one game to queue.");
            }
            return;
        }
        setQueueLoading(true);
        setQueueError(null);
        setQueueMessage(null);
        try {
            const response = await enqueueMatchmaking({
                participantId,
                games: selected,
            });
            setQueueMessage(
                `${response.message} Estimated wait: ${response.estimatedWaitSeconds}s (position ${response.position}).`
            );
            setActiveTicket(response.ticketId);
        } catch (error) {
            console.error(error);
            setQueueError(
                error instanceof Error
                    ? error.message
                    : "Failed to queue selected games."
            );
        } finally {
            setQueueLoading(false);
        }
    };

    const formatPlayers = (min?: number, max?: number) => {
        if (!min && !max) return null;
        if (min && max) {
            if (min === max) return `${min} players`;
            return `${min}-${max} players`;
        }
        return `${min ?? max} players`;
    };

    const formatDuration = (minutes?: number) => {
        if (!minutes) return null;
        return `~${minutes} min`;
    };

    const queueDisabled = selected.length === 0 || queueLoading;

    const handleCancelTicket = async () => {
        if (!activeTicket) return;
        try {
            await cancelTicket(activeTicket);
            setQueueMessage("Ticket cancelled.");
            setActiveTicket(null);
        } catch (error) {
            console.error(error);
            setQueueError(
                error instanceof Error
                    ? error.message
                    : "Failed to cancel ticket."
            );
        }
    };

    return (
        <main className="mx-auto flex min-h-screen max-w-5xl flex-col gap-12 px-6 py-16">
            <section className="space-y-6 text-center">
                <p className="text-sm uppercase tracking-[0.3em] text-muted-foreground">
                    Sotopia Social Game Arena
                </p>
                <h1 className="text-4xl font-semibold sm:text-5xl">
                    Choose a research game to play with humans + LLM agents
                </h1>
                <p className="mx-auto max-w-3xl text-base text-muted-foreground sm:text-lg">
                    Each experience blends experimental storytelling with structured
                    social deduction mechanics. Select a title below to review the
                    consent form, set up the lobby, and jump into a live match.
                </p>
                <div className="flex justify-center gap-3 text-sm">
                    <Button variant="ghost" size="sm" onClick={() => router.push("/leaderboard")}>
                        View Leaderboard
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => router.push("/history")}>
                        Match History
                    </Button>
                </div>
            </section>

            <section className="rounded-2xl border border-border bg-card/60 p-6">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
                    <div className="flex-1 space-y-2">
                        <p className="text-sm uppercase tracking-[0.3em] text-muted-foreground">
                            Your Arena Identity
                        </p>
                        <input
                            type="text"
                            value={
                                identity.pendingParticipantId ||
                                identity.identity?.participantId ||
                                ""
                            }
                            onChange={(e) =>
                                identity.setPendingParticipantId(e.target.value)
                            }
                            placeholder="Participant ID"
                            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        />
                        <input
                            type="text"
                            value={
                                identity.pendingDisplayName ||
                                identity.identity?.displayName ||
                                ""
                            }
                            onChange={(e) =>
                                identity.setPendingDisplayName(e.target.value)
                            }
                            placeholder="Display name (optional)"
                            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        />
                        {identity.identity && (
                            <p className="text-xs text-muted-foreground">
                                Signed in as {identity.identity.displayName} (
                                {identity.identity.participantId})
                            </p>
                        )}
                    </div>
                    <div className="flex gap-2">
                        <Button
                            onClick={() =>
                                identity.register(
                                    identity.pendingParticipantId ||
                                        identity.identity?.participantId ||
                                        queueParticipantId ||
                                        "guest",
                                    identity.pendingDisplayName ||
                                        identity.identity?.displayName ||
                                        undefined
                                )
                            }
                            disabled={identity.registerLoading}
                        >
                            {identity.registerLoading ? "Saving…" : "Save Identity"}
                        </Button>
                        {identity.identity && (
                            <Button
                                variant="ghost"
                                onClick={identity.clearIdentity}
                                className="text-muted-foreground"
                            >
                                Clear
                            </Button>
                        )}
                    </div>
                </div>
                {identity.registerError && (
                    <p className="mt-2 text-sm text-destructive">
                        {identity.registerError}
                    </p>
                )}
            </section>

            <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <div className="rounded-2xl border border-border bg-card/50 p-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">
                        Avg. Queue Time
                    </p>
                    <p className="mt-1 text-2xl font-semibold">
                        {matchmakingStatus?.avgWaitSeconds
                            ? `${matchmakingStatus.avgWaitSeconds}s`
                            : matchmakingStatusLoading
                            ? "…"
                            : "—"}
                    </p>
                </div>
                <div className="rounded-2xl border border-border bg-card/50 p-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">
                        Active Sessions
                    </p>
                    <p className="mt-1 text-2xl font-semibold">
                        {matchmakingStatus?.activeSessions ??
                            (matchmakingStatusLoading ? "…" : "—")}
                    </p>
                </div>
                <div className="rounded-2xl border border-border bg-card/50 p-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">
                        Queue Depth
                    </p>
                    <p className="mt-1 text-2xl font-semibold">
                        {matchmakingStatus?.queueDepth ??
                            (matchmakingStatusLoading ? "…" : "—")}
                    </p>
                </div>
                <div className="rounded-2xl border border-border bg-card/50 p-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">
                        Server Status
                    </p>
                    <p className="mt-1 text-lg font-semibold capitalize">
                        {matchmakingStatus?.serverStatus ??
                            (matchmakingStatusLoading ? "…" : "Unknown")}
                    </p>
                    <p className="text-xs text-muted-foreground">
                        Updated{" "}
                        {matchmakingStatus?.lastUpdated
                            ? new Date(
                                  matchmakingStatus.lastUpdated * 1000
                              ).toLocaleTimeString()
                            : ""}
                    </p>
                </div>
            </section>

            <section className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex gap-2">
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={selectAll}
                            disabled={selected.length === allSlugs.length}
                        >
                            Select All
                        </Button>
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={clearSelection}
                            disabled={selected.length === 0}
                        >
                            Clear
                        </Button>
                    </div>
                    <span className="text-sm text-muted-foreground">
                        {selected.length} game
                        {selected.length === 1 ? "" : "s"} selected
                    </span>
                </div>

                <div className="grid gap-6 sm:grid-cols-2">
                    {games.map((game) => {
                        const queueGame = queueOverview?.games.find(
                            (entry) => entry.slug === game.slug
                        );
                        const isEnabled =
                            queueGame?.enabled ??
                            (game.status !== "coming-soon" &&
                                game.status !== "maintenance");
                        return (
                        <button
                            key={game.slug}
                            type="button"
                            onClick={() => toggleGame(game.slug)}
                            disabled={!isEnabled}
                            className={`group rounded-2xl border p-6 text-left shadow-sm transition focus-visible:outline-none focus-visible:ring-2 ${
                                selected.includes(game.slug)
                                    ? "border-primary bg-primary/5 shadow-lg"
                                : "border-border bg-card/40 hover:-translate-y-0.5 hover:border-primary"
                            }`}
                        >
                            <div className="flex items-center justify-between">
                                <h2 className="text-2xl font-semibold">
                                    {game.title}
                                </h2>
                                <span className="text-sm text-primary opacity-80 group-hover:opacity-100">
                                    {selected.includes(game.slug)
                                        ? "Selected"
                                    : "Tap to select"}
                            </span>
                        </div>
                            <p className="mt-3 text-sm text-muted-foreground">
                                {game.summary}
                            </p>
                            {game.tags && (
                            <ul className="mt-4 flex flex-wrap gap-2 text-xs font-medium text-muted-foreground">
                                {game.tags.map((tag) => (
                                    <li
                                        key={tag}
                                            className="rounded-full border border-border px-3 py-1"
                                        >
                                            {tag}
                                        </li>
                                    ))}
                                </ul>
                            )}
                        <div className="mt-4 flex flex-wrap gap-2 text-xs font-medium text-muted-foreground">
                                {formatPlayers(game.minPlayers, game.maxPlayers) && (
                                    <span className="rounded-full bg-muted px-3 py-1">
                                        {formatPlayers(
                                            game.minPlayers,
                                            game.maxPlayers
                                        )}
                                    </span>
                                )}
                                {formatDuration(game.estDurationMinutes) && (
                                    <span className="rounded-full bg-muted px-3 py-1">
                                        {formatDuration(game.estDurationMinutes)}
                                    </span>
                                )}
                            {game.status && (
                                <span
                                    className={`rounded-full px-3 py-1 ${
                                        statusStyles[game.status] ??
                                        "bg-slate-200 text-slate-700"
                                    }`}
                                >
                                    {game.status === "online"
                                        ? "Available"
                                        : game.status === "maintenance"
                                        ? "Maintenance"
                                        : "Coming soon"}
                                </span>
                            )}
                            {queueOverview?.games
                                ?.find((entry) => entry.slug === game.slug)
                                ?.gamesPlayed ? (
                                <span className="rounded-full bg-muted px-3 py-1">
                                    {queueOverview.games.find(
                                        (entry) => entry.slug === game.slug
                                    )?.gamesPlayed ?? 0}{" "}
                                    matches
                                </span>
                            ) : null}
                            {game.features?.teamChat && (
                                <span className="rounded-full bg-muted px-3 py-1">
                                    Team chat
                                </span>
                            )}
                            {game.features?.spectators && (
                                <span className="rounded-full bg-muted px-3 py-1">
                                    Spectators
                                </span>
                            )}
                        </div>
                        {!isEnabled && (
                            <p className="mt-2 text-xs text-destructive">
                                Disabled by admin
                            </p>
                        )}
                    </button>
                );
                })}
                </div>
            </section>

            <section className="space-y-3 rounded-2xl border border-border bg-card/50 p-6">
                <div className="flex flex-col gap-2 sm:flex-row">
                    <input
                        type="text"
                        value={queueParticipantId}
                        onChange={(e) => setQueueParticipantId(e.target.value)}
                        placeholder="Participant or model ID (optional)"
                        className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    />
                </div>
                <div className="flex flex-wrap items-center gap-3">
                    <Button
                        onClick={handleQueueSelected}
                        disabled={queueDisabled}
                        variant="secondary"
                    >
                        {queueLoading ? "Queueing…" : "Queue Selected"}
                    </Button>
                    <Button
                        onClick={handlePlaySelected}
                        disabled={selected.length === 0}
                    >
                        Play Selected
                    </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                    Queueing lets us match you to live experiments once matchmaking is available.
                    Play jumps straight into the first selected game.
                </p>
                {queueMessage && (
                    <div className="rounded-md border border-dashed border-primary/60 bg-primary/5 px-3 py-2 text-sm text-primary">
                        {queueMessage}
                    </div>
                )}
                {queueError && (
                    <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                        {queueError}
                    </div>
                )}
                {ticketStatusError && (
                    <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                        Failed to refresh ticket status.
                    </div>
                )}
                {matchmakingStatusError && (
                    <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                        {matchmakingStatusError instanceof Error
                            ? matchmakingStatusError.message
                            : "Unable to load matchmaking stats."}
                    </div>
                )}
                {matchmakingStatus?.issues?.length ? (
                    <div className="space-y-2">
                        {matchmakingStatus.issues.map((issue) => (
                            <div
                                key={issue}
                                className="rounded-md border border-amber-400/60 bg-amber-50 px-3 py-2 text-sm text-amber-900"
                            >
                                {issue}
                            </div>
                        ))}
                    </div>
                ) : null}
            </section>

            {activeTicket && (
                <section className="rounded-2xl border border-border bg-card/40 p-6">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                            <p className="text-sm font-semibold">
                                Ticket {activeTicket.slice(0, 8)}
                            </p>
                            <p className="text-xs text-muted-foreground">
                                Status:{" "}
                                {ticketStatusLoading
                                    ? "Refreshing…"
                                    : ticketStatus?.status ?? "unknown"}
                            </p>
                        </div>
                        <div className="flex gap-2">
                            <Button
                                variant="secondary"
                                size="sm"
                                onClick={() => router.push("/history")}
                            >
                                View History
                            </Button>
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleCancelTicket}
                            >
                                Cancel Ticket
                            </Button>
                        </div>
                    </div>
                    {ticketStatus?.matchedGame && (
                        <p className="mt-2 text-sm text-muted-foreground">
                            Matched to {ticketStatus.matchedGame}. Launching session soon…
                        </p>
                    )}
                </section>
            )}
        </main>
    );
}
