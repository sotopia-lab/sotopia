"use client";

import { useState } from "react";
import useSWR from "swr";
import {
    fetchPersonalHistory,
    type PersonalHistoryResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";

export default function HistoryPage() {
    const [participantId, setParticipantId] = useState("");
    const [queryId, setQueryId] = useState<string | null>(null);

    const { data, error, isLoading } = useSWR<PersonalHistoryResponse>(
        queryId ? ["history", queryId] : null,
        () => fetchPersonalHistory(queryId as string),
        { refreshInterval: 10000 }
    );

    const submitSearch = (event: React.FormEvent) => {
        event.preventDefault();
        if (participantId.trim()) {
            setQueryId(participantId.trim());
        }
    };

    return (
        <main className="mx-auto max-w-4xl px-4 py-10">
            <header className="mb-6 space-y-2">
                <p className="text-sm uppercase tracking-[0.3em] text-muted-foreground">
                    Match History
                </p>
                <h1 className="text-3xl font-semibold">Participant Timeline</h1>
                <p className="text-sm text-muted-foreground">
                    Look up recent games for a specific participant or agent.
                </p>
            </header>

            <form onSubmit={submitSearch} className="mb-6 flex gap-2">
                <input
                    type="text"
                    value={participantId}
                    onChange={(e) => setParticipantId(e.target.value)}
                    placeholder="Enter participant ID"
                    className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                />
                <Button type="submit" disabled={!participantId.trim()}>
                    Fetch History
                </Button>
            </form>

            {isLoading && queryId && (
                <div className="rounded-xl border border-border bg-card/60 p-4 text-sm text-muted-foreground">
                    Loading history…
                </div>
            )}

            {error && queryId && (
                <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
                    Failed to load history. Participant may have no recent games.
                </div>
            )}

            {data && (
                <section className="space-y-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">
                        Showing latest {data.history.length} matches for{" "}
                        <span className="font-semibold">{data.participantId}</span>
                    </p>
                    <ul className="space-y-3">
                        {data.history.map((entry, idx) => (
                            <li
                                key={`${entry.recordedAt}-${idx}`}
                                className="rounded-2xl border border-border bg-card/50 p-4 shadow-sm"
                            >
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm font-semibold">
                                            {entry.game}
                                        </p>
                                        <p className="text-xs text-muted-foreground">
                                            vs. {entry.opponentModel}
                                        </p>
                                    </div>
                                    <span
                                        className={`rounded-full px-3 py-1 text-xs font-medium ${
                                            entry.winner === "human"
                                                ? "bg-emerald-100 text-emerald-800"
                                                : "bg-slate-200 text-slate-700"
                                        }`}
                                    >
                                        Winner: {entry.winner === "human" ? "You" : "AI"}
                                    </span>
                                </div>
                                <div className="mt-3 flex justify-between text-xs text-muted-foreground">
                                    <span>
                                        Duration: {entry.durationSeconds.toFixed(0)}s
                                    </span>
                                    <span>
                                        {new Date(entry.recordedAt * 1000).toLocaleString()}
                                    </span>
                                </div>
                            </li>
                        ))}
                        {!data.history.length && (
                            <li className="rounded-2xl border border-border bg-muted/40 p-4 text-sm text-muted-foreground">
                                No recent matches recorded for this participant.
                            </li>
                        )}
                    </ul>
                </section>
            )}
        </main>
    );
}
