"use client";

import { useEffect } from "react";
import useSWR from "swr";
import {
    fetchPersonalHistory,
    type PersonalHistoryResponse,
} from "@/lib/api";
import { getStoredUser } from "@/lib/auth-api";
import { useRouter } from "next/navigation";

export default function HistoryPage() {
    const router = useRouter();
    const user = getStoredUser();
    const participantId = user?.pk ?? null;

    const { data, error, isLoading } = useSWR<PersonalHistoryResponse>(
        participantId ? ["history", participantId] : null,
        () => fetchPersonalHistory(participantId as string),
        { refreshInterval: 10000 }
    );

    useEffect(() => {
        if (typeof window !== "undefined" && !participantId) {
            router.replace("/login");
        }
    }, [participantId, router]);

    if (!participantId) {
        return (
            <main className="mx-auto max-w-4xl px-4 py-10">
                <p className="text-sm text-muted-foreground">Redirecting to login…</p>
            </main>
        );
    }

    return (
        <main className="mx-auto max-w-4xl px-4 py-10">
            <header className="mb-6 space-y-2">
                <p className="text-sm uppercase tracking-[0.3em] text-muted-foreground">
                    Match History
                </p>
                <h1 className="text-3xl font-semibold">Your Game History</h1>
                <p className="text-sm text-muted-foreground">
                    Recent games you have played.
                </p>
            </header>

            {isLoading && participantId && (
                <div className="rounded-xl border border-border bg-card/60 p-4 text-sm text-muted-foreground">
                    Loading history…
                </div>
            )}

            {error && participantId && (
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
                                        {new Date(entry.recordedAt).toLocaleString()}
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
