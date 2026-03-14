"use client";

import useSWR from "swr";
import { fetchLeaderboard, type LeaderboardResponse } from "@/lib/api";

export default function LeaderboardPage() {
    const { data, error, isLoading } = useSWR<LeaderboardResponse>(
        "leaderboard",
        fetchLeaderboard,
        { refreshInterval: 10000 }
    );

    return (
        <main className="mx-auto max-w-5xl px-4 py-10">
            <header className="mb-8 space-y-2">
                <p className="text-sm uppercase tracking-[0.3em] text-muted-foreground">
                    Arena Rankings
                </p>
                <h1 className="text-3xl font-semibold">Game Leaderboard</h1>
                <p className="text-sm text-muted-foreground">
                    Tracking aggregate human vs. AI performance across all games.
                </p>
            </header>

            {isLoading && (
                <div className="rounded-2xl border border-border bg-card/60 p-6 text-sm text-muted-foreground">
                    Loading leaderboard…
                </div>
            )}

            {error && (
                <div className="rounded-2xl border border-destructive/40 bg-destructive/10 p-6 text-sm text-destructive">
                    Failed to load leaderboard. Try again later.
                </div>
            )}

            {data && (
                <section className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
                    <table className="w-full border-collapse text-sm">
                        <thead className="bg-muted/60 text-xs uppercase tracking-wide text-muted-foreground">
                            <tr>
                                <th className="px-4 py-3 text-left">Game</th>
                                <th className="px-4 py-3 text-left">Matches</th>
                                <th className="px-4 py-3 text-left">Human Win %</th>
                                <th className="px-4 py-3 text-left">
                                    Avg Duration (s)
                                </th>
                                <th className="px-4 py-3 text-left">Human Wins</th>
                                <th className="px-4 py-3 text-left">AI Wins</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.entries.map((entry) => (
                                <tr
                                    key={entry.game}
                                    className="border-b border-border/80 last:border-b-0"
                                >
                                    <td className="px-4 py-3 font-medium">
                                        {entry.game}
                                    </td>
                                    <td className="px-4 py-3">
                                        {entry.totalMatches}
                                    </td>
                                    <td className="px-4 py-3">
                                        {(entry.humanWinRate * 100).toFixed(1)}%
                                    </td>
                                    <td className="px-4 py-3">
                                        {entry.avgDurationSeconds.toFixed(0)}
                                    </td>
                                    <td className="px-4 py-3">
                                        {entry.humanWins}
                                    </td>
                                    <td className="px-4 py-3">{entry.aiWins}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <p className="px-4 py-3 text-xs text-muted-foreground">
                        Last updated:{" "}
                        {new Date(data.lastUpdated * 1000).toLocaleTimeString()}
                    </p>
                </section>
            )}
        </main>
    );
}
