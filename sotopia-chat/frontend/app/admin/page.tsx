"use client";

import { useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import {
    fetchAdminStatus,
    setGameEnabled,
    type AdminStatus,
} from "@/lib/api";

export default function AdminPage() {
    const [token, setToken] = useState("");
    const [submittedToken, setSubmittedToken] = useState<string | null>(null);

    const { data, error, isLoading, mutate } = useSWR<AdminStatus>(
        submittedToken ? ["admin-status", submittedToken] : null,
        () => fetchAdminStatus(submittedToken as string),
        { refreshInterval: 5000 }
    );

    const submitToken = (event: React.FormEvent) => {
        event.preventDefault();
        if (token.trim()) {
            setSubmittedToken(token.trim());
        }
    };

    const toggleGame = async (slug: string, enabled: boolean) => {
        if (!submittedToken) return;
        await setGameEnabled(slug, enabled, submittedToken);
        mutate();
    };

    return (
        <main className="mx-auto max-w-5xl space-y-8 px-4 py-10">
            <header className="space-y-2">
                <p className="text-sm uppercase tracking-[0.3em] text-muted-foreground">
                    Admin Console
                </p>
                <h1 className="text-3xl font-semibold">Platform Health</h1>
                <p className="text-sm text-muted-foreground">
                    Requires admin token. Monitor queue saturation, Redis health, and toggle games.
                </p>
            </header>

            <form onSubmit={submitToken} className="flex gap-2">
                <input
                    type="password"
                    value={token}
                    onChange={(e) => setToken(e.target.value)}
                    placeholder="Enter admin token"
                    className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                />
                <Button type="submit">Authenticate</Button>
            </form>

            {isLoading && submittedToken && (
                <div className="rounded-2xl border border-border bg-card/60 p-4 text-sm text-muted-foreground">
                    Loading status…
                </div>
            )}

            {error && submittedToken && (
                <div className="rounded-2xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
                    Failed to load status. Verify admin token.
                </div>
            )}

            {data && (
                <section className="space-y-6">
                    <div className="grid gap-4 sm:grid-cols-3">
                        <StatCard
                            label="Uptime"
                            value={`${Math.floor(data.uptimeSeconds / 60)} min`}
                        />
                        <StatCard
                            label="Redis"
                            value={data.redisAlive ? "Connected" : "Offline"}
                        />
                        <StatCard
                            label="Active Sessions"
                            value={`${data.globalStats.activeSessions}`}
                        />
                    </div>

                    <div className="rounded-2xl border border-border bg-card/50">
                        <table className="w-full border-collapse text-sm">
                            <thead className="bg-muted/60 text-xs uppercase tracking-wide text-muted-foreground">
                                <tr>
                                    <th className="px-4 py-3 text-left">Game</th>
                                    <th className="px-4 py-3 text-left">Queue Depth</th>
                                    <th className="px-4 py-3 text-left">Avg Wait</th>
                                    <th className="px-4 py-3 text-left">Status</th>
                                    <th className="px-4 py-3 text-left">Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.games.map((game) => (
                                    <tr
                                        key={game.slug}
                                        className="border-b border-border/80 last:border-b-0"
                                    >
                                        <td className="px-4 py-3 font-medium">{game.title}</td>
                                        <td className="px-4 py-3">{game.queueDepth}</td>
                                        <td className="px-4 py-3">{game.avgWaitSeconds}s</td>
                                        <td className="px-4 py-3">
                                            {game.enabled ? (
                                                <span className="rounded-full bg-emerald-100 px-2 py-1 text-xs text-emerald-800">
                                                    Enabled
                                                </span>
                                            ) : (
                                                <span className="rounded-full bg-red-100 px-2 py-1 text-xs text-red-800">
                                                    Disabled
                                                </span>
                                            )}
                                        </td>
                                        <td className="px-4 py-3">
                                            <Button
                                                variant="secondary"
                                                size="sm"
                                                onClick={() => toggleGame(game.slug, !game.enabled)}
                                            >
                                                {game.enabled ? "Disable" : "Enable"}
                                            </Button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </section>
            )}
        </main>
    );
}

function StatCard({ label, value }: { label: string; value: string }) {
    return (
        <div className="rounded-2xl border border-border bg-card/50 p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">
                {label}
            </p>
            <p className="mt-2 text-2xl font-semibold">{value}</p>
        </div>
    );
}
