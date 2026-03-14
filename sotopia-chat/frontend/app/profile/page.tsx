"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import useSWR from "swr";
import { useAuth } from "@/contexts/auth-context";
import { Button } from "@/components/ui/button";
import {
    getProfile,
    getEloHistory,
    getMatchHistory,
    type ProfileStats,
    type EloHistoryEntry,
    type MatchHistoryEntry,
} from "@/lib/auth-api";

export default function ProfilePage() {
    const router = useRouter();
    const { user, isAuthenticated, isLoading: authLoading, logout } = useAuth();
    const [activeTab, setActiveTab] = useState<"overview" | "elo" | "matches">("overview");

    // Redirect if not authenticated
    useEffect(() => {
        if (!authLoading && !isAuthenticated) {
            router.push("/login");
        }
    }, [authLoading, isAuthenticated, router]);

    const {
        data: profile,
        error: profileError,
        isLoading: isProfileLoading,
    } = useSWR<ProfileStats>(
        isAuthenticated ? "profile" : null,
        getProfile,
        { refreshInterval: 30000 }
    );

    const {
        data: eloHistory,
        error: eloError,
        isLoading: eloLoading,
    } = useSWR<EloHistoryEntry[]>(
        isAuthenticated && activeTab === "elo" ? "elo-history" : null,
        () => getEloHistory(50)
    );

    const {
        data: matchHistory,
        error: matchError,
        isLoading: matchLoading,
    } = useSWR<MatchHistoryEntry[]>(
        isAuthenticated && activeTab === "matches" ? "match-history" : null,
        () => getMatchHistory(50)
    );

    if (authLoading || !isAuthenticated) {
        return (
            <main className="flex min-h-screen items-center justify-center bg-background">
                <div className="text-muted-foreground">Loading...</div>
            </main>
        );
    }

    const handleLogout = () => {
        logout();
        router.push("/");
    };

    return (
        <main className="mx-auto max-w-4xl px-4 py-10">
            {/* Header */}
            <header className="mb-8 flex items-center justify-between">
                <div>
                    <p className="text-sm uppercase tracking-[0.3em] text-muted-foreground">
                        Player Profile
                    </p>
                    <h1 className="text-3xl font-semibold">
                        {isProfileLoading ? "Loading..." : (profile?.username || user?.username || "—")}
                    </h1>
                </div>
                <div className="flex gap-2">
                    <Link href="/">
                        <Button variant="outline">← Arena</Button>
                    </Link>
                    <Button variant="destructive" onClick={handleLogout}>
                        Logout
                    </Button>
                </div>
            </header>

            {profileError && (
                <div className="mb-6 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
                    Failed to load profile. Please try again.
                </div>
            )}

            {/* Stats Cards */}
            {profile && (
                <section className="mb-8 grid gap-4 sm:grid-cols-4">
                    <StatCard
                        label="ELO Rating"
                        value={profile.elo_rating.toString()}
                        subtext={`${profile.rank_emoji} ${profile.rank_tier}`}
                    />
                    <StatCard
                        label="Games Played"
                        value={profile.games_played.toString()}
                    />
                    <StatCard
                        label="Games Won"
                        value={profile.games_won.toString()}
                    />
                    <StatCard
                        label="Win Rate"
                        value={`${(profile.win_rate * 100).toFixed(1)}%`}
                    />
                </section>
            )}

            {/* Tabs */}
            <div className="mb-6 flex gap-2 border-b border-border">
                <TabButton
                    active={activeTab === "overview"}
                    onClick={() => setActiveTab("overview")}
                >
                    Overview
                </TabButton>
                <TabButton
                    active={activeTab === "elo"}
                    onClick={() => setActiveTab("elo")}
                >
                    ELO History
                </TabButton>
                <TabButton
                    active={activeTab === "matches"}
                    onClick={() => setActiveTab("matches")}
                >
                    Match History
                </TabButton>
            </div>

            {/* Tab Content */}
            {activeTab === "overview" && profile && (
                <section className="space-y-6">
                    <div className="rounded-2xl border border-border bg-card/50 p-6">
                        <h2 className="mb-4 text-lg font-semibold">Account Info</h2>
                        <dl className="space-y-3">
                            <div className="flex justify-between">
                                <dt className="text-muted-foreground">Email</dt>
                                <dd>{profile.email}</dd>
                            </div>
                            <div className="flex justify-between">
                                <dt className="text-muted-foreground">Auth Provider</dt>
                                <dd className="capitalize">{profile.auth_provider}</dd>
                            </div>
                            <div className="flex justify-between">
                                <dt className="text-muted-foreground">Member Since</dt>
                                <dd>{new Date(profile.created_at).toLocaleDateString()}</dd>
                            </div>
                        </dl>
                    </div>

                    <div className="rounded-2xl border border-border bg-card/50 p-6">
                        <h2 className="mb-4 text-lg font-semibold">Rank Progress</h2>
                        <div className="flex items-center gap-4">
                            <span className="text-4xl">{profile.rank_emoji}</span>
                            <div>
                                <p className="text-xl font-bold">{profile.rank_tier}</p>
                                <p className="text-sm text-muted-foreground">
                                    {profile.elo_rating} ELO
                                </p>
                            </div>
                        </div>
                        <div className="mt-4">
                            <RankProgressBar elo={profile.elo_rating} />
                        </div>
                    </div>
                </section>
            )}

            {activeTab === "elo" && (
                <section className="space-y-4">
                    {eloLoading && (
                        <div className="rounded-xl border border-border bg-card/60 p-4 text-sm text-muted-foreground">
                            Loading ELO history...
                        </div>
                    )}
                    {eloError && (
                        <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
                            Failed to load ELO history.
                        </div>
                    )}
                    {eloHistory && eloHistory.length === 0 && (
                        <div className="rounded-xl border border-border bg-card/60 p-6 text-center text-muted-foreground">
                            No ELO history yet. Play some games to see your progress!
                        </div>
                    )}
                    {eloHistory && eloHistory.length > 0 && (
                        <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
                            <table className="w-full border-collapse text-sm">
                                <thead className="bg-muted/60 text-xs uppercase tracking-wide text-muted-foreground">
                                    <tr>
                                        <th className="px-4 py-3 text-left">Date</th>
                                        <th className="px-4 py-3 text-left">Game</th>
                                        <th className="px-4 py-3 text-left">Result</th>
                                        <th className="px-4 py-3 text-left">Change</th>
                                        <th className="px-4 py-3 text-left">Rating</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {eloHistory.map((entry, idx) => (
                                        <tr
                                            key={`${entry.game_pk}-${idx}`}
                                            className="border-b border-border/80 last:border-b-0"
                                        >
                                            <td className="px-4 py-3 text-muted-foreground">
                                                {new Date(entry.created_at).toLocaleDateString()}
                                            </td>
                                            <td className="px-4 py-3 capitalize">
                                                {entry.game_type}
                                            </td>
                                            <td className="px-4 py-3">
                                                <span
                                                    className={`rounded-full px-2 py-1 text-xs font-medium ${
                                                        entry.won
                                                            ? "bg-emerald-100 text-emerald-800"
                                                            : "bg-red-100 text-red-800"
                                                    }`}
                                                >
                                                    {entry.won ? "Win" : "Loss"}
                                                </span>
                                            </td>
                                            <td className="px-4 py-3">
                                                <span
                                                    className={
                                                        entry.elo_change > 0
                                                            ? "text-emerald-600"
                                                            : "text-red-600"
                                                    }
                                                >
                                                    {entry.elo_change > 0 ? "+" : ""}
                                                    {entry.elo_change}
                                                </span>
                                            </td>
                                            <td className="px-4 py-3 font-medium">
                                                {entry.elo_after}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </section>
            )}

            {activeTab === "matches" && (
                <section className="space-y-4">
                    {matchLoading && (
                        <div className="rounded-xl border border-border bg-card/60 p-4 text-sm text-muted-foreground">
                            Loading match history...
                        </div>
                    )}
                    {matchError && (
                        <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
                            Failed to load match history.
                        </div>
                    )}
                    {matchHistory && matchHistory.length === 0 && (
                        <div className="rounded-xl border border-border bg-card/60 p-6 text-center text-muted-foreground">
                            No matches yet. Start playing to build your history!
                        </div>
                    )}
                    {matchHistory && matchHistory.length > 0 && (
                        <ul className="space-y-3">
                            {matchHistory.map((match, idx) => (
                                <li
                                    key={`${match.game_pk}-${idx}`}
                                    className="rounded-2xl border border-border bg-card/50 p-4 shadow-sm"
                                >
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <p className="font-semibold capitalize">
                                                {match.game_type}
                                            </p>
                                            <p className="text-sm text-muted-foreground">
                                                vs. {match.opponent_name} ({match.opponent_type})
                                            </p>
                                        </div>
                                        <div className="text-right">
                                            <span
                                                className={`rounded-full px-3 py-1 text-xs font-medium ${
                                                    match.won
                                                        ? "bg-emerald-100 text-emerald-800"
                                                        : "bg-red-100 text-red-800"
                                                }`}
                                            >
                                                {match.won ? "Victory" : "Defeat"}
                                            </span>
                                            <p
                                                className={`mt-1 text-sm font-medium ${
                                                    match.elo_change > 0
                                                        ? "text-emerald-600"
                                                        : "text-red-600"
                                                }`}
                                            >
                                                {match.elo_change > 0 ? "+" : ""}
                                                {match.elo_change} ELO
                                            </p>
                                        </div>
                                    </div>
                                    <p className="mt-2 text-xs text-muted-foreground">
                                        {new Date(match.played_at).toLocaleString()}
                                    </p>
                                </li>
                            ))}
                        </ul>
                    )}
                </section>
            )}
        </main>
    );
}

function StatCard({
    label,
    value,
    subtext,
}: {
    label: string;
    value: string;
    subtext?: string;
}) {
    return (
        <div className="rounded-2xl border border-border bg-card/50 p-4 shadow-sm">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">
                {label}
            </p>
            <p className="mt-1 text-2xl font-bold">{value}</p>
            {subtext && (
                <p className="mt-1 text-sm text-muted-foreground">{subtext}</p>
            )}
        </div>
    );
}

function TabButton({
    active,
    onClick,
    children,
}: {
    active: boolean;
    onClick: () => void;
    children: React.ReactNode;
}) {
    return (
        <button
            onClick={onClick}
            className={`border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
                active
                    ? "border-primary text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
        >
            {children}
        </button>
    );
}

function RankProgressBar({ elo }: { elo: number }) {
    const ranks = [
        { name: "Bronze", min: 0, max: 1199, color: "bg-amber-600" },
        { name: "Silver", min: 1200, max: 1399, color: "bg-gray-400" },
        { name: "Gold", min: 1400, max: 1599, color: "bg-yellow-500" },
        { name: "Platinum", min: 1600, max: 1799, color: "bg-cyan-400" },
        { name: "Diamond", min: 1800, max: 1999, color: "bg-blue-500" },
        { name: "Master", min: 2000, max: 3000, color: "bg-purple-600" },
    ];

    const currentRank = ranks.find((r) => elo >= r.min && elo <= r.max) || ranks[0];
    const progress = ((elo - currentRank.min) / (currentRank.max - currentRank.min)) * 100;
    const nextRank = ranks[ranks.indexOf(currentRank) + 1];

    return (
        <div>
            <div className="mb-2 flex justify-between text-xs text-muted-foreground">
                <span>{currentRank.name} ({currentRank.min})</span>
                <span>{nextRank ? `${nextRank.name} (${nextRank.min})` : "Max Rank"}</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-muted">
                <div
                    className={`h-full ${currentRank.color} transition-all`}
                    style={{ width: `${Math.min(progress, 100)}%` }}
                />
            </div>
            {nextRank && (
                <p className="mt-2 text-center text-xs text-muted-foreground">
                    {nextRank.min - elo} ELO to {nextRank.name}
                </p>
            )}
        </div>
    );
}
