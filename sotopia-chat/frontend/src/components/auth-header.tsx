"use client";

import Link from "next/link";
import { useAuth } from "@/contexts/auth-context";
import { Button } from "@/components/ui/button";

export function AuthHeader() {
    const { user, isAuthenticated, isLoading, logout } = useAuth();

    return (
        <header className="fixed top-0 left-0 right-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-6">
                <Link href="/" className="font-semibold">
                    Sotopia Arena
                </Link>

                <nav className="flex items-center gap-4">
                    <Link
                        href="/leaderboard"
                        className="text-sm text-muted-foreground hover:text-foreground"
                    >
                        Leaderboard
                    </Link>

                    {isLoading ? (
                        <span className="text-sm text-muted-foreground">...</span>
                    ) : isAuthenticated && user ? (
                        <div className="flex items-center gap-3">
                            <Link
                                href="/profile"
                                className="flex items-center gap-2 text-sm hover:text-primary"
                            >
                                <span className="rounded-full bg-primary/10 px-2 py-1 text-xs font-medium text-primary">
                                    {user.elo_rating} ELO
                                </span>
                                <span className="font-medium">{user.username}</span>
                            </Link>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={logout}
                                className="text-muted-foreground"
                            >
                                Logout
                            </Button>
                        </div>
                    ) : (
                        <div className="flex items-center gap-2">
                            <Link href="/login">
                                <Button variant="ghost" size="sm">
                                    Sign In
                                </Button>
                            </Link>
                            <Link href="/register">
                                <Button size="sm">Sign Up</Button>
                            </Link>
                        </div>
                    )}
                </nav>
            </div>
        </header>
    );
}
