"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/contexts/auth-context";
import { AuthError } from "@/lib/auth-api";
import { Button } from "@/components/ui/button";

export default function LoginPage() {
    const router = useRouter();
    const { login, isAuthenticated, isLoading, error, clearError } = useAuth();
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [submitting, setSubmitting] = useState(false);
    const [localError, setLocalError] = useState<string | null>(null);

    // Redirect if already authenticated
    useEffect(() => {
        if (isAuthenticated && !isLoading) {
            router.push("/");
        }
    }, [isAuthenticated, isLoading, router]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLocalError(null);
        clearError();

        if (!username.trim() || !password) {
            setLocalError("Username and password are required");
            return;
        }

        setSubmitting(true);
        try {
            await login({ username: username.trim(), password });
            router.push("/");
        } catch (err) {
            // Extract error detail from AuthError
            if (err instanceof AuthError && err.detail) {
                const detail = err.detail as { detail?: string };
                setLocalError(detail.detail || "Login failed");
            } else if (err instanceof Error) {
                setLocalError("Invalid username or password");
            }
        } finally {
            setSubmitting(false);
        }
    };

    if (isLoading) {
        return (
            <main className="flex min-h-screen items-center justify-center bg-background">
                <div className="text-muted-foreground">Loading...</div>
            </main>
        );
    }

    return (
        <main className="flex min-h-screen items-center justify-center bg-background px-4">
            <div className="w-full max-w-md space-y-8">
                <header className="text-center">
                    <h1 className="text-3xl font-bold tracking-tight">Welcome Back</h1>
                    <p className="mt-2 text-sm text-muted-foreground">
                        Sign in to your Sotopia Arena account
                    </p>
                </header>

                <form onSubmit={handleSubmit} className="space-y-6">
                    {(localError || error) && (
                        <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
                            {localError || error}
                        </div>
                    )}

                    <div className="space-y-4">
                        <div>
                            <label
                                htmlFor="username"
                                className="block text-sm font-medium text-foreground"
                            >
                                Username
                            </label>
                            <input
                                id="username"
                                name="username"
                                type="text"
                                autoComplete="username"
                                required
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                className="mt-1 block w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring"
                                placeholder="your_username"
                            />
                        </div>

                        <div>
                            <label
                                htmlFor="password"
                                className="block text-sm font-medium text-foreground"
                            >
                                Password
                            </label>
                            <input
                                id="password"
                                name="password"
                                type="password"
                                autoComplete="current-password"
                                required
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="mt-1 block w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring"
                                placeholder="••••••••"
                            />
                        </div>
                    </div>

                    <Button
                        type="submit"
                        disabled={submitting}
                        className="w-full"
                    >
                        {submitting ? "Signing in..." : "Sign In"}
                    </Button>
                </form>

                <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                        <span className="w-full border-t border-border" />
                    </div>
                    <div className="relative flex justify-center text-xs uppercase">
                        <span className="bg-background px-2 text-muted-foreground">
                            Or continue with
                        </span>
                    </div>
                </div>

                <div className="grid grid-cols-3 gap-3">
                    <Button
                        type="button"
                        variant="outline"
                        onClick={() => window.location.href = `${process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8800'}/oauth/login/google`}
                        className="w-full"
                    >
                        Google
                    </Button>
                    <Button
                        type="button"
                        variant="outline"
                        onClick={() => window.location.href = `${process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8800'}/oauth/login/github`}
                        className="w-full"
                    >
                        GitHub
                    </Button>
                    <Button
                        type="button"
                        variant="outline"
                        onClick={() => window.location.href = `${process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8800'}/oauth/login/discord`}
                        className="w-full"
                    >
                        Discord
                    </Button>
                </div>

                <p className="text-center text-sm text-muted-foreground">
                    Don&apos;t have an account?{" "}
                    <Link
                        href="/register"
                        className="font-medium text-primary hover:underline"
                    >
                        Sign up
                    </Link>
                </p>

                <p className="text-center text-sm text-muted-foreground">
                    <Link href="/" className="hover:underline">
                        ← Back to Arena
                    </Link>
                </p>
            </div>
        </main>
    );
}
