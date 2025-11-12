import Link from "next/link";
import { games } from "@/core/config/games";

export default function GamesLandingPage() {
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
            </section>

            <section className="grid gap-6 sm:grid-cols-2">
                {games.map((game) => (
                    <Link
                        key={game.slug}
                        href={`/games/${game.slug}`}
                        className="group rounded-2xl border border-border bg-card/40 p-6 shadow-sm transition hover:-translate-y-0.5 hover:border-primary hover:shadow-lg"
                        style={
                            game.accentColor
                                ? { borderColor: `${game.accentColor}33` }
                                : undefined
                        }
                    >
                        <div className="flex items-center justify-between">
                            <h2 className="text-2xl font-semibold">
                                {game.title}
                            </h2>
                            <span className="text-sm text-primary opacity-80 group-hover:opacity-100">
                                Play →
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
                    </Link>
                ))}
            </section>
        </main>
    );
}
