import type { WerewolfSessionState } from "@/lib/types";

interface SessionSidebarProps {
    state?: WerewolfSessionState;
}

export function SessionSidebar({ state }: SessionSidebarProps) {
    if (!state) {
        return (
            <aside className="flex h-full flex-col gap-4 rounded-lg border bg-card p-4 text-sm text-muted-foreground">
                <span>Fetching session state…</span>
            </aside>
        );
    }

    return (
        <aside className="flex h-full flex-col gap-4 rounded-lg border bg-card p-4">
            <header>
                <h3 className="text-lg font-semibold">Session Overview</h3>
                <p className="text-xs text-muted-foreground">
                    Phase: {state.phase.phase}
                </p>
            </header>
            <section className="space-y-2">
                <h4 className="text-sm font-medium uppercase tracking-wide text-muted-foreground">
                    Players
                </h4>
                <ul className="space-y-2 text-sm">
                    {state.players.map((player) => (
                        <li
                            key={player.id}
                            className="flex items-center justify-between rounded border border-border px-3 py-2 text-foreground"
                        >
                            <div className="flex flex-col">
                                <span className="font-medium">
                                    {player.displayName}
                                </span>
                                <span className="text-xs text-muted-foreground">
                                    {player.role}
                                    {player.isHost ? " • host" : ""}
                                </span>
                            </div>
                            <span
                                className={`text-xs uppercase ${player.isAlive ? "text-emerald-600" : "text-destructive"}`}
                            >
                                {player.isAlive ? "alive" : "eliminated"}
                            </span>
                        </li>
                    ))}
                </ul>
            </section>
            <section className="mt-auto space-y-2 text-sm">
                <h4 className="font-medium">Available actions</h4>
                {state.availableActions.length ? (
                    <ul className="flex flex-wrap gap-2 text-xs uppercase text-muted-foreground">
                        {state.availableActions.map((action) => (
                            <li
                                key={action}
                                className="rounded border border-border px-2 py-1"
                            >
                                {action}
                            </li>
                        ))}
                    </ul>
                ) : (
                    <p className="text-xs text-muted-foreground">
                        Waiting for your turn…
                    </p>
                )}
            </section>
        </aside>
    );
}
