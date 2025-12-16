import type { GameSummary } from "@/core/types/game-module";

export const games: GameSummary[] = [
    {
        slug: "werewolf",
        title: "Werewolf",
        summary:
            "Classic social deduction: survive the night, debate during the day, and outwit the pack.",
        tags: ["social deduction", "LLM agents", "turn-based"],
        accentColor: "#9333ea",
        minPlayers: 6,
        maxPlayers: 6,
        estDurationMinutes: 15,
        status: "online",
        features: {
            teamChat: true,
            spectators: false,
            hasLeaderboard: true,
        },
    },
    {
        slug: "secret-mafia",
        title: "Secret Mafia",
        summary:
            "Hidden roles meet deception-heavy strategy. Humans and models collaborate to expose the pack.",
        tags: ["social deduction", "party", "coming soon"],
        accentColor: "#f97316",
        minPlayers: 8,
        maxPlayers: 12,
        estDurationMinutes: 25,
        status: "coming-soon",
        features: {
            teamChat: true,
            hasLeaderboard: true,
        },
    },
    {
        slug: "codenames-duo",
        title: "Codenames Duo",
        summary:
            "Word-association co-op where humans pair with LLM spymasters to guess the right agents.",
        tags: ["word", "co-op", "maintenance"],
        accentColor: "#0ea5e9",
        minPlayers: 4,
        maxPlayers: 6,
        estDurationMinutes: 12,
        status: "maintenance",
        features: {
            teamChat: true,
            spectators: true,
            hasLeaderboard: false,
        },
    },
    {
        slug: "colonel-blotto",
        title: "Colonel Blotto",
        summary:
            "Allocate troops across fronts against clever agents. Competitive resource allocation at scale.",
        tags: ["strategy", "multi-agent"],
        accentColor: "#22c55e",
        minPlayers: 2,
        maxPlayers: 4,
        estDurationMinutes: 8,
        status: "coming-soon",
        features: {
            spectators: false,
            hasLeaderboard: true,
        },
    },
];

export function getGameBySlug(slug: string): GameSummary | undefined {
    return games.find((game) => game.slug === slug);
}
