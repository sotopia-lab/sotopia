import type { GameSummary } from "@/core/types/game-module";

export const games: GameSummary[] = [
    {
        slug: "werewolf",
        title: "Werewolf",
        summary:
            "Classic social deduction: survive the night, debate during the day, and outwit the pack.",
        tags: ["social deduction", "LLM agents", "turn-based"],
        accentColor: "#9333ea",
    },
];

export function getGameBySlug(slug: string): GameSummary | undefined {
    return games.find((game) => game.slug === slug);
}
