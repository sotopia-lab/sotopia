#!/usr/bin/env python3
"""
Utility to scaffold a new game module in the arena frontend.

Usage:
    python scripts/scaffold_game_module.py werewolf "Werewolf"
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = ROOT / "sotopia-chat" / "frontend" / "src"
GAMES_DIR = FRONTEND_SRC / "games"

TEMPLATE_FILES = {
    "api.ts": """export async function create{Title}Game() {
    throw new Error("API not implemented yet for {slug}");
}
""",
    "types.ts": """export interface {Title}SessionState {
    phase: string;
}
""",
    "use-session.ts": """export function use{Title}Session(
    sessionId: string | null,
    participantId: string | null
) {
    return {
        session: undefined,
        isLoading: false,
        error: undefined,
    };
}
""",
    "use-actions.ts": """export function use{Title}Actions(
    sessionId: string | null,
    participantId: string | null
) {
    return [
        { isSubmitting: false },
        {
            submitAction: async () => {
                throw new Error("Not implemented");
            },
            clearError: () => undefined,
        },
    ] as const;
}
""",
    "components/game-board.tsx": """export function {Title}GameBoard() {
    return <div>{Title} placeholder board.</div>;
}
""",
    "components/lobby.tsx": """export function {Title}Lobby() {
    return <div>{Title} lobby placeholder.</div>;
}
""",
    "index.ts": """import { {Title}GameBoard } from "./components/game-board";

export const {slug_camel}Game = {{
    slug: "{slug}",
    title: "{Title}",
    summary: "{Title} summary",
    components: {{
        Consent: () => null,
        Lobby: {Title}Lobby,
        GameBoard: {Title}GameBoard,
    }},
    hooks: {{
        useSession: () => ({{ isLoading: false, session: undefined }}),
        useActions: () => [{{ isSubmitting: false }}, {{ submitAction: async () => undefined, clearError: () => undefined }}],
    }},
}};
""",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new game module.")
    parser.add_argument("slug", help="game slug, e.g. werewolf")
    parser.add_argument("title", nargs="?", help="human readable title")
    args = parser.parse_args()

    slug = args.slug.lower()
    title = args.title or slug.replace("-", " ").title()
    target_dir = GAMES_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "components").mkdir(exist_ok=True)

    slug_camel = "".join(part.capitalize() for part in slug.split("-"))

    for relative, template in TEMPLATE_FILES.items():
        dest = target_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        content = template.format(slug=slug, Title=title, slug_camel=slug_camel)
        dest.write_text(content)

    print(f"Scaffolded game module in {target_dir}")


if __name__ == "__main__":
    main()
