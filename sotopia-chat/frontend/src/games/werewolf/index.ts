import { WerewolfGameBoard } from "@/games/werewolf/components/game-board";
import { WerewolfLobby } from "@/games/werewolf/components/lobby";
import { useWerewolfSession } from "@/games/werewolf/use-session";
import { useWerewolfActions } from "@/games/werewolf/use-actions";
import type { WerewolfSessionState } from "@/games/werewolf/types";
import type { GameDefinition } from "@/core/types/game-module";

// No-op consent component - consent is now integrated in the lobby
function InlineConsent({ onAccept }: { onAccept: () => void }) {
    // Auto-accept immediately - real consent is shown in the lobby
    if (typeof window !== 'undefined') {
        setTimeout(() => onAccept(), 0);
    }
    return null;
}

export const werewolfGame: GameDefinition<WerewolfSessionState> = {
    slug: "werewolf",
    title: "Werewolf",
    summary:
        "Classic social deduction: survive the night, debate during the day, and outwit the pack.",
    tags: ["social deduction", "LLM agents", "turn-based"],
    accentColor: "#9333ea",
    components: {
        Consent: InlineConsent,
        Lobby: WerewolfLobby,
        GameBoard: WerewolfGameBoard,
    },
    hooks: {
        useSession: useWerewolfSession,
        useActions: useWerewolfActions,
    },
};
