import useSWR from "swr";
import { useMemo } from "react";

import { getWerewolfSessionState } from "@/lib/werewolf-api";
import type { WerewolfSessionState } from "@/lib/types";

export function useWerewolfSession(
    sessionId: string | null,
    participantId: string | null
) {
    const key = sessionId ? ["werewolf-session", sessionId] : null;

    const { data, error, isLoading } = useSWR<WerewolfSessionState>(
        key,
        () => (sessionId ? getWerewolfSessionState(sessionId) : Promise.reject()),
        {
            refreshInterval: 2000,
        }
    );

    const derived = useMemo(() => {
        if (!data) return undefined;
        const me =
            data.players.find((player) => player.isHost) ??
            data.me ??
            null;
        return {
            ...data,
            me,
        };
    }, [data]);

    return {
        session: derived,
        isLoading,
        error,
    };
}
