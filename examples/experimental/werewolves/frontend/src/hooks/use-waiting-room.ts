import { useCallback, useEffect, useRef, useState } from "react";
import { enterWaitingRoom } from "@/lib/api";

export type WaitingRoomStatus =
    | "idle"
    | "matching"
    | "matched"
    | "error"
    | "cancelled";

export interface WaitingRoomState {
    status: WaitingRoomStatus;
    sessionId: string | null;
    error?: string;
    isMatching: boolean;
}

export interface WaitingRoomControls {
    join: () => Promise<void>;
    cancel: () => void;
    reset: () => void;
}

export function useWaitingRoom(playerId: string | null): [
    WaitingRoomState,
    WaitingRoomControls
] {
    const [status, setStatus] = useState<WaitingRoomStatus>("idle");
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [error, setError] = useState<string | undefined>(undefined);
    const abortRef = useRef<{ cancelled: boolean }>({ cancelled: false });

    useEffect(() => {
        return () => {
            abortRef.current.cancelled = true;
        };
    }, []);

    const join = useCallback(async () => {
        if (!playerId) {
            setError("Missing participant identifier.");
            setStatus("error");
            return;
        }
        if (status === "matching") return;
        abortRef.current.cancelled = false;
        setStatus("matching");
        setError(undefined);

        try {
            while (!abortRef.current.cancelled) {
                const matchedSessionId = await enterWaitingRoom(playerId);
                if (matchedSessionId) {
                    setSessionId(matchedSessionId);
                    setStatus("matched");
                    return;
                }
                await new Promise((resolve) => setTimeout(resolve, 500));
            }
            setStatus("cancelled");
        } catch (err) {
            console.error(err);
            setError(
                err instanceof Error ? err.message : "Failed to join waiting room."
            );
            setStatus("error");
        }
    }, [playerId, status]);

    const cancel = useCallback(() => {
        abortRef.current.cancelled = true;
        if (status === "matching") {
            setStatus("cancelled");
        }
    }, [status]);

    const reset = useCallback(() => {
        abortRef.current.cancelled = true;
        setStatus("idle");
        setSessionId(null);
        setError(undefined);
    }, []);

    return [
        {
            status,
            sessionId,
            error,
            isMatching: status === "matching"
        },
        { join, cancel, reset }
    ];
}
