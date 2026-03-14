/**
 * Hook for submitting werewolf game actions via WebSocket
 */

import { useCallback, useState, useRef } from "react";
import type { SimulationWebSocket } from "@/lib/websocket-utils";

export interface WerewolfActionsState {
    isSubmitting: boolean;
    lastError?: string;
}

export interface WerewolfActionsControls {
    submitAction: (actionType: string, argument: string) => Promise<void>;
    clearError: () => void;
    setWebSocket: (ws: unknown) => void;
}

export function useWerewolfActions(
    sessionId: string | null,
    participantId: string | null
): [WerewolfActionsState, WerewolfActionsControls] {
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [lastError, setLastError] = useState<string | undefined>(undefined);
    const wsRef = useRef<SimulationWebSocket | null>(null);

    const setWebSocket = useCallback((ws: unknown) => {
        wsRef.current = ws as SimulationWebSocket | null;
    }, []);

    const submitAction = useCallback(
        async (actionType: string, argument: string) => {
            if (!wsRef.current) {
                setLastError("WebSocket not connected");
                console.error("[Actions] WebSocket not connected");
                return;
            }

            if (!sessionId || !participantId) {
                setLastError("Session or participant ID missing");
                console.error("[Actions] Session or participant ID missing");
                return;
            }

            setIsSubmitting(true);
            setLastError(undefined);

            try {
                // Send CLIENT_MSG with the action
                const message = {
                    session_id: sessionId,
                    participant_id: participantId,
                    action_type: actionType,
                    content: argument,
                    timestamp: new Date().toISOString(),
                };

                wsRef.current.sendClientMessage(message);
            } catch (error) {
                const errorMsg = error instanceof Error ? error.message : "Failed to send action";
                setLastError(errorMsg);
                console.error("[Actions] Error sending action:", error);
            } finally {
                setIsSubmitting(false);
            }
        },
        [sessionId, participantId]
    );

    const clearError = useCallback(() => {
        setLastError(undefined);
    }, []);

    return [
        { isSubmitting, lastError },
        { submitAction, clearError, setWebSocket },
    ];
}
