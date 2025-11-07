/**
 * Hook for submitting werewolf game actions
 */

import { useCallback, useState } from "react";
import { submitWerewolfAction } from "@/lib/werewolf-api";
import type { WerewolfAction } from "@/lib/werewolf-api";

export interface WerewolfActionsState {
    isSubmitting: boolean;
    lastError?: string;
}

export interface WerewolfActionsControls {
    submitAction: (actionType: string, argument: string) => Promise<void>;
    clearError: () => void;
}

export function useWerewolfActions(
    sessionId: string | null,
    participantId: string | null
): [WerewolfActionsState, WerewolfActionsControls] {
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [lastError, setLastError] = useState<string | undefined>(undefined);

    const submitAction = useCallback(
        async (actionType: string, argument: string) => {
            if (!sessionId || !participantId) {
                setLastError("Missing session or participant information");
                return;
            }

            try {
                setIsSubmitting(true);
                setLastError(undefined);

                const action: WerewolfAction = {
                    action_type: actionType,
                    argument: argument.trim(),
                };

                await submitWerewolfAction(sessionId, participantId, action);
            } catch (error) {
                console.error("Failed to submit action:", error);
                setLastError(
                    error instanceof Error
                        ? error.message
                        : "Failed to submit action"
                );
            } finally {
                setIsSubmitting(false);
            }
        },
        [sessionId, participantId]
    );

    const clearError = useCallback(() => {
        setLastError(undefined);
    }, []);

    return [{ isSubmitting, lastError }, { submitAction, clearError }];
}