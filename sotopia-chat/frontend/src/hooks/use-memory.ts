"use client";

import { useState, useCallback } from "react";
import useSWR from "swr";
import { fetchMemory, updateMemory, type MemoryPayload } from "@/lib/api";

export function usePlayerMemory(participantId: string | null) {
    const {
        data,
        error,
        isLoading,
        mutate,
    } = useSWR<MemoryPayload>(
        participantId ? ["memory", participantId] : null,
        () => fetchMemory(participantId as string)
    );

    const [pendingNotes, setPendingNotes] = useState("");
    const [saveError, setSaveError] = useState<string | null>(null);
    const [isSaving, setIsSaving] = useState(false);

    const save = useCallback(
        async (updates: Partial<MemoryPayload>) => {
            if (!participantId) return;
            setSaveError(null);
            setIsSaving(true);
            try {
                const next = await updateMemory(participantId, {
                    participantId,
                    notes: updates.notes ?? data?.notes,
                    roleHistory: updates.roleHistory ?? data?.roleHistory ?? [],
                    voteHistory: updates.voteHistory ?? data?.voteHistory ?? [],
                    custom: updates.custom ?? data?.custom ?? {},
                });
                mutate(next, false);
            } catch (error) {
                console.error(error);
                setSaveError(
                    error instanceof Error
                        ? error.message
                        : "Failed to save memory"
                );
            } finally {
                setIsSaving(false);
            }
        },
        [participantId, data, mutate]
    );

    return {
        memory: data,
        isLoading,
        error,
        save,
        pendingNotes,
        setPendingNotes,
        saveError,
        isSaving,
    };
}
