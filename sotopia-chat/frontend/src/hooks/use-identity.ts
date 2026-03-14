"use client";

import { useCallback, useEffect, useState } from "react";
import useSWR from "swr";
import {
    registerIdentity,
    fetchIdentity,
    type IdentityRecord,
} from "@/lib/api";
import { getStoredToken, getStoredUser } from "@/lib/auth-api";

const STORAGE_KEY = "arena_identity_token";

export function useIdentity() {
    const [token, setToken] = useState<string | null>(null);
    const [pendingParticipantId, setPendingParticipantId] = useState("");
    const [pendingDisplayName, setPendingDisplayName] = useState("");
    const [registerError, setRegisterError] = useState<string | null>(null);
    const [registerLoading, setRegisterLoading] = useState(false);

    useEffect(() => {
        if (typeof window === "undefined") return;
        
        // First check if user is authenticated with JWT
        const jwtToken = getStoredToken();
        const jwtUser = getStoredUser();
        
        if (jwtToken && jwtUser) {
            // User is authenticated - auto-register identity using their pk
            const autoRegister = async () => {
                try {
                    const record = await registerIdentity({
                        participantId: jwtUser.pk,
                        displayName: jwtUser.username,
                    });
                    setToken(record.token);
                    window.localStorage.setItem(STORAGE_KEY, record.token);
                } catch (err) {
                    console.error("Auto-register identity failed:", err);
                    // Fall back to stored identity token
                    const stored = window.localStorage.getItem(STORAGE_KEY);
                    if (stored) {
                        setToken(stored);
                    }
                }
            };
            autoRegister();
        } else {
            // Not authenticated - use stored identity token
            const stored = window.localStorage.getItem(STORAGE_KEY);
            if (stored) {
                setToken(stored);
            }
        }
    }, []);

    const {
        data,
        error,
        isLoading: identityLoading,
        mutate,
    } = useSWR<IdentityRecord>(
        token ? ["identity", token] : null,
        () => fetchIdentity(token as string)
    );

    const saveToken = useCallback(
        (value: string) => {
            setToken(value);
            if (typeof window !== "undefined") {
                window.localStorage.setItem(STORAGE_KEY, value);
            }
        },
        []
    );

    const register = useCallback(
        async (participantId: string, displayName?: string) => {
            setRegisterError(null);
            if (!participantId.trim()) {
                setRegisterError("Participant ID is required");
                return;
            }
            setRegisterLoading(true);
            try {
                const record = await registerIdentity({
                    participantId: participantId.trim(),
                    displayName: displayName?.trim(),
                });
                saveToken(record.token);
                mutate(record, false);
            } catch (err) {
                console.error(err);
                setRegisterError(
                    err instanceof Error ? err.message : "Failed to register identity"
                );
            } finally {
                setRegisterLoading(false);
            }
        },
        [mutate, saveToken]
    );

    const clearIdentity = useCallback(() => {
        setToken(null);
        if (typeof window !== "undefined") {
            window.localStorage.removeItem(STORAGE_KEY);
        }
        mutate(undefined, false);
    }, [mutate]);

    return {
        token,
        identity: data,
        isLoading: identityLoading,
        error,
        pendingParticipantId,
        pendingDisplayName,
        setPendingParticipantId,
        setPendingDisplayName,
        register,
        registerError,
        registerLoading,
        clearIdentity,
    };
}
