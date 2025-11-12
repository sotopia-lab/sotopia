import { useCallback, useEffect, useRef, useState } from "react";
import {
    connectSessionAsClient,
    getClientLock,
    getSessionMessages,
    sendMessage
} from "@/lib/api";
import type { MessageTransaction } from "@/lib/types";

export interface ChatMessage {
    id: string;
    sender: "client" | "server" | "system";
    text: string;
    raw: MessageTransaction;
}

export interface ChatSessionState {
    messages: ChatMessage[];
    isConnected: boolean;
    isTurn: boolean;
    isSending: boolean;
    lastError?: string;
}

export interface ChatSessionControls {
    send: (content: string) => Promise<void>;
    clearErrors: () => void;
}

const POLL_INTERVAL_MS = 750;
const LOCK_INTERVAL_MS = 1200;

export function useChatSession(
    sessionId: string | null,
    participantId: string | null
): [ChatSessionState, ChatSessionControls] {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const [isTurn, setIsTurn] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [lastError, setLastError] = useState<string | undefined>(undefined);

    const pollTimer = useRef<NodeJS.Timeout | null>(null);
    const lockTimer = useRef<NodeJS.Timeout | null>(null);
    const connectingRef = useRef(false);

    const reset = useCallback(() => {
        pollTimer.current && clearInterval(pollTimer.current);
        lockTimer.current && clearInterval(lockTimer.current);
        pollTimer.current = null;
        lockTimer.current = null;

        setMessages([]);
        setIsConnected(false);
        setIsTurn(false);
        setIsSending(false);
        setLastError(undefined);
        connectingRef.current = false;
    }, []);

    useEffect(() => {
        reset();
        if (!sessionId || !participantId) {
            return;
        }
        let isMounted = true;
        if (!connectingRef.current) {
            connectingRef.current = true;
            connectSessionAsClient(sessionId, participantId)
                .then((initialMessages) => {
                    if (!isMounted) return;
                    setIsConnected(true);
                    setMessages(convertMessages(initialMessages));
                })
                .catch((err) => {
                    console.error(err);
                    if (!isMounted) return;
                    setLastError(
                        err instanceof Error ? err.message : "Failed to connect."
                    );
                })
                .finally(() => {
                    connectingRef.current = false;
                });
        }

        pollTimer.current = setInterval(() => {
            getSessionMessages(sessionId)
                .then((data) => {
                    if (!isMounted) return;
                    setMessages(convertMessages(data));
                })
                .catch((err) => {
                    console.error(err);
                    if (!isMounted) return;
                    setLastError(
                        err instanceof Error
                            ? err.message
                            : "Failed to refresh messages."
                    );
                });
        }, POLL_INTERVAL_MS);

        lockTimer.current = setInterval(() => {
            getClientLock(sessionId)
                .then((lock) => {
                    if (!isMounted) return;
                    setIsTurn(lock === "action");
                })
                .catch((err) => {
                    console.error(err);
                });
        }, LOCK_INTERVAL_MS);

        return () => {
            isMounted = false;
            reset();
        };
    }, [sessionId, participantId, reset]);

    const send = useCallback(
        async (content: string) => {
            if (!sessionId || !participantId) {
                setLastError("Missing session or participant information.");
                return;
            }
            if (!content.trim()) {
                return;
            }
            try {
                setIsSending(true);
                const response = await sendMessage(sessionId, participantId, content);
                setMessages(convertMessages(response));
            } catch (err) {
                console.error(err);
                setLastError(
                    err instanceof Error ? err.message : "Failed to send message."
                );
            } finally {
                setIsSending(false);
            }
        },
        [participantId, sessionId]
    );

    const clearErrors = useCallback(() => setLastError(undefined), []);

    return [
        { messages, isConnected, isTurn, isSending, lastError },
        { send, clearErrors }
    ];
}

function convertMessages(entries: MessageTransaction[]): ChatMessage[] {
    return entries.map((entry) => ({
        id: entry.timestamp_str,
        sender:
            entry.sender === "client"
                ? "client"
                : entry.sender === "server"
                ? "server"
                : "system",
        text: entry.message,
        raw: entry
    }));
}
