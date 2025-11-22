"use client";

import { useEffect, useState } from "react";
import { toast } from "react-hot-toast";

import { ConsentCard } from "@/core/components/consent-card";
import { WaitingRoomCard } from "@/components/waiting-room-card";
import { SessionPane } from "@/components/chat/session-pane";
import { useWaitingRoom } from "@/hooks/use-waiting-room";
import { useChatSession } from "@/hooks/use-chat-session";
import { useWerewolfSession } from "@/games/werewolf/use-session";
import { SessionSidebar } from "@/components/session-sidebar";

export function ExperimentRoot() {
    const [consentAccepted, setConsentAccepted] = useState(false);
    const [participantId, setParticipantId] = useState<string>("");

    const [waitingRoomState, waitingRoomControls] = useWaitingRoom(
        consentAccepted && participantId ? participantId : null
    );

    const sessionId = waitingRoomState.sessionId;
    const participantIdentifier = participantId || null;

    const [chatState, chatControls] = useChatSession(
        sessionId,
        participantIdentifier
    );

    const { session, error: sessionError } = useWerewolfSession(
        sessionId,
        participantIdentifier
    );

    useEffect(() => {
        if (waitingRoomState.status === "matched" && waitingRoomState.sessionId) {
            toast.success(`Matched! Session ID: ${waitingRoomState.sessionId}`);
        }
    }, [waitingRoomState.status, waitingRoomState.sessionId]);

    useEffect(() => {
        if (chatState.lastError) {
            toast.error(chatState.lastError);
        }
    }, [chatState.lastError]);

    useEffect(() => {
        if (sessionError) {
            toast.error(
                sessionError instanceof Error
                    ? sessionError.message
                    : "Failed to load session state."
            );
        }
    }, [sessionError]);

    const canShowWaitingRoom = consentAccepted && !sessionId;
    const canShowChat = consentAccepted && Boolean(sessionId);

    return (
        <div className="flex w-full max-w-5xl flex-col gap-8">
            {!consentAccepted ? (
                <ConsentCard onAccept={() => setConsentAccepted(true)} />
            ) : null}

            {canShowWaitingRoom ? (
                <WaitingRoomCard
                    playerId={participantId}
                    onPlayerIdChange={setParticipantId}
                    waitingRoom={waitingRoomState}
                    onJoin={waitingRoomControls.join}
                    onCancel={waitingRoomControls.cancel}
                />
            ) : null}

            {canShowChat ? (
                <div className="grid gap-6 md:grid-cols-[2fr_1fr]">
                    <SessionPane
                        chatState={chatState}
                        onSend={chatControls.send}
                        participantId={participantIdentifier}
                    />
                    <SessionSidebar state={session} />
                </div>
            ) : null}
        </div>
    );
}
