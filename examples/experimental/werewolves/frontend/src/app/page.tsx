"use client";

import { useState } from "react";
import { WerewolfLobby } from "@/components/lobby";
import { WerewolfGameBoard } from "@/components/game-board";
import { ConsentCard } from "@/components/consent-card";

export default function WerewolfPage() {
    const [consentAccepted, setConsentAccepted] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [playerId, setPlayerId] = useState<string | null>(null);

    const handleGameCreated = (newSessionId: string, newPlayerId: string) => {
        setSessionId(newSessionId);
        setPlayerId(newPlayerId);
    };

    return (
        <div className="flex min-h-screen items-center justify-center bg-background p-4">
            {!consentAccepted ? (
                <ConsentCard onAccept={() => setConsentAccepted(true)} />
            ) : !sessionId || !playerId ? (
                <WerewolfLobby onGameCreated={handleGameCreated} />
            ) : (
                <WerewolfGameBoard
                    sessionId={sessionId}
                    participantId={playerId}
                />
            )}
        </div>
    );
}
