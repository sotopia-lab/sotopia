 "use client";

import { useState } from "react";
import { ConsentCard } from "@/components/consent-card";
import { WerewolfLobby } from "@/components/lobby";
import { WerewolfGameBoard } from "@/components/game-board";

export default function WerewolfPage() {
    const [consentAccepted, setConsentAccepted] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [playerId, setPlayerId] = useState<string | null>(null);

    const handleGameCreated = (newSessionId: string, newPlayerId: string) => {
        setSessionId(newSessionId);
        setPlayerId(newPlayerId);
    };

    return (
        <main className="flex min-h-screen flex-col items-center justify-center bg-background p-4">
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
        </main>
    );
}
