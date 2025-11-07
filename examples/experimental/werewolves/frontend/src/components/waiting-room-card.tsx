"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import type { WaitingRoomState } from "@/hooks/use-waiting-room";

interface WaitingRoomCardProps {
    playerId: string;
    onPlayerIdChange: (value: string) => void;
    waitingRoom: WaitingRoomState;
    onJoin: () => Promise<void>;
    onCancel: () => void;
}

export function WaitingRoomCard({
    playerId,
    onPlayerIdChange,
    waitingRoom,
    onJoin,
    onCancel
}: WaitingRoomCardProps) {
    const [pendingId, setPendingId] = useState(playerId);

    useEffect(() => setPendingId(playerId), [playerId]);

    const matching = waitingRoom.isMatching;

    return (
        <div className="mx-auto flex w-full max-w-xl flex-col gap-4 rounded-lg border bg-card p-6 shadow-sm">
            <header>
                <h2 className="text-xl font-medium">
                    Enter the Waiting Room
                </h2>
                <p className="text-sm text-muted-foreground">
                    Provide the identifier your facilitator gave you. We&apos;ll
                    match you with another participant automatically.
                </p>
            </header>
            <label className="flex flex-col gap-2 text-sm">
                Participant identifier
                <input
                    className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    value={pendingId}
                    onChange={(event) => setPendingId(event.target.value)}
                    placeholder="example@andrew.cmu.edu"
                    disabled={matching}
                />
            </label>
            <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>
                    Status:{" "}
                    <strong className="uppercase">
                        {waitingRoom.status}
                    </strong>
                </span>
                {waitingRoom.error ? (
                    <span className="text-destructive">
                        {waitingRoom.error}
                    </span>
                ) : null}
            </div>
            <div className="flex justify-end gap-2">
                <Button
                    variant="outline"
                    onClick={() => {
                        onPlayerIdChange(pendingId.trim());
                        onCancel();
                    }}
                    disabled={!matching}
                >
                    Cancel
                </Button>
                <Button
                    onClick={async () => {
                        onPlayerIdChange(pendingId.trim());
                        await onJoin();
                    }}
                    disabled={!pendingId.trim()}
                >
                    {matching ? "Matching…" : "Enter waiting room"}
                </Button>
            </div>
        </div>
    );
}
