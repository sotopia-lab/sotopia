import { MessageList } from "@/components/chat/message-list";
import { ComposeForm } from "@/components/chat/compose-form";
import { ChatSessionState } from "@/hooks/use-chat-session";

interface SessionPaneProps {
    chatState: ChatSessionState;
    onSend: (content: string) => Promise<void>;
    participantId: string | null;
}

export function SessionPane({
    chatState,
    onSend,
    participantId
}: SessionPaneProps) {
    return (
        <div className="flex h-full flex-col rounded-lg border bg-card shadow-sm">
            <header className="border-b px-4 py-3">
                <div className="text-sm font-medium">
                    Session Transcript
                </div>
                <div className="text-xs text-muted-foreground">
                    {chatState.isConnected
                        ? "Connected to session."
                        : "Connecting…"}
                </div>
                {chatState.lastError ? (
                    <div className="mt-2 rounded bg-destructive/10 px-2 py-1 text-xs text-destructive">
                        {chatState.lastError}
                    </div>
                ) : null}
            </header>
            <MessageList
                messages={chatState.messages}
                participantId={participantId}
            />
            <ComposeForm
                onSend={onSend}
                disabled={chatState.isSending || !chatState.isConnected}
                isTurn={chatState.isTurn}
            />
        </div>
    );
}
