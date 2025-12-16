import { ChatMessage } from "@/hooks/use-chat-session";
import { formatTimestamp } from "@/lib/utils";

interface MessageListProps {
    messages: ChatMessage[];
    participantId: string | null;
}

export function MessageList({ messages, participantId }: MessageListProps) {
    if (!messages.length) {
        return (
            <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
                Waiting for the conversation to begin…
            </div>
        );
    }

    return (
        <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-4">
            {messages.map((msg) => (
                <div
                    key={msg.id}
                    className={`flex flex-col ${msg.sender === "client" ? "items-end" : "items-start"}`}
                >
                    <span className="text-xs uppercase tracking-wide text-muted-foreground">
                        {labelForSender(msg.sender, participantId)}
                    </span>
                    <div
                        className={`mt-1 max-w-[80%] rounded-lg px-3 py-2 text-sm shadow-sm ${
                            msg.sender === "client"
                                ? "bg-primary text-primary-foreground"
                                : msg.sender === "server"
                                ? "bg-secondary text-secondary-foreground"
                                : "bg-muted text-muted-foreground"
                        }`}
                    >
                        {msg.text}
                    </div>
                    <span className="mt-1 text-[10px] uppercase text-muted-foreground">
                        {formatTimestamp(msg.id)}
                    </span>
                </div>
            ))}
        </div>
    );
}

function labelForSender(
    sender: ChatMessage["sender"],
    participantId: string | null
) {
    switch (sender) {
        case "client":
            return participantId ?? "You";
        case "server":
            return "Partner";
        default:
            return "System";
    }
}
