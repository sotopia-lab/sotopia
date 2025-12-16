"use client";

import { FormEvent, useState } from "react";
import TextareaAutosize from "react-textarea-autosize";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ComposeFormProps {
    onSend: (value: string) => Promise<void>;
    disabled?: boolean;
    isTurn: boolean;
}

export function ComposeForm({
    onSend,
    disabled,
    isTurn
}: ComposeFormProps) {
    const [value, setValue] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!value.trim() || disabled || isSubmitting) return;
        try {
            setIsSubmitting(true);
            await onSend(value.trim());
            setValue("");
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <form
            onSubmit={handleSubmit}
            className="flex flex-col gap-3 border-t bg-background p-4"
        >
            <div
                className={cn(
                    "text-xs uppercase tracking-wide",
                    isTurn ? "text-emerald-600" : "text-muted-foreground"
                )}
            >
                {isTurn
                    ? "Your turn — you may submit an action."
                    : "Waiting for partner…"}
            </div>
            <TextareaAutosize
                minRows={2}
                maxRows={6}
                value={value}
                onChange={(event) => setValue(event.target.value)}
                className="w-full resize-none rounded-md border border-border bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                placeholder={
                    isTurn ? "Describe your action or message…" : "Please wait…"
                }
                disabled={disabled || !isTurn}
            />
            <div className="flex justify-end gap-2">
                <Button
                    type="submit"
                    disabled={disabled || isSubmitting || !isTurn}
                >
                    {isSubmitting ? "Sending…" : "Send"}
                </Button>
            </div>
        </form>
    );
}
