import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

export function formatTimestamp(timestamp: string) {
    const date = new Date(Number(timestamp) * 1000);
    if (Number.isNaN(date.getTime())) {
        return timestamp;
    }
    return date.toLocaleTimeString();
}
