import "./globals.css";
import type { Metadata } from "next";
import { Providers } from "@/components/providers";

export const metadata: Metadata = {
    title: "Werewolves Experiment",
    description: "Experimental lobby for Sotopia social games"
};

export default function RootLayout({
    children
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" suppressHydrationWarning>
            <body>
                <Providers
                    attribute="class"
                    defaultTheme="system"
                    enableSystem
                >
                    <div className="min-h-screen bg-background text-foreground">
                        {children}
                    </div>
                </Providers>
            </body>
        </html>
    );
}
