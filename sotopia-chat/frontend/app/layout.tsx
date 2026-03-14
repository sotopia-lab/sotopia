import "./globals.css";
import type { Metadata } from "next";
import { Providers } from "@/components/providers";

export const metadata: Metadata = {
    title: "Sotopia Social Game Arena",
    description: "Modular Next.js frontend for Sotopia research games",
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
