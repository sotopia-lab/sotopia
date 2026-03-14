"use client";

import { Toaster } from "react-hot-toast";
import { ThemeProvider } from "next-themes";
import type { ThemeProviderProps } from "next-themes/dist/types";
import { AuthProvider } from "@/contexts/auth-context";

export function Providers({
    children,
    ...themeProps
}: ThemeProviderProps) {
    return (
        <ThemeProvider {...themeProps}>
            <AuthProvider>
                <Toaster position="top-right" />
                {children}
            </AuthProvider>
        </ThemeProvider>
    );
}
