"use client";

import { Toaster } from "react-hot-toast";
import { ThemeProvider } from "next-themes";
import type { ThemeProviderProps } from "next-themes/dist/types";

export function Providers({
    children,
    ...themeProps
}: ThemeProviderProps) {
    return (
        <ThemeProvider {...themeProps}>
            <Toaster position="top-right" />
            {children}
        </ThemeProvider>
    );
}
