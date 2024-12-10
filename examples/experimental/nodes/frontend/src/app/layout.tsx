import '../styles/globals.css';
import { ReactNode } from 'react';

export const metadata = {
  title: 'AI Code Editor',
  description: 'A modern, interactive code editor interface designed to work with AI agents.',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}