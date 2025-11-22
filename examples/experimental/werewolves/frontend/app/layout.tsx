import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Duskmire Werewolves',
  description: 'A social deduction game powered by AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
