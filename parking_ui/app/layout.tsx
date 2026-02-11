import React from "react"
import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'

import './globals.css'
import { BackgroundStream } from '@/components/background-stream'

const _geist = Geist({ subsets: ['latin'] })
const _geistMono = Geist_Mono({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Smart Parking Detection',
  description: 'Real-time parking availability monitoring system',
  generator: 'v0.app',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased relative">
        {/* Background stream is low-opacity and pointer-events-none so it won't block UI */}
        <BackgroundStream />
        {/* Main content should appear above the background */}
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  )
}
