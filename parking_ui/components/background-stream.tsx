'use client'

import React from 'react'
import { useEffect, useRef, useState } from 'react'

/**
 * BackgroundStream (improved)
 * - Prefer polling `/api/snapshot` which returns a single JPEG. This is
 *   compatible across browsers and avoids MJPEG rendering quirks.
 * - Falls back to direct MJPEG `/api/video-stream` if snapshot polling
 *   repeatedly fails.
 * - The image element uses `pointer-events: none` so it won't block UI.
 */
export function BackgroundStream() {
  const API_BASE = (process.env.NEXT_PUBLIC_API_BASE as string) || 'http://localhost:5000'
  const snapshotUrl = `${API_BASE}/api/snapshot`
  const streamUrl = `${API_BASE}/api/video-stream`

  const [src, setSrc] = useState<string>(streamUrl)
  const [status, setStatus] = useState<'loading' | 'ok' | 'error'>('loading')
  const [lastError, setLastError] = useState<string | null>(null)
  const blobUrlRef = useRef<string | null>(null)
  const failCountRef = useRef(0)

  useEffect(() => {
    let mounted = true
    // Poll snapshot every second
    const interval = setInterval(async () => {
      try {
        const url = `${snapshotUrl}?t=${Date.now()}`
        const res = await fetch(url, { cache: 'no-store' })
        if (!res.ok) throw new Error(`snapshot ${res.status}`)
        const blob = await res.blob()
        const bUrl = URL.createObjectURL(blob)
        // revoke previous blob URL
        if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current)
        blobUrlRef.current = bUrl
        if (mounted) {
          setSrc(bUrl)
          setStatus('ok')
          setLastError(null)
        }
        failCountRef.current = 0
      } catch (e) {
        const msg = (e as any)?.message || String(e)
        if (mounted) {
          setStatus('error')
          setLastError(msg)
        }
        failCountRef.current += 1
        // After a few failures, fall back to MJPEG stream
        if (failCountRef.current >= 3 && mounted) setSrc(streamUrl)
      }
    }, 1000)

    return () => {
      mounted = false
      clearInterval(interval)
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current)
    }
  }, [snapshotUrl, streamUrl])

  return (
    <>
      <img
        src={src}
        alt="background-stream"
        className="fixed inset-0 w-full h-full object-cover opacity-60 pointer-events-none z-0"
      />
      <div className="fixed top-4 right-4 z-30 pointer-events-auto">
        <div className="px-3 py-1 rounded-md text-sm font-medium bg-black/60 text-white">
          Video: {status.toUpperCase()}
          {lastError ? <div className="text-xs text-red-300">{lastError}</div> : null}
        </div>
      </div>
    </>
  )
}
