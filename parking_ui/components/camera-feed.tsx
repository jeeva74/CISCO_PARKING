'use client'

import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Camera, CircleDot } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

interface CameraFeedProps {
  camereName?: string
  isLive?: boolean
  fps?: number
}

export function CameraFeed({
  camereName = 'Lot A - Camera 1',
  isLive = true,
  fps = 30,
}: CameraFeedProps) {
  const API_BASE = (process.env.NEXT_PUBLIC_API_BASE as string) || 'http://localhost:5000'
  const snapshotUrl = `${API_BASE}/api/snapshot`
  const [src, setSrc] = useState<string | null>(null)
  const blobRef = useRef<string | null>(null)

  useEffect(() => {
    let mounted = true
    const id = setInterval(async () => {
      try {
        const res = await fetch(`${snapshotUrl}?t=${Date.now()}`, { cache: 'no-store' })
        if (!res.ok) return
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        if (blobRef.current) URL.revokeObjectURL(blobRef.current)
        blobRef.current = url
        if (mounted) setSrc(url)
      } catch (e) {
        // ignore; UI will show placeholder
      }
    }, 1000)

    // initial fetch
    void (async () => {
      try {
        const res = await fetch(`${snapshotUrl}?t=${Date.now()}`, { cache: 'no-store' })
        if (!res.ok) return
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        blobRef.current = url
        if (mounted) setSrc(url)
      } catch (_) {}
    })()

    return () => {
      mounted = false
      clearInterval(id)
      if (blobRef.current) URL.revokeObjectURL(blobRef.current)
    }
  }, [snapshotUrl])

  return (
    <Card className="overflow-hidden border border-border/20 bg-gradient-to-br from-card to-card/60 h-full flex flex-col relative">
      <div className="bg-black/40 backdrop-blur-sm p-6 border-b border-border/20 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Camera className="w-5 h-5 text-primary" />
          <div>
            <h3 className="text-lg font-semibold text-foreground">{camereName}</h3>
            <p className="text-sm text-muted-foreground">Live Detection Feed</p>
          </div>
        </div>
        {isLive && (
          <Badge className="bg-destructive/90 hover:bg-destructive text-white gap-2 animate-pulse">
            <CircleDot className="w-3 h-3 fill-current" />
            LIVE
          </Badge>
        )}
      </div>

      {/* Feed Container */}
      <div className="flex-1 relative overflow-hidden bg-black/60">
        {/* Snapshot image fills the container when available */}
        {src ? (
          <img src={src} alt="camera-snapshot" className="absolute inset-0 w-full h-full object-cover" />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center">
              <Camera className="w-16 h-16 text-muted-foreground/30 mx-auto mb-4" />
              <p className="text-muted-foreground text-sm">Camera feed will display here</p>
              <p className="text-xs text-muted-foreground mt-2">Real-time parking detection visualization</p>
            </div>
          </div>
        )}

        {/* Grid overlay for parking spaces visualization */}
        <div className="absolute inset-0 opacity-10 pointer-events-none">
          <div className="grid grid-cols-4 h-full gap-1 p-4">
            {Array.from({ length: 20 }).map((_, i) => (
              <div
                key={i}
                className={`border-2 rounded ${
                  i % 3 === 0 ? 'border-destructive bg-destructive/5' : 'border-primary bg-primary/5'
                }`}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Footer Stats */}
      <div className="bg-black/40 backdrop-blur-sm p-4 border-t border-border/20 flex items-center justify-between">
        <div className="flex gap-6">
          <div>
            <p className="text-xs text-muted-foreground mb-1">FPS</p>
            <p className="text-sm font-semibold text-foreground">{fps}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground mb-1">Resolution</p>
            <p className="text-sm font-semibold text-foreground">1920x1080</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground mb-1">Status</p>
            <p className="text-sm font-semibold text-primary">Connected</p>
          </div>
        </div>
      </div>
    </Card>
  )
}
