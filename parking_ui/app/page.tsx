'use client'

import { ParkingStatsCard } from '@/components/parking-stats-card'
import { CameraFeed } from '@/components/camera-feed'
import { OverallStatsCard } from '@/components/overall-stats-card'
import { Bike, Car, Activity } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { RefreshCw } from 'lucide-react'
import { useParkingData } from '@/hooks/use-parking'

export default function Dashboard() {
  // Use backend-driven state via custom hook
  const { data, loading, error, refresh } = useParkingData(5000)

  // Fallbacks while data loads
  const two = data?.twoWheeler ?? { available: 0, occupied: 0, total: 0 }
  const four = data?.fourWheeler ?? { available: 0, occupied: 0, total: 0 }

  const totalAvailable = two.available + four.available
  const totalOccupied = two.occupied + four.occupied
  const totalSpaces = (two.total || 0) + (four.total || 0) || 0
  const capacity = totalSpaces > 0 ? Math.round((totalOccupied / totalSpaces) * 100) : 0

  return (
    <main className="min-h-screen bg-gradient-to-br from-background via-background to-background/95">
      {/* Header */}
      <header className="border-b border-border/10 bg-background/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-primary to-secondary rounded-lg">
                <Activity className="w-6 h-6 text-black" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">Smart Parking</h1>
                <p className="text-sm text-muted-foreground">Real-time Detection System</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Button
                onClick={() => refresh()}
                disabled={loading}
                className="bg-primary hover:bg-primary/90 text-black gap-2"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              {error && <div className="text-sm text-destructive">Error: {error}</div>}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Overall Stats */}
        <div className="mb-8">
          <OverallStatsCard
            totalAvailable={totalAvailable}
            totalOccupied={totalOccupied}
            totalSpaces={totalSpaces}
            capacity={capacity}
          />
        </div>

        {/* Two Column Layout: Stats on Left, Feed on Right */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Sidebar - Stats */}
          <div className="lg:col-span-1 space-y-6 flex flex-col">
            {/* Two Wheeler Stats */}
            <ParkingStatsCard
              title="Two Wheelers"
              icon={<Bike className="w-6 h-6 text-black" />}
              available={two.available}
              occupied={two.occupied}
              total={two.total}
              color="bg-primary/20 text-primary"
            />

            {/* Four Wheeler Stats */}
            <ParkingStatsCard
              title="Four Wheelers"
              icon={<Car className="w-6 h-6 text-black" />}
              available={four.available}
              occupied={four.occupied}
              total={four.total}
              color="bg-secondary/20 text-secondary"
            />
          </div>

          {/* Right Side - Camera Feed */}
          <div className="lg:col-span-2">
            <CameraFeed camereName="Lot A - Main Camera" isLive={true} fps={30} />
          </div>
        </div>

        {/* Footer Stats */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 text-center text-sm text-muted-foreground border-t border-border/10 pt-8">
          <div>
            <p className="font-semibold text-foreground">Last Updated</p>
            <p className="text-xs mt-1">{data?.timestamp ?? '—'}</p>
          </div>
          <div>
            <p className="font-semibold text-foreground">Detection Model</p>
            <p className="text-xs mt-1">YOLOv8 Real-time</p>
          </div>
          <div>
            <p className="font-semibold text-foreground">System Status</p>
            <p className="text-xs mt-1 text-secondary">● Operating Normally</p>
          </div>
        </div>
      </div>
    </main>
  )
}
