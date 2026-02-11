'use client'

import { Card } from '@/components/ui/card'
import { ParkingCircle, TrendingUp } from 'lucide-react'

interface OverallStatsCardProps {
  totalAvailable: number
  totalOccupied: number
  totalSpaces: number
  capacity: number
}

export function OverallStatsCard({
  totalAvailable,
  totalOccupied,
  totalSpaces,
  capacity,
}: OverallStatsCardProps) {
  const occupancyPercent = Math.round((totalOccupied / totalSpaces) * 100)
  const trend = occupancyPercent > 70 ? 'high' : occupancyPercent > 40 ? 'medium' : 'low'

  return (
    <Card className="overflow-hidden border border-border/20 bg-gradient-to-br from-card to-card/60 backdrop-blur-sm col-span-full">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-xl font-bold text-foreground flex items-center gap-2">
              <ParkingCircle className="w-6 h-6 text-primary" />
              Total Parking Capacity
            </h3>
            <p className="text-sm text-muted-foreground mt-1">
              System-wide availability overview
            </p>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-2 justify-end mb-2">
              <span className="text-sm text-muted-foreground">Occupancy Trend:</span>
              <TrendingUp
                className={`w-5 h-5 ${
                  trend === 'high'
                    ? 'text-destructive'
                    : trend === 'medium'
                      ? 'text-yellow-500'
                      : 'text-secondary'
                }`}
              />
            </div>
            <span
              className={`text-sm font-semibold px-3 py-1 rounded-full ${
                trend === 'high'
                  ? 'bg-destructive/10 text-destructive'
                  : trend === 'medium'
                    ? 'bg-yellow-500/10 text-yellow-500'
                    : 'bg-secondary/10 text-secondary'
              }`}
            >
              {trend === 'high' ? 'High' : trend === 'medium' ? 'Medium' : 'Low'} Occupancy
            </span>
          </div>
        </div>

        {/* Main Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Total Spaces */}
          <div className="bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg p-4 border border-primary/20">
            <p className="text-sm text-muted-foreground mb-2">Total Spaces</p>
            <p className="text-3xl font-bold text-foreground">{totalSpaces}</p>
            <p className="text-xs text-muted-foreground mt-2">Parking slots</p>
          </div>

          {/* Available */}
          <div className="bg-gradient-to-br from-secondary/10 to-secondary/5 rounded-lg p-4 border border-secondary/20">
            <p className="text-sm text-muted-foreground mb-2">Available Now</p>
            <p className="text-3xl font-bold text-secondary">{totalAvailable}</p>
            <p className="text-xs text-muted-foreground mt-2">
              {Math.round((totalAvailable / totalSpaces) * 100)}% free
            </p>
          </div>

          {/* Occupied */}
          <div className="bg-gradient-to-br from-destructive/10 to-destructive/5 rounded-lg p-4 border border-destructive/20">
            <p className="text-sm text-muted-foreground mb-2">Currently Occupied</p>
            <p className="text-3xl font-bold text-destructive">{totalOccupied}</p>
            <p className="text-xs text-muted-foreground mt-2">{occupancyPercent}% full</p>
          </div>

          {/* Capacity */}
          <div className="bg-gradient-to-br from-blue-500/10 to-blue-500/5 rounded-lg p-4 border border-blue-500/20">
            <p className="text-sm text-muted-foreground mb-2">System Capacity</p>
            <p className="text-3xl font-bold text-blue-500">{capacity}%</p>
            <p className="text-xs text-muted-foreground mt-2">
              {capacity > 90 ? 'Near full' : capacity > 70 ? 'Moderate' : 'Available'}
            </p>
          </div>
        </div>

        {/* Occupancy Bar */}
        <div className="mt-6 pt-6 border-t border-border/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-foreground">Occupancy Rate</span>
            <span className="text-sm font-bold text-foreground">{occupancyPercent}%</span>
          </div>
          <div className="w-full h-3 bg-secondary/10 rounded-full overflow-hidden border border-border/20">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                occupancyPercent > 80
                  ? 'bg-gradient-to-r from-destructive to-destructive/60'
                  : occupancyPercent > 50
                    ? 'bg-gradient-to-r from-yellow-500 to-yellow-500/60'
                    : 'bg-gradient-to-r from-secondary to-secondary/60'
              }`}
              style={{ width: `${occupancyPercent}%` }}
            />
          </div>
        </div>
      </div>
    </Card>
  )
}
