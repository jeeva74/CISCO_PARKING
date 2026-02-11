'use client'

import React from "react"

import { Card } from '@/components/ui/card'
import { Bike, Car } from 'lucide-react'

interface ParkingStatsCardProps {
  title: string
  icon: React.ReactNode
  available: number
  occupied: number
  total: number
  color: string
}

export function ParkingStatsCard({
  title,
  icon,
  available,
  occupied,
  total,
  color,
}: ParkingStatsCardProps) {
  const availablePercent = Math.round((available / total) * 100)
  const occupiedPercent = 100 - availablePercent

  return (
    <Card className="overflow-hidden border border-border/20 bg-gradient-to-br from-card to-card/60 backdrop-blur-sm">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-foreground">{title}</h3>
          <div className={`p-3 rounded-lg ${color}`}>{icon}</div>
        </div>

        {/* Circular Progress */}
        <div className="flex items-center justify-center mb-6">
          <div className="relative w-32 h-32">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
              {/* Background circle */}
              <circle
                cx="60"
                cy="60"
                r="54"
                fill="none"
                stroke="hsl(220, 13%, 20%)"
                strokeWidth="8"
              />
              {/* Available circle */}
              <circle
                cx="60"
                cy="60"
                r="54"
                fill="none"
                stroke="hsl(200, 100%, 50%)"
                strokeWidth="8"
                strokeDasharray={`${(availablePercent / 100) * 339.29} 339.29`}
                strokeLinecap="round"
                className="transition-all duration-500"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-3xl font-bold text-primary">{availablePercent}%</span>
              <span className="text-xs text-muted-foreground">Available</span>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-primary/10 rounded-lg p-4 border border-primary/20">
            <div className="text-sm text-muted-foreground mb-1">Available</div>
            <div className="text-2xl font-bold text-primary">{available}</div>
            <div className="text-xs text-muted-foreground mt-2">out of {total}</div>
          </div>
          <div className="bg-destructive/10 rounded-lg p-4 border border-destructive/20">
            <div className="text-sm text-muted-foreground mb-1">Occupied</div>
            <div className="text-2xl font-bold text-destructive">{occupied}</div>
            <div className="text-xs text-muted-foreground mt-2">{occupiedPercent}%</div>
          </div>
        </div>
      </div>
    </Card>
  )
}
