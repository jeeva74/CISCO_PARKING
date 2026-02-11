"use client"

import { useEffect, useRef, useState, useCallback } from 'react'

const DEFAULT_POLL_MS = 5000

export interface VehicleCounts {
  available: number
  occupied: number
  total: number
}

export interface ParkingApiResponse {
  twoWheeler: VehicleCounts
  fourWheeler: VehicleCounts
  predictions?: any
  timestamp?: string
}

/**
 * Hook to fetch parking data from backend and keep it in sync.
 * - Polls the backend every `pollMs` milliseconds
 * - Provides manual `refresh()` and exposes `loading`/`error` states
 */
export function useParkingData(pollMs: number = DEFAULT_POLL_MS) {
  const API_BASE = (process.env.NEXT_PUBLIC_API_BASE as string) || 'http://localhost:5000'
  const endpoint = `${API_BASE}/api/parking-details`

  const [data, setData] = useState<ParkingApiResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const fetchData = useCallback(async (signal?: AbortSignal) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(endpoint, { signal })
      if (!res.ok) throw new Error(`API error ${res.status}`)
      const json = (await res.json()) as ParkingApiResponse
      setData(json)
    } catch (err: any) {
      if (err.name === 'AbortError') return
      setError(err.message || String(err))
    } finally {
      setLoading(false)
    }
  }, [endpoint])

  // Manual refresh API for UI buttons
  const refresh = useCallback(() => {
    // cancel any in-flight request before starting a new one
    if (abortRef.current) abortRef.current.abort()
    const ac = new AbortController()
    abortRef.current = ac
    void fetchData(ac.signal)
  }, [fetchData])

  useEffect(() => {
    const ac = new AbortController()
    abortRef.current = ac
    void fetchData(ac.signal)

    const id = setInterval(() => {
      const a = new AbortController()
      abortRef.current = a
      void fetchData(a.signal)
    }, pollMs)

    return () => {
      clearInterval(id)
      if (abortRef.current) abortRef.current.abort()
    }
  }, [fetchData, pollMs])

  return { data, loading, error, refresh }
}
