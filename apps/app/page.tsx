"use client"

import { DashboardSidebar } from "@/components/dashboard/sidebar"
import { DashboardHeader } from "@/components/dashboard/header"
import { ConsciousnessCard } from "@/components/dashboard/consciousness-card"
import { MetricsCard } from "@/components/dashboard/metrics-card"
import { PhiEvolutionChart } from "@/components/dashboard/phi-chart"
import { EmotionalDistribution } from "@/components/dashboard/emotional-distribution"
import { SystemStatus } from "@/components/dashboard/system-status"
import { RecentActivity } from "@/components/dashboard/recent-activity"
import { useDashboardStatus, useUserTokens, useDashboardRealtime } from "@/hooks/use-api"
import { hackMemoriAPI } from "@/lib/api"
import { MessageSquare, Database, Cpu, Zap, Brain, Activity } from "lucide-react"
import { useState, useEffect } from "react"

export default function DashboardPage() {
  const { data: statusData } = useDashboardStatus()
  const { data: tokenData } = useUserTokens()
  const { data: realtimeData } = useDashboardRealtime()

  const [hackMemoriStats, setHackMemoriStats] = useState({
    activeSessions: 0,
    totalQuestions: 0,
    totalResponses: 0,
    acceptanceRate: 0
  })

  const metrics = statusData?.metrics
  const performance = realtimeData?.performance

  // Load Hack-Memori stats
  useEffect(() => {
    const loadHackMemoriStats = async () => {
      try {
        const sessions = await hackMemoriAPI.getSessions()
        const activeSessions = sessions.filter(s => s.status === 'RUNNING').length

        let totalQuestions = 0
        let totalResponses = 0
        let acceptedResponses = 0

        for (const session of sessions) {
          const questions = await hackMemoriAPI.getQuestions(session.id)
          const responses = await hackMemoriAPI.getResponses(session.id)

          totalQuestions += questions.length
          totalResponses += responses.length
          acceptedResponses += responses.filter(r => r.acceptedForTraining).length
        }

        const acceptanceRate = totalResponses > 0 ? (acceptedResponses / totalResponses) * 100 : 0

        setHackMemoriStats({
          activeSessions,
          totalQuestions,
          totalResponses,
          acceptanceRate
        })
      } catch (error) {
        console.error('Failed to load Hack-Memori stats:', error)
      }
    }

    loadHackMemoriStats()

    // Refresh stats every 30 seconds
    const interval = setInterval(loadHackMemoriStats, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex h-screen bg-background">
      <DashboardSidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <DashboardHeader />
        <main className="flex-1 overflow-auto p-6">
          <div className="mb-6">
            <h1 className="text-2xl font-bold text-foreground">Dashboard Overview</h1>
            <p className="text-muted-foreground">Monitor your AI consciousness system in real-time</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
            <MetricsCard
              title="Active Conversations"
              value={metrics?.total_conversations ?? 0}
              icon={MessageSquare}
              subtitle="Total interactions"
            />
            <MetricsCard
              title="Datasets Generated"
              value={metrics?.datasets_generated ?? 0}
              icon={Database}
              subtitle="From training exercises"
            />
            <MetricsCard
              title="Hack-Memori Sessions"
              value={hackMemoriStats.activeSessions}
              icon={Brain}
              subtitle={`${hackMemoriStats.totalResponses} responses`}
            />
            <MetricsCard
              title="Response Time"
              value={`${performance?.response_time?.toFixed(1) ?? 0}s`}
              icon={Cpu}
              subtitle="Average latency"
            />
            <MetricsCard
              title="Token Balance"
              value={tokenData?.balance?.toLocaleString() ?? 0}
              icon={Zap}
              subtitle={`Level ${tokenData?.level ?? 1}`}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            <ConsciousnessCard />
            <SystemStatus />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <PhiEvolutionChart />
            <EmotionalDistribution />
            <RecentActivity />
          </div>
        </main>
      </div>
    </div>
  )
}
