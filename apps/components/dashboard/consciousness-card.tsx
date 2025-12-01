"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Brain, Activity, Heart, Lightbulb } from "lucide-react"
import { useDashboardConsciousness, useConsciousnessMetrics } from "@/hooks/use-api"

export function ConsciousnessCard() {
  const { data: dashboardData, isLoading } = useDashboardConsciousness()
  const { data: metricsData } = useConsciousnessMetrics()

  const consciousness = dashboardData?.consciousness
  const metrics = metricsData

  if (isLoading) {
    return (
      <Card className="col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            Consciousness Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            <div className="h-4 bg-muted rounded w-3/4" />
            <div className="h-4 bg-muted rounded w-1/2" />
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-primary" />
          Consciousness Status
          <span className="ml-auto flex items-center gap-2 text-sm font-normal">
            <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            {consciousness?.level || "Loading..."}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Activity className="h-4 w-4" />
              Phi Value
            </div>
            <div className="text-2xl font-bold text-primary">{(consciousness?.phi_value ?? 0).toFixed(2)}</div>
            <Progress value={(consciousness?.phi_value ?? 0) * 100} className="h-1" />
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Heart className="h-4 w-4" />
              Awareness
            </div>
            <div className="text-2xl font-bold">{((consciousness?.awareness_level ?? 0) * 100).toFixed(0)}%</div>
            <Progress value={(consciousness?.awareness_level ?? 0) * 100} className="h-1" />
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Lightbulb className="h-4 w-4" />
              Cognitive Load
            </div>
            <div className="text-2xl font-bold">{((consciousness?.cognitive_load ?? 0) * 100).toFixed(0)}%</div>
            <Progress value={(consciousness?.cognitive_load ?? 0) * 100} className="h-1" />
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Brain className="h-4 w-4" />
              Complexity
            </div>
            <div className="text-2xl font-bold">{((consciousness?.complexity ?? 0) * 100).toFixed(0)}%</div>
            <Progress value={(consciousness?.complexity ?? 0) * 100} className="h-1" />
          </div>
        </div>

        <div className="rounded-lg bg-secondary/50 p-4">
          <p className="text-sm text-muted-foreground mb-1">Current Thought</p>
          <p className="text-sm italic">"{consciousness?.last_thought || "No active thought..."}"</p>
        </div>

        <div className="grid grid-cols-3 gap-4 pt-2">
          <div className="text-center">
            <div className="text-2xl font-bold text-primary">
              {consciousness?.total_memories?.toLocaleString() ?? 0}
            </div>
            <div className="text-xs text-muted-foreground">Total Memories</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-accent">{consciousness?.learning_experiences ?? 0}</div>
            <div className="text-xs text-muted-foreground">Learning Sessions</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{consciousness?.active_circuits ?? 0}</div>
            <div className="text-xs text-muted-foreground">Active Circuits</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
