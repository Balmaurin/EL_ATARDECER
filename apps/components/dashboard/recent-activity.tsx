"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useConsciousnessDreams } from "@/hooks/use-api"
import { Clock, Brain } from "lucide-react"

export function RecentActivity() {
  const { data, isLoading } = useConsciousnessDreams(5)

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Clock className="h-4 w-4" />
            Recent Activity
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="space-y-2">
                <div className="h-4 bg-muted rounded w-3/4" />
                <div className="h-3 bg-muted rounded w-1/2" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  const dreams = data?.dreams || []

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Clock className="h-4 w-4" />
          Cognitive History
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {dreams.length === 0 ? (
            <p className="text-sm text-muted-foreground">No recent activity</p>
          ) : (
            dreams.map((dream: any, index: number) => (
              <div key={dream.id || index} className="flex gap-3 pb-3 border-b border-border last:border-0">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/20">
                  <Brain className="h-4 w-4 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm truncate">{dream.description}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Depth: {dream.insights_gained?.toFixed(2) || "N/A"} | {dream.type}
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  )
}
