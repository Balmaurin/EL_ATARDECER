"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useDashboardConsciousness } from "@/hooks/use-api"
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts"
import { Heart } from "lucide-react"

const EMOTION_COLORS: Record<string, string> = {
  neutral: "#6b7280",
  joy: "#22c55e",
  curiosity: "#3b82f6",
  reflection: "#8b5cf6",
  concern: "#ef4444",
}

export function EmotionalDistribution() {
  const { data } = useDashboardConsciousness()

  // Use real data only - no mocks
  const emotionData = Object.keys(data?.analytics?.emotional_distribution || {}).length > 0 && data?.analytics?.emotional_distribution
    ? Object.entries(data.analytics.emotional_distribution).map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value: value as number,
      color: EMOTION_COLORS[name] || "#6b7280",
    }))
    : []

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Heart className="h-4 w-4 text-accent" />
          Emotional Distribution
        </CardTitle>
      </CardHeader>
      <CardContent>
        {emotionData.length === 0 ? (
          <div className="h-[200px] w-full flex items-center justify-center text-muted-foreground">
            <p className="text-sm">No emotional distribution data available</p>
          </div>
        ) : (
          <div className="h-[200px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={emotionData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={70}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {emotionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, "Percentage"]}
                />
                <Legend
                  verticalAlign="bottom"
                  height={36}
                  formatter={(value) => <span className="text-xs text-foreground">{value}</span>}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
