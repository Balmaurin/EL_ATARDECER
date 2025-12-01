"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useDashboardConsciousness } from "@/hooks/use-api"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { TrendingUp } from "lucide-react"

export function PhiEvolutionChart() {
  const { data } = useDashboardConsciousness()

  // Use real data only - no mocks
  let chartData: { date: string; phi: number }[] = []
  if (
    data?.analytics?.phi_evolution?.labels &&
    data.analytics.phi_evolution.labels.length > 0 &&
    data?.analytics?.phi_evolution?.values
  ) {
    const labels = data.analytics.phi_evolution.labels
    const values = data.analytics.phi_evolution.values
    chartData = labels.map((label: string, index: number) => ({
      date: label,
      phi: (values[index] as number) || 0,
    }))
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <TrendingUp className="h-4 w-4 text-primary" />
          Phi Evolution
        </CardTitle>
      </CardHeader>
      <CardContent>
        {chartData.length === 0 ? (
          <div className="h-[200px] w-full flex items-center justify-center text-muted-foreground">
            <p className="text-sm">No phi evolution data available</p>
          </div>
        ) : (
          <div className="h-[200px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="phi"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={{ fill: "hsl(var(--primary))" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
