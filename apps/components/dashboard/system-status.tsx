"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { useDashboardStatus } from "@/hooks/use-api"
import { Server, CheckCircle2, AlertCircle, XCircle } from "lucide-react"

const statusIcons = {
  active: CheckCircle2,
  operational: CheckCircle2,
  healthy: CheckCircle2,
  integrated: CheckCircle2,
  persistent: CheckCircle2,
  evolving: CheckCircle2,
  degraded: AlertCircle,
  error: XCircle,
}

const statusColors = {
  active: "bg-green-500/20 text-green-500 border-green-500/30",
  operational: "bg-green-500/20 text-green-500 border-green-500/30",
  healthy: "bg-green-500/20 text-green-500 border-green-500/30",
  integrated: "bg-blue-500/20 text-blue-500 border-blue-500/30",
  persistent: "bg-blue-500/20 text-blue-500 border-blue-500/30",
  evolving: "bg-purple-500/20 text-purple-500 border-purple-500/30",
  degraded: "bg-yellow-500/20 text-yellow-500 border-yellow-500/30",
  error: "bg-red-500/20 text-red-500 border-red-500/30",
}

export function SystemStatus() {
  const { data, isLoading } = useDashboardStatus()

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Server className="h-4 w-4" />
            System Services
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-8 bg-muted rounded" />
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  const services = data?.services || {}

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Server className="h-4 w-4" />
          System Services
          <Badge variant="outline" className="ml-auto bg-green-500/20 text-green-500 border-green-500/30">
            {data?.system?.health || "Unknown"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {Object.entries(services).map(([name, status]) => {
            const statusStr = String(status).toLowerCase()
            const IconComponent = statusIcons[statusStr as keyof typeof statusIcons] || AlertCircle
            const colorClass = statusColors[statusStr as keyof typeof statusColors] || statusColors.degraded

            return (
              <div key={name} className="flex items-center justify-between py-2 border-b border-border last:border-0">
                <div className="flex items-center gap-2">
                  <IconComponent
                    className="h-4 w-4"
                    style={{
                      color:
                        statusStr === "active" || statusStr === "operational" || statusStr === "healthy"
                          ? "#22c55e"
                          : "#eab308",
                    }}
                  />
                  <span className="text-sm capitalize">{name}</span>
                </div>
                <Badge variant="outline" className={colorClass}>
                  {String(status)}
                </Badge>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
