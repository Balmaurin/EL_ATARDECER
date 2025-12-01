"use client"

import { DashboardSidebar } from "@/components/dashboard/sidebar"
import { DashboardHeader } from "@/components/dashboard/header"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import {
  Cpu,
  HardDrive,
  MemoryStick,
  Network,
  Server,
  Clock,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Gauge,
  Globe,
} from "lucide-react"
import { useSystemStats, useSystemStatus } from "@/hooks/use-api"
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"

export default function SystemPage() {
  const { data: statsData, isLoading: statsLoading, mutate: mutateStats } = useSystemStats()
  const { data: statusData, isLoading: statusLoading, mutate: mutateStatus } = useSystemStatus()

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400)
    const hours = Math.floor((seconds % 86400) / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${days}d ${hours}h ${minutes}m`
  }

  // Mock performance history for charts
  const performanceHistory = [
    { time: "00:00", cpu: 12, memory: 45 },
    { time: "04:00", cpu: 15, memory: 48 },
    { time: "08:00", cpu: 25, memory: 52 },
    { time: "12:00", cpu: 35, memory: 58 },
    { time: "16:00", cpu: 28, memory: 55 },
    { time: "20:00", cpu: 18, memory: 50 },
    { time: "Now", cpu: statsData?.cpu_usage_percent || 15, memory: statsData?.memory_usage_percent || 48 },
  ]

  const servicesList = statusData?.services ? Object.entries(statusData.services) : []

  return (
    <div className="flex h-screen bg-background">
      <DashboardSidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <DashboardHeader />
        <main className="flex-1 overflow-auto p-6">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-foreground">System Monitoring</h1>
              <p className="text-muted-foreground">Real-time system health and performance metrics</p>
            </div>
            <div className="flex items-center gap-2">
              <Badge
                variant="outline"
                className={
                  statusData?.status === "healthy"
                    ? "bg-green-500/20 text-green-500 border-green-500/30"
                    : "bg-yellow-500/20 text-yellow-500 border-yellow-500/30"
                }
              >
                {statusData?.status || "Loading..."}
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  mutateStats()
                  mutateStatus()
                }}
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-blue-500" />
                  CPU Usage
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-blue-500">{statsData?.cpu_usage_percent?.toFixed(1) || 0}%</div>
                <Progress value={statsData?.cpu_usage_percent || 0} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <MemoryStick className="h-4 w-4 text-purple-500" />
                  Memory Usage
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-purple-500">
                  {statsData?.memory_usage_percent?.toFixed(1) || 0}%
                </div>
                <Progress value={statsData?.memory_usage_percent || 0} className="mt-2" />
                <p className="text-xs text-muted-foreground mt-1">
                  {statsData?.memory_used_gb?.toFixed(1) || 0} / {statsData?.memory_total_gb?.toFixed(1) || 0} GB
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <HardDrive className="h-4 w-4 text-green-500" />
                  Disk Usage
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-green-500">
                  {statsData?.disk_usage_percent?.toFixed(1) || 0}%
                </div>
                <Progress value={statsData?.disk_usage_percent || 0} className="mt-2" />
                <p className="text-xs text-muted-foreground mt-1">
                  {statsData?.disk_used_gb?.toFixed(1) || 0} / {statsData?.disk_total_gb?.toFixed(1) || 0} GB
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Clock className="h-4 w-4 text-orange-500" />
                  Uptime
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-orange-500">{formatUptime(statsData?.uptime_seconds || 0)}</div>
                <p className="text-xs text-muted-foreground mt-1">Version: {statusData?.version || "2.0.0"}</p>
              </CardContent>
            </Card>
          </div>

          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList className="bg-secondary">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="services">Services</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
              <TabsTrigger value="network">Network</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">System Performance</CardTitle>
                    <CardDescription>CPU and Memory usage over time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[250px] w-full">
                      <ResponsiveContainer width="100%" height={200} aspect={undefined}>
                        <AreaChart data={performanceHistory}>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                          <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                          <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[0, 100]} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "hsl(var(--card))",
                              border: "1px solid hsl(var(--border))",
                              borderRadius: "8px",
                            }}
                          />
                          <Area
                            type="monotone"
                            dataKey="cpu"
                            stroke="#3b82f6"
                            fill="#3b82f6"
                            fillOpacity={0.2}
                            name="CPU %"
                          />
                          <Area
                            type="monotone"
                            dataKey="memory"
                            stroke="#8b5cf6"
                            fill="#8b5cf6"
                            fillOpacity={0.2}
                            name="Memory %"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Quick Stats</CardTitle>
                    <CardDescription>Real-time system metrics</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
                        <div className="flex items-center gap-3">
                          <Network className="h-5 w-5 text-blue-500" />
                          <span className="text-sm">Active Connections</span>
                        </div>
                        <span className="font-bold">{statsData?.active_connections || 0}</span>
                      </div>

                      <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
                        <div className="flex items-center gap-3">
                          <Globe className="h-5 w-5 text-green-500" />
                          <span className="text-sm">Total Requests</span>
                        </div>
                        <span className="font-bold">{statsData?.total_requests?.toLocaleString() || 0}</span>
                      </div>

                      <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
                        <div className="flex items-center gap-3">
                          <Gauge className="h-5 w-5 text-orange-500" />
                          <span className="text-sm">Avg Response Time</span>
                        </div>
                        <span className="font-bold">{statsData?.average_response_time_ms?.toFixed(0) || 0}ms</span>
                      </div>

                      <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
                        <div className="flex items-center gap-3">
                          <AlertTriangle className="h-5 w-5 text-red-500" />
                          <span className="text-sm">Error Rate</span>
                        </div>
                        <span className="font-bold">{statsData?.error_rate_percent?.toFixed(2) || 0}%</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="services" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Server className="h-5 w-5" />
                    System Services
                  </CardTitle>
                  <CardDescription>Status of all system services and components</CardDescription>
                </CardHeader>
                <CardContent>
                  {statusLoading ? (
                    <div className="flex items-center justify-center py-8">
                      <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {servicesList.map(([name, details]: [string, any]) => {
                        const status = typeof details === "string" ? details : details.status
                        const isHealthy = [
                          "active",
                          "operational",
                          "healthy",
                          "integrated",
                          "persistent",
                          "evolving",
                        ].includes(status?.toLowerCase())

                        return (
                          <div
                            key={name}
                            className="flex items-center justify-between p-4 rounded-lg bg-secondary/50 border border-border"
                          >
                            <div className="flex items-center gap-4">
                              <div
                                className={`flex h-10 w-10 items-center justify-center rounded-full ${isHealthy ? "bg-green-500/20" : "bg-yellow-500/20"
                                  }`}
                              >
                                {isHealthy ? (
                                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                                ) : (
                                  <AlertTriangle className="h-5 w-5 text-yellow-500" />
                                )}
                              </div>
                              <div>
                                <p className="font-medium capitalize">{name.replace(/_/g, " ")}</p>
                                {typeof details === "object" && details.response_time_ms && (
                                  <p className="text-sm text-muted-foreground">
                                    Response: {details.response_time_ms}ms
                                  </p>
                                )}
                              </div>
                            </div>
                            <Badge
                              variant="outline"
                              className={
                                isHealthy
                                  ? "bg-green-500/20 text-green-500 border-green-500/30"
                                  : "bg-yellow-500/20 text-yellow-500 border-yellow-500/30"
                              }
                            >
                              {status}
                            </Badge>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="performance" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Response Time</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-primary">
                      {statsData?.average_response_time_ms?.toFixed(0) || 0}ms
                    </div>
                    <p className="text-xs text-muted-foreground">Average latency</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Throughput</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">
                      {Math.round(
                        (statsData?.total_requests || 0) / Math.max(1, (statsData?.uptime_seconds || 1) / 3600),
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">Requests/hour</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-green-500">
                      {(100 - (statsData?.error_rate_percent || 0)).toFixed(2)}%
                    </div>
                    <p className="text-xs text-muted-foreground">Successful requests</p>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="network" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Network className="h-5 w-5 text-blue-500" />
                    Network Statistics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 rounded-lg bg-secondary/50 text-center">
                      <div className="text-2xl font-bold">{statsData?.active_connections || 0}</div>
                      <div className="text-xs text-muted-foreground">Active Connections</div>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50 text-center">
                      <div className="text-2xl font-bold">{statsData?.total_requests?.toLocaleString() || 0}</div>
                      <div className="text-xs text-muted-foreground">Total Requests</div>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50 text-center">
                      <div className="text-2xl font-bold">{statsData?.average_response_time_ms?.toFixed(0) || 0}ms</div>
                      <div className="text-xs text-muted-foreground">Avg Response</div>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50 text-center">
                      <div className="text-2xl font-bold">{(statsData?.error_rate_percent || 0).toFixed(2)}%</div>
                      <div className="text-xs text-muted-foreground">Error Rate</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </div>
  )
}
