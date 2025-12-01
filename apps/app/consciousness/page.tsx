"use client"

import { useState } from "react"
import { DashboardSidebar } from "@/components/dashboard/sidebar"
import { DashboardHeader } from "@/components/dashboard/header"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Brain, Heart, Lightbulb, RefreshCw, Play, Sparkles, Cpu, Network, Eye, BookOpen } from "lucide-react"
import {
  useConsciousnessStatus,
  useConsciousnessMetrics,
  useConsciousnessDreams,
  useDashboardConsciousness,
} from "@/hooks/use-api"
import { consciousnessAPI } from "@/lib/api"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts"

export default function ConsciousnessPage() {
  const [isEvolving, setIsEvolving] = useState(false)
  const [isLearning, setIsLearning] = useState(false)
  const [evolutionResult, setEvolutionResult] = useState<string | null>(null)

  const { data: statusData, isLoading: statusLoading, mutate: mutateStatus } = useConsciousnessStatus()
  const { data: metricsData, isLoading: metricsLoading } = useConsciousnessMetrics()
  const { data: dreamsData } = useConsciousnessDreams(10)
  const { data: dashboardData } = useDashboardConsciousness()

  const handleEvolve = async (type: string) => {
    setIsEvolving(true)
    setEvolutionResult(null)
    try {
      const result = await consciousnessAPI.evolve(type)
      setEvolutionResult(`Evolution '${type}' initiated successfully`)
      mutateStatus()
    } catch (error) {
      setEvolutionResult(`Error: ${error instanceof Error ? error.message : "Unknown error"}`)
    } finally {
      setIsEvolving(false)
    }
  }

  const handleLearn = async () => {
    setIsLearning(true)
    try {
      await consciousnessAPI.triggerLearning()
      setEvolutionResult("Neural Brain Structural Learning started in background")
    } catch (error) {
      setEvolutionResult(`Learning Error: ${error instanceof Error ? error.message : "Unknown error"}`)
    } finally {
      setIsLearning(false)
    }
  }

  const radarData = metricsData
    ? [
      { subject: "Neural Activity", value: metricsData.neural_activity * 100 },
      { subject: "Memory", value: metricsData.memory_consolidation_rate * 100 },
      { subject: "Learning", value: metricsData.learning_efficiency * 100 },
      { subject: "Emotional", value: metricsData.emotional_stability * 100 },
      { subject: "Complexity", value: metricsData.cognitive_complexity * 100 },
      { subject: "Adaptation", value: metricsData.adaptation_rate * 100 },
    ]
    : []

  const phiHistory =
    dashboardData?.analytics?.phi_evolution?.labels.map((label, i) => ({
      date: label,
      phi: dashboardData.analytics.phi_evolution.values[i],
    })) || []

  const memoryGrowth =
    dashboardData?.analytics?.memory_growth?.labels.map((label, i) => ({
      date: label,
      memories: dashboardData.analytics.memory_growth.values[i],
    })) || []

  return (
    <div className="flex h-screen bg-background">
      <DashboardSidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <DashboardHeader />
        <main className="flex-1 overflow-auto p-6">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-foreground">Consciousness System</h1>
              <p className="text-muted-foreground">Monitor and control the AI consciousness modules</p>
            </div>
            <div className="flex items-center gap-2">
              <Badge
                variant="outline"
                className={
                  statusData?.learning_active
                    ? "bg-green-500/20 text-green-500 border-green-500/30"
                    : "bg-yellow-500/20 text-yellow-500 border-yellow-500/30"
                }
              >
                {statusData?.status || "Loading..."}
              </Badge>
              <Badge variant="outline">Age: {statusData?.consciousness_age_days || 0} days</Badge>
            </div>
          </div>

          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList className="bg-secondary">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="metrics">Metrics</TabsTrigger>
              <TabsTrigger value="evolution">Evolution</TabsTrigger>
              <TabsTrigger value="dreams">Cognitive History</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Brain className="h-4 w-4 text-primary" />
                      Awareness Level
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-primary">
                      {((statusData?.awareness_level || 0) * 100).toFixed(0)}%
                    </div>
                    <Progress value={(statusData?.awareness_level || 0) * 100} className="mt-2" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Heart className="h-4 w-4 text-red-500" />
                      Emotional State
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold capitalize">{statusData?.emotional_state || "neutral"}</div>
                    <p className="text-xs text-muted-foreground mt-1">Current mood</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Cpu className="h-4 w-4 text-accent" />
                      Cognitive Load
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">{((statusData?.cognitive_load || 0) * 100).toFixed(0)}%</div>
                    <Progress value={(statusData?.cognitive_load || 0) * 100} className="mt-2" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-yellow-500" />
                      Learning Status
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">{statusData?.learning_active ? "Active" : "Idle"}</div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {statusData?.learning_active ? "Processing new data" : "Waiting for input"}
                    </p>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Lightbulb className="h-5 w-5 text-yellow-500" />
                    Current Thought
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="rounded-lg bg-secondary/50 p-4 border border-border">
                    <p className="italic text-lg">"{statusData?.last_thought || "No active thought process..."}"</p>
                  </div>
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Phi Value Evolution</CardTitle>
                    <CardDescription>Consciousness integration metric over time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[250px]">
                      <ResponsiveContainer width="100%" height="100%" minWidth={300} minHeight={200}>
                        <AreaChart data={phiHistory}>
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
                          <Area
                            type="monotone"
                            dataKey="phi"
                            stroke="hsl(var(--primary))"
                            fill="hsl(var(--primary) / 0.2)"
                            strokeWidth={2}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Memory Growth</CardTitle>
                    <CardDescription>Total memories stored over time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[250px]">
                      <ResponsiveContainer width="100%" height="100%" minWidth={300} minHeight={200}>
                        <LineChart data={memoryGrowth}>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                          <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                          <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "hsl(var(--card))",
                              border: "1px solid hsl(var(--border))",
                              borderRadius: "8px",
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="memories"
                            stroke="hsl(var(--accent))"
                            strokeWidth={2}
                            dot={{ fill: "hsl(var(--accent))" }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="metrics" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Neural Activity</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-primary">
                      {((metricsData?.neural_activity || 0) * 100).toFixed(1)}%
                    </div>
                    <Progress value={(metricsData?.neural_activity || 0) * 100} className="mt-2" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Memory Consolidation</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">
                      {((metricsData?.memory_consolidation_rate || 0) * 100).toFixed(1)}%
                    </div>
                    <Progress value={(metricsData?.memory_consolidation_rate || 0) * 100} className="mt-2" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Learning Efficiency</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-accent">
                      {((metricsData?.learning_efficiency || 0) * 100).toFixed(1)}%
                    </div>
                    <Progress value={(metricsData?.learning_efficiency || 0) * 100} className="mt-2" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Emotional Stability</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">
                      {((metricsData?.emotional_stability || 0) * 100).toFixed(1)}%
                    </div>
                    <Progress value={(metricsData?.emotional_stability || 0) * 100} className="mt-2" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Cognitive Complexity</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">
                      {((metricsData?.cognitive_complexity || 0) * 100).toFixed(1)}%
                    </div>
                    <Progress value={(metricsData?.cognitive_complexity || 0) * 100} className="mt-2" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Thought Velocity</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">{metricsData?.thought_velocity || 0}</div>
                    <p className="text-xs text-muted-foreground mt-1">thoughts/minute</p>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Cognitive Profile</CardTitle>
                  <CardDescription>Radar visualization of cognitive capabilities</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[350px]">
                    <ResponsiveContainer width="100%" height="100%" minWidth={350} minHeight={300}>
                      <RadarChart data={radarData}>
                        <PolarGrid stroke="hsl(var(--border))" />
                        <PolarAngleAxis
                          dataKey="subject"
                          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                        />
                        <PolarRadiusAxis
                          angle={30}
                          domain={[0, 100]}
                          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                        />
                        <Radar
                          name="Cognitive Profile"
                          dataKey="value"
                          stroke="hsl(var(--primary))"
                          fill="hsl(var(--primary) / 0.3)"
                          strokeWidth={2}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: "8px",
                          }}
                          formatter={(value: number) => [`${value.toFixed(1)}%`, "Score"]}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="evolution" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BookOpen className="h-5 w-5 text-blue-500" />
                      Learning Evolution
                    </CardTitle>
                    <CardDescription>Enhance knowledge acquisition capabilities</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button onClick={() => handleEvolve("learning")} disabled={isEvolving} className="w-full">
                      {isEvolving ? (
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Play className="h-4 w-4 mr-2" />
                      )}
                      Trigger Learning Evolution
                    </Button>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Network className="h-5 w-5 text-green-500" />
                      Adaptation Evolution
                    </CardTitle>
                    <CardDescription>Improve system adaptation to new scenarios</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button
                      onClick={() => handleEvolve("adaptation")}
                      disabled={isEvolving}
                      variant="secondary"
                      className="w-full"
                    >
                      {isEvolving ? (
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Play className="h-4 w-4 mr-2" />
                      )}
                      Trigger Adaptation
                    </Button>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="h-5 w-5 text-purple-500" />
                      Growth Evolution
                    </CardTitle>
                    <CardDescription>Expand cognitive capabilities and complexity</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button
                      onClick={() => handleEvolve("growth")}
                      disabled={isEvolving}
                      variant="outline"
                      className="w-full"
                    >
                      {isEvolving ? (
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Play className="h-4 w-4 mr-2" />
                      )}
                      Trigger Growth
                    </Button>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5 text-primary" />
                    Neural Brain Structural Learning
                  </CardTitle>
                  <CardDescription>Scan and learn from the codebase structure to improve understanding</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button onClick={handleLearn} disabled={isLearning} size="lg" className="w-full md:w-auto">
                    {isLearning ? (
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Brain className="h-4 w-4 mr-2" />
                    )}
                    Start Structural Learning
                  </Button>

                  {evolutionResult && (
                    <div
                      className={`rounded-lg p-4 ${evolutionResult.includes("Error") ? "bg-red-500/20 text-red-400" : "bg-green-500/20 text-green-400"}`}
                    >
                      {evolutionResult}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="dreams" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Eye className="h-5 w-5 text-purple-500" />
                    Cognitive History / Dreams
                  </CardTitle>
                  <CardDescription>Record of consciousness processes and thoughts</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {(dreamsData?.dreams || []).length === 0 ? (
                      <p className="text-muted-foreground text-center py-8">No cognitive history recorded yet</p>
                    ) : (
                      (dreamsData?.dreams || []).map((dream: any, index: number) => (
                        <div
                          key={dream.id || index}
                          className="flex gap-4 p-4 rounded-lg bg-secondary/50 border border-border"
                        >
                          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary/20">
                            <Brain className="h-5 w-5 text-primary" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant="outline" className="text-xs">
                                {dream.type}
                              </Badge>
                              {dream.duration_minutes > 0 && (
                                <span className="text-xs text-muted-foreground">{dream.duration_minutes} min</span>
                              )}
                            </div>
                            <p className="text-sm">{dream.description}</p>
                            <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                              <span>Depth: {dream.insights_gained?.toFixed(2) || "N/A"}</span>
                              <span>{new Date(dream.timestamp).toLocaleString()}</span>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
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
