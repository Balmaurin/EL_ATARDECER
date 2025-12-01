"use client"

import { useTrainingStatus, useComponentTrainingStatus, useValidationResults } from "@/hooks/use-api"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Activity,
  CheckCircle2,
  XCircle,
  Clock,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Play,
  Pause,
  Loader2,
} from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts"

export function TrainingMonitorTab() {
  const { data: status, isLoading, mutate } = useTrainingStatus()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  const isTraining = status?.isTraining || false
  const progress = status?.progressPercent || 0
  const currentComponent = status?.currentComponent
  const componentsCompleted = status?.componentsCompleted || 0
  const totalComponents = status?.totalComponents || 0
  const trainingId = status?.trainingId || status?.lastTrainingId

  return (
    <div className="space-y-6">
      {/* Status Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Training Status
              </CardTitle>
              <CardDescription>
                Monitoreo en tiempo real del entrenamiento del sistema
              </CardDescription>
            </div>
            <Badge
              variant={isTraining ? "default" : status?.status === "completed" ? "default" : "secondary"}
              className={
                isTraining
                  ? "bg-green-500"
                  : status?.status === "completed"
                  ? "bg-blue-500"
                  : status?.status === "failed"
                  ? "bg-red-500"
                  : "bg-gray-500"
              }
            >
              {isTraining ? (
                <>
                  <Play className="h-3 w-3 mr-1" />
                  Running
                </>
              ) : status?.status === "completed" ? (
                <>
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Completed
                </>
              ) : status?.status === "failed" ? (
                <>
                  <XCircle className="h-3 w-3 mr-1" />
                  Failed
                </>
              ) : (
                <>
                  <Pause className="h-3 w-3 mr-1" />
                  Idle
                </>
              )}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Progress */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Overall Progress</span>
              <span className="text-sm text-muted-foreground">{progress.toFixed(1)}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>

          {/* Components Progress */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-muted-foreground mb-1">Components</div>
              <div className="text-2xl font-bold">
                {componentsCompleted} / {totalComponents}
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">Q&A Available</div>
              <div className="text-2xl font-bold">{status?.qaCount || 0}</div>
              <div className="text-xs text-muted-foreground">
                {status?.qaUnusedCount || 0} unused
              </div>
            </div>
          </div>

          {/* Current Component */}
          {currentComponent && (
            <div className="flex items-center gap-2 p-3 bg-secondary rounded-lg">
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
              <span className="text-sm">
                Training: <strong>{currentComponent}</strong>
              </span>
            </div>
          )}

          {/* Timestamps */}
          {status?.startedAt && (
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                Started: {new Date(status.startedAt).toLocaleString()}
              </div>
              {status?.estimatedCompletion && (
                <div className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  ETA: {new Date(status.estimatedCompletion).toLocaleString()}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Detailed Tabs */}
      {trainingId && (
        <Tabs defaultValue="components" className="space-y-4">
          <TabsList>
            <TabsTrigger value="components">Components</TabsTrigger>
            <TabsTrigger value="validation">Validation</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
          </TabsList>

          <TabsContent value="components">
            <ComponentStatusList trainingId={trainingId} />
          </TabsContent>

          <TabsContent value="validation">
            <ValidationResultsList trainingId={trainingId} />
          </TabsContent>

          <TabsContent value="metrics">
            <TrainingMetricsChart trainingId={trainingId} />
          </TabsContent>
        </Tabs>
      )}
    </div>
  )
}

function ComponentStatusList({ trainingId }: { trainingId: string }) {
  const { data: components, isLoading } = useComponentTrainingStatus(trainingId)

  if (isLoading) {
    return <div className="text-center py-8">Loading components...</div>
  }

  if (!components || components.length === 0) {
    return <div className="text-center py-8 text-muted-foreground">No components trained yet</div>
  }

  return (
    <div className="space-y-2">
      {components.map((comp: any) => (
        <Card key={comp.componentName}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="font-medium">{comp.componentName}</div>
                <div className="text-sm text-muted-foreground mt-1">
                  {comp.status === "completed" ? (
                    <span className="text-green-500">Completed</span>
                  ) : comp.status === "running" ? (
                    <span className="text-blue-500">Running...</span>
                  ) : comp.error ? (
                    <span className="text-red-500">Failed: {comp.error}</span>
                  ) : (
                    <span className="text-gray-500">Pending</span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-sm font-medium">{comp.progress.toFixed(0)}%</div>
                  <Progress value={comp.progress} className="w-24 h-2" />
                </div>
                {comp.status === "completed" ? (
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                ) : comp.error ? (
                  <XCircle className="h-5 w-5 text-red-500" />
                ) : (
                  <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

function ValidationResultsList({ trainingId }: { trainingId: string }) {
  const { data: validations, isLoading } = useValidationResults(trainingId)

  if (isLoading) {
    return <div className="text-center py-8">Loading validations...</div>
  }

  if (!validations || validations.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No validation results available yet
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {validations.map((val: any) => (
        <Card key={val.componentName}>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">{val.componentName}</CardTitle>
              <Badge
                variant={val.validationPassed ? "default" : "destructive"}
                className={val.validationPassed ? "bg-green-500" : "bg-red-500"}
              >
                {val.validationPassed ? (
                  <>
                    <CheckCircle2 className="h-3 w-3 mr-1" />
                    Passed
                  </>
                ) : (
                  <>
                    <XCircle className="h-3 w-3 mr-1" />
                    Failed
                  </>
                )}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Improvement Score */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Improvement Score</span>
                <div className="flex items-center gap-1">
                  {val.improvementScore > 0 ? (
                    <TrendingUp className="h-4 w-4 text-green-500" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-red-500" />
                  )}
                  <span
                    className={`text-sm font-bold ${
                      val.improvementScore > 0 ? "text-green-500" : "text-red-500"
                    }`}
                  >
                    {(val.improvementScore * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <Progress
                value={Math.abs(val.improvementScore) * 100}
                className={`h-2 ${val.improvementScore > 0 ? "bg-green-500" : "bg-red-500"}`}
              />
            </div>

            {/* Metrics Comparison */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-muted-foreground mb-2">Before</div>
                <div className="space-y-1">
                  {Object.entries(val.beforeMetrics || {}).map(([key, value]: [string, any]) => (
                    <div key={key} className="text-sm">
                      <span className="text-muted-foreground">{key}:</span>{" "}
                      <span className="font-medium">{typeof value === "number" ? value.toFixed(2) : value}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-2">After</div>
                <div className="space-y-1">
                  {Object.entries(val.afterMetrics || {}).map(([key, value]: [string, any]) => (
                    <div key={key} className="text-sm">
                      <span className="text-muted-foreground">{key}:</span>{" "}
                      <span className="font-medium">{typeof value === "number" ? value.toFixed(2) : value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Test Results */}
            {val.testResults && (
              <div>
                <div className="text-xs text-muted-foreground mb-2">Test Results</div>
                <div className="space-y-1">
                  {Object.entries(val.testResults).map(([test, passed]: [string, any]) => (
                    <div key={test} className="flex items-center gap-2 text-sm">
                      {passed ? (
                        <CheckCircle2 className="h-3 w-3 text-green-500" />
                      ) : (
                        <XCircle className="h-3 w-3 text-red-500" />
                      )}
                      <span>{test}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

function TrainingMetricsChart({ trainingId }: { trainingId: string }) {
  const { data: components } = useComponentTrainingStatus(trainingId)

  if (!components || components.length === 0) {
    return <div className="text-center py-8 text-muted-foreground">No metrics available</div>
  }

  const chartData = components.map((comp: any) => ({
    name: comp.componentName,
    progress: comp.progress,
    status: comp.status === "completed" ? 100 : comp.status === "running" ? comp.progress : 0,
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle>Training Progress by Component</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="progress" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

