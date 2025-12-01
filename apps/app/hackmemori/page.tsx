"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/hooks/use-toast"
import { hackMemoriAPI } from "@/lib/api"
import { Play, Square, Plus, Brain, MessageSquare, Database, Clock, CheckCircle, XCircle, Eye, EyeOff, Trash2 } from "lucide-react"
import { DashboardLayout } from "@/components/dashboard/dashboard-layout"

interface HackMemoriSession {
    id: string
    name: string
    createdAt: string
    startedAt?: string
    stoppedAt?: string
    status: string
    config: Record<string, any>
}

interface HackMemoriQuestion {
    id: string
    text: string
    origin: string
    createdAt: string
}

interface HackMemoriResponse {
    id: string
    questionId?: string
    sessionId?: string
    prompt: string
    response: string
    acceptedForTraining: boolean
    createdAt: string
}

export default function HackMemoriPage() {
    const [sessions, setSessions] = useState<HackMemoriSession[]>([])
    const [selectedSession, setSelectedSession] = useState<HackMemoriSession | null>(null)
    const [questions, setQuestions] = useState<HackMemoriQuestion[]>([])
    const [responses, setResponses] = useState<HackMemoriResponse[]>([])
    const [isLoading, setIsLoading] = useState(false)
    const [newSessionName, setNewSessionName] = useState("")
    const [newQuestion, setNewQuestion] = useState("")
    const [visibleResponses, setVisibleResponses] = useState<Set<string>>(new Set())
    const [startDateTime, setStartDateTime] = useState("")
    const [endDateTime, setEndDateTime] = useState("")
    const { toast } = useToast()

    // Load sessions on mount
    useEffect(() => {
        loadSessions()
    }, [])

    // Load session details when selected
    useEffect(() => {
        if (selectedSession) {
            loadSessionData(selectedSession.id)
        }
    }, [selectedSession])

    const loadSessions = async () => {
        try {
            const data = await hackMemoriAPI.getSessions()
            setSessions(data)
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to load Hack-Memori sessions",
                variant: "destructive",
            })
        }
    }

    const loadSessionData = async (sessionId: string) => {
        try {
            const [questionsData, responsesData] = await Promise.all([
                hackMemoriAPI.getQuestions(sessionId),
                hackMemoriAPI.getResponses(sessionId)
            ])
            setQuestions(questionsData)
            setResponses(responsesData)
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to load session data",
                variant: "destructive",
            })
        }
    }

    const handleCreateSession = async () => {
        if (!newSessionName.trim()) return

        setIsLoading(true)
        try {
            const config: any = {
                frequency: 5,
                max_questions: 100
            }

            // Add scheduling if provided
            if (startDateTime) {
                config.start_datetime = startDateTime
            }
            if (endDateTime) {
                config.end_datetime = endDateTime
            }

            const session = await hackMemoriAPI.startSession(newSessionName, config)
            setSessions(prev => [session, ...prev])
            setNewSessionName("")
            setStartDateTime("")
            setEndDateTime("")
            toast({
                title: "Success",
                description: "Hack-Memori session created and scheduled",
            })
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to create session",
                variant: "destructive",
            })
        } finally {
            setIsLoading(false)
        }
    }

    const toggleResponseVisibility = (responseId: string) => {
        const newVisible = new Set(visibleResponses)
        if (newVisible.has(responseId)) {
            newVisible.delete(responseId)
        } else {
            newVisible.add(responseId)
        }
        setVisibleResponses(newVisible)
    }

    const getQuestionForResponse = (response: HackMemoriResponse) => {
        // Find the question that matches this response
        if (response.questionId) {
            return questions.find(q => q.id === response.questionId) || null
        }
        return questions.find(q => q.text === response.prompt) || null
    }

    const handleStopSession = async (sessionId: string) => {
        try {
            await hackMemoriAPI.stopSession(sessionId)
            await loadSessions()
            if (selectedSession?.id === sessionId) {
                await loadSessionData(sessionId)
            }
            toast({
                title: "Success",
                description: "Session stopped",
            })
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to stop session",
                variant: "destructive",
            })
        }
    }

    const handleAddQuestion = async () => {
        if (!selectedSession || !newQuestion.trim()) return

        try {
            await hackMemoriAPI.addQuestion(selectedSession.id, newQuestion)
            setNewQuestion("")
            await loadSessionData(selectedSession.id)
            toast({
                title: "Success",
                description: "Question added to session",
            })
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to add question",
                variant: "destructive",
            })
        }
    }

    const handleDeleteSession = async (sessionId: string, sessionName: string) => {
        try {
            await hackMemoriAPI.deleteSession(sessionId)

            setSessions(prev => prev.filter(s => s.id !== sessionId))

            // If the deleted session was selected, clear selection
            if (selectedSession?.id === sessionId) {
                setSelectedSession(null)
                setQuestions([])
                setResponses([])
            }

            toast({
                title: "Sesión eliminada",
                description: `La sesión "${sessionName}" ha sido eliminada exitosamente`,
            })
        } catch (error) {
            toast({
                title: "Error al eliminar",
                description: `No se pudo eliminar la sesión: ${error instanceof Error ? error.message : 'Error desconocido'}`,
                variant: "destructive",
            })
        }
    }

    const getStatusColor = (status: string) => {
        switch (status) {
            case "RUNNING": return "bg-green-500"
            case "STOPPED": return "bg-red-500"
            default: return "bg-gray-500"
        }
    }

    const getStatusText = (status: string) => {
        switch (status) {
            case "RUNNING": return "Running"
            case "STOPPED": return "Stopped"
            default: return "Unknown"
        }
    }

    return (
        <DashboardLayout>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold">Hack-Memori</h1>
                        <p className="text-muted-foreground">Automated Q&A generation for AI training</p>
                    </div>
                    <div className="flex items-center gap-2">
                        <Brain className="h-8 w-8 text-primary" />
                    </div>
                </div>

                <Tabs defaultValue="sessions" className="space-y-6">
                    <TabsList>
                        <TabsTrigger value="sessions">Sessions</TabsTrigger>
                        <TabsTrigger value="create">Create Session</TabsTrigger>
                        {selectedSession && <TabsTrigger value="manage">Manage Session</TabsTrigger>}
                    </TabsList>

                    <TabsContent value="sessions" className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {sessions.map((session) => (
                                <Card
                                    key={session.id}
                                    className={`cursor-pointer transition-colors ${selectedSession?.id === session.id ? 'ring-2 ring-primary' : ''
                                        }`}
                                    onClick={() => setSelectedSession(session)}
                                >
                                    <CardHeader className="pb-3">
                                        <div className="flex items-center justify-between">
                                            <CardTitle className="text-lg">{session.name}</CardTitle>
                                            <div className={`w-3 h-3 rounded-full ${getStatusColor(session.status)}`} />
                                        </div>
                                        <CardDescription>
                                            Created: {new Date(session.createdAt).toLocaleDateString()}
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between text-sm">
                                                <span>Status:</span>
                                                <Badge variant={session.status === "RUNNING" ? "default" : "secondary"}>
                                                    {getStatusText(session.status)}
                                                </Badge>
                                            </div>
                                            {session.status === "RUNNING" && (
                                                <div className="flex items-center justify-between text-sm">
                                                    <span>Started:</span>
                                                    <span>{new Date(session.startedAt!).toLocaleTimeString()}</span>
                                                </div>
                                            )}
                                            <div className="flex items-center justify-between text-sm">
                                                <span>Frequency:</span>
                                                <span>{session.config?.frequency || 5}s</span>
                                            </div>
                                        </div>
                                        <div className="flex gap-2 mt-3">
                                            {session.status === "RUNNING" && (
                                                <Button
                                                    variant="destructive"
                                                    size="sm"
                                                    className="flex-1"
                                                    onClick={(e) => {
                                                        e.stopPropagation()
                                                        handleStopSession(session.id)
                                                    }}
                                                >
                                                    <Square className="h-4 w-4 mr-2" />
                                                    Stop Session
                                                </Button>
                                            )}
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                className={session.status === "RUNNING" ? "px-3" : "flex-1"}
                                                onClick={(e) => {
                                                    e.stopPropagation()
                                                    handleDeleteSession(session.id, session.name)
                                                }}
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}

                            {sessions.length === 0 && (
                                <Card className="col-span-full">
                                    <CardContent className="flex flex-col items-center justify-center py-12">
                                        <Brain className="h-12 w-12 text-muted-foreground mb-4" />
                                        <h3 className="text-lg font-semibold mb-2">No Sessions Yet</h3>
                                        <p className="text-muted-foreground text-center mb-4">
                                            Create your first Hack-Memori session to start automated Q&A generation
                                        </p>
                                        <Button onClick={() => (document.querySelector('[value="create"]') as HTMLButtonElement)?.click()}>
                                            <Plus className="h-4 w-4 mr-2" />
                                            Create Session
                                        </Button>
                                    </CardContent>
                                </Card>
                            )}
                        </div>
                    </TabsContent>

                    <TabsContent value="create" className="space-y-4">
                        <Card>
                            <CardHeader>
                                <CardTitle>Create New Hack-Memori Session</CardTitle>
                                <CardDescription>
                                    Start an automated Q&A generation session using real LLM
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div>
                                    <label className="text-sm font-medium">Session Name</label>
                                    <Input
                                        placeholder="e.g., Consciousness Research Session"
                                        value={newSessionName}
                                        onChange={(e) => setNewSessionName(e.target.value)}
                                        className="mt-1"
                                    />
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="text-sm font-medium">Start Date & Time (Optional)</label>
                                        <Input
                                            type="datetime-local"
                                            value={startDateTime}
                                            onChange={(e) => setStartDateTime(e.target.value)}
                                            className="mt-1"
                                            placeholder="Leave empty to start immediately"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-sm font-medium">End Date & Time (Optional)</label>
                                        <Input
                                            type="datetime-local"
                                            value={endDateTime}
                                            onChange={(e) => setEndDateTime(e.target.value)}
                                            className="mt-1"
                                            placeholder="Leave empty for manual stop"
                                        />
                                    </div>
                                </div>
                                <Button
                                    onClick={handleCreateSession}
                                    disabled={isLoading || !newSessionName.trim()}
                                    className="w-full"
                                >
                                    <Play className="h-4 w-4 mr-2" />
                                    {isLoading ? "Creating..." : "Start Session"}
                                </Button>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    {selectedSession && (
                        <TabsContent value="manage" className="space-y-4">
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                {/* Session Info */}
                                <Card>
                                    <CardHeader>
                                        <CardTitle className="flex items-center gap-2">
                                            <Brain className="h-5 w-5" />
                                            {selectedSession.name}
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="space-y-3">
                                        <div className="flex items-center justify-between">
                                            <span className="text-sm font-medium">Status:</span>
                                            <Badge variant={selectedSession.status === "RUNNING" ? "default" : "secondary"}>
                                                {getStatusText(selectedSession.status)}
                                            </Badge>
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <span className="text-sm font-medium">Questions:</span>
                                            <span className="text-sm">{questions.length}</span>
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <span className="text-sm font-medium">Responses:</span>
                                            <span className="text-sm">{responses.length}</span>
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <span className="text-sm font-medium">Accepted:</span>
                                            <span className="text-sm">
                                                {responses.filter(r => r.acceptedForTraining).length}
                                            </span>
                                        </div>
                                    </CardContent>
                                </Card>

                                {/* Add Question */}
                                <Card>
                                    <CardHeader>
                                        <CardTitle className="flex items-center gap-2">
                                            <Plus className="h-5 w-5" />
                                            Add Question
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="space-y-3">
                                        <Textarea
                                            placeholder="Enter a question for the LLM..."
                                            value={newQuestion}
                                            onChange={(e) => setNewQuestion(e.target.value)}
                                            rows={3}
                                        />
                                        <Button
                                            onClick={handleAddQuestion}
                                            disabled={!newQuestion.trim()}
                                            className="w-full"
                                        >
                                            Add Question
                                        </Button>
                                    </CardContent>
                                </Card>

                                {/* Stats */}
                                <Card>
                                    <CardHeader>
                                        <CardTitle className="flex items-center gap-2">
                                            <Database className="h-5 w-5" />
                                            Training Data
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="space-y-3">
                                        <div>
                                            <div className="flex justify-between text-sm mb-1">
                                                <span>Acceptance Rate</span>
                                                <span>
                                                    {responses.length > 0
                                                        ? Math.round((responses.filter(r => r.acceptedForTraining).length / responses.length) * 100)
                                                        : 0}%
                                                </span>
                                            </div>
                                            <Progress
                                                value={responses.length > 0
                                                    ? (responses.filter(r => r.acceptedForTraining).length / responses.length) * 100
                                                    : 0}
                                                className="h-2"
                                            />
                                        </div>
                                        <div className="text-sm text-muted-foreground">
                                            {responses.filter(r => r.acceptedForTraining).length} of {responses.length} responses accepted for training
                                        </div>
                                    </CardContent>
                                </Card>
                            </div>

                            {/* Recent Activity */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <Clock className="h-5 w-5" />
                                        Recent Activity
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3 max-h-96 overflow-y-auto">
                                        {responses.slice(0, 10).map((response, index) => {
                                            const question = getQuestionForResponse(response)
                                            const isVisible = visibleResponses.has(response.id)

                                            return (
                                                <div
                                                    key={response.id || `${response.questionId ?? "response"}-${response.createdAt}-${index}`}
                                                    className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg"
                                                >
                                                    <div className="flex-shrink-0 mt-1">
                                                        {response.acceptedForTraining ? (
                                                            <CheckCircle className="h-4 w-4 text-green-500" />
                                                        ) : (
                                                            <XCircle className="h-4 w-4 text-red-500" />
                                                        )}
                                                    </div>
                                                    <div className="flex-1 min-w-0">
                                                        <p className="text-sm font-medium truncate">
                                                            {question ? question.text.substring(0, 100) : response.prompt.substring(0, 100)}
                                                            {question && question.text.length > 100 ? "..." : ""}
                                                        </p>
                                                        <div className="flex items-center justify-between mt-2">
                                                            <p className="text-xs text-muted-foreground">
                                                                {new Date(response.createdAt).toLocaleString()}
                                                            </p>
                                                            <Button
                                                                variant="ghost"
                                                                size="sm"
                                                                onClick={() => toggleResponseVisibility(response.id)}
                                                                className="h-6 px-2 text-xs"
                                                            >
                                                                {isVisible ? (
                                                                    <>
                                                                        <EyeOff className="h-3 w-3 mr-1" />
                                                                        Ocultar
                                                                    </>
                                                                ) : (
                                                                    <>
                                                                        <Eye className="h-3 w-3 mr-1" />
                                                                        Ver Respuesta
                                                                    </>
                                                                )}
                                                            </Button>
                                                        </div>
                                                        {isVisible && (
                                                            <div className="mt-3 p-3 bg-background rounded border text-sm">
                                                                <p className="font-medium mb-2">Respuesta generada:</p>
                                                                <p className="whitespace-pre-wrap">{response.response}</p>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            )
                                        })}

                                        {responses.length === 0 && (
                                            <div className="text-center py-8 text-muted-foreground">
                                                <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
                                                <p>No activity yet</p>
                                                <p className="text-xs">Add questions to start generating responses</p>
                                            </div>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>
                        </TabsContent>
                    )}
                </Tabs>
            </div>
        </DashboardLayout>
    )
}
