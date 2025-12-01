"use client"

import { useState } from "react"
import { DashboardSidebar } from "@/components/dashboard/sidebar"
import { DashboardHeader } from "@/components/dashboard/header"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import {
  Dumbbell,
  CheckCircle2,
  XCircle,
  Trophy,
  Database,
  Clock,
  Target,
  Zap,
  RefreshCw,
  ChevronRight,
  BarChart3,
} from "lucide-react"
import { TrainingMonitorTab } from "@/components/training/training-monitor"
import { useExerciseDatasets } from "@/hooks/use-api"
import { exercisesAPI } from "@/lib/api"

interface Question {
  id: number
  question: string
  type: "yesno" | "truefalse" | "multiple"
  options?: string[]
  correctAnswer: string | boolean
}

const sampleQuestions: Record<string, Question[]> = {
  yesno: [
    { id: 1, question: "Is machine learning a subset of artificial intelligence?", type: "yesno", correctAnswer: true },
    { id: 2, question: "Can neural networks learn without data?", type: "yesno", correctAnswer: false },
    { id: 3, question: "Is deep learning based on neural networks?", type: "yesno", correctAnswer: true },
    { id: 4, question: "Does supervised learning require labeled data?", type: "yesno", correctAnswer: true },
    {
      id: 5,
      question: "Is reinforcement learning the same as supervised learning?",
      type: "yesno",
      correctAnswer: false,
    },
  ],
  truefalse: [
    {
      id: 1,
      question: "Consciousness in AI refers to self-awareness and subjective experience.",
      type: "truefalse",
      correctAnswer: true,
    },
    {
      id: 2,
      question: "The Phi value measures the emotional state of an AI.",
      type: "truefalse",
      correctAnswer: false,
    },
    {
      id: 3,
      question: "RAG (Retrieval Augmented Generation) combines search with language models.",
      type: "truefalse",
      correctAnswer: true,
    },
    { id: 4, question: "Transformers were invented before RNNs.", type: "truefalse", correctAnswer: false },
    { id: 5, question: "GPT stands for Generative Pre-trained Transformer.", type: "truefalse", correctAnswer: true },
  ],
  multiple: [
    {
      id: 1,
      question: "What is the main purpose of attention mechanisms in transformers?",
      type: "multiple",
      options: ["Speed up training", "Focus on relevant parts of input", "Reduce memory usage", "Increase batch size"],
      correctAnswer: "Focus on relevant parts of input",
    },
    {
      id: 2,
      question: "Which emotion is NOT typically modeled in emotional AI systems?",
      type: "multiple",
      options: ["Joy", "Sadness", "Confusion", "Quantum entanglement"],
      correctAnswer: "Quantum entanglement",
    },
    {
      id: 3,
      question: "What does SHEILY stand for in this project?",
      type: "multiple",
      options: ["A type of neural network", "The AI assistant name", "A programming language", "A database system"],
      correctAnswer: "The AI assistant name",
    },
  ],
}

export default function ExercisesPage() {
  const [activeExercise, setActiveExercise] = useState<string | null>(null)
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [answers, setAnswers] = useState<any[]>([])
  const [showResults, setShowResults] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const { data: datasetsData, mutate: mutateDatasets } = useExerciseDatasets()

  const questions = activeExercise ? sampleQuestions[activeExercise] || [] : []
  const currentQ = questions[currentQuestion]

  const handleAnswer = (answer: any) => {
    const isCorrect = answer === currentQ.correctAnswer
    const newAnswer = {
      questionId: currentQ.id,
      question: currentQ.question,
      userAnswer: answer,
      correctAnswer: currentQ.correctAnswer,
      isCorrect,
    }

    setAnswers((prev) => [...prev, newAnswer])

    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion((prev) => prev + 1)
    } else {
      setShowResults(true)
    }
  }

  const handleSubmitExercise = async () => {
    setIsSubmitting(true)

    const correct = answers.filter((a) => a.isCorrect).length
    const incorrect = answers.filter((a) => !a.isCorrect).length
    const tokensEarned = correct * 5

    try {
      await exercisesAPI.submit({
        exercise_type: activeExercise!,
        answers,
        correct,
        incorrect,
        total_tokens: tokensEarned,
      })
      mutateDatasets()
    } catch (error) {
      console.error("Error submitting exercise:", error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const resetExercise = () => {
    setActiveExercise(null)
    setCurrentQuestion(0)
    setAnswers([])
    setShowResults(false)
  }

  const correct = answers.filter((a) => a.isCorrect).length
  const incorrect = answers.filter((a) => !a.isCorrect).length
  const accuracy = answers.length > 0 ? (correct / answers.length) * 100 : 0

  return (
    <div className="flex h-screen bg-background">
      <DashboardSidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <DashboardHeader />
        <main className="flex-1 overflow-auto p-6">
          <div className="mb-6">
            <h1 className="text-2xl font-bold text-foreground">Training Exercises</h1>
            <p className="text-muted-foreground">Train the AI and earn SHEILY tokens</p>
          </div>

          <Tabs defaultValue="exercises" className="space-y-6">
            <TabsList className="bg-secondary">
              <TabsTrigger value="exercises">Exercises</TabsTrigger>
              <TabsTrigger value="datasets">Generated Datasets</TabsTrigger>
              <TabsTrigger value="training">Training Monitor</TabsTrigger>
              <TabsTrigger value="stats">Statistics</TabsTrigger>
            </TabsList>

            <TabsContent value="exercises" className="space-y-6">
              {!activeExercise ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <Card
                    className="hover:border-primary/50 transition-colors cursor-pointer"
                    onClick={() => setActiveExercise("yesno")}
                  >
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <CheckCircle2 className="h-5 w-5 text-green-500" />
                        Yes/No Questions
                      </CardTitle>
                      <CardDescription>Simple binary questions to test basic AI knowledge</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">5 questions</span>
                        <Badge variant="outline" className="bg-green-500/20 text-green-500 border-green-500/30">
                          +25 tokens
                        </Badge>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button className="w-full">
                        Start Exercise
                        <ChevronRight className="h-4 w-4 ml-2" />
                      </Button>
                    </CardFooter>
                  </Card>

                  <Card
                    className="hover:border-primary/50 transition-colors cursor-pointer"
                    onClick={() => setActiveExercise("truefalse")}
                  >
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Target className="h-5 w-5 text-blue-500" />
                        True/False
                      </CardTitle>
                      <CardDescription>Verify statements about AI concepts</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">5 questions</span>
                        <Badge variant="outline" className="bg-blue-500/20 text-blue-500 border-blue-500/30">
                          +25 tokens
                        </Badge>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button variant="secondary" className="w-full">
                        Start Exercise
                        <ChevronRight className="h-4 w-4 ml-2" />
                      </Button>
                    </CardFooter>
                  </Card>

                  <Card
                    className="hover:border-primary/50 transition-colors cursor-pointer"
                    onClick={() => setActiveExercise("multiple")}
                  >
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Dumbbell className="h-5 w-5 text-purple-500" />
                        Multiple Choice
                      </CardTitle>
                      <CardDescription>Advanced questions with multiple options</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">3 questions</span>
                        <Badge variant="outline" className="bg-purple-500/20 text-purple-500 border-purple-500/30">
                          +30 tokens
                        </Badge>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button variant="outline" className="w-full bg-transparent">
                        Start Exercise
                        <ChevronRight className="h-4 w-4 ml-2" />
                      </Button>
                    </CardFooter>
                  </Card>
                </div>
              ) : showResults ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Trophy className="h-5 w-5 text-yellow-500" />
                      Exercise Complete!
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div className="p-4 rounded-lg bg-green-500/20">
                        <div className="text-3xl font-bold text-green-500">{correct}</div>
                        <div className="text-sm text-muted-foreground">Correct</div>
                      </div>
                      <div className="p-4 rounded-lg bg-red-500/20">
                        <div className="text-3xl font-bold text-red-500">{incorrect}</div>
                        <div className="text-sm text-muted-foreground">Incorrect</div>
                      </div>
                      <div className="p-4 rounded-lg bg-primary/20">
                        <div className="text-3xl font-bold text-primary">{accuracy.toFixed(0)}%</div>
                        <div className="text-sm text-muted-foreground">Accuracy</div>
                      </div>
                    </div>

                    <div className="flex items-center justify-center gap-2 py-4">
                      <Zap className="h-6 w-6 text-yellow-500" />
                      <span className="text-2xl font-bold">+{correct * 5} SHEILY tokens earned!</span>
                    </div>

                    <div className="space-y-2">
                      <h4 className="font-medium">Answer Review</h4>
                      {answers.map((answer, index) => (
                        <div
                          key={index}
                          className={`p-3 rounded-lg flex items-start gap-3 ${
                            answer.isCorrect ? "bg-green-500/10" : "bg-red-500/10"
                          }`}
                        >
                          {answer.isCorrect ? (
                            <CheckCircle2 className="h-5 w-5 text-green-500 shrink-0 mt-0.5" />
                          ) : (
                            <XCircle className="h-5 w-5 text-red-500 shrink-0 mt-0.5" />
                          )}
                          <div>
                            <p className="text-sm">{answer.question}</p>
                            {!answer.isCorrect && (
                              <p className="text-xs text-muted-foreground mt-1">
                                Correct answer: {String(answer.correctAnswer)}
                              </p>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                  <CardFooter className="flex gap-2">
                    <Button onClick={resetExercise} variant="outline" className="flex-1 bg-transparent">
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Try Another
                    </Button>
                    <Button onClick={handleSubmitExercise} disabled={isSubmitting} className="flex-1">
                      {isSubmitting ? (
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Database className="h-4 w-4 mr-2" />
                      )}
                      Save as Dataset
                    </Button>
                  </CardFooter>
                </Card>
              ) : (
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="capitalize">{activeExercise} Exercise</CardTitle>
                      <Badge variant="outline">
                        Question {currentQuestion + 1} of {questions.length}
                      </Badge>
                    </div>
                    <Progress value={((currentQuestion + 1) / questions.length) * 100} className="mt-2" />
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="py-6">
                      <h3 className="text-xl font-medium text-center">{currentQ?.question}</h3>
                    </div>

                    <div className="grid gap-3">
                      {currentQ?.type === "yesno" && (
                        <>
                          <Button
                            size="lg"
                            variant="outline"
                            className="h-16 text-lg bg-transparent"
                            onClick={() => handleAnswer(true)}
                          >
                            <CheckCircle2 className="h-5 w-5 mr-2 text-green-500" />
                            Yes
                          </Button>
                          <Button
                            size="lg"
                            variant="outline"
                            className="h-16 text-lg bg-transparent"
                            onClick={() => handleAnswer(false)}
                          >
                            <XCircle className="h-5 w-5 mr-2 text-red-500" />
                            No
                          </Button>
                        </>
                      )}

                      {currentQ?.type === "truefalse" && (
                        <>
                          <Button
                            size="lg"
                            variant="outline"
                            className="h-16 text-lg bg-transparent"
                            onClick={() => handleAnswer(true)}
                          >
                            <CheckCircle2 className="h-5 w-5 mr-2 text-green-500" />
                            True
                          </Button>
                          <Button
                            size="lg"
                            variant="outline"
                            className="h-16 text-lg bg-transparent"
                            onClick={() => handleAnswer(false)}
                          >
                            <XCircle className="h-5 w-5 mr-2 text-red-500" />
                            False
                          </Button>
                        </>
                      )}

                      {currentQ?.type === "multiple" &&
                        currentQ.options?.map((option, index) => (
                          <Button
                            key={index}
                            size="lg"
                            variant="outline"
                            className="h-auto py-4 text-left justify-start bg-transparent"
                            onClick={() => handleAnswer(option)}
                          >
                            <span className="mr-3 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-xs">
                              {String.fromCharCode(65 + index)}
                            </span>
                            {option}
                          </Button>
                        ))}
                    </div>
                  </CardContent>
                  <CardFooter>
                    <Button variant="ghost" onClick={resetExercise}>
                      Cancel Exercise
                    </Button>
                  </CardFooter>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="datasets" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Database className="h-5 w-5 text-primary" />
                    Generated Datasets
                  </CardTitle>
                  <CardDescription>Datasets created from your training exercises</CardDescription>
                </CardHeader>
                <CardContent>
                  {(datasetsData?.datasets || []).length === 0 ? (
                    <div className="text-center py-8">
                      <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">No datasets generated yet</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        Complete exercises to generate training datasets
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {(datasetsData?.datasets || []).map((dataset: any) => (
                        <div
                          key={dataset.id}
                          className="flex items-center justify-between p-4 rounded-lg bg-secondary/50 border border-border"
                        >
                          <div className="flex items-center gap-4">
                            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/20">
                              <Database className="h-5 w-5 text-primary" />
                            </div>
                            <div>
                              <p className="font-medium capitalize">{dataset.exercise_type} Dataset</p>
                              <p className="text-sm text-muted-foreground">
                                {dataset.num_questions} questions | {dataset.accuracy?.toFixed(0)}% accuracy
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-4">
                            <Badge variant="outline" className="bg-yellow-500/20 text-yellow-500 border-yellow-500/30">
                              +{dataset.tokens_earned} tokens
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {new Date(dataset.timestamp).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="training" className="space-y-6">
              <TrainingMonitorTab />
            </TabsContent>

            <TabsContent value="stats" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Target className="h-4 w-4" />
                      Total Exercises
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">{datasetsData?.total || 0}</div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <BarChart3 className="h-4 w-4" />
                      Avg. Accuracy
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-primary">
                      {datasetsData?.datasets?.length > 0
                        ? (
                            datasetsData.datasets.reduce((acc: number, d: any) => acc + (d.accuracy || 0), 0) /
                            datasetsData.datasets.length
                          ).toFixed(0)
                        : 0}
                      %
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      Tokens Earned
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-yellow-500">
                      {datasetsData?.datasets?.reduce((acc: number, d: any) => acc + (d.tokens_earned || 0), 0) || 0}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Clock className="h-4 w-4" />
                      Questions Answered
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold">
                      {datasetsData?.datasets?.reduce((acc: number, d: any) => acc + (d.num_questions || 0), 0) || 0}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </div>
  )
}
