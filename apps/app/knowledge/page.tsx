"use client"

import { useState } from "react"
import { DashboardSidebar } from "@/components/dashboard/sidebar"
import { DashboardHeader } from "@/components/dashboard/header"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Search,
  Plus,
  FileText,
  Database,
  Upload,
  RefreshCw,
  Trash2,
  ExternalLink,
  Lightbulb,
  HardDrive,
  Cpu,
} from "lucide-react"
import { useKnowledgeStats, useKnowledgeDocuments } from "@/hooks/use-api"
import { knowledgeAPI } from "@/lib/api"

export default function KnowledgePage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [newKnowledge, setNewKnowledge] = useState("")
  const [isAdding, setIsAdding] = useState(false)
  const [addResult, setAddResult] = useState<{ success: boolean; message: string } | null>(null)

  const { data: statsData, mutate: mutateStats } = useKnowledgeStats()
  const { data: documentsData, isLoading: docsLoading } = useKnowledgeDocuments()

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setIsSearching(true)
    setSearchResults([])

    try {
      const result = await knowledgeAPI.search(searchQuery, 10)
      setSearchResults(result.results || [])
    } catch (error) {
      console.error("Search error:", error)
      setSearchResults([])
    } finally {
      setIsSearching(false)
    }
  }

  const handleAddKnowledge = async () => {
    if (!newKnowledge.trim()) return

    setIsAdding(true)
    setAddResult(null)

    try {
      const result = await knowledgeAPI.add(newKnowledge, {
        source: "manual_input",
        category: "general",
        timestamp: new Date().toISOString(),
      })
      setAddResult({ success: true, message: "Knowledge added successfully!" })
      setNewKnowledge("")
      mutateStats()
    } catch (error) {
      setAddResult({ success: false, message: `Error: ${error instanceof Error ? error.message : "Unknown error"}` })
    } finally {
      setIsAdding(false)
    }
  }

  return (
    <div className="flex h-screen bg-background">
      <DashboardSidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <DashboardHeader />
        <main className="flex-1 overflow-auto p-6">
          <div className="mb-6">
            <h1 className="text-2xl font-bold text-foreground">Knowledge Base</h1>
            <p className="text-muted-foreground">Manage RAG documents and search the knowledge base</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Database className="h-4 w-4 text-primary" />
                  Documents
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-primary">{statsData?.documents_count || 0}</div>
                <p className="text-xs text-muted-foreground">Indexed documents</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <HardDrive className="h-4 w-4" />
                  Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xl font-bold">{statsData?.available ? "Online" : "Offline"}</div>
                <Badge
                  variant="outline"
                  className={
                    statsData?.available
                      ? "bg-green-500/20 text-green-500 border-green-500/30"
                      : "bg-red-500/20 text-red-500 border-red-500/30"
                  }
                >
                  {statsData?.available ? "Connected" : "Disconnected"}
                </Badge>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-accent" />
                  Method
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-sm font-medium">TF-IDF + SVD</div>
                <p className="text-xs text-muted-foreground">Cosine Similarity</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Processing
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xl font-bold">{documentsData?.indexed || 0}</div>
                <p className="text-xs text-muted-foreground">Files processed</p>
              </CardContent>
            </Card>
          </div>

          <Tabs defaultValue="search" className="space-y-6">
            <TabsList className="bg-secondary">
              <TabsTrigger value="search">Search</TabsTrigger>
              <TabsTrigger value="add">Add Knowledge</TabsTrigger>
              <TabsTrigger value="documents">Documents</TabsTrigger>
            </TabsList>

            <TabsContent value="search" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Search className="h-5 w-5 text-primary" />
                    Semantic Search
                  </CardTitle>
                  <CardDescription>Search the knowledge base using natural language</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex gap-2">
                    <Input
                      placeholder="Enter your search query..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                      className="flex-1"
                    />
                    <Button onClick={handleSearch} disabled={isSearching}>
                      {isSearching ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                    </Button>
                  </div>

                  {searchResults.length > 0 && (
                    <div className="space-y-3">
                      <h4 className="font-medium text-sm">Found {searchResults.length} results</h4>
                      <ScrollArea className="h-[400px]">
                        <div className="space-y-3 pr-4">
                          {searchResults.map((result, index) => (
                            <Card key={index} className="bg-secondary/50">
                              <CardContent className="p-4">
                                <div className="flex items-start justify-between gap-4">
                                  <div className="flex-1 min-w-0">
                                    <p className="text-sm">{result.document}</p>
                                    {result.metadata && Object.keys(result.metadata).length > 0 && (
                                      <div className="flex flex-wrap gap-1 mt-2">
                                        {Object.entries(result.metadata).map(([key, value]) => (
                                          <Badge key={key} variant="outline" className="text-xs">
                                            {key}: {String(value)}
                                          </Badge>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                  <Badge variant="outline" className="shrink-0">
                                    Score: {(result.score * 100).toFixed(1)}%
                                  </Badge>
                                </div>
                              </CardContent>
                            </Card>
                          ))}
                        </div>
                      </ScrollArea>
                    </div>
                  )}

                  {searchResults.length === 0 && searchQuery && !isSearching && (
                    <div className="text-center py-8">
                      <Lightbulb className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">No results found</p>
                      <p className="text-sm text-muted-foreground mt-1">Try different keywords or add more knowledge</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="add" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Plus className="h-5 w-5 text-green-500" />
                    Add New Knowledge
                  </CardTitle>
                  <CardDescription>Add text content to the knowledge base for RAG retrieval</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    placeholder="Enter knowledge content here... This could be documentation, facts, instructions, or any text you want the AI to be able to retrieve."
                    value={newKnowledge}
                    onChange={(e) => setNewKnowledge(e.target.value)}
                    className="min-h-[200px]"
                  />

                  {addResult && (
                    <div
                      className={`rounded-lg p-4 ${
                        addResult.success ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {addResult.message}
                    </div>
                  )}
                </CardContent>
                <CardFooter>
                  <Button onClick={handleAddKnowledge} disabled={isAdding || !newKnowledge.trim()}>
                    {isAdding ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : <Plus className="h-4 w-4 mr-2" />}
                    Add to Knowledge Base
                  </Button>
                </CardFooter>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Upload className="h-5 w-5 text-blue-500" />
                    Upload Document
                  </CardTitle>
                  <CardDescription>Upload PDF, DOCX, or TXT files for indexing</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="border-2 border-dashed border-border rounded-lg p-8 text-center">
                    <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground mb-2">Drag and drop files here, or click to browse</p>
                    <p className="text-xs text-muted-foreground">Supported formats: PDF, DOCX, TXT, MD</p>
                    <Button variant="outline" className="mt-4 bg-transparent">
                      Select Files
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="documents" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5 text-primary" />
                    Indexed Documents
                  </CardTitle>
                  <CardDescription>Documents currently in the RAG knowledge base</CardDescription>
                </CardHeader>
                <CardContent>
                  {docsLoading ? (
                    <div className="flex items-center justify-center py-8">
                      <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                    </div>
                  ) : (documentsData?.documents || []).length === 0 ? (
                    <div className="text-center py-8">
                      <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">No documents indexed</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        Add knowledge or upload documents to get started
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {(documentsData?.documents || []).map((doc: any, index: number) => (
                        <div
                          key={doc.id || index}
                          className="flex items-center justify-between p-4 rounded-lg bg-secondary/50 border border-border"
                        >
                          <div className="flex items-center gap-4">
                            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/20">
                              <FileText className="h-5 w-5 text-primary" />
                            </div>
                            <div>
                              <p className="font-medium">{doc.filename}</p>
                              <p className="text-sm text-muted-foreground">
                                {doc.chunks} chunks | {doc.size}
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge
                              variant="outline"
                              className={
                                doc.status === "indexed"
                                  ? "bg-green-500/20 text-green-500 border-green-500/30"
                                  : "bg-yellow-500/20 text-yellow-500 border-yellow-500/30"
                              }
                            >
                              {doc.status}
                            </Badge>
                            <Button variant="ghost" size="icon">
                              <ExternalLink className="h-4 w-4" />
                            </Button>
                            <Button variant="ghost" size="icon" className="text-destructive">
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </div>
  )
}
