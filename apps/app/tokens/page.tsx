"use client"

import { useState, useEffect } from "react"
import { DashboardSidebar } from "@/components/dashboard/sidebar"
import { DashboardHeader } from "@/components/dashboard/header"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import {
  Send,
  Wallet,
  TrendingUp,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw,
  Gift,
  Copy,
  ExternalLink,
  Zap,
  Trophy,
  Target,
} from "lucide-react"
import { useBlockchainBalance, useTokenBalance } from "@/hooks/use-api"
import { blockchainAPI } from "@/lib/api"

export default function TokensPage() {
  const [recipient, setRecipient] = useState("")
  const [amount, setAmount] = useState("")
  const [memo, setMemo] = useState("")
  const [isSending, setIsSending] = useState(false)
  const [isRequestingFaucet, setIsRequestingFaucet] = useState(false)
  const [transactionResult, setTransactionResult] = useState<{ success: boolean; message: string } | null>(null)
  const [lastFaucetRequest, setLastFaucetRequest] = useState<string | null>(null)
  const [canUseFaucet, setCanUseFaucet] = useState(true)

  const { data: balanceData, isLoading: balanceLoading, mutate: mutateBalance } = useBlockchainBalance()
  const { data: tokenData, isLoading: tokenLoading, mutate: mutateTokens } = useTokenBalance()

  // Check if user can use faucet (once per day limit)
  useEffect(() => {
    const lastRequest = localStorage.getItem('lastFaucetRequest')
    if (lastRequest) {
      const lastRequestDate = new Date(lastRequest)
      const now = new Date()
      const timeDiff = now.getTime() - lastRequestDate.getTime()
      const daysDiff = timeDiff / (1000 * 3600 * 24)

      if (daysDiff < 1) {
        setCanUseFaucet(false)
        setLastFaucetRequest(lastRequest)
      }
    }
  }, [])

  const handleSend = async () => {
    if (!recipient || !amount) return

    setIsSending(true)
    setTransactionResult(null)

    try {
      const result = await blockchainAPI.send(recipient, Number.parseFloat(amount), memo)
      setTransactionResult({
        success: true,
        message: `Successfully sent ${amount} SHEILY tokens. TX: ${result.transaction_id}`,
      })
      setRecipient("")
      setAmount("")
      setMemo("")
      mutateBalance()
      mutateTokens()
    } catch (error) {
      setTransactionResult({
        success: false,
        message: `Error: ${error instanceof Error ? error.message : "Transaction failed"}`,
      })
    } finally {
      setIsSending(false)
    }
  }

  const handleFaucet = async () => {
    // Check if user can use faucet
    if (!canUseFaucet) {
      setTransactionResult({
        success: false,
        message: "Faucet can only be used once per day. Please try again tomorrow.",
      })
      return
    }

    setIsRequestingFaucet(true)
    setTransactionResult(null)

    try {
      const result = await blockchainAPI.faucet(100)

      // Mark faucet as used
      const now = new Date().toISOString()
      localStorage.setItem('lastFaucetRequest', now)
      setLastFaucetRequest(now)
      setCanUseFaucet(false)

      setTransactionResult({
        success: true,
        message: `Received ${result.amount} SHEILY tokens from faucet! New balance: ${result.new_balance}`,
      })
      mutateBalance()
      mutateTokens()
    } catch (error) {
      setTransactionResult({
        success: false,
        message: `Error: ${error instanceof Error ? error.message : "Faucet request failed"}`,
      })
    } finally {
      setIsRequestingFaucet(false)
    }
  }

  const copyAddress = () => {
    if (balanceData?.address) {
      navigator.clipboard.writeText(balanceData.address)
    }
  }

  const levelProgress = tokenData ? (tokenData.experience / tokenData.next_level_experience) * 100 : 0

  return (
    <div className="flex h-screen bg-background">
      <DashboardSidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <DashboardHeader />
        <main className="flex-1 overflow-auto p-6">
          <div className="mb-6">
            <h1 className="text-2xl font-bold text-foreground">SHEILY Tokens</h1>
            <p className="text-muted-foreground">Manage your blockchain tokens and transactions</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <Card className="bg-gradient-to-br from-primary/20 to-primary/5 border-primary/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Wallet className="h-4 w-4 text-primary" />
                  Balance Total
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-primary">
                  {tokenLoading
                    ? "..."
                    : (tokenData?.combined_balance || 0).toLocaleString()}
                </div>
                <p className="text-xs text-muted-foreground">
                  {tokenData?.total_tokens || 0} confirmados + {tokenData?.provisional_tokens || 0} provisorios
                </p>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-orange/20 to-orange/5 border-orange/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Clock className="h-4 w-4 text-orange-500" />
                  Tokens Provisorios
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-orange-500">
                  {tokenLoading ? "..." : tokenData?.provisional_tokens || 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  Pendientes de training
                </p>
                <div className="mt-2">
                  <Progress
                    value={Math.min(100, ((tokenData?.provisional_tokens || 0) / 100) * 100)}
                    className="h-1"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    {100 - Math.min(100, tokenData?.provisional_tokens || 0)} más para training
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Zap className="h-4 w-4 text-green-500" />
                  Threshold de Training
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{tokenData?.next_training_threshold || 100}</div>
                <p className="text-xs text-muted-foreground">Q&A para activar training</p>
                <div className="mt-2">
                  <Progress
                    value={((tokenData?.provisional_tokens || 0) % 8) * 12.5} // Mostra ~1/8 increments
                    className="h-1"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  Transactions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{balanceData?.total_transactions || 0}</div>
                <p className="text-xs text-muted-foreground">{balanceData?.pending_transactions || 0} pending</p>
              </CardContent>
            </Card>
          </div>

          <Tabs defaultValue="wallet" className="space-y-6">
            <TabsList className="bg-secondary">
              <TabsTrigger value="wallet">Wallet</TabsTrigger>
              <TabsTrigger value="send">Send</TabsTrigger>
              <TabsTrigger value="history">History</TabsTrigger>
              <TabsTrigger value="earn">Earn</TabsTrigger>
            </TabsList>

            <TabsContent value="wallet" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Wallet className="h-5 w-5 text-primary" />
                      Wallet Address
                    </CardTitle>
                    <CardDescription>Your SHEILY blockchain address</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-2 p-3 rounded-lg bg-secondary/50 border border-border">
                      <code className="flex-1 text-sm font-mono truncate">{balanceData?.address || "Loading..."}</code>
                      <Button variant="ghost" size="icon" onClick={copyAddress}>
                        <Copy className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon">
                        <ExternalLink className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                  <CardFooter>
                    <p className="text-xs text-muted-foreground">
                      Last transaction: {balanceData?.last_transaction || "N/A"}
                    </p>
                  </CardFooter>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Gift className="h-5 w-5 text-green-500" />
                      Test Faucet
                    </CardTitle>
                    <CardDescription>Request free tokens for testing</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p className="text-sm text-muted-foreground">
                      Get 100 SHEILY tokens for testing purposes. Limited to one request per day.
                    </p>
                    {!canUseFaucet && lastFaucetRequest && (
                      <div className="p-3 rounded-lg bg-yellow-500/20 text-yellow-400">
                        <p className="text-sm">
                          Already used today. Last request: {new Date(lastFaucetRequest).toLocaleString()}
                        </p>
                        <p className="text-xs text-yellow-300">
                          Next request available: {new Date(new Date(lastFaucetRequest).getTime() + 24 * 60 * 60 * 1000).toLocaleString()}
                        </p>
                      </div>
                    )}
                    <Button
                      onClick={handleFaucet}
                      disabled={isRequestingFaucet || !canUseFaucet}
                      className="w-full"
                    >
                      {isRequestingFaucet ? (
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Gift className="h-4 w-4 mr-2" />
                      )}
                      {!canUseFaucet ? "Already Used Today" : "Request 100 SHEILY"}
                    </Button>
                  </CardContent>
                </Card>
              </div>

              {transactionResult && (
                <div
                  className={`rounded-lg p-4 ${transactionResult.success ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                    }`}
                >
                  {transactionResult.message}
                </div>
              )}

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-accent" />
                    Token Economics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 rounded-lg bg-secondary/50 text-center">
                      <div className="text-2xl font-bold text-primary">1:1</div>
                      <div className="text-xs text-muted-foreground">USD Peg</div>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50 text-center">
                      <div className="text-2xl font-bold">10M</div>
                      <div className="text-xs text-muted-foreground">Total Supply</div>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50 text-center">
                      <div className="text-2xl font-bold text-green-500">2.5%</div>
                      <div className="text-xs text-muted-foreground">Staking APY</div>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50 text-center">
                      <div className="text-2xl font-bold">5</div>
                      <div className="text-xs text-muted-foreground">Per Exercise</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="send" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Send className="h-5 w-5 text-primary" />
                    Send SHEILY Tokens
                  </CardTitle>
                  <CardDescription>Transfer tokens to another wallet address</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="recipient">Recipient Address</Label>
                    <Input
                      id="recipient"
                      placeholder="Enter wallet address or user ID"
                      value={recipient}
                      onChange={(e) => setRecipient(e.target.value)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="amount">Amount</Label>
                    <div className="relative">
                      <Input
                        id="amount"
                        type="number"
                        placeholder="0.00"
                        value={amount}
                        onChange={(e) => setAmount(e.target.value)}
                        className="pr-20"
                      />
                      <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
                        SHEILY
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Available:{" "}
                      {balanceData?.balance_sheily?.toLocaleString() || tokenData?.balance?.toLocaleString() || 0}{" "}
                      SHEILY
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="memo">Memo (optional)</Label>
                    <Input
                      id="memo"
                      placeholder="Add a note to this transaction"
                      value={memo}
                      onChange={(e) => setMemo(e.target.value)}
                    />
                  </div>

                  {transactionResult && (
                    <div
                      className={`rounded-lg p-4 ${transactionResult.success ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                        }`}
                    >
                      {transactionResult.message}
                    </div>
                  )}
                </CardContent>
                <CardFooter>
                  <Button onClick={handleSend} disabled={isSending || !recipient || !amount} className="w-full">
                    {isSending ? (
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4 mr-2" />
                    )}
                    Send Tokens
                  </Button>
                </CardFooter>
              </Card>
            </TabsContent>

            <TabsContent value="history" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="h-5 w-5" />
                    Transaction History
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {[
                      {
                        type: "receive",
                        amount: 100,
                        description: "Faucet request",
                        time: "2 hours ago",
                        status: "confirmed",
                      },
                      {
                        type: "receive",
                        amount: 25,
                        description: "Exercise reward",
                        time: "1 day ago",
                        status: "confirmed",
                      },
                      {
                        type: "receive",
                        amount: 100,
                        description: "Welcome bonus",
                        time: "3 days ago",
                        status: "confirmed",
                      },
                    ].map((tx, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-4 rounded-lg bg-secondary/50 border border-border"
                      >
                        <div className="flex items-center gap-4">
                          <div
                            className={`flex h-10 w-10 items-center justify-center rounded-full ${tx.type === "receive" ? "bg-green-500/20" : "bg-red-500/20"
                              }`}
                          >
                            {tx.type === "receive" ? (
                              <ArrowDownRight className="h-5 w-5 text-green-500" />
                            ) : (
                              <ArrowUpRight className="h-5 w-5 text-red-500" />
                            )}
                          </div>
                          <div>
                            <p className="font-medium">{tx.description}</p>
                            <p className="text-sm text-muted-foreground">{tx.time}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className={`font-bold ${tx.type === "receive" ? "text-green-500" : "text-red-500"}`}>
                            {tx.type === "receive" ? "+" : "-"}
                            {tx.amount} SHEILY
                          </p>
                          <Badge
                            variant="outline"
                            className="bg-green-500/20 text-green-500 border-green-500/30 text-xs"
                          >
                            {tx.status}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="earn" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <Zap className="h-5 w-5 text-yellow-500" />
                      Complete Exercises
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground mb-4">
                      Earn 5 tokens per correct answer in training exercises
                    </p>
                    <Badge variant="outline" className="bg-yellow-500/20 text-yellow-500 border-yellow-500/30">
                      +5 SHEILY per answer
                    </Badge>
                  </CardContent>
                  <CardFooter>
                    <Button variant="outline" className="w-full bg-transparent" asChild>
                      <a href="/exercises">Start Exercises</a>
                    </Button>
                  </CardFooter>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <TrendingUp className="h-5 w-5 text-green-500" />
                      Stake Tokens
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground mb-4">Earn 2.5% APY by staking your SHEILY tokens</p>
                    <Badge variant="outline" className="bg-green-500/20 text-green-500 border-green-500/30">
                      2.5% APY
                    </Badge>
                  </CardContent>
                  <CardFooter>
                    <Button variant="outline" className="w-full bg-transparent" disabled>
                      Coming Soon
                    </Button>
                  </CardFooter>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <Gift className="h-5 w-5 text-purple-500" />
                      Referral Program
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground mb-4">Invite friends and earn 50 tokens per referral</p>
                    <Badge variant="outline" className="bg-purple-500/20 text-purple-500 border-purple-500/30">
                      +50 SHEILY per referral
                    </Badge>
                  </CardContent>
                  <CardFooter>
                    <Button variant="outline" className="w-full bg-transparent" disabled>
                      Coming Soon
                    </Button>
                  </CardFooter>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </div>
  )
}
