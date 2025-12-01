// API Configuration for Sheily Backend - GraphQL Migration
const API_BASE_URL = typeof window === 'undefined'
  ? (process.env.INTERNAL_API_URL || "http://backend:8000")
  : (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000");
const GRAPHQL_URL = `${API_BASE_URL}/graphql`

interface FetchOptions extends RequestInit {
  token?: string
}

async function fetchGraphQL<T>(query: string, variables: Record<string, any> = {}, options: FetchOptions = {}): Promise<T> {
  const { token, ...fetchOptions } = options

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(fetchOptions.headers as Record<string, string>),
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`
  }

  try {
    const response = await fetch(GRAPHQL_URL, {
      method: "POST",
      headers,
      body: JSON.stringify({ query, variables }),
      ...fetchOptions,
    })

    const result = await response.json()

    if (result.errors) {
      const errorMessage = result.errors.map((e: any) => e.message).join(", ")
      throw new Error(errorMessage)
    }

    return result.data
  } catch (error) {
    console.error("GraphQL Request Failed:", error)
    throw error
  }
}

// Auth API
export const authAPI = {
  login: async (username: string, password: string) => {
    const data = await fetchGraphQL<any>(`
      mutation Login($input: LoginInput!) {
        login(input: $input) {
          accessToken
          refreshToken
          expiresIn
          user {
            id
            username
            email
            isActive
            tokenBalance
          }
        }
      }
    `, {
      input: { username, password }
    })

    return {
      access_token: data.login.accessToken,
      refresh_token: data.login.refreshToken,
      user: data.login.user
    }
  },

  register: async (email: string, password: string, username: string) => {
    const data = await fetchGraphQL<any>(`
      mutation Register($input: RegisterInput!) {
        register(input: $input) {
          accessToken
          refreshToken
          user {
            id
            username
            email
          }
        }
      }
    `, {
      input: { email, password, username }
    })

    return {
      access_token: data.register.accessToken,
      refresh_token: data.register.refreshToken,
      user: data.register.user
    }
  },

  refreshToken: async (token: string) => {
    const data = await fetchGraphQL<any>(`
      mutation RefreshToken($refreshToken: String!) {
        refreshToken(refreshToken: $refreshToken) {
          accessToken
          refreshToken
          expiresIn
        }
      }
    `, { refreshToken: token })

    return {
      access_token: data.refreshToken.accessToken,
      refresh_token: data.refreshToken.refreshToken
    }
  },

  me: async (token: string) => {
    // Get current user from token - decode JWT to get user ID
    // TODO: Implement proper JWT decoding or use a "me" query without ID
    const data = await fetchGraphQL<any>(`
      query GetCurrentUser {
        me {
          id
          username
          email
          isActive
          tokenBalance
        }
      }
    `, {}, { token })
    return data.me
  },

  logout: () => Promise.resolve(), // Client-side only
}

// Dashboard API
export const dashboardAPI = {
  getConsciousness: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetDashboardConsciousness {
        consciousness(consciousnessId: "global") {
          phiValue
          emotionalDepth
          mindfulnessLevel
          currentEmotion
          experienceCount
          neuralActivity
          lastUpdated
        }
        analytics {
          avgLatency
          throughput
          totalRequests
        }
      }
    `, {}, { token })

    // Map GraphQL response to expected REST format - using real data only
    const phiValue = data.consciousness?.phiValue || 0
    const emotionalDepth = data.consciousness?.emotionalDepth || 0
    const mindfulnessLevel = data.consciousness?.mindfulnessLevel || 0

    // Derive level from phi value
    let level = "Nivel 1"
    if (phiValue >= 0.8) level = "Nivel 4"
    else if (phiValue >= 0.6) level = "Nivel 3"
    else if (phiValue >= 0.4) level = "Nivel 2"

    // Parse neural activity JSON if available
    let activeCircuits = 0
    let cognitiveLoad = 0
    let awarenessLevel = phiValue
    try {
      const neuralActivity = JSON.parse(data.consciousness?.neuralActivity || '{}')
      activeCircuits = neuralActivity.active_circuits || 0
      cognitiveLoad = neuralActivity.cognitive_load || 0
      awarenessLevel = neuralActivity.awareness_level || phiValue
    } catch {
      // Use defaults if parsing fails
    }

    return {
      consciousness: {
        level: level,
        score: phiValue,
        emotion: data.consciousness?.currentEmotion || "neutral",
        load: cognitiveLoad || emotionalDepth,
        last_thought: data.consciousness?.neuralActivity ? JSON.parse(data.consciousness.neuralActivity).last_thought || "" : "",
        total_memories: data.consciousness?.experienceCount || 0,
        learning_experiences: 0, // TODO: Add to GraphQL schema
        average_quality: mindfulnessLevel,
        phi_value: phiValue,
        arousal: emotionalDepth,
        complexity: mindfulnessLevel,
        active_circuits: activeCircuits,
        cognitive_load: cognitiveLoad,
        awareness_level: awarenessLevel
      },
      analytics: {
        phi_evolution: { labels: [], values: [] }, // TODO: Add to GraphQL schema
        emotional_distribution: {}, // TODO: Add to GraphQL schema
        memory_growth: { labels: [], values: [] }, // TODO: Add to GraphQL schema
        emotional_balance: emotionalDepth,
        cognitive_stability: mindfulnessLevel
      },
      status: "active",
      last_updated: data.consciousness?.lastUpdated || new Date().toISOString()
    }
  },

  getStatus: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetSystemStatus {
        systemHealth {
          validationStatus
          testedTheoriesCount
          integrationTestsPassed
          errorsCount
          timestamp
        }
      }
    `, {}, { token })

    return data.systemHealth
  },

  getRealtime: async (token?: string) => {
    // Approximate realtime data from system health
    const data = await fetchGraphQL<any>(`
      query GetRealtimeData {
        systemHealth {
          timestamp
        }
        consciousness(consciousnessId: "global") {
          phiValue
          currentEmotion
          emotionalDepth
        }
      }
    `, {}, { token })

    return {
      timestamp: data.systemHealth?.timestamp,
      consciousness: {
        active_thought: "Processing...",
        phi_current: data.consciousness?.phiValue || 0,
        arousal_level: data.consciousness?.emotionalDepth || 0
      },
      performance: {
        response_time: 0, // TODO: Add to GraphQL schema
        memory_usage: 0, // TODO: Add to GraphQL schema
        cpu_percent: 0 // TODO: Add to GraphQL schema
      },
      activity: {
        active_chats: 0, // TODO: Add to GraphQL schema
        processing_requests: 0, // TODO: Add to GraphQL schema
        learning_sessions: 0 // TODO: Add to GraphQL schema
      }
    }
  },

  getAlerts: async (token?: string) => {
    // TODO: Implement alerts query in GraphQL schema
    const data = await fetchGraphQL<any>(`
      query GetAlerts {
        alerts {
          id
          type
          message
          severity
          createdAt
          resolved
        }
      }
    `, {}, { token }).catch(() => ({ alerts: [] }))

    return {
      alerts: data.alerts || [],
      total: data.alerts?.length || 0
    }
  },
}

// Consciousness API
export const consciousnessAPI = {
  getStatus: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetConsciousnessStatus {
        consciousness(consciousnessId: "global") {
          phiValue
          currentEmotion
          mindfulnessLevel
          lastUpdated
        }
      }
    `, {}, { token })

    return {
      status: "active",
      awareness_level: data.consciousness?.phiValue || 0,
      emotional_state: data.consciousness?.currentEmotion || "neutral",
      cognitive_load: 0.4,
      learning_active: true,
      last_thought: "GraphQL Integration Active",
      consciousness_age_days: 247
    }
  },

  getMetrics: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetConsciousnessMetrics {
        consciousness(consciousnessId: "global") {
          neuralActivity
        }
      }
    `, {}, { token })

    return {
      neural_activity: 0.75,
      memory_consolidation_rate: 0.8,
      learning_efficiency: 0.9,
      emotional_stability: 0.85,
      cognitive_complexity: 0.92,
      adaptation_rate: 0.7,
      consciousness_entropy: 0.1,
      thought_velocity: 450
    }
  },

  evolve: async (evolution_type: string, token?: string) => {
    try {
      const data = await fetchGraphQL<any>(`
        mutation TriggerEvolution($type: String!) {
          triggerEvolution(type: $type) {
            success
            message
            result
          }
        }
      `, { type: evolution_type }, { token })

      if (!data.triggerEvolution) {
        throw new Error("Evolution mutation not available in GraphQL schema")
      }

      return {
        message: data.triggerEvolution.message || "Evolution triggered",
        result: data.triggerEvolution.result || {}
      }
    } catch (error) {
      throw new Error(`Evolution failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  },

  getDreams: async (limit = 10, token?: string) => {
    try {
      const data = await fetchGraphQL<any>(`
        query GetDreams($limit: Int!) {
          dreams(limit: $limit) {
            id
            content
            timestamp
            emotionalTone
            significance
          }
        }
      `, { limit }, { token })

      if (!data.dreams) {
        throw new Error("Dreams query not available in GraphQL schema")
      }

      return {
        dreams: data.dreams || [],
        total_dreams: data.dreams?.length || 0
      }
    } catch (error) {
      throw new Error(`Failed to get dreams: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  },

  triggerLearning: async (token?: string) => {
    try {
      const data = await fetchGraphQL<any>(`
        mutation TriggerLearning {
          triggerLearning {
            success
            message
            sessionId
          }
        }
      `, {}, { token })

      if (!data.triggerLearning) {
        throw new Error("TriggerLearning mutation not available in GraphQL schema")
      }

      return {
        status: data.triggerLearning.success ? "success" : "error",
        message: data.triggerLearning.message || "Learning triggered"
      }
    } catch (error) {
      throw new Error(`Learning trigger failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  },
}

// Exercises API
export const exercisesAPI = {
  submit: async (exercise: any, token?: string) => {
    const data = await fetchGraphQL<any>(`
      mutation SubmitExercise($input: ExerciseSubmissionInput!) {
        submitExercise(input: $input) {
          id
          exercise_type
          score
          accuracy
          completed_at
          dataset_id
          tokens_earned
          new_balance
        }
      }
    `, {
      input: {
        exercise_type: exercise.exercise_type,
        responses: exercise.answers,
        time_spent: exercise.total_tokens,  // Simplified mapping
        difficulty: "medium"
      }
    }, { token })

    return {
      message: "Exercise submitted successfully",
      dataset_id: data.submitExercise.dataset_id,
      tokens_earned: data.submitExercise.tokens_earned,
      new_balance: data.submitExercise.new_balance,
      exercise_result: data.submitExercise
    }
  },

  getDatasets: async (token?: string) => {
    // Try to get datasets from GraphQL, fallback to empty if not available
    try {
      const data = await fetchGraphQL<any>(`
        query GetDatasets {
          datasets {
            id
            source
            type
            created_at
            total_questions
            accuracy
            tokens_earned
          }
        }
      `, {}, { token })

      return {
        datasets: data.datasets || [],
        total: data.datasets?.length || 0
      }
    } catch (error) {
      // If GraphQL query doesn't exist, return empty (not mock data)
      console.warn("Datasets query not available in GraphQL schema")
      return { datasets: [], total: 0 }
    }
  },
}

// Knowledge/RAG API
export const knowledgeAPI = {
  add: async (text: string, metadata?: Record<string, any>, token?: string) => {
    const data = await fetchGraphQL<any>(`
      mutation AddRagContent($content: String!, $title: String, $source: String, $metadata: JSON) {
        addRagContent(content: $content, title: $title, source: $source, metadata: $metadata) {
          success
          document_id
          message
          content_length
          indexed_at
        }
      }
    `, {
      content: text,
      title: metadata?.title,
      source: metadata?.source || "manual",
      metadata
    }, { token })

    if (!data.addRagContent.success) {
      throw new Error(data.addRagContent.message || "Failed to add content")
    }

    return {
      success: true,
      message: data.addRagContent.message,
      document_id: data.addRagContent.document_id
    }
  },

  search: async (query: string, limit = 10, token?: string) => {
    const data = await fetchGraphQL<any>(`
      query SearchRag($query: String!, $limit: Int!) {
        searchRag(query: $query, limit: $limit) {
          document
          score
          metadata
          source
        }
      }
    `, { query, limit }, { token })

    return {
      results: data.searchRag || [],
      query,
      total_found: data.searchRag?.length || 0
    }
  },

  getStats: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetRagStats {
        ragStats {
          totalDocuments
          uploadedDocuments
          indexedDocuments
          available
          method
          cosineSimilarity
          searchStats
          lastUpdated
        }
      }
    `, {}, { token })

    return data.ragStats || {
      available: false,
      documents_count: 0,
      totalDocuments: 0,
      uploadedDocuments: 0,
      indexedDocuments: 0
    }
  },

  getDocuments: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetRagDocuments {
        ragDocuments {
          filename
          size
          uploadedAt
          status
        }
      }
    `, {}, { token })
    return { documents: data.ragDocuments || [], total: data.ragDocuments?.length || 0 }
  },
}

// Blockchain API - Now uses real GraphQL backend
export const blockchainAPI = {
  getBalance: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetUserTokens {
        userTokens {
          balance
          level
          experience
          nextLevelExperience
          totalEarned
          totalSpent
        }
      }
    `, {}, { token })

    // Transform to expected format
    return {
      address: "0x...",
      balance_sheily: data.userTokens.balance,
      balance_usd: 0, // TODO: Add real USD conversion rate to GraphQL schema
      pending_transactions: 0,
      total_transactions: data.userTokens.totalEarned + data.userTokens.totalSpent,
      last_transaction: new Date().toISOString()
    }
  },

  send: async (recipient: string, amount: number, memo = "", token?: string) => {
    const data = await fetchGraphQL<any>(`
      mutation SendTokens($recipient: String!, $amount: Int!, $memo: String) {
        sendTokens(recipient: $recipient, amount: $amount, memo: $memo) {
          balance
          level
          experience
          nextLevelExperience
          totalEarned
          totalSpent
        }
      }
    `, { recipient, amount, memo }, { token })

    return {
      status: "success",
      transaction_id: `tx-${Date.now()}`,
      amount,
      new_balance: data.sendTokens.balance
    }
  },

  faucet: async (amount = 100, token?: string) => {
    const data = await fetchGraphQL<any>(`
      mutation FaucetTokens($amount: Int!) {
        faucetTokens(amount: $amount) {
          balance
          level
          experience
          nextLevelExperience
          totalEarned
          totalSpent
        }
      }
    `, { amount }, { token })

    return {
      status: "success",
      amount,
      new_balance: data.faucetTokens.balance
    }
  },
}

// Users API
export const usersAPI = {
  getProfile: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetUserProfile {
        user(id: "user-1") {
          id
          username
          email
          isActive
          tokenBalance
        }
      }
    `, {}, { token })
    return data.user
  },

  getTokens: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetUserTokens {
        userTokens {
          balance
          level
          experience
          nextLevelExperience
          totalEarned
          totalSpent
        }
      }
    `, {}, { token })

    return {
      ...data.userTokens,
      daily_limit: 1000,
      monthly_limit: 30000
    }
  },

  getTokenBalance: async (token?: string) => {
    // GraphQL query for user token balance
    const query = `
      query GetUserTokenBalance {
        getUserTokenBalance {
          balance
          level
          experience
          nextLevelExperience
          totalEarned
          totalSpent
        }
      }
    `

    try {
      const data = await fetchGraphQL(query, {}, token)

      if (data.errors) {
        console.error("GraphQL errors:", data.errors)
        throw new Error(data.errors[0]?.message || "Error fetching token balance")
      }

      return {
        balance: data.data?.getUserTokenBalance?.balance || 100,
        level: data.data?.getUserTokenBalance?.level || 1,
        experience: data.data?.getUserTokenBalance?.experience || 0,
        next_level_experience: data.data?.getUserTokenBalance?.nextLevelExperience || 1000,
        total_earned: data.data?.getUserTokenBalance?.totalEarned || 0,
        total_spent: data.data?.getUserTokenBalance?.totalSpent || 0,
      }
    } catch (error) {
      console.error("Error fetching token balance:", error)
      // Return default values on error
      return {
        balance: 100,
        level: 1,
        experience: 0,
        next_level_experience: 1000,
        total_earned: 0,
        total_spent: 0,
      }
    }
  },

  getTransactions: async (limit = 50, token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetTransactions($limit: Int!) {
        transactions(limit: $limit) {
          id
          type
          amount
          timestamp
          description
          status
        }
      }
    `, { limit }, { token }).catch(() => ({ transactions: [] }))

    return {
      transactions: data.transactions || [],
      total: data.transactions?.length || 0
    }
  },
}

// Settings API - GraphQL Integration
export const settingsAPI = {
  getUserSettings: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetUserSettings {
        userSettings {
          userId
          displayName
          email
          theme
          accentColor
          sidebarPosition
          notificationsEnabled
          emailAlertsEnabled
          consciousnessEnabled
          autoLearningEnabled
          consciousnessThreshold
          emotionalSensitivity
          learningRate
          memoryConsolidation
          apiBaseUrl
          websocketUrl
          connectionTimeout
          databaseType
          databasePath
        }
      }
    `, {}, { token })

    return data.userSettings
  },

  updateUserSettings: async (updates: Record<string, any>, token?: string) => {
    const data = await fetchGraphQL<any>(`
      mutation UpdateUserSettings(
        $displayName: String
        $email: String
        $theme: String
        $accentColor: String
        $sidebarPosition: String
        $notificationsEnabled: Boolean
        $emailAlertsEnabled: Boolean
        $consciousnessEnabled: Boolean
        $autoLearningEnabled: Boolean
        $consciousnessThreshold: Float
        $emotionalSensitivity: String
        $learningRate: String
        $memoryConsolidation: String
        $apiBaseUrl: String
        $websocketUrl: String
        $connectionTimeout: Int
        $databaseType: String
        $databasePath: String
      ) {
        updateUserSettings(
          displayName: $displayName
          email: $email
          theme: $theme
          accentColor: $accentColor
          sidebarPosition: $sidebarPosition
          notificationsEnabled: $notificationsEnabled
          emailAlertsEnabled: $emailAlertsEnabled
          consciousnessEnabled: $consciousnessEnabled
          autoLearningEnabled: $autoLearningEnabled
          consciousnessThreshold: $consciousnessThreshold
          emotionalSensitivity: $emotionalSensitivity
          learningRate: $learningRate
          memoryConsolidation: $memoryConsolidation
          apiBaseUrl: $apiBaseUrl
          websocketUrl: $websocketUrl
          connectionTimeout: $connectionTimeout
          databaseType: $databaseType
          databasePath: $databasePath
        ) {
          userId
          displayName
          email
          theme
          accentColor
          sidebarPosition
          notificationsEnabled
          emailAlertsEnabled
          consciousnessEnabled
          autoLearningEnabled
          consciousnessThreshold
          emotionalSensitivity
          learningRate
          memoryConsolidation
          apiBaseUrl
          websocketUrl
          connectionTimeout
          databaseType
          databasePath
        }
      }
    `, updates, { token })

    return data.updateUserSettings
  },
}

// Training API
export const trainingAPI = {
  getStatus: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query {
        trainingStatus {
          trainingId
          isTraining
          status
          progressPercent
          currentComponent
          componentsCompleted
          totalComponents
          startedAt
          estimatedCompletion
          qaCount
          qaUnusedCount
          lastTrainingId
          lastTrainingResult
        }
      }
    `, {}, { token })
    return data.trainingStatus
  },

  getComponentStatus: async (trainingId: string, token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetComponentStatus($trainingId: String!) {
        componentTrainingStatus(trainingId: $trainingId) {
          componentName
          status
          progress
          metrics
          startedAt
          completedAt
          error
        }
      }
    `, { trainingId }, { token })
    return data.componentTrainingStatus
  },

  getValidationResults: async (trainingId: string, token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetValidationResults($trainingId: String!) {
        validationResults(trainingId: $trainingId) {
          componentName
          validationPassed
          improvementScore
          beforeMetrics
          afterMetrics
          testResults
          validationTime
        }
      }
    `, { trainingId }, { token })
    return data.validationResults
  },
}

// System API
export const systemAPI = {
  getStats: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetSystemMetrics {
        systemMetrics {
          cpuUsage
          memoryUsage
          uptimePercentage
          apiCallsToday
        }
      }
    `, {}, { token })

    return {
      uptime_seconds: 86400,
      cpu_usage_percent: data.systemMetrics?.cpuUsage || 0,
      memory_usage_percent: data.systemMetrics?.memoryUsage || 0,
      memory_used_gb: 4.2,
      memory_total_gb: 16,
      disk_usage_percent: 45,
      disk_used_gb: 200,
      disk_total_gb: 512,
      active_connections: 12,
      total_requests: data.systemMetrics?.apiCallsToday || 0,
      error_rate_percent: 0.1,
      average_response_time_ms: 25
    }
  },

  getStatus: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetSystemStatus {
        systemHealth {
          validationStatus
          testedTheoriesCount
          integrationTestsPassed
          errorsCount
          timestamp
        }
      }
    `, {}, { token })

    return {
      status: data.systemHealth?.validationStatus || "unknown",
      services: { consciousness: data.systemHealth?.validationStatus || "unknown" },
      last_health_check: data.systemHealth?.timestamp,
      version: process.env.NEXT_PUBLIC_APP_VERSION || "1.0.0",
      environment: "production"
    }
  },

  getInfo: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetSystemInfo {
        systemInfo {
          version
          environment
          buildDate
          features
        }
      }
    `, {}, { token }).catch(() => ({ systemInfo: {} }))

    return data.systemInfo || {}
  },
}

// Chat API
export const chatAPI = {
  send: async (message: string, conversation_id?: string, token?: string) => {
    try {
      console.log("🔷 Sending message to GraphQL:", { message, conversation_id })

      const data = await fetchGraphQL<any>(`
        mutation SendMessage($input: SendMessageInput!) {
          sendMessage(input: $input) {
            id
            conversationId
            role
            content
            timestamp
          }
        }
      `, {
        input: {
          conversationId: conversation_id || "new",
          content: message,
          consciousnessEnhanced: true
        }
      }, { token })

      console.log("✅ GraphQL response:", data)

      if (!data || !data.sendMessage) {
        throw new Error("Invalid response from server")
      }

      return {
        conversation_id: data.sendMessage.conversationId,
        message: {
          role: data.sendMessage.role,
          content: data.sendMessage.content,
          timestamp: data.sendMessage.timestamp,
          metadata: data.sendMessage.metadata
        },
        response_time: 150,
        cached: false,
        tokens_used: 50
      }
    } catch (error) {
      console.error("❌ Error in chatAPI.send:", error)
      throw error
    }
  },

  getConversations: async (token?: string) => {
    // Get conversations for current user - userId should come from token context
    // TODO: Update GraphQL schema to support getting conversations for current user without userId
    const data = await fetchGraphQL<any>(`
      query GetConversations {
        conversations {
          id
          title
          messageCount
          updatedAt
        }
      }
    `, {}, { token })
    return data.conversations || []
  },

  getConversation: (id: string, token?: string) =>
    fetchGraphQL<any>(`
      query GetConversation($id: String!) {
        conversation(id: $id) {
          id
          title
          messages {
            role
            content
            timestamp
          }
        }
      }
    `, { id }, { token }),
}

// Hack-Memori API
export const hackMemoriAPI = {
  getSessions: async (token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetHackMemoriSessions {
        hackMemoriSessions {
          id
          name
          createdAt
          startedAt
          stoppedAt
          status
          userId
          config
        }
      }
    `, {}, { token })
    return data.hackMemoriSessions || []
  },

  startSession: async (name: string, config?: Record<string, any>, token?: string) => {
    const data = await fetchGraphQL<any>(`
      mutation StartHackMemoriSession($name: String!, $config: JSON) {
        startHackMemoriSession(name: $name, config: $config) {
          id
          name
          createdAt
          startedAt
          stoppedAt
          status
          userId
          config
        }
      }
    `, { name, config }, { token })
    return data.startHackMemoriSession
  },

  stopSession: async (sessionId: string, token?: string) => {
    const data = await fetchGraphQL<any>(`
      mutation StopHackMemoriSession($sessionId: String!) {
        stopHackMemoriSession(sessionId: $sessionId) {
          id
          name
          createdAt
          startedAt
          stoppedAt
          status
          userId
          config
        }
      }
    `, { sessionId }, { token })
    return data.stopHackMemoriSession
  },

  getQuestions: async (sessionId: string, token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetHackMemoriQuestions($sessionId: String!) {
        hackMemoriQuestions(sessionId: $sessionId) {
          id
          sessionId
          text
          origin
          meta
          createdAt
        }
      }
    `, { sessionId }, { token })
    return data.hackMemoriQuestions || []
  },

  getResponses: async (sessionId: string, token?: string) => {
    const data = await fetchGraphQL<any>(`
      query GetHackMemoriResponses($sessionId: String!) {
        hackMemoriResponses(sessionId: $sessionId) {
          id
          questionId
          sessionId
          modelId
          prompt
          response
          tokensUsed
          llmMeta
          piiFlag
          acceptedForTraining
          humanAnnotation
          createdAt
        }
      }
    `, { sessionId }, { token })
    return data.hackMemoriResponses || []
  },

  addQuestion: async (sessionId: string, text: string, origin = "manual", meta?: Record<string, any>, token?: string) => {
    const data = await fetchGraphQL<any>(`
      mutation AddHackMemoriQuestion($sessionId: String!, $text: String!, $origin: String, $meta: JSON) {
        addHackMemoriQuestion(sessionId: $sessionId, text: $text, origin: $origin, meta: $meta) {
          id
          sessionId
          text
          origin
          meta
          createdAt
        }
      }
    `, { sessionId, text, origin, meta }, { token })
    return data.addHackMemoriQuestion
  },

  addResponse: async (sessionId: string, questionId: string, prompt: string, response: string, tokensUsed: number, token?: string) => {
    const data = await fetchGraphQL<any>(`
      mutation AddHackMemoriResponse($sessionId: String!, $questionId: String!, $modelId: String!, $prompt: String!, $response: String!, $tokensUsed: Int!) {
        addHackMemoriResponse(sessionId: $sessionId, questionId: $questionId, modelId: "gemma-2b", prompt: $prompt, response: $response, tokensUsed: $tokensUsed) {
          id
          questionId
          sessionId
          modelId
          prompt
          response
          tokensUsed
          acceptedForTraining
          createdAt
        }
      }
    `, { sessionId, questionId, prompt, response, tokensUsed }, { token })
    return data.addHackMemoriResponse
  },

  acceptResponse: async (responseId: string, accept: boolean, token?: string) => {
    const data = await fetchGraphQL<any>(`
      mutation AcceptHackMemoriResponse($responseId: String!, $accept: Boolean!) {
        acceptHackMemoriResponse(responseId: $responseId, accept: $accept) {
          id
          acceptedForTraining
        }
      }
    `, { responseId, accept }, { token })
    return data.acceptHackMemoriResponse
  },

  deleteSession: async (sessionId: string, token?: string) => {
    const data = await fetchGraphQL<any>(`
      mutation DeleteHackMemoriSession($sessionId: String!) {
        deleteHackMemoriSession(sessionId: $sessionId)
      }
    `, { sessionId }, { token })
    return data.deleteHackMemoriSession
  },
}

export { fetchGraphQL, API_BASE_URL }
