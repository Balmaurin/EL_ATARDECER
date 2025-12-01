"use client"

import useSWR, { type SWRConfiguration } from "swr"
import {
  dashboardAPI,
  consciousnessAPI,
  exercisesAPI,
  knowledgeAPI,
  blockchainAPI,
  usersAPI,
  systemAPI,
  trainingAPI,
} from "@/lib/api"

const defaultConfig: SWRConfiguration = {
  refreshInterval: 30000, // Refresh every 30 seconds
  revalidateOnFocus: true,
  dedupingInterval: 5000,
}

// Dashboard hooks
export function useDashboardConsciousness(token?: string) {
  return useSWR("dashboard-consciousness", () => dashboardAPI.getConsciousness(token), {
    ...defaultConfig,
    refreshInterval: 10000,
  })
}

export function useDashboardStatus(token?: string) {
  return useSWR("dashboard-status", () => dashboardAPI.getStatus(token), defaultConfig)
}

export function useDashboardRealtime(token?: string) {
  return useSWR("dashboard-realtime", () => dashboardAPI.getRealtime(token), {
    ...defaultConfig,
    refreshInterval: 5000,
  })
}

export function useDashboardAlerts(token?: string) {
  return useSWR("dashboard-alerts", () => dashboardAPI.getAlerts(token), defaultConfig)
}

// Consciousness hooks
export function useConsciousnessStatus(token?: string) {
  return useSWR("consciousness-status", () => consciousnessAPI.getStatus(token), {
    ...defaultConfig,
    refreshInterval: 10000,
  })
}

export function useConsciousnessMetrics(token?: string) {
  return useSWR("consciousness-metrics", () => consciousnessAPI.getMetrics(token), {
    ...defaultConfig,
    refreshInterval: 15000,
  })
}

export function useConsciousnessDreams(limit = 10, token?: string) {
  return useSWR(["consciousness-dreams", limit], () => consciousnessAPI.getDreams(limit, token), defaultConfig)
}

// Exercises hooks
export function useExerciseDatasets(token?: string) {
  return useSWR("exercise-datasets", () => exercisesAPI.getDatasets(token), defaultConfig)
}

// Knowledge hooks
export function useKnowledgeStats(token?: string) {
  return useSWR("knowledge-stats", () => knowledgeAPI.getStats(token), defaultConfig)
}

export function useKnowledgeDocuments(token?: string) {
  return useSWR("knowledge-documents", () => knowledgeAPI.getDocuments(token), defaultConfig)
}

// Blockchain hooks
export function useBlockchainBalance(token?: string) {
  return useSWR("blockchain-balance", () => blockchainAPI.getBalance(token), defaultConfig)
}

// Users hooks
export function useUserTokens() {
  return useSWR("user-tokens", () => usersAPI.getTokens(), defaultConfig)
}

export function useTokenBalance() {
  return useSWR("token-balance", () => usersAPI.getTokenBalance(), defaultConfig)
}

// System hooks
export function useSystemStats(token?: string) {
  return useSWR("system-stats", () => systemAPI.getStats(token), { ...defaultConfig, refreshInterval: 10000 })
}

export function useSystemStatus(token?: string) {
  return useSWR("system-status", () => systemAPI.getStatus(token), { ...defaultConfig, refreshInterval: 15000 })
}

// Training hooks
export function useTrainingStatus(token?: string) {
  return useSWR("training-status", () => trainingAPI.getStatus(token), { ...defaultConfig, refreshInterval: 2000 })
}

export function useComponentTrainingStatus(trainingId: string, token?: string) {
  return useSWR(
    ["component-training-status", trainingId],
    () => trainingAPI.getComponentStatus(trainingId, token),
    { ...defaultConfig, refreshInterval: 2000 }
  )
}

export function useValidationResults(trainingId: string, token?: string) {
  return useSWR(
    ["validation-results", trainingId],
    () => trainingAPI.getValidationResults(trainingId, token),
    { ...defaultConfig, refreshInterval: 5000 }
  )
}
