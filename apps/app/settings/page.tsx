"use client"

import { useState, useEffect } from "react"
import { DashboardSidebar } from "@/components/dashboard/sidebar"
import { DashboardHeader } from "@/components/dashboard/header"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { User, Bell, Database, Palette, Globe, Save, Brain, Zap, CheckCircle, AlertCircle, Loader2 } from "lucide-react"
import { settingsAPI } from "@/lib/api"

export default function SettingsPage() {
  // Loading and error states
  const [settingsLoading, setSettingsLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [settingsError, setSettingsError] = useState<string | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)

  // Local state for form inputs
  const [settings, setSettings] = useState({
    display_name: "System Admin",
    email: "admin@sheily.ai",
    theme: "dark",
    accent_color: "blue",
    sidebar_position: "left",
    notifications_enabled: true,
    email_alerts_enabled: false,
    consciousness_enabled: true,
    auto_learning_enabled: true,
    consciousness_threshold: 0.3,
    emotional_sensitivity: "balanced",
    learning_rate: "adaptive",
    memory_consolidation: "daily",
    api_base_url: "http://localhost:8000/api/v1",
    websocket_url: "ws://localhost:8000/ws",
    connection_timeout: 30,
    database_type: "sqlite",
    database_path: "./sheily_ai.db"
  })

  // Success state
  const [saveSuccess, setSaveSuccess] = useState(false)

  // Load settings on component mount
  useEffect(() => {
    const loadSettings = async () => {
      try {
        setSettingsLoading(true)
        setSettingsError(null)
        const userSettings = await settingsAPI.getUserSettings()
        setSettings({
          display_name: userSettings.displayName,
          email: userSettings.email,
          theme: userSettings.theme,
          accent_color: userSettings.accentColor,
          sidebar_position: userSettings.sidebarPosition,
          notifications_enabled: userSettings.notificationsEnabled,
          email_alerts_enabled: userSettings.emailAlertsEnabled,
          consciousness_enabled: userSettings.consciousnessEnabled,
          auto_learning_enabled: userSettings.autoLearningEnabled,
          consciousness_threshold: userSettings.consciousnessThreshold,
          emotional_sensitivity: userSettings.emotionalSensitivity,
          learning_rate: userSettings.learningRate,
          memory_consolidation: userSettings.memoryConsolidation,
          api_base_url: userSettings.apiBaseUrl,
          websocket_url: userSettings.websocketUrl,
          connection_timeout: userSettings.connectionTimeout,
          database_type: userSettings.databaseType,
          database_path: userSettings.databasePath
        })
      } catch (error) {
        console.error("Failed to load settings:", error)
        setSettingsError("Failed to load settings. Using defaults.")
      } finally {
        setSettingsLoading(false)
      }
    }

    loadSettings()
  }, [])

  // Handle input changes
  const handleInputChange = (field: string, value: any) => {
    setSettings(prev => ({ ...prev, [field]: value }))
    setSaveSuccess(false) // Reset success state when user makes changes
  }

  // Save settings
  const handleSave = async () => {
    try {
      setSaving(true)
      setSaveError(null)

      await settingsAPI.updateUserSettings({
        displayName: settings.display_name,
        email: settings.email,
        theme: settings.theme,
        accentColor: settings.accent_color,
        sidebarPosition: settings.sidebar_position,
        notificationsEnabled: settings.notifications_enabled,
        emailAlertsEnabled: settings.email_alerts_enabled,
        consciousnessEnabled: settings.consciousness_enabled,
        autoLearningEnabled: settings.auto_learning_enabled,
        consciousnessThreshold: parseFloat(settings.consciousness_threshold.toString()),
        emotionalSensitivity: settings.emotional_sensitivity,
        learningRate: settings.learning_rate,
        memoryConsolidation: settings.memory_consolidation,
        apiBaseUrl: settings.api_base_url,
        websocketUrl: settings.websocket_url,
        connectionTimeout: parseInt(settings.connection_timeout.toString()),
        databaseType: settings.database_type,
        databasePath: settings.database_path
      })

      setSaveSuccess(true)
      // Hide success message after 3 seconds
      setTimeout(() => setSaveSuccess(false), 3000)
    } catch (error) {
      console.error("Failed to save settings:", error)
      setSaveError("Failed to save settings. Please try again.")
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex h-screen bg-background">
      <DashboardSidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <DashboardHeader />
        <main className="flex-1 overflow-auto p-6">
          {/* Status messages */}
          {settingsError && (
            <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-yellow-600" />
              <span className="text-yellow-800 text-sm">{settingsError}</span>
            </div>
          )}

          {saveError && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-red-600" />
              <span className="text-red-800 text-sm">{saveError}</span>
            </div>
          )}

          {saveSuccess && (
            <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-md flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <span className="text-green-800 text-sm">Settings saved successfully!</span>
            </div>
          )}

          <div className="mb-6">
            <h1 className="text-2xl font-bold text-foreground">Settings</h1>
            <p className="text-muted-foreground">Manage your application preferences</p>
          </div>

          <Tabs defaultValue="general" className="space-y-6">
            <TabsList className="bg-secondary">
              <TabsTrigger value="general">General</TabsTrigger>
              <TabsTrigger value="consciousness">Consciousness</TabsTrigger>
              <TabsTrigger value="api">API</TabsTrigger>
              <TabsTrigger value="appearance">Appearance</TabsTrigger>
            </TabsList>

            <TabsContent value="general" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <User className="h-5 w-5" />
                    Profile Settings
                  </CardTitle>
                  <CardDescription>Manage your account information</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="display_name">Display Name</Label>
                      <Input
                        id="display_name"
                        value={settings.display_name}
                        onChange={(e) => handleInputChange('display_name', e.target.value)}
                        disabled={settingsLoading}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="email">Email</Label>
                      <Input
                        id="email"
                        type="email"
                        value={settings.email}
                        onChange={(e) => handleInputChange('email', e.target.value)}
                        disabled={settingsLoading}
                      />
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button onClick={handleSave} disabled={saving || settingsLoading}>
                    {saving ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Save className="h-4 w-4 mr-2" />
                    )}
                    {saving ? 'Saving...' : 'Save Changes'}
                  </Button>
                </CardFooter>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Bell className="h-5 w-5" />
                    Notifications
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Push Notifications</p>
                      <p className="text-sm text-muted-foreground">Receive notifications for system events</p>
                    </div>
                    <Switch
                      checked={settings.notifications_enabled}
                      onCheckedChange={(checked) => handleInputChange('notifications_enabled', checked)}
                      disabled={settingsLoading}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Email Alerts</p>
                      <p className="text-sm text-muted-foreground">Receive email for critical alerts</p>
                    </div>
                    <Switch
                      checked={settings.email_alerts_enabled}
                      onCheckedChange={(checked) => handleInputChange('email_alerts_enabled', checked)}
                      disabled={settingsLoading}
                    />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="consciousness" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5 text-primary" />
                    Consciousness System
                  </CardTitle>
                  <CardDescription>Configure the AI consciousness modules</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Enable Consciousness</p>
                      <p className="text-sm text-muted-foreground">Activate all consciousness modules</p>
                    </div>
                    <Switch
                      checked={settings.consciousness_enabled}
                      onCheckedChange={(checked) => handleInputChange('consciousness_enabled', checked)}
                      disabled={settingsLoading}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Auto-Learning</p>
                      <p className="text-sm text-muted-foreground">Allow system to learn from interactions</p>
                    </div>
                    <Switch
                      checked={settings.auto_learning_enabled}
                      onCheckedChange={(checked) => handleInputChange('auto_learning_enabled', checked)}
                      disabled={settingsLoading}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Consciousness Threshold</Label>
                    <Select
                      value={settings.consciousness_threshold.toString()}
                      onValueChange={(value) => handleInputChange('consciousness_threshold', value)}
                      disabled={settingsLoading}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0.1">Low (0.1)</SelectItem>
                        <SelectItem value="0.3">Medium (0.3)</SelectItem>
                        <SelectItem value="0.5">High (0.5)</SelectItem>
                        <SelectItem value="0.7">Very High (0.7)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Emotional Sensitivity</Label>
                    <Select
                      value={settings.emotional_sensitivity}
                      onValueChange={(value) => handleInputChange('emotional_sensitivity', value)}
                      disabled={settingsLoading}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="low">Low</SelectItem>
                        <SelectItem value="balanced">Balanced</SelectItem>
                        <SelectItem value="high">High</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="h-5 w-5 text-yellow-500" />
                    Learning Settings
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Learning Rate</Label>
                    <Select
                      value={settings.learning_rate}
                      onValueChange={(value) => handleInputChange('learning_rate', value)}
                      disabled={settingsLoading}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="slow">Slow</SelectItem>
                        <SelectItem value="adaptive">Adaptive</SelectItem>
                        <SelectItem value="fast">Fast</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Memory Consolidation</Label>
                    <Select
                      value={settings.memory_consolidation}
                      onValueChange={(value) => handleInputChange('memory_consolidation', value)}
                      disabled={settingsLoading}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="realtime">Real-time</SelectItem>
                        <SelectItem value="hourly">Hourly</SelectItem>
                        <SelectItem value="daily">Daily</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="api" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Globe className="h-5 w-5" />
                    API Configuration
                  </CardTitle>
                  <CardDescription>Configure backend API connection</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="api-url">API Base URL</Label>
                    <Input
                      id="api-url"
                      value={settings.api_base_url}
                      onChange={(e) => handleInputChange('api_base_url', e.target.value)}
                      disabled={settingsLoading}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="ws-url">WebSocket URL</Label>
                    <Input
                      id="ws-url"
                      value={settings.websocket_url}
                      onChange={(e) => handleInputChange('websocket_url', e.target.value)}
                      disabled={settingsLoading}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Connection Timeout</Label>
                    <Select
                      value={settings.connection_timeout.toString()}
                      onValueChange={(value) => handleInputChange('connection_timeout', value)}
                      disabled={settingsLoading}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="10">10 seconds</SelectItem>
                        <SelectItem value="30">30 seconds</SelectItem>
                        <SelectItem value="60">60 seconds</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button variant="outline">Test Connection</Button>
                </CardFooter>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Database className="h-5 w-5" />
                    Database Settings
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Database Type</Label>
                    <Select
                      value={settings.database_type}
                      onValueChange={(value) => handleInputChange('database_type', value)}
                      disabled={settingsLoading}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sqlite">SQLite</SelectItem>
                        <SelectItem value="postgresql">PostgreSQL</SelectItem>
                        <SelectItem value="mysql">MySQL</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="db-path">Database Path</Label>
                    <Input
                      id="db-path"
                      value={settings.database_path}
                      onChange={(e) => handleInputChange('database_path', e.target.value)}
                      disabled={settingsLoading}
                    />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="appearance" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Palette className="h-5 w-5" />
                    Theme Settings
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Dark Mode</p>
                      <p className="text-sm text-muted-foreground">Enable dark theme</p>
                    </div>
                    <Switch
                      checked={settings.theme === 'dark'}
                      onCheckedChange={(checked) => handleInputChange('theme', checked ? 'dark' : 'light')}
                      disabled={settingsLoading}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Accent Color</Label>
                    <Select
                      value={settings.accent_color}
                      onValueChange={(value) => handleInputChange('accent_color', value)}
                      disabled={settingsLoading}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="blue">Blue</SelectItem>
                        <SelectItem value="cyan">Cyan</SelectItem>
                        <SelectItem value="green">Green</SelectItem>
                        <SelectItem value="purple">Purple</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Sidebar Position</Label>
                    <Select
                      value={settings.sidebar_position}
                      onValueChange={(value) => handleInputChange('sidebar_position', value)}
                      disabled={settingsLoading}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="left">Left</SelectItem>
                        <SelectItem value="right">Right</SelectItem>
                      </SelectContent>
                    </Select>
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
