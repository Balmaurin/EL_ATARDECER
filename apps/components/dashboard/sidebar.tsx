"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Brain,
  LayoutDashboard,
  Dumbbell,
  Coins,
  Settings,
  MessageSquare,
  Activity,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  BookOpen,
  Zap,
} from "lucide-react"

const navItems = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/consciousness", label: "Consciousness", icon: Brain },
  { href: "/chat", label: "Chat AI", icon: MessageSquare },
  { href: "/hackmemori", label: "Hack-Memori", icon: Zap },
  { href: "/exercises", label: "Training", icon: Dumbbell },
  { href: "/knowledge", label: "Knowledge Base", icon: BookOpen },
  { href: "/tokens", label: "Tokens", icon: Coins },
  { href: "/system", label: "System", icon: Activity },
  { href: "/settings", label: "Settings", icon: Settings },
]

export function DashboardSidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const pathname = usePathname()
  const router = useRouter()

  const handleNavigation = (href: string, event: React.MouseEvent) => {
    // Prevenir comportamiento por defecto y navegación externa
    event.preventDefault()
    event.stopPropagation()

    // Solo permitir navegación interna
    router.push(href)
  }

  return (
    <aside
      className={cn(
        "flex flex-col border-r border-sidebar-border bg-sidebar transition-all duration-300",
        collapsed ? "w-16" : "w-64",
      )}
    >
      <div className="flex h-16 items-center justify-between border-b border-sidebar-border px-4">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <Sparkles className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="font-semibold text-sidebar-foreground">Sheily AI</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-sidebar-foreground"
          onClick={() => setCollapsed(!collapsed)}
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </Button>
      </div>

      <ScrollArea className="flex-1 px-2 py-4">
        <nav className="flex flex-col gap-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href
            const Icon = item.icon

            return (
              <Button
                key={item.href}
                variant={isActive ? "secondary" : "ghost"}
                className={cn("w-full justify-start text-sidebar-foreground", collapsed && "justify-center px-2")}
                onClick={(event) => handleNavigation(item.href, event)}
              >
                <Icon className={cn("h-4 w-4", !collapsed && "mr-2")} />
                {!collapsed && <span>{item.label}</span>}
              </Button>
            )
          })}
        </nav>
      </ScrollArea>

      <div className="border-t border-sidebar-border p-4">
        {!collapsed && (
          <div className="flex items-center gap-2 rounded-lg bg-sidebar-accent p-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20">
              <Zap className="h-4 w-4 text-primary" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs text-sidebar-foreground truncate">Consciousness</p>
              <p className="text-xs text-muted-foreground">Active</p>
            </div>
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse-glow" />
          </div>
        )}
      </div>
    </aside>
  )
}
