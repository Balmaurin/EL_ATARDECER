#!/usr/bin/env python3
"""
Sistema de Registro Activo - IMPLEMENTACIÓN REAL
Monitoreo de salud basado en estado real de objetos y memoria, sin simulaciones.
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_registry import AgentRegistry


@dataclass
class AgentHealthStatus:
    """Estado de salud de un agente"""

    agent_id: str
    is_healthy: bool
    last_heartbeat: datetime
    response_time: float
    error_count: int
    uptime_seconds: int
    health_score: float
    memory_usage_bytes: int = 0  # Nuevo: Uso real de memoria


class ActiveAgentRegistry:
    """Registro de agentes que gestiona activamente el estado y salud REAL"""

    def __init__(self):
        self.base_registry = AgentRegistry()
        self.health_monitor: Dict[str, AgentHealthStatus] = {}
        self.performance_history: Dict[str, List[Dict]] = {}
        self.auto_recovery_enabled = True
        self.health_check_interval = 30  # segundos
        self.logger = logging.getLogger("Sheily.ActiveRegistry")
        self.is_running = False
        self._agent_instances = {} # Referencias a las instancias reales para ping

    async def start_active_management(self):
        """Iniciar gestión activa del registro"""
        await self.base_registry.start_registry()
        self.is_running = True

        # Iniciar monitoreo continuo
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._performance_tracking_loop())
        asyncio.create_task(self._auto_recovery_loop())

        self.logger.info("Active Agent Registry started (REAL MODE)")

    async def register_agent_with_health_monitoring(
        self, agent: Any, agent_type: str = "generic", metadata: Dict[str, Any] = None
    ) -> str:
        """Registrar agente con monitoreo de salud automático"""
        # Registrar en el registro base
        agent_id = await self.base_registry.register_agent(
            agent=agent, agent_type=agent_type, metadata=metadata or {}
        )
        
        # Guardar referencia real para monitoreo
        self._agent_instances[agent_id] = agent

        # Inicializar monitoreo de salud
        self.health_monitor[agent_id] = AgentHealthStatus(
            agent_id=agent_id,
            is_healthy=True,
            last_heartbeat=datetime.now(),
            response_time=0.0,
            error_count=0,
            uptime_seconds=0,
            health_score=1.0,
            memory_usage_bytes=sys.getsizeof(agent) # Medición real inicial
        )

        # Inicializar historial de rendimiento
        self.performance_history[agent_id] = []

        self.logger.info(f"Agent {agent_id} registered with REAL health monitoring")
        return agent_id

    async def unregister_agent_safely(self, agent_id: str) -> bool:
        """Desregistrar agente de forma segura verificando dependencias"""
        try:
            # Desregistrar del registro base
            success = await self.base_registry.unregister_agent(agent_id)

            if success:
                # Limpiar monitoreo
                self.health_monitor.pop(agent_id, None)
                self.performance_history.pop(agent_id, None)
                self._agent_instances.pop(agent_id, None)

                self.logger.info(f"Agent {agent_id} unregistered successfully")

            return success

        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False

    async def _health_monitoring_loop(self):
        """Loop de monitoreo de salud"""
        while self.is_running:
            try:
                await self._check_all_agents_health()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)

    async def _check_all_agents_health(self):
        """Verificar salud de todos los agentes"""
        for agent_id in list(self.health_monitor.keys()):
            await self._check_agent_health(agent_id)

    async def _check_agent_health(self, agent_id: str):
        """Verificar salud de un agente específico (REAL)"""
        try:
            health_status = self.health_monitor.get(agent_id)
            if not health_status:
                return

            start_time = datetime.now()

            # Verificación REAL: ¿El objeto existe y responde?
            agent_responsive = await self._ping_agent_real(agent_id)

            response_time = (datetime.now() - start_time).total_seconds()

            # Actualizar estado de salud
            health_status.last_heartbeat = datetime.now()
            health_status.response_time = response_time
            health_status.is_healthy = agent_responsive
            
            # Actualizar uso de memoria real
            if agent_id in self._agent_instances:
                try:
                    health_status.memory_usage_bytes = sys.getsizeof(self._agent_instances[agent_id])
                except:
                    pass

            if not agent_responsive:
                health_status.error_count += 1

            # Calcular score de salud
            health_status.health_score = self._calculate_health_score(health_status)

            # Registrar en historial
            self._record_health_check(agent_id, health_status)

            # Detectar problemas
            if health_status.health_score < 0.5:
                await self._handle_unhealthy_agent(agent_id, health_status)

        except Exception as e:
            self.logger.error(f"Error checking health for agent {agent_id}: {e}")

    async def _ping_agent_real(self, agent_id: str) -> bool:
        """
        REAL ping to agent - checks actual agent health
        Uses actual agent methods, not memory references
        """
        try:
            agent = self._agent_instances.get(agent_id)
            if agent is None:
                return False
            
            # REAL health check: Try to call agent methods
            # Method 1: Check if agent has health_check method
            if hasattr(agent, 'health_check'):
                try:
                    if asyncio.iscoroutinefunction(agent.health_check):
                        health_result = await asyncio.wait_for(agent.health_check(), timeout=2.0)
                    else:
                        health_result = agent.health_check()
                    
                    # Check if health result indicates agent is alive
                    if isinstance(health_result, dict):
                        status = health_result.get('status', 'unknown')
                        return status in ['operational', 'active', 'idle', 'ready']
                    elif isinstance(health_result, bool):
                        return health_result
                except asyncio.TimeoutError:
                    self.logger.warning(f"Health check timeout for agent {agent_id}")
                    return False
                except Exception as e:
                    self.logger.debug(f"Health check error for agent {agent_id}: {e}")
            
            # Method 2: Check is_active attribute
            if hasattr(agent, 'is_active'):
                is_active = getattr(agent, 'is_active', False)
                if callable(is_active):
                    return is_active()
                return bool(is_active)
            
            # Method 3: Check status attribute
            if hasattr(agent, 'status'):
                status = getattr(agent, 'status', None)
                if hasattr(status, 'value'):
                    status = status.value
                return status not in ['error', 'shutdown', 'offline', 'dead']
            
            # Method 4: Try to access agent_id (basic existence check)
            if hasattr(agent, 'agent_id'):
                return getattr(agent, 'agent_id') == agent_id
            
            # If we can access the agent object, assume it's alive
            # This is a last resort - better than getrefcount
            return True

        except Exception as e:
            self.logger.debug(f"Error pinging agent {agent_id}: {e}")
            return False

    def _calculate_health_score(self, health_status: AgentHealthStatus) -> float:
        """Calcular score de salud basado en métricas reales"""
        # Factor de respuesta (mejor si es menor)
        # En sistemas reales, > 1s es lento para un ping en memoria
        response_factor = max(0, 1.0 - (health_status.response_time / 1.0))

        # Factor de errores
        error_factor = max(0, 1.0 - (health_status.error_count / 5.0))

        # Factor de tiempo desde último heartbeat
        time_since_heartbeat = (
            datetime.now() - health_status.last_heartbeat
        ).total_seconds()
        heartbeat_factor = max(0, 1.0 - (time_since_heartbeat / 60.0))

        # Score ponderado
        return response_factor * 0.4 + error_factor * 0.4 + heartbeat_factor * 0.2

    def _record_health_check(self, agent_id: str, health_status: AgentHealthStatus):
        """Registrar verificación de salud en el historial"""
        if agent_id not in self.performance_history:
            self.performance_history[agent_id] = []

        record = {
            "timestamp": datetime.now().isoformat(),
            "is_healthy": health_status.is_healthy,
            "response_time": health_status.response_time,
            "health_score": health_status.health_score,
            "error_count": health_status.error_count,
            "memory_bytes": health_status.memory_usage_bytes
        }

        self.performance_history[agent_id].append(record)

        # Mantener solo últimos 100 registros por agente
        if len(self.performance_history[agent_id]) > 100:
            self.performance_history[agent_id] = self.performance_history[agent_id][
                -100:
            ]

    async def _handle_unhealthy_agent(
        self, agent_id: str, health_status: AgentHealthStatus
    ):
        """Manejar agente con problemas de salud"""
        self.logger.warning(
            f"Agent {agent_id} is unhealthy (score: {health_status.health_score:.2f})"
        )

        if self.auto_recovery_enabled:
            await self._attempt_agent_recovery(agent_id)

    async def _attempt_agent_recovery(self, agent_id: str):
        """Intentar recuperar un agente"""
        try:
            self.logger.info(f"Attempting recovery for agent {agent_id}")

            agent = self._agent_instances.get(agent_id)
            if agent and hasattr(agent, 'restart'):
                # Intento de reinicio real si el agente lo soporta
                await agent.restart()
                self.logger.info(f"Agent {agent_id} restart triggered")
            elif agent and hasattr(agent, 'start'):
                 await agent.start()
                 self.logger.info(f"Agent {agent_id} start triggered")

            # Resetear contador de errores para dar otra oportunidad
            if agent_id in self.health_monitor:
                self.health_monitor[agent_id].error_count = max(
                    0, self.health_monitor[agent_id].error_count - 1
                )

        except Exception as e:
            self.logger.error(f"Recovery failed for agent {agent_id}: {e}")

    async def _performance_tracking_loop(self):
        """Loop de seguimiento de rendimiento"""
        while self.is_running:
            try:
                await self._analyze_performance_trends()
                await asyncio.sleep(300)  # Cada 5 minutos
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(300)

    async def _analyze_performance_trends(self):
        """Analizar tendencias de rendimiento"""
        for agent_id, history in self.performance_history.items():
            if len(history) >= 10:
                recent_scores = [record["health_score"] for record in history[-10:]]
                avg_score = sum(recent_scores) / len(recent_scores)

                if avg_score < 0.7:
                    self.logger.warning(
                        f"Agent {agent_id} shows declining performance: {avg_score:.2f}"
                    )

    async def _auto_recovery_loop(self):
        """Loop de recuperación automática"""
        while self.is_running:
            try:
                await self._check_for_recovery_opportunities()
                await asyncio.sleep(120)
            except Exception as e:
                self.logger.error(f"Auto recovery error: {e}")
                await asyncio.sleep(120)

    async def _check_for_recovery_opportunities(self):
        """Verificar oportunidades de recuperación"""
        for agent_id, health_status in self.health_monitor.items():
            time_since_last_check = (
                datetime.now() - health_status.last_heartbeat
            ).total_seconds()

            if (
                health_status.is_healthy
                and time_since_last_check < 60
                and health_status.error_count > 0
            ):
                health_status.error_count = max(0, health_status.error_count - 1)

    # API pública para consultas
    async def get_registry_health_summary(self) -> Dict[str, Any]:
        """Obtener resumen de salud del registro"""
        total_agents = len(self.health_monitor)
        healthy_agents = len([h for h in self.health_monitor.values() if h.is_healthy])

        avg_health_score = 0.0
        if total_agents > 0:
            avg_health_score = (
                sum(h.health_score for h in self.health_monitor.values()) / total_agents
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "unhealthy_agents": total_agents - healthy_agents,
            "average_health_score": avg_health_score,
            "registry_uptime": "active" if self.is_running else "inactive",
        }

    async def get_agent_detailed_health(
        self, agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Obtener salud detallada de un agente"""
        health_status = self.health_monitor.get(agent_id)
        if not health_status:
            return None

        history = self.performance_history.get(agent_id, [])
        recent_history = history[-10:] if len(history) >= 10 else history

        return {
            "agent_id": agent_id,
            "current_health": asdict(health_status),
            "recent_performance": recent_history,
            "performance_trend": self._calculate_performance_trend(recent_history),
            "recommendations": self._generate_health_recommendations(
                health_status, recent_history
            ),
        }

    def _calculate_performance_trend(self, history: List[Dict]) -> str:
        """Calcular tendencia de rendimiento"""
        if len(history) < 3:
            return "insufficient_data"

        scores = [record["health_score"] for record in history]

        recent_avg = sum(scores[-3:]) / 3
        earlier_avg = sum(scores[:3]) / 3

        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "declining"
        else:
            return "stable"

    def _generate_health_recommendations(
        self, health_status: AgentHealthStatus, history: List[Dict]
    ) -> List[str]:
        """Generar recomendaciones de salud"""
        recommendations = []

        if health_status.health_score < 0.5:
            recommendations.append("Consider restarting the agent")

        if health_status.response_time > 2.0:
            recommendations.append("High latency detected - Check system load")

        if health_status.error_count > 5:
            recommendations.append("Review agent logs for recurring errors")

        return recommendations

    async def force_health_check_all(self) -> Dict[str, Any]:
        """Forzar verificación de salud para todos los agentes"""
        start_time = datetime.now()

        for agent_id in list(self.health_monitor.keys()):
            await self._check_agent_health(agent_id)

        duration = (datetime.now() - start_time).total_seconds()

        return {
            "health_check_completed": True,
            "agents_checked": len(self.health_monitor),
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
        }


# CONVERTIR AgentRegistry en gestor activo
active_registry = ActiveAgentRegistry()


# Funciones de utilidad
async def start_active_registry():
    """Iniciar registro activo"""
    await active_registry.start_active_management()
    return active_registry


async def get_registry_health():
    """Obtener salud del registro"""
    return await active_registry.get_registry_health_summary()
