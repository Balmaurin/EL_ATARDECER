"""
MCP Agent System for Sheily Enterprise Master
==============================================

Sistema unificado de agentes MCP especializados para Sheily.
Integra los 4 agentes especializados core con el sistema de IA MCP existente.

Características principales:
- 4 agentes core especializados (Finance, Security, Healthcare, Business)
- 1 sistema MCP (Model Context Protocol) para coordinación
- Integración con Gemini/AGP de Google (opcional)
- Coordinación inteligente con machine learning
- Integración seamless con Sheily Core
- Auto-escalado y recuperación automática de fallos
- Comunicación inter-agente en tiempo real

Estructura del sistema:
- base/: Clases base y interfaces comunes
- coordination/: Sistema de coordinación y gestión MCP
- specialized/: Agentes especializados por dominio
- integration/: Integración con Sheily Core

Agentes Core Implementados:
1. Financial Analyst - Análisis financiero y predicciones
2. Security Expert - Seguridad y compliance
3. Medical Advisor - Asesoramiento médico y diagnóstico
4. General Assistant - Asistencia empresarial general

Author: MCP Enterprise Master
Version: 1.0.0
"""

import asyncio
import logging
import uuid

# Configuración de logging para el módulo
logging.getLogger(__name__).addHandler(logging.NullHandler())

from .base.base_agent import (
    AgentCapability,
    AgentMessage,
    AgentMetrics,
    AgentStatus,
    AgentTask,
    BaseMCPAgent,
    MessageBus,
    get_global_message_bus,
)
from .coordination.agent_coordinator import (
    MCPAgentCoordinator,
    get_global_agent_coordinator,
    initialize_mcp_agent_system,
)

# Agentes especializados disponibles
try:
    from .specialized.research.scientific_research_agent import ScientificResearchAgent

    _SCIENTIFIC_RESEARCH_AVAILABLE = True
except ImportError:
    _SCIENTIFIC_RESEARCH_AVAILABLE = False
    ScientificResearchAgent = None

# Información del sistema
__version__ = "1.0.0"
__author__ = "MCP Enterprise Master"
__description__ = "Sistema unificado de agentes MCP para Sheily Enterprise"

# Estado de agentes disponibles
AVAILABLE_AGENTS = {
    "scientific_research": _SCIENTIFIC_RESEARCH_AVAILABLE,
    "finance_agents": False,  # Próximamente
    "security_agents": False,  # Próximamente
    "healthcare_agents": False,  # Próximamente
    "education_agents": False,  # Próximamente
}

# Capacidades del sistema
SYSTEM_CAPABILITIES = [
    "intelligent_task_assignment",
    "multi_agent_coordination",
    "real_time_communication",
    "automatic_scaling",
    "failure_recovery",
    "performance_monitoring",
    "research_synthesis",
    "predictive_analytics",
]


def get_available_agents() -> dict:
    """Obtener estado de agentes disponibles"""
    return AVAILABLE_AGENTS.copy()


def create_agent_coordinator() -> MCPAgentCoordinator:
    """Crear una nueva instancia del coordinador MCP"""
    return MCPAgentCoordinator()


async def get_system_status() -> dict:
    """Obtener estado global del sistema de agentes MCP"""
    try:
        coordinator = await get_global_agent_coordinator()
        return await coordinator.get_system_status()
    except Exception as e:
        return {"status": "error", "error": str(e), "system_available": False}


def get_system_info() -> dict:
    """Obtener información general del sistema MCP"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "available_agents": get_available_agents(),
        "total_agents_implemented": sum(AVAILABLE_AGENTS.values()),
        "total_agents_planned": len(AVAILABLE_AGENTS),
        "system_capabilities": SYSTEM_CAPABILITIES,
        "coordination_features": [
            "load_balancing",
            "failover_management",
            "resource_monitoring",
            "task_scheduling",
            "performance_optimization",
        ],
    }


# ========== FUNCIONES DE UTILIDAD ==========


async def assign_research_task(task_type: str, parameters: dict) -> dict:
    """
    Asignar tarea de investigación al agente especializado

    Args:
        task_type: Tipo de tarea (analyze_data, generate_hypothesis, etc.)
        parameters: Parámetros específicos de la tarea

    Returns:
        Resultado de la asignación y ejecución
    """
    try:
        coordinator = await get_global_agent_coordinator()
        result = await coordinator.assign_task_intelligently(
            task_type=task_type,
            parameters=parameters,
            required_capabilities=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS],
        )
        return result
    except Exception as e:
        return {
            "error": f"Failed to assign research task: {str(e)}",
            "task_type": task_type,
            "available_agents": get_available_agents(),
        }


async def get_coordination_stats() -> dict:
    """Obtener estadísticas del sistema de coordinación"""
    try:
        coordinator = await get_global_agent_coordinator()
        return coordinator.coordination_stats
    except Exception as e:
        return {"error": str(e), "stats_available": False}


async def execute_parallel_tasks(tasks: list) -> dict:
    """
    Ejecutar múltiples tareas en paralelo

    Args:
        tasks: Lista de tareas a ejecutar en paralelo

    Returns:
        Resultados de la ejecución paralela
    """
    try:
        coordinator = await get_global_agent_coordinator()

        # Crear tarea de coordinación multi-agente
        coordination_task = AgentTask(
            task_id=f"parallel_{uuid.uuid4()}",
            task_type="coordinate_multi_agent",
            parameters={"sub_tasks": tasks, "strategy": "parallel"},
        )

        return await coordinator.execute_task(coordination_task)
    except Exception as e:
        return {"error": f"Parallel execution failed: {str(e)}"}
