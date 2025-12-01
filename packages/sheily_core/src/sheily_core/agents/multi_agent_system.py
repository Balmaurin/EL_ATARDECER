#!/usr/bin/env python3
"""
Sistema Multi-Agente para Sheily AI
Implementa coordinación y comunicación entre agentes especializados
Basado en patrones de Google: Hierarchical, Collaborative, Peer-to-Peer
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import contextlib

logger = logging.getLogger(__name__)

# REAL tracing - use actual tracing system
try:
    from ...agent_tracing import trace_agent_execution
    TRACING_AVAILABLE = True
except ImportError:
    # Fallback to basic tracing if module not available
    TRACING_AVAILABLE = False
    @contextlib.contextmanager
    def trace_agent_execution(agent_name, operation):
        """Basic tracing when full system not available"""
        logger.info(f"🔍 Trace: {agent_name}.{operation}")
        trace_obj = type('Trace', (), {
            'add_event': lambda self, n, d: logger.debug(f"Trace event: {n} - {d}")
        })()
        yield trace_obj

# =============================================================================
# MODELOS DE DATOS PARA MULTI-AGENTE
# =============================================================================


class AgentRole(Enum):
    """Roles posibles para agentes en el sistema"""

    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    EVALUATOR = "evaluator"
    MEDIATOR = "mediator"
    EXECUTOR = "executor"


class CommunicationProtocol(Enum):
    """Protocolos de comunicación entre agentes"""

    DIRECT = "direct"  # Comunicación directa síncrona
    MESSAGE_QUEUE = "queue"  # Cola de mensajes asíncrona
    EVENT_DRIVEN = "event"  # Sistema basado en eventos
    SHARED_MEMORY = "memory"  # Memoria compartida


class TaskStatus(Enum):
    """Estados posibles de una tarea"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCapability:
    """Capacidad específica de un agente"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    confidence_score: float = 1.0
    execution_time_estimate: float = 1.0  # segundos


@dataclass
class AgentProfile:
    """Perfil completo de un agente"""

    agent_id: str
    name: str
    role: AgentRole
    capabilities: List[AgentCapability] = field(default_factory=list)
    specialization_domains: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    communication_protocols: List[CommunicationProtocol] = field(default_factory=list)
    is_active: bool = True
    last_seen: Optional[datetime] = None


@dataclass
class Task:
    """Tarea en el sistema multi-agente"""

    task_id: str
    description: str
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1  # 1-10, 10 es máxima prioridad
    dependencies: List[str] = field(default_factory=list)  # IDs de tareas requeridas
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    quality_score: Optional[float] = None


@dataclass
class AgentMessage:
    """Mensaje entre agentes"""

    message_id: str
    sender_id: str
    receiver_id: str
    message_type: (
        str  # "task_assignment", "status_update", "collaboration_request", etc.
    )
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # Para rastrear conversaciones


# =============================================================================
# AGENTE BASE PARA EL SISTEMA MULTI-AGENTE
# =============================================================================


class MultiAgentBase:
    """Clase base para agentes en el sistema multi-agente"""

    def __init__(self, agent_id: str, name: str, role: AgentRole):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.capabilities: List[AgentCapability] = []
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.is_active = True
        self.last_activity = datetime.now()
        self.performance_stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_response_time": 0.0,
            "success_rate": 1.0,
        }

    async def start(self):
        """Iniciar el agente"""
        self.is_active = True
        logger.info(f"Agente {self.name} ({self.agent_id}) iniciado")

    async def stop(self):
        """Detener el agente"""
        self.is_active = False
        logger.info(f"Agente {self.name} ({self.agent_id}) detenido")

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Procesar un mensaje recibido"""
        raise NotImplementedError("Subclasses must implement process_message")

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Ejecutar una tarea asignada"""
        raise NotImplementedError("Subclasses must implement execute_task")

    def get_profile(self) -> AgentProfile:
        """Obtener perfil del agente"""
        return AgentProfile(
            agent_id=self.agent_id,
            name=self.name,
            role=self.role,
            capabilities=self.capabilities,
            performance_metrics=self.performance_stats,
            is_active=self.is_active,
            last_seen=self.last_activity,
        )

    def update_performance_stats(self, task_duration: float, success: bool):
        """Actualizar estadísticas de rendimiento"""
        self.performance_stats["tasks_completed"] += 1 if success else 0
        self.performance_stats["tasks_failed"] += 0 if success else 1

        # Actualizar tiempo promedio de respuesta
        current_avg = self.performance_stats["average_response_time"]
        total_tasks = (
            self.performance_stats["tasks_completed"]
            + self.performance_stats["tasks_failed"]
        )

        if total_tasks > 0:
            self.performance_stats["average_response_time"] = (
                (current_avg * (total_tasks - 1)) + task_duration
            ) / total_tasks

        # Actualizar tasa de éxito
        if total_tasks > 0:
            self.performance_stats["success_rate"] = (
                self.performance_stats["tasks_completed"] / total_tasks
            )


# =============================================================================
# AGENTE COORDINADOR (HIERARCHICAL PATTERN)
# =============================================================================


class CoordinatorAgent(MultiAgentBase):
    """Agente coordinador que gestiona y asigna tareas a agentes especializados"""

    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name, AgentRole.COORDINATOR)
        self.registered_agents: Dict[str, AgentProfile] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.coordination_strategy = (
            "load_balanced"  # "load_balanced", "specialization", "performance"
        )
        self.pending_messages: List[Tuple[str, AgentMessage]] = []  # Queue for messages

    async def start(self):
        await super().start()
        # Iniciar procesamiento de tareas
        asyncio.create_task(self._process_task_queue())

    async def register_agent(self, agent_profile: AgentProfile):
        """Registrar un agente en el sistema"""
        self.registered_agents[agent_profile.agent_id] = agent_profile
        logger.info(
            f"Agente registrado: {agent_profile.name} ({agent_profile.agent_id})"
        )

    async def unregister_agent(self, agent_id: str):
        """Desregistrar un agente"""
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]
            logger.info(f"Agente desregistrado: {agent_id}")

    async def submit_task(self, task: Task) -> str:
        """Enviar una tarea para procesamiento"""
        await self.task_queue.put(task)
        self.active_tasks[task.task_id] = task
        logger.info(f"Tarea enviada: {task.task_id} - {task.description}")
        return task.task_id

    async def _process_task_queue(self):
        """Procesar cola de tareas"""
        while self.is_active:
            try:
                task = await self.task_queue.get()

                # Asignar agente apropiado
                assigned_agent = await self._assign_task_to_agent(task)

                if assigned_agent:
                    # Enviar tarea al agente asignado
                    await self._send_task_to_agent(task, assigned_agent)
                else:
                    # Marcar tarea como fallida si no hay agente disponible
                    task.status = TaskStatus.FAILED
                    task.error_message = "No suitable agent available"
                    logger.warning(f"No se pudo asignar tarea {task.task_id}")

                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"Error procesando tarea: {e}")

    async def _assign_task_to_agent(self, task: Task) -> Optional[str]:
        """Asignar tarea al agente más apropiado"""
        if self.coordination_strategy == "load_balanced":
            return await self._assign_load_balanced(task)
        elif self.coordination_strategy == "specialization":
            return await self._assign_by_specialization(task)
        elif self.coordination_strategy == "performance":
            return await self._assign_by_performance(task)
        else:
            return await self._assign_load_balanced(task)

    async def _assign_load_balanced(self, task: Task) -> Optional[str]:
        """REAL load-balanced assignment based on actual agent metrics"""
        available_agents = [
            agent_id
            for agent_id, profile in self.registered_agents.items()
            if profile.is_active
        ]

        if not available_agents:
            return None

        # Get REAL load metrics for each agent
        agent_loads = []
        for agent_id in available_agents:
            profile = self.registered_agents[agent_id]
            
            # Calculate load score based on real metrics
            # Lower score = less loaded = better choice
            
            # Factor 1: Current active tasks (if available)
            current_tasks = profile.performance_metrics.get("current_tasks", 0)
            max_tasks = profile.performance_metrics.get("max_concurrent_tasks", 10)
            task_load = current_tasks / max_tasks if max_tasks > 0 else 0.0
            
            # Factor 2: Recent performance (success rate)
            success_rate = profile.performance_metrics.get("success_rate", 1.0)
            performance_factor = 1.0 - success_rate  # Lower is better (inverse)
            
            # Factor 3: Average response time (normalized)
            avg_response_time = profile.performance_metrics.get("average_response_time", 1.0)
            # Normalize: assume 10s is max acceptable, scale to 0-1
            time_factor = min(1.0, avg_response_time / 10.0)
            
            # Combined load score (weighted)
            load_score = (
                task_load * 0.5 +      # Current load is most important
                performance_factor * 0.3 +  # Performance matters
                time_factor * 0.2      # Response time matters
            )
            
            agent_loads.append((agent_id, load_score, profile))
        
        # Sort by load score (lowest first = least loaded)
        agent_loads.sort(key=lambda x: x[1])
        
        # Select agent with lowest load
        best_agent_id, best_load_score, best_profile = agent_loads[0]
        
        logger.debug(
            f"Load-balanced assignment: {task.task_id} -> {best_agent_id} "
            f"(load_score: {best_load_score:.3f})"
        )
        
        return best_agent_id

    async def _assign_by_specialization(self, task: Task) -> Optional[str]:
        """Asignación por especialización"""
        task_keywords = task.description.lower().split()

        best_agent = None
        best_score = 0

        for agent_id, profile in self.registered_agents.items():
            if not profile.is_active:
                continue

            # Calcular coincidencia con dominios de especialización
            score = 0
            for keyword in task_keywords:
                for domain in profile.specialization_domains:
                    if keyword in domain.lower():
                        score += 1

            if score > best_score:
                best_score = score
                best_agent = agent_id

        return best_agent

    async def _assign_by_performance(self, task: Task) -> Optional[str]:
        """Asignación por rendimiento"""
        best_agent = None
        best_score = 0

        for agent_id, profile in self.registered_agents.items():
            if not profile.is_active:
                continue

            # Usar métricas de rendimiento para decidir
            success_rate = profile.performance_metrics.get("success_rate", 0.5)
            avg_response_time = profile.performance_metrics.get(
                "avg_response_time", 10.0
            )

            # Score combinado: éxito alto + tiempo de respuesta bajo
            score = success_rate * (1.0 / (1.0 + avg_response_time))

            if score > best_score:
                best_score = score
                best_agent = agent_id

        return best_agent

    async def _send_task_to_agent(self, task: Task, agent_id: str):
        """REAL task sending to agent - actual message delivery"""
        message = AgentMessage(
            message_id=f"task_{task.task_id}_{int(time.time())}",
            sender_id=self.agent_id,
            receiver_id=agent_id,
            message_type="task_assignment",
            content={"task": task.__dict__},
            correlation_id=task.task_id,
        )

        # REAL message sending - find agent and send message
        if agent_id in self.registered_agents:
            agent_profile = self.registered_agents[agent_id]
            
            # Get actual agent instance if available
            agent_instance = getattr(agent_profile, 'instance', None)
            if agent_instance and hasattr(agent_instance, 'process_message'):
                # Send message directly to agent
                try:
                    response = await agent_instance.process_message(message)
                    if response:
                        # Handle response if agent sends one back
                        await self.process_message(response)
                    logger.info(f"✅ Tarea {task.task_id} enviada y procesada por agente {agent_id}")
                except Exception as e:
                    logger.error(f"Error sending task to agent {agent_id}: {e}")
                    raise
            else:
                # Fallback: use message bus if available
                if hasattr(self, 'message_bus') and self.message_bus:
                    await self.message_bus.publish(message)
                    logger.info(f"✅ Tarea {task.task_id} enviada a agente {agent_id} vía message bus")
                else:
                    # Last resort: queue for processing
                    if not hasattr(self, 'pending_messages'):
                        self.pending_messages = []
                    self.pending_messages.append((agent_id, message))
                    logger.warning(f"⚠️ Agente {agent_id} no tiene process_message, mensaje en cola")
        else:
            logger.error(f"❌ Agente {agent_id} no encontrado en registro")
            raise ValueError(f"Agent {agent_id} not found in registry")

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Procesar mensajes del coordinador"""
        if message.message_type == "task_completed":
            task_id = message.correlation_id
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.output_data = message.content.get("result")
                task.completed_at = datetime.now()

                logger.info(f"Tarea completada: {task_id}")

        elif message.message_type == "task_failed":
            task_id = message.correlation_id
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.FAILED
                task.error_message = message.content.get("error")
                task.completed_at = datetime.now()

                logger.warning(f"Tarea fallida: {task_id} - {task.error_message}")

        return None

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """El coordinador no ejecuta tareas directamente"""
        return {"error": "Coordinator agent does not execute tasks directly"}


# =============================================================================
# AGENTE ESPECIALIZADO
# =============================================================================


class SpecializedAgent(MultiAgentBase):
    """Agente especializado en una tarea específica"""

    def __init__(self, agent_id: str, name: str, specialization: str):
        super().__init__(agent_id, name, AgentRole.SPECIALIST)
        self.specialization = specialization
        self.coordinator_id: Optional[str] = None

    async def start(self):
        await super().start()
        # Registrar capacidades basadas en especialización
        await self._register_capabilities()

    async def _register_capabilities(self):
        """Registrar capacidades del agente especializado"""
        if self.specialization == "code_analysis":
            self.capabilities.append(
                AgentCapability(
                    name="analyze_code",
                    description="Analizar código fuente para calidad y bugs",
                    input_schema={
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"issues": {"type": "array"}},
                    },
                )
            )
        elif self.specialization == "data_processing":
            self.capabilities.append(
                AgentCapability(
                    name="process_data",
                    description="Procesar y analizar datos",
                    input_schema={
                        "type": "object",
                        "properties": {"data": {"type": "array"}},
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"insights": {"type": "object"}},
                    },
                )
            )
        # Añadir más especializaciones según sea necesario

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Procesar mensajes del agente especializado"""
        if message.message_type == "task_assignment":
            task_data = message.content.get("task", {})
            task = Task(**task_data)

            # Ejecutar tarea
            try:
                result = await self.execute_task(task)

                # Enviar resultado de vuelta al coordinador
                response_message = AgentMessage(
                    message_id=f"result_{task.task_id}_{int(time.time())}",
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type="task_completed",
                    content={"result": result},
                    correlation_id=task.task_id,
                )

                return response_message

            except Exception as e:
                # Enviar error de vuelta
                error_message = AgentMessage(
                    message_id=f"error_{task.task_id}_{int(time.time())}",
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type="task_failed",
                    content={"error": str(e)},
                    correlation_id=task.task_id,
                )

                return error_message

        return None

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Ejecutar tarea especializada"""
        start_time = time.time()

        try:
            with trace_agent_execution(
                self.name, f"execute_task_{task.task_id}"
            ) as trace:
                trace.add_event("task_started", {"task_description": task.description})

                # REAL processing based on specialization
                if self.specialization == "code_analysis":
                    result = await self._analyze_code(task.input_data)
                elif self.specialization == "data_processing":
                    result = await self._process_data(task.input_data)
                else:
                    # Generic task execution - process input data based on type
                    result = await self._execute_generic_task(task.input_data, self.specialization)

                trace.add_event("task_completed", {"result_keys": list(result.keys())})

                # Actualizar estadísticas
                duration = time.time() - start_time
                self.update_performance_stats(duration, True)

                return result

        except Exception as e:
            duration = time.time() - start_time
            self.update_performance_stats(duration, False)
            raise e

    async def _analyze_code(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """REAL code analysis - comprehensive static analysis"""
        import ast
        import re
        
        code = input_data.get("code", "")
        if not code:
            return {"issues": [], "complexity_score": 0.0, "error": "No code provided"}
        
        issues = []
        complexity_factors = 0
        
        try:
            # REAL analysis: Parse AST for structure
            try:
                tree = ast.parse(code)
                
                # Count complexity factors
                complexity_factors += len(list(ast.walk(tree)))
                
                # Find function definitions
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                complexity_factors += len(functions) * 2
                
                # Find control flow (if, for, while, try)
                control_flow = [
                    node for node in ast.walk(tree)
                    if isinstance(node, (ast.If, ast.For, ast.While, ast.Try))
                ]
                complexity_factors += len(control_flow) * 3
                
                # Check for nested structures
                max_depth = self._calculate_ast_depth(tree)
                if max_depth > 5:
                    issues.append({
                        "type": "warning",
                        "message": f"Deep nesting detected (depth: {max_depth})",
                        "severity": "medium"
                    })
                
            except SyntaxError as e:
                issues.append({
                    "type": "error",
                    "message": f"Syntax error: {e.msg} at line {e.lineno}",
                    "severity": "high"
                })
                return {"issues": issues, "complexity_score": 0.0, "syntax_error": True}
            
            # REAL pattern analysis
            # Check for TODO/FIXME comments
            todo_pattern = r'(TODO|FIXME|XXX|HACK):?\s*(.+)'
            todos = re.findall(todo_pattern, code, re.IGNORECASE)
            for todo_type, todo_msg in todos:
                issues.append({
                    "type": "info",
                    "message": f"{todo_type} comment: {todo_msg.strip()}",
                    "severity": "low"
                })
            
            # Check for long lines
            lines = code.split('\n')
            long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
            if long_lines:
                issues.append({
                    "type": "warning",
                    "message": f"Long lines detected: {len(long_lines)} lines exceed 120 characters",
                    "severity": "low",
                    "line_numbers": long_lines[:10]  # Limit to first 10
                })
            
            # Check for large functions
            if functions:
                large_functions = [
                    f.name for f in functions
                    if f.end_lineno and f.lineno and (f.end_lineno - f.lineno) > 50
                ]
                if large_functions:
                    issues.append({
                        "type": "warning",
                        "message": f"Large functions detected: {', '.join(large_functions)}",
                        "severity": "medium"
                    })
            
            # Check for potential security issues
            dangerous_patterns = [
                (r'eval\s*\(', "Use of eval() - security risk"),
                (r'exec\s*\(', "Use of exec() - security risk"),
                (r'__import__\s*\(', "Dynamic import - potential security risk"),
                (r'pickle\.(loads?|dumps?)', "Pickle usage - security risk if loading untrusted data"),
            ]
            
            for pattern, message in dangerous_patterns:
                if re.search(pattern, code):
                    issues.append({
                        "type": "error",
                        "message": message,
                        "severity": "high"
                    })
            
            # Calculate complexity score (normalized)
            complexity_score = min(1.0, complexity_factors / 100.0)
            
            return {
                "issues": issues,
                "complexity_score": round(complexity_score, 3),
                "statistics": {
                    "total_lines": len(lines),
                    "functions": len(functions),
                    "control_flow_statements": len(control_flow),
                    "max_nesting_depth": max_depth
                }
            }
            
        except Exception as e:
            logger.error(f"Error in code analysis: {e}")
            return {
                "issues": [{"type": "error", "message": f"Analysis error: {str(e)}"}],
                "complexity_score": 0.0,
                "error": str(e)
            }
    
    def _calculate_ast_depth(self, node, depth=0):
        """Calculate maximum AST depth"""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._calculate_ast_depth(child, depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth

    async def _process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """REAL data processing - comprehensive statistical analysis"""
        import statistics
        
        data = input_data.get("data", [])
        if not data:
            return {
                "insights": {
                    "total_records": 0,
                    "error": "No data provided"
                }
            }
        
        try:
            # REAL statistical analysis
            if not isinstance(data, list):
                data = list(data) if hasattr(data, '__iter__') else [data]
            
            numeric_data = []
            for item in data:
                try:
                    # Try to convert to numeric
                    if isinstance(item, (int, float)):
                        numeric_data.append(float(item))
                    elif isinstance(item, str):
                        # Try to parse as number
                        try:
                            numeric_data.append(float(item))
                        except ValueError:
                            pass
                except (ValueError, TypeError):
                    continue
            
            insights = {
                "total_records": len(data),
                "numeric_records": len(numeric_data),
            }
            
            if numeric_data:
                # REAL statistical calculations
                insights.update({
                    "average_value": round(statistics.mean(numeric_data), 4),
                    "median_value": round(statistics.median(numeric_data), 4),
                    "max_value": max(numeric_data),
                    "min_value": min(numeric_data),
                    "range": max(numeric_data) - min(numeric_data),
                })
                
                # Standard deviation if enough data
                if len(numeric_data) > 1:
                    insights["std_deviation"] = round(statistics.stdev(numeric_data), 4)
                
                # Percentiles
                sorted_data = sorted(numeric_data)
                n = len(sorted_data)
                insights["percentiles"] = {
                    "p25": sorted_data[int(n * 0.25)] if n > 0 else None,
                    "p50": sorted_data[int(n * 0.50)] if n > 0 else None,
                    "p75": sorted_data[int(n * 0.75)] if n > 0 else None,
                    "p90": sorted_data[int(n * 0.90)] if n > 0 else None,
                }
                
                # Data distribution analysis
                if len(numeric_data) > 10:
                    # Check for outliers (using IQR method)
                    q1 = sorted_data[int(n * 0.25)]
                    q3 = sorted_data[int(n * 0.75)]
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = [x for x in numeric_data if x < lower_bound or x > upper_bound]
                    insights["outliers_count"] = len(outliers)
                    if outliers:
                        insights["outliers"] = sorted(outliers)[:10]  # First 10
            else:
                # Non-numeric data analysis
                insights["data_type"] = "non_numeric"
                if data:
                    # Count unique values
                    unique_values = len(set(str(item) for item in data))
                    insights["unique_values"] = unique_values
                    insights["duplication_rate"] = round(1.0 - (unique_values / len(data)), 4)
            
            return {"insights": insights}
            
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            return {
                "insights": {
                    "total_records": len(data) if data else 0,
                    "error": str(e)
                }
            }
    
    async def _execute_generic_task(self, input_data: Dict[str, Any], specialization: str) -> Dict[str, Any]:
        """REAL generic task execution - processes any input based on specialization"""
        try:
            # Analyze input data structure
            result = {
                "specialization": specialization,
                "input_type": type(input_data).__name__,
                "input_keys": list(input_data.keys()) if isinstance(input_data, dict) else None,
                "processed_at": datetime.now().isoformat(),
            }
            
            # Process based on input type
            if isinstance(input_data, dict):
                # Process dictionary input
                result["data_summary"] = {
                    "total_keys": len(input_data),
                    "key_types": {k: type(v).__name__ for k, v in list(input_data.items())[:10]},
                }
                
                # Try to extract meaningful information
                if "text" in input_data:
                    text = str(input_data["text"])
                    result["text_analysis"] = {
                        "length": len(text),
                        "word_count": len(text.split()),
                        "has_sentences": "." in text or "!" in text or "?" in text,
                    }
                
                if "data" in input_data:
                    data = input_data["data"]
                    if isinstance(data, (list, tuple)):
                        result["data_analysis"] = {
                            "count": len(data),
                            "sample": data[:5] if len(data) > 5 else data,
                        }
            
            elif isinstance(input_data, (list, tuple)):
                result["list_analysis"] = {
                    "count": len(input_data),
                    "element_types": list(set(type(item).__name__ for item in input_data[:10])),
                    "sample": input_data[:5] if len(input_data) > 5 else input_data,
                }
            
            elif isinstance(input_data, str):
                result["string_analysis"] = {
                    "length": len(input_data),
                    "word_count": len(input_data.split()),
                    "is_json": False,
                }
                # Try to parse as JSON
                try:
                    parsed = json.loads(input_data)
                    result["string_analysis"]["is_json"] = True
                    result["parsed_content"] = parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            
            result["status"] = "completed"
            return result
            
        except Exception as e:
            logger.error(f"Error in generic task execution: {e}")
            return {
                "specialization": specialization,
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }


# =============================================================================
# SISTEMA MULTI-AGENTE PRINCIPAL
# =============================================================================


class MultiAgentSystem:
    """Sistema principal de agentes múltiples"""

    def __init__(self):
        self.coordinator: Optional[CoordinatorAgent] = None
        self.agents: Dict[str, MultiAgentBase] = {}
        self.message_bus: asyncio.Queue = asyncio.Queue()
        self.is_running = False

    async def initialize(self):
        """Inicializar el sistema multi-agente"""
        logger.info("Inicializando sistema multi-agente...")

        # Crear coordinador
        self.coordinator = CoordinatorAgent("coordinator_001", "Master Coordinator")
        await self.coordinator.start()

        # Registrar coordinador como agente
        self.agents[self.coordinator.agent_id] = self.coordinator

        # Iniciar procesamiento de mensajes
        self.is_running = True
        asyncio.create_task(self._process_message_bus())

        logger.info("Sistema multi-agente inicializado")

    async def shutdown(self):
        """Apagar el sistema multi-agente"""
        logger.info("Apagando sistema multi-agente...")

        self.is_running = False

        # Detener todos los agentes
        for agent in self.agents.values():
            await agent.stop()

        self.agents.clear()
        logger.info("Sistema multi-agente apagado")

    async def register_agent(self, agent: MultiAgentBase):
        """Registrar un agente en el sistema"""
        self.agents[agent.agent_id] = agent
        await agent.start()

        # Registrar en el coordinador
        if self.coordinator:
            await self.coordinator.register_agent(agent.get_profile())

        logger.info(f"Agente registrado: {agent.name} ({agent.agent_id})")

    async def submit_task(
        self, description: str, input_data: Dict[str, Any] = None, priority: int = 1
    ) -> str:
        """Enviar una tarea al sistema"""
        task = Task(
            task_id=f"task_{int(time.time())}_{len(self.coordinator.active_tasks) if self.coordinator else 0}",
            description=description,
            input_data=input_data or {},
            priority=priority,
        )

        if self.coordinator:
            return await self.coordinator.submit_task(task)
        else:
            raise RuntimeError("Coordinator not initialized")

    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Obtener estado de una tarea"""
        if self.coordinator and task_id in self.coordinator.active_tasks:
            return self.coordinator.active_tasks[task_id]
        return None

    async def send_message(self, message: AgentMessage):
        """Enviar mensaje entre agentes"""
        await self.message_bus.put(message)

    async def _process_message_bus(self):
        """Procesar bus de mensajes"""
        while self.is_running:
            try:
                message = await self.message_bus.get()

                # Encontrar agente receptor
                if message.receiver_id in self.agents:
                    receiver = self.agents[message.receiver_id]

                    # Procesar mensaje
                    response = await receiver.process_message(message)

                    # Enviar respuesta si existe
                    if response:
                        await self.send_message(response)

                self.message_bus.task_done()

            except Exception as e:
                logger.error(f"Error procesando mensaje: {e}")

    async def evaluate_system_performance(self) -> Dict[str, Any]:
        """Evaluar rendimiento del sistema multi-agente"""
        if not self.coordinator:
            return {"error": "Coordinator not initialized"}

        # Recopilar métricas de todos los agentes
        system_metrics = {
            "total_agents": len(self.agents),
            "active_agents": sum(
                1 for agent in self.agents.values() if agent.is_active
            ),
            "total_tasks": len(self.coordinator.active_tasks),
            "completed_tasks": sum(
                1
                for task in self.coordinator.active_tasks.values()
                if task.status == TaskStatus.COMPLETED
            ),
            "failed_tasks": sum(
                1
                for task in self.coordinator.active_tasks.values()
                if task.status == TaskStatus.FAILED
            ),
            "agent_performance": {},
        }

        # Métricas por agente
        for agent_id, agent in self.agents.items():
            system_metrics["agent_performance"][agent_id] = {
                "name": agent.name,
                "role": agent.role.value,
                "performance_stats": agent.performance_stats,
                "is_active": agent.is_active,
            }

        # Calcular métricas globales
        total_tasks = system_metrics["total_tasks"]
        if total_tasks > 0:
            system_metrics["task_success_rate"] = (
                system_metrics["completed_tasks"] / total_tasks
            )
        else:
            system_metrics["task_success_rate"] = 0.0

        return system_metrics


# =============================================================================
# FUNCIONES DE UTILIDAD Y EJEMPLOS
# =============================================================================

# Instancia global del sistema multi-agente
multi_agent_system = MultiAgentSystem()


async def initialize_multi_agent_system():
    """Inicializar sistema multi-agente con agentes de ejemplo"""
    await multi_agent_system.initialize()

    # Crear agentes especializados de ejemplo
    code_agent = SpecializedAgent("code_001", "Code Analyzer", "code_analysis")
    data_agent = SpecializedAgent("data_001", "Data Processor", "data_processing")

    # Registrar agentes
    await multi_agent_system.register_agent(code_agent)
    await multi_agent_system.register_agent(data_agent)

    logger.info("Sistema multi-agente inicializado con agentes de ejemplo")


async def demo_multi_agent_workflow():
    """Demostración de workflow multi-agente"""
    # Inicializar sistema
    await initialize_multi_agent_system()

    # Enviar tarea de análisis de código
    task_id = await multi_agent_system.submit_task(
        "Analyze the following Python code for potential issues",
        {
            "code": """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers) if numbers else 0

# TODO: Add error handling
result = calculate_average([1, 2, 3, 4, 5])
print(result)
"""
        },
    )

    # Esperar un poco para procesamiento
    await asyncio.sleep(2)

    # Verificar estado
    status = await multi_agent_system.get_task_status(task_id)
    if status:
        print(f"Task {task_id} status: {status.status.value}")
        if status.output_data:
            print(f"Result: {status.output_data}")

    # Evaluar rendimiento del sistema
    performance = await multi_agent_system.evaluate_system_performance()
    print(f"System performance: {performance}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Clases principales
    "MultiAgentBase",
    "CoordinatorAgent",
    "SpecializedAgent",
    "MultiAgentSystem",
    # Modelos de datos
    "AgentRole",
    "CommunicationProtocol",
    "TaskStatus",
    "AgentCapability",
    "AgentProfile",
    "Task",
    "AgentMessage",
    # Instancia global
    "multi_agent_system",
    # Funciones de utilidad
    "initialize_multi_agent_system",
    "demo_multi_agent_workflow",
]

# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Multi-Agent System"
