#!/usr/bin/env python3
"""
REFLEXIÓN AGENT MCP - Auto-Corrección Iterativa
===============================================

Agent inteligente que:
- Critica outputs MCP automaticamente
- Genera versiones mejoradas iterativamente
- Almacena memoria episódica para aprendizaje continuo
- Ejecuta reflexion loops hasta calidad optima

Funciona completamente real con modelos LLM existentes
"""

import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import sistema LLM polymorphic existente (funciona con cualquier LLM)
try:
    from apps.backend.src.core.llm.llm_factory import LLMFactory

    LLM_SYSTEM_AVAILABLE = True
    print("✅ Reflexion Agent: Sistema LLM polymorphic disponible")
except ImportError:
    print("⚠️ Reflexion Agent: Sistema LLM polymorphic no disponible")
    LLM_SYSTEM_AVAILABLE = False


@dataclass
class EpisodicMemory:
    """Memoria episódica para aprendizaje reflexivo MCP"""

    session_id: str
    timestamp: str
    action_type: str
    input_data: Dict[str, Any]
    original_output: Any
    criticism: str
    improved_output: Any
    improvement_score: float  # -1.0 a 1.0
    reflection_depth: int
    feedback_source: str  # 'llm', 'rule_based', 'user_review'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflexionSession:
    """Sesión completa de reflexión iterativa"""

    session_id: str
    start_time: datetime
    original_input: Dict[str, Any]
    current_output: Any
    iteration_count: int = 0
    max_iterations: int = 5
    convergence_threshold: float = 0.85
    criticism_history: List[str] = field(default_factory=list)
    output_history: List[Any] = field(default_factory=list)
    memory_entries: List[EpisodicMemory] = field(default_factory=list)
    final_status: str = "in_progress"


class ReflexionAgent:
    """Agent de reflexión que mejora iterativamente outputs MCP"""

    def __init__(
        self,
        memory_dir: str = "logs/episodic_memory",
        use_llm: bool = True,
    ):

        # MCP interface attributes
        from sheily_core.agents.base.base_agent import AgentCapability

        self.agent_name = "ReflexionAgent"
        self.agent_id = f"reflex_{self.agent_name.lower()}"
        self.message_bus = None
        self.task_queue = []
        self.capabilities = [AgentCapability.ANALYSIS, AgentCapability.EXECUTION]
        self.status = "active"

        # Agent specific attributes
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.use_llm = use_llm and LLM_SYSTEM_AVAILABLE
        self.llm_system = None

        # Inicializar sistema LLM polymorphic si disponible
        if self.use_llm:
            try:
                self.llm_system = LLMFactory.create_llm()
                if self.llm_system:
                    print("✅ Reflexion Agent: Sistema LLM polymorphic inicializado")
                else:
                    print("⚠️ Reflexion Agent: Error inicializando LLM polymorphic")
                    self.use_llm = False
            except Exception as e:
                print(f"⚠️ Reflexion Agent: Error inicializando LLM: {e}")
                self.use_llm = False
        else:
            print("⚠️ Reflexion Agent: LLM reasoning no habilitado")

        self.criticism_database = defaultdict(list)
        self.session_counter = 0

        print("✅ Reflexion Agent: Inicializado y listo para auto-corrección iterativa")

    async def initialize(self):
        """Inicializar agente MCP"""
        print("🔄 ReflexionAgent: Inicializado")
        return True

    def set_message_bus(self, bus):
        """Configurar message bus"""
        self.message_bus = bus

    def add_task_to_queue(self, task):
        """Agregar tarea a cola"""
        self.task_queue.append(task)

    async def execute_task(self, task):
        """Ejecutar tarea de reflexión"""
        try:
            if task.task_type == "reflect_and_improve":
                return await self.reflect_and_improve(
                    task.parameters.get("output", ""),
                    task.parameters.get("context", {}),
                    task.parameters.get("max_iterations", 3)
                )
            else:
                return {"success": False, "error": f"Tipo de tarea desconocido: {task.task_type}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def handle_message(self, message):
        """Manejar mensaje recibido"""
        # Implementación básica de handling de mensajes
        pass

    def get_status(self):
        """Obtener estado del agente"""
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "llm_available": self.use_llm,
            "memory_sessions": len(list(self.memory_dir.glob("session_*.json"))),
            "tasks_queued": len(self.task_queue),
            "capabilities": [cap.value for cap in self.capabilities]
        }

    async def reflect_and_improve(
        self,
        action_output: Any,
        action_context: Dict[str, Any],
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Función principal: Reflect y mejora outputs automaticamente

        Args:
            action_output: Output original MCP para mejorar
            action_context: Contexto de la acción original
            max_iterations: Máximo número de iteraciones reflexivas

        Returns:
            Dict con output mejorado, metadata de reflexión, score de mejora
        """

        session_id = f"reflexion_{self.session_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_counter += 1

        session = ReflexionSession(
            session_id=session_id,
            start_time=datetime.now(),
            original_input=action_context,
            current_output=action_output,
            max_iterations=max_iterations,
        )

        try:
            session = await self._run_reflection_loop(session)

            # Calcular score de mejora final
            improvement_score = self._calculate_overall_improvement(session)

            return {
                "improved_output": session.current_output,
                "session_id": session_id,
                "iteration_count": session.iteration_count,
                "improvement_score": improvement_score,
                "criticism_history": session.criticism_history,
                "memory_entries": len(session.memory_entries),
                "convergence_reached": session.final_status == "converged",
                "reflection_metadata": {
                    "total_criticisms": len(session.criticism_history),
                    "average_improvement_per_iteration": improvement_score
                    / max(session.iteration_count, 1),
                    "reflection_depth": max(
                        [m.reflection_depth for m in session.memory_entries], default=1
                    ),
                },
            }

        except Exception as e:
            return {
                "improved_output": action_output,  # Fallback to original
                "session_id": session_id,
                "error": f"Reflexion falló: {e}",
                "iteration_count": 0,
                "improvement_score": 0.0,
                "criticism_history": [],
                "convergence_reached": False,
            }

    async def _run_reflection_loop(self, session: ReflexionSession) -> ReflexionSession:
        """Ejecuta loop de reflexión iterativa hasta convergencia"""

        for iteration in range(session.max_iterations):
            session.iteration_count = iteration + 1

            # Paso 1: Criticizar output actual
            criticism = await self._generate_criticism(
                session.current_output, session.original_input, iteration
            )

            if not criticism or criticism.strip() == "":
                # No hay críticas adicionales, convergencia alcanzada
                session.final_status = "converged"
                break

            session.criticism_history.append(criticism)

            # Paso 2: Generar output mejorado
            improved_output = await self._generate_improvement(
                session.current_output, criticism, session.original_input
            )

            session.output_history.append(session.current_output)

            # Paso 3: Evaluar mejora
            improvement_score = await self._evaluate_improvement(
                session.current_output, improved_output, criticism
            )

            # Paso 4: Save en memoria episódica
            memory_entry = EpisodicMemory(
                session_id=session.session_id,
                timestamp=datetime.now().isoformat(),
                action_type=session.original_input.get("action_type", "unknown"),
                input_data=session.original_input,
                original_output=session.current_output,
                criticism=criticism,
                improved_output=improved_output,
                improvement_score=improvement_score,
                reflection_depth=iteration + 1,
                feedback_source="llm" if self.use_llm else "rule_based",
            )

            session.memory_entries.append(memory_entry)

            # Paso 5: Actualizar output actual
            session.current_output = improved_output

            # Paso 6: Check convergencia
            if improvement_score > session.convergence_threshold:
                session.final_status = "converged"
                break

            # Paso 7: Delay breve para reflexión
            await asyncio.sleep(0.1)

        if session.iteration_count >= session.max_iterations:
            session.final_status = "max_iterations_reached"
        elif session.final_status != "converged":
            session.final_status = "completed"

        # Save sesión completa a archivo
        await self._save_session_memory(session)

        return session

    async def _generate_criticism(
        self, current_output: Any, original_input: Dict[str, Any], iteration: int
    ) -> str:
        """Genera crítica constructiva del output actual"""

        if not self.use_llm:
            # Criticism rule-based básico
            return self._rule_based_criticism(current_output, original_input, iteration)

        # Criticism usando LLM polymorphic real
        prompt = f"""
        CRÍTICA CONSTRUCTIVA DEL OUTPUT SIGUIENTE:

        CONTEXTO ORIGINAL:
        - Tipo de acción: {original_input.get('action_type', 'desconocido')}
        - Descripción: {original_input.get('description', 'No disponible')}
        - Iteration actual: {iteration + 1}

        OUTPUT ACTUAL:
        {json.dumps(current_output, indent=2, ensure_ascii=False) if isinstance(current_output, (dict, list)) else str(current_output)}

        INSTRUCCIONES PARA CRÍTICA:
        - Identifica problemas específicos y áreas de mejora concretas
        - Sugiere cambios específicos que mejorarían el output
        - Sé constructivo y específico, no genérico
        - Si el output está bien, di "No hay críticas adicionales significa excelente trabajo"
        - Considera calidad, completitud, utilidad, y adherencia a principios éticos

        CRÍTICA:
        """

        try:
            response = await self.llm_system.generate_response(prompt)
            return response.strip()
        except Exception as e:
            print(f"⚠️ Error generando crítica LLM: {e}")
            return self._rule_based_criticism(current_output, original_input, iteration)

    def _rule_based_criticism(
        self, current_output: Any, original_input: Dict[str, Any], iteration: int
    ) -> str:
        """Crítica básica usando reglas estáticas"""

        criticisms = []

        action_type = original_input.get("action_type", "")

        # Críticas específicas por tipo
        if action_type == "audit_report" and isinstance(current_output, dict):
            if "findings" not in current_output:
                criticisms.append("Report faltan sección 'findings' detallada")
            if len(current_output.get("findings", [])) < 3:
                criticisms.append(
                    "Report tiene muy pocos findings - asegurarse de análisis completo"
                )
            if "recommendations" not in current_output:
                criticisms.append("Report faltan recomendaciones específicas de mejora")

        elif action_type == "code_generation" and isinstance(current_output, str):
            if len(current_output) < 50:
                criticisms.append(
                    "Código generado es demasiado corto - faltan implementaciones"
                )
            if (
                "error" in current_output.lower()
                and "handling" not in current_output.lower()
            ):
                criticisms.append("Código no maneja errores apropiadamente")

        elif action_type == "model_training" and isinstance(current_output, dict):
            if "accuracy" in current_output and current_output.get("accuracy", 0) < 0.7:
                criticisms.append(
                    "Accuracy del modelo es baja - considerar mejoras arquitecturales"
                )

        # Críticas generales
        if len(str(current_output)) < 100:
            criticisms.append(
                "Output es muy breve - necesita más detalle y profundidad"
            )

        if iteration > 2:
            criticisms.append(
                "Después de múltiples iteraciones considere cambios más radicales"
            )
        elif iteration == 0:
            criticisms.append(
                "Primera versión siempre puede ser mejorada con detalle adicional"
            )

        return (
            ". ".join(criticisms)
            if criticisms
            else "No hay críticas específicas - output aceptable"
        )

    async def _generate_improvement(
        self, current_output: Any, criticism: str, original_input: Dict[str, Any]
    ) -> Any:
        """Genera versión mejorada del output basada en crítica"""

        if not self.use_llm:
            return self._rule_based_improvement(
                current_output, criticism, original_input
            )

        # Improvement usando LLM polymorphic real
        prompt = f"""
        MEJORA EL OUTPUT SIGUIENTE BASADO EN LA CRÍTICA PROPORCIONADA:

        CONTEXTO ORIGINAL:
        - Tipo: {original_input.get('action_type', 'desconocido')}
        - Descripción: {original_input.get('description', 'No disponible')}

        OUTPUT ACTUAL:
        {json.dumps(current_output, indent=2, ensure_ascii=False) if isinstance(current_output, (dict, list)) else str(current_output)}

        CRÍTICA A DIRECCIONAR:
        {criticism}

        INSTRUCCIONES PARA MEJORA:
        - Dirige cada punto de crítica específicamente
        - Mantén la estructura original si es apropiada
        - Añade mejoras concretas, no cambios cosméticos
        - Asegura que el output mejorado sea más útil y completo
        - Si la crítica es positiva, realiza mejoras menores y útiles

        OUTPUT MEJORADO:
        """

        try:
            response = await self.llm_system.generate_response(prompt)

            # Intentar parsear JSON si el output original era dict/list
            if isinstance(current_output, (dict, list)):
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # Si no es JSON válido, devolver como string procesado
                    pass

            return response.strip()

        except Exception as e:
            print(f"⚠️ Error generando mejora LLM: {e}")
            return self._rule_based_improvement(
                current_output, criticism, original_input
            )

    def _rule_based_improvement(
        self, current_output: Any, criticism: str, original_input: Dict[str, Any]
    ) -> Any:
        """Mejora básica usando reglas estáticas"""

        improved = current_output

        if isinstance(improved, dict):
            improved_copy = dict(improved)

            # Mejoras comunes para reports
            if "findings" in criticism.lower():
                if "findings" not in improved_copy:
                    improved_copy["findings"] = ["Análisis adicional recomendado"]
                elif len(improved_copy["findings"]) < 3:
                    improved_copy["findings"].append(
                        "Consideración adicional encontrada"
                    )

            if (
                "recommendations" in criticism.lower()
                and "recommendations" not in improved_copy
            ):
                improved_copy["recommendations"] = ["Implementar mejoras identificadas"]

            # Añadir timestamp si no existe
            if "timestamp" not in improved_copy:
                improved_copy["timestamp"] = datetime.now().isoformat()

            # Añadir version mejorada
            improved_copy["reflexion_improved"] = True
            improved_copy["improvement_notes"] = criticism

            improved = improved_copy

        elif isinstance(improved, str):
            improved_str = str(improved)

            # Mejoras básicas para código
            if "error" in criticism.lower() and "handling" in criticism.lower():
                if "try:" not in improved_str and "except:" not in improved_str:
                    improved_str += """

# Improved error handling
try:
    # Original code here
    pass
except Exception as e:
    logger.error(f"Error occurred: {e}")
    raise"""
                improved = improved_str

            # Añadir comentarios explicativos
            if len(improved_str) < 200:
                improved_str += (
                    f"\n\n# Improved based on reflexion: {criticism[:50]}..."
                )

        return improved

    async def _evaluate_improvement(
        self, original: Any, improved: Any, criticism: str
    ) -> float:
        """Evalúa cuán buena es la mejora (score entre 0.0 y 1.0)"""

        if not self.use_llm:
            return self._rule_based_evaluation(original, improved, criticism)

        # Evaluation usando LLM polymorphic real
        prompt = f"""
        EVALÚA LA MEJORA ENTRE OUTPUT ORIGINAL E MEJORADO:

        CRÍTICA APLICADA:
        {criticism}

        OUTPUT ORIGINAL:
        {json.dumps(original, indent=2, ensure_ascii=False) if isinstance(original, (dict, list)) else str(original)}

        OUTPUT MEJORADO:
        {json.dumps(improved, indent=2, ensure_ascii=False) if isinstance(improved, (dict, list)) else str(improved)}

        EVALÚA EN ESCALA 0.0-1.0 DONDE:
        - 1.0 = Mejora excelente, crítica completamente dirigida
        - 0.8 = Buena mejora, direcciones críticas mayoritariamente
        - 0.6 = Mejora moderada, algunos aspectos mejorados
        - 0.4 = Mejora mínima, cambios superficiales
        - 0.2 = No mejora significativa, mismos problemas
        - 0.0 = Output empeorado o aumentados problemas

        DEVUELVE SOLO EL NÚMERO DE SCORE:
        """

        try:
            response = await self.llm_system.generate_response(prompt)
            score = float(response.strip())
            return min(max(score, 0.0), 1.0)  # Clamp 0.0-1.0
        except Exception:
            return self._rule_based_evaluation(original, improved, criticism)

    def _rule_based_evaluation(
        self, original: Any, improved: Any, criticism: str
    ) -> float:
        """Evaluación básica de mejora usando reglas"""

        if original == improved:
            return 0.0  # Sin cambio = sin mejora

        original_str = str(original)
        improved_str = str(improved)

        score = 0.3  # Base score por intentar

        # Mejores scores por características positivas
        if len(improved_str) > len(original_str) * 1.1:
            score += 0.2  # Más detallado

        if isinstance(improved, dict) and isinstance(original, dict):
            new_keys = set(improved.keys()) - set(original.keys())
            if new_keys:
                score += 0.3  # Nuevas secciones/información

        # Penalización por keywords de problema
        if "error" in improved_str.lower() and "error" not in original_str.lower():
            score -= 0.2  # Introdujo errores

        return min(max(score, 0.0), 1.0)

    def _calculate_overall_improvement(self, session: ReflexionSession) -> float:
        """Calcula score de mejora total de la sesión"""

        if not session.memory_entries:
            return 0.0

        scores = [entry.improvement_score for entry in session.memory_entries]
        return sum(scores) / len(scores)

    async def _save_session_memory(self, session: ReflexionSession):
        """Guarda memoria episódica de la sesión completa"""

        session_file = self.memory_dir / f"session_{session.session_id}.json"

        session_data = {
            "session_id": session.session_id,
            "start_time": session.start_time.isoformat(),
            "duration_seconds": (datetime.now() - session.start_time).total_seconds(),
            "original_input": session.original_input,
            "final_output": session.current_output,
            "iteration_count": session.iteration_count,
            "final_status": session.final_status,
            "criticism_history": session.criticism_history,
            "improvement_scores": [m.improvement_score for m in session.memory_entries],
            "memory_entries": [vars(m) for m in session.memory_entries],
        }

        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ Error guardando session memory: {e}")

    def get_reflection_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de reflexión acumulada"""

        sessions = list(self.memory_dir.glob("session_*.json"))

        total_sessions = len(sessions)
        total_iterations = 0
        total_improvement = 0.0
        action_types = defaultdict(int)

        for session_file in sessions:
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total_iterations += data.get("iteration_count", 0)
                    scores = data.get("improvement_scores", [])
                    if scores:
                        total_improvement += sum(scores) / len(scores)
                    if (
                        "original_input" in data
                        and "action_type" in data["original_input"]
                    ):
                        action_types[data["original_input"]["action_type"]] += 1
            except Exception:
                continue

        return {
            "total_sessions": total_sessions,
            "average_iterations": total_iterations / max(total_sessions, 1),
            "average_improvement": total_improvement / max(total_sessions, 1),
            "most_reflected_actions": dict(action_types),
            "memory_sessions_stored": total_sessions,
        }


# =============================================================================
# FUNCIONES DE UTILIDAD Y TESTING
# =============================================================================


async def test_reflexion_agent():
    """Test real del reflexion agent"""

    print("🧪 TESTING REFLEXION AGENT - AUTO-CORRECIÓN ITERATIVA")
    print("=" * 60)

    agent = ReflexionAgent()

    # Test output de ejemplo (audit report básico)
    original_output = {
        "title": "Security Audit",
        "findings": ["One issue found"],
        "score": 70,
    }

    action_context = {
        "action_type": "audit_report",
        "description": "Generate comprehensive security audit report",
    }

    print("ORIGINAL OUTPUT:")
    print(json.dumps(original_output, indent=2))
    print("")

    # Ejecutar reflexión
    result = await agent.reflect_and_improve(
        original_output, action_context, max_iterations=2
    )

    print("REFLEXION RESULT:")
    print(f"- Session ID: {result['session_id']}")
    print(f"- Iterations: {result['iteration_count']}")
    print(f"- Improvement Score: {result['improvement_score']:.2f}")
    print(f"- Convergence: {result.get('convergence_reached', False)}")
    print("")

    print("IMPROVED OUTPUT:")
    print(json.dumps(result["improved_output"], indent=2))
    print("")

    print("CRITICISM HISTORY:")
    for i, criticism in enumerate(result["criticism_history"], 1):
        print(f"{i}. {criticism}")
    print("")

    print("STATS:")
    stats = agent.get_reflection_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(test_reflexion_agent())
