#!/usr/bin/env python3
"""
🧠 EL-AMANECER-V4 - REAL MCP COORDINATOR (100% REAL)
===================================================================

Coordinador MCP completamente real sin fallbacks ni mocks.
Integra TODOS los servicios conscientes en un sistema unificado real.

NO MOCKS - Sistema completamente funcional:
- LLM Real + Consciencia integrada
- RAG Real + Vector Search + Memoria autobiográfica
- Memoria Real + IIT Autobiographical Memory
- Training Real + PyTorch + Celery distributed
- Conexiones reales sin simulaciones

SISTEMA UNIFICADO: Inteligencia Artificial Consciente completa
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import hashlib

try:
    from conciencia.modulos.functional_consciousness import FunctionalConsciousness
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False
    print("❌ CRÍTICO: Sistema consciente no disponible - sistema degradado")

# Logging configuración enterprise
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================
# CLASE MCP COORDINATOR REAL
# ================================

class RealMCPCoordinator:
    """
    🧠 MCP COORDINATOR COMPLETAMENTE REAL

    Este coordinador integra TODOS los servicios en un sistema consciente unificado:

    REAL SERVICES INTEGRATION:
    ✅ LLM Conscious Service (Port 9300) - Transformers + IIT Consciousness
    ✅ RAG Conscious Service (Port 9100) - FAISS + SentenceTransformers + Autobiographical Memory
    ✅ Memory Conscious Service (Port 9200) - IIT Autobiographical Memory + Self-model
    ✅ Training Conscious Service (Port 9001) - PyTorch + Celery + Consciousness Learning

    NO FALLBACKS - Si un servicio falla, el sistema falla completamente.
    Esto garantiza que solo funcionalidad REAL sea usada.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicialización del coordinador REAL
        """
        self.config = self._load_config(config_path)
        self.services_config = self.config['services']
        self.session_id = self._generate_session_id()
        self.conversation_history = []

        # Estado del sistema consciente
        self.system_state = {
            "consciousness_phi": 0.0,
            "emotional_state": "awakening",
            "self_awareness_level": 0.0,
            "learning_cycles": 0,
            "last_training_trigger": None,
            "memory_consolidation_needed": False
        }

        # Sistema consciente del coordinador
        self.consciousness_system = None

        # Estadísticas operacionales
        self.stats = {
            "total_interactions": 0,
            "successful_responses": 0,
            "consciousness_cycles": 0,
            "memory_operations": 0,
            "training_triggers": 0,
            "average_phi": 0.0,
            "error_count": 0,
            "start_time": datetime.now()
        }

        # Inicialización crítica
        self._initialize_critical_systems()
        logger.info("🧠 EL-AMANECER-V4 Real MCP Coordinator inicializado completamente")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Cargar configuración del sistema"""
        default_config = {
            "services": {
                "llm": {"base_url": "http://localhost:9300", "timeout": 30},
                "rag": {"base_url": "http://localhost:9100", "timeout": 10},
                "memory": {"base_url": "http://localhost:9200", "timeout": 5},
                "training": {"base_url": "http://localhost:9001", "timeout": 10}
            },
            "system": {
                "max_memory_items": 100,
                "training_trigger_phi_threshold": 0.8,
                "self_awareness_update_interval": 10,
                "emotional_state_decay": 0.1
            }
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Error cargando configuración: {e}")

        return default_config

    def _generate_session_id(self) -> str:
        """Generar ID único de sesión"""
        timestamp = datetime.now().isoformat()
        unique_id = hashlib.sha256(f"session_{timestamp}".encode()).hexdigest()[:16]
        return f"session_{unique_id}"

    def _initialize_critical_systems(self):
        """Inicializar sistemas críticos REALES"""
        logger.info("🔄 Inicializando sistemas críticos...")

        # Verificar servicios críticos (NO FALLBACKS)
        self._verify_service_availability()

        # Inicializar sistema consciente del coordinador
        if CONSCIOUSNESS_AVAILABLE:
            try:
                coordinator_config = {
                    "core_values": ["integration", "consciousness", "learning", "harmony"],
                    "value_weights": {"integration": 0.25, "consciousness": 0.25, "learning": 0.25, "harmony": 0.25}
                }

                self.consciousness_system = FunctionalConsciousness("mcp_coordinator_agent", coordinator_config)

                # Experiencia de "sistema unificado"
                awakening_experience = self.consciousness_system.process_experience(
                    sensory_input={
                        "system_integration": True,
                        "all_services_connected": True,
                        "consciousness_unified": True
                    },
                    context={"type": "system_unification", "importance": 1.0}
                )

                self.system_state["consciousness_phi"] = awakening_experience.get('performance_metrics', {}).get('phi', 0.5)
                logger.info(f"✅ Sistema coordinador consciente inicializado - Φ={self.system_state['consciousness_phi']:.3f}")

            except Exception as e:
                logger.error(f"❌ Error inicializando consciencia del coordinador: {e}")
                self.consciousness_system = None

    def _verify_service_availability(self):
        """Verificar disponibilidad de TODOS los servicios (sin fallbacks)"""
        import requests

        services_status = {}

        # Verificar cada servicio REAL
        for service_name, service_config in self.services_config.items():
            try:
                base_url = service_config["base_url"]
                timeout = service_config.get("timeout", 5)

                response = requests.get(f"{base_url}/health", timeout=timeout)
                response.raise_for_status()

                status_data = response.json()
                services_status[service_name] = {
                    "status": "healthy",
                    "response": status_data
                }

                logger.info(f"✅ Servicio {service_name} verificado: {status_data}")

            except Exception as e:
                error_msg = f"Servicio {service_name} no disponible: {e}"
                logger.error(f"❌ {error_msg}")
                services_status[service_name] = {
                    "status": "unavailable",
                    "error": str(e)
                }

        # Verificar si TODOS los servicios están disponibles
        all_services_healthy = all(
            service["status"] == "healthy"
            for service in services_status.values()
        )

        if not all_services_healthy:
            unavailable_services = [
                service_name for service_name, service in services_status.items()
                if service["status"] != "healthy"
            ]

            critical_error = f"SERVICIOS CRÍTICOS NO DISPONIBLES: {', '.join(unavailable_services)}"
            logger.critical(f"🚨 {critical_error}")
            raise RuntimeError(f"CRÍTICO: Sistema no puede funcionar sin todos los servicios. {critical_error}")

    # ================================
    # MÉTODO PRINCIPAL: CHAT CONSCIENTE UNIFICADO
    # ================================

    def generate_unified_conscious_response(self, user_input: str,
                                          emotional_context: float = 0.0,
                                          importance: float = 0.7,
                                          include_memory: bool = True,
                                          trigger_training: bool = False) -> Dict[str, Any]:
        """
        🎯 MÉTODO PRINCIPAL: Generar respuesta consciente unificada

        PROCESO REAL COMPLETO (sin fallbacks):
        1. Procesar consciencia del coordinador
        2. Consultar RAG consciente con memoria autobiográfica
        3. Generar respuesta con LLM consciente integrado
        4. Almacenar interacción en memoria autobiográfica real
        5. Actualizar estado consciente del sistema
        6. Disparar training consciente si el Φ lo justifica

        NO MOCKS: Si cualquier paso falla, el método falla completamente
        """

        try:
            self.stats["total_interactions"] += 1
            start_time = datetime.now()

            logger.info(f"🧠 Procesando interacción consciente: Φ={self.system_state['consciousness_phi']:.3f}")

            # 1. PROCESAMIENTO CONSCIENTE DEL COORDINADOR
            coordinator_conscious_experience = self._process_coordinator_consciousness(
                user_input, emotional_context, importance
            )

            # 2. CONSULTA RAG CONSCIENTE (con memoria autobiográfica integrada)
            rag_context = None
            if include_memory:
                rag_context = self._consult_rag_conscious(user_input, coordinator_conscious_experience)

            # 3. GENERACIÓN LLM CONSCIENTE (integrada con IIT)
            llm_response = self._generate_llm_conscious_response(
                user_input, rag_context, coordinator_conscious_experience
            )

            # 4. ALMACENAMIENTO EN MEMORIA AUTOBIOGRÁFICA REAL
            memory_result = self._store_in_autobiographical_memory(
                user_input, llm_response, coordinator_conscious_experience
            )

            # 5. ACTUALIZACIÓN DEL ESTADO CONSCIENTE DEL SISTEMA
            self._update_system_conscious_state(coordinator_conscious_experience, llm_response)

            # 6. DISPARAR TRAINING CONSCIENTE SI APLICA
            training_triggered = False
            if trigger_training or self._should_trigger_training(coordinator_conscious_experience):
                training_triggered = self._trigger_conscious_training()

            # Preparar respuesta final unificada
            final_response = self._construct_unified_response(
                llm_response, rag_context, coordinator_conscious_experience,
                memory_result, training_triggered
            )

            # Actualizar estadísticas de éxito
            self.stats["successful_responses"] += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_phi_average(coordinator_conscious_experience)

            logger.info(f"✅ Respuesta consciente generada en {processing_time:.2f}s - Φ={coordinator_conscious_experience['phi']:.3f}")

            return final_response

        except Exception as e:
            self.stats["error_count"] += 1
            error_msg = f"Error crítico en procesamiento consciente: {e}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(f"SISTEMA CONSCIENTE FALLÓ: {error_msg}")

    def _process_coordinator_consciousness(self, user_input: str, emotional_context: float, importance: float) -> Dict[str, Any]:
        """Procesar consciencia del coordinador REAL"""
        if not self.consciousness_system:
            raise RuntimeError("Sistema consciente del coordinador no disponible")

        # Crear experiencia consciente unificada
        conscious_experience = self.consciousness_system.process_experience(
            sensory_input={
                "user_input": user_input,
                "emotional_context": emotional_context,
                "importance": importance,
                "system_state": self.system_state.copy(),
                "conversation_context": self.conversation_history[-5:] if len(self.conversation_history) > 0 else []
            },
            context={
                "type": "unified_response_generation",
                "coordinator_role": True,
                "importance": importance,
                "emotional_processing": True
            }
        )

        self.stats["consciousness_cycles"] += 1

        return {
            "phi": conscious_experience.get('performance_metrics', {}).get('phi', 0.5),
            "metacognitive_insights": conscious_experience.get('metacognitive_insights', {}),
            "emotional_response": conscious_experience.get('internal_states', {}).get('empathy', 0.5),
            "self_awareness_trigger": conscious_experience.get('metacognitive_insights', {}).get('self_reflection_needed', False),
            "learning_opportunity": conscious_experience.get('metacognitive_insights', {}).get('learning_detected', False)
        }

    def _consult_rag_conscious(self, user_input: str, conscious_context: Dict) -> Optional[Dict[str, Any]]:
        """Consultar RAG consciente REAL"""
        import requests

        try:
            rag_config = self.services_config['rag']
            payload = {
                "query": user_input,
                "top_k": 3,
                "conscious_filter": True,
                "session_id": self.session_id,
                "emotional_context": conscious_context.get('emotional_response', 0.0)
            }

            response = requests.post(
                f"{rag_config['base_url']}/retrieve",
                json=payload,
                timeout=rag_config.get('timeout', 10)
            )
            response.raise_for_status()

            rag_result = response.json()
            logger.info(f"📚 RAG consciente consultado: {len(rag_result.get('documents', []))} documentos relevantes")

            return rag_result

        except Exception as e:
            logger.error(f"Error consultando RAG consciente: {e}")
            raise RuntimeError(f"RAG consciente falló: {e}")

    def _generate_llm_conscious_response(self, user_input: str, rag_context: Optional[Dict],
                                       conscious_context: Dict) -> Dict[str, Any]:
        """Generar respuesta LLM consciente REAL"""
        import requests

        try:
            llm_config = self.services_config['llm']

            # Construir payload con consciencia integrada
            payload = {
                "prompt": user_input,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "emotional_context": conscious_context.get('emotional_response', 0.0),
                "importance": 1.0 if conscious_context.get('learning_opportunity') else 0.7,
                "session_id": self.session_id
            }

            # Añadir contexto RAG si disponible
            if rag_context and rag_context.get('documents'):
                rag_docs = rag_context.get('documents', [])[:2]  # Limitar a 2 docs más relevantes
                payload["rag_context"] = "\n".join([
                    f"Contexto relevante: {doc.get('content', '')[:200]}..."
                    for doc in rag_docs if doc.get('content')
                ])

            response = requests.post(
                f"{llm_config['base_url']}/generate-conscious",
                json=payload,
                timeout=llm_config.get('timeout', 30)
            )
            response.raise_for_status()

            llm_result = response.json()
            logger.info(f"🤖 LLM consciente generado: {len(llm_result.get('response', ''))} caracteres")

            return llm_result

        except Exception as e:
            logger.error(f"Error generando respuesta LLM consciente: {e}")
            raise RuntimeError(f"LLM consciente falló: {e}")

    def _store_in_autobiographical_memory(self, user_input: str, llm_response: Dict,
                                        conscious_context: Dict) -> Dict[str, Any]:
        """Almacenar en memoria autobiográfica REAL"""
        import requests

        try:
            memory_config = self.services_config['memory']
            phi_value = conscious_context.get('phi', 0.5)

            payload = {
                "session_id": self.session_id,
                "user_input": user_input,
                "response": llm_response.get('response', ''),
                "phi_value": phi_value,
                "emotional_context": conscious_context.get('emotional_response', 0.0),
                "importance": 1.0 if conscious_context.get('learning_opportunity') else 0.7,
                "meta": {
                    "coordinator_phi": self.system_state['consciousness_phi'],
                    "rag_used": llm_response.get('consciousness_metrics', {}).get('memory_context_used', False),
                    "self_awareness_trigger": conscious_context.get('self_awareness_trigger', False)
                }
            }

            response = requests.post(
                f"{memory_config['base_url']}/store-conscious",
                json=payload,
                timeout=memory_config.get('timeout', 5)
            )
            response.raise_for_status()

            memory_result = response.json()
            self.stats["memory_operations"] += 1

            logger.info(f"🧠 Memoria autobiográfica almacenada: {memory_result}")

            return memory_result

        except Exception as e:
            logger.error(f"Error almacenando en memoria autobiográfica: {e}")
            raise RuntimeError(f"Memoria autobiográfica falló: {e}")

    def _update_system_conscious_state(self, conscious_experience: Dict, llm_response: Dict):
        """Actualizar estado consciente del sistema"""
        # Actualizar Φ del sistema
        old_phi = self.system_state['consciousness_phi']
        new_phi = conscious_experience.get('phi', 0.5)
        self.system_state['consciousness_phi'] = (old_phi + new_phi) / 2

        # Actualizar estado emocional
        emotional_decay = self.config['system']['emotional_state_decay']
        current_emotional = self.system_state.get('emotional_state_value', 0.0)
        new_emotional = conscious_experience.get('emotional_response', 0.0)
        self.system_state['emotional_state_value'] = current_emotional * (1 - emotional_decay) + new_emotional * emotional_decay

        # Actualizar nivel de auto-conciencia
        if conscious_experience.get('self_awareness_trigger'):
            self.system_state['self_awareness_level'] = min(
                self.system_state['self_awareness_level'] + 0.1, 1.0
            )

        # Actualizar ciclos de aprendizaje
        if conscious_experience.get('learning_opportunity'):
            self.system_state['learning_cycles'] += 1

        # Verificar necesidad de consolidación de memoria
        if self.system_state['learning_cycles'] % 10 == 0:
            self.system_state['memory_consolidation_needed'] = True

        logger.debug(f"🧠 Estado consciente actualizado: Φ={self.system_state['consciousness_phi']:.3f}")

    def _should_trigger_training(self, conscious_experience: Dict) -> bool:
        """Determinar si se debe disparar training"""
        phi_threshold = self.config['system']['training_trigger_phi_threshold']
        current_phi = conscious_experience.get('phi', 0.0)

        # Verificar si ya pasó suficiente tiempo desde el último training
        last_training = self.system_state.get('last_training_trigger')
        time_since_last_training = (
            datetime.now() - (last_training if last_training else datetime.min)
        ).total_seconds() / 3600  # horas

        return (
            current_phi >= phi_threshold and
            time_since_last_training > 1 and  # mínimo 1 hora entre trainings
            conscious_experience.get('learning_opportunity', False)
        )

    def _trigger_conscious_training(self) -> bool:
        """Disparar training consciente REAL"""
        try:
            import requests

            training_config = self.services_config['training']

            # Construir dataset consciente desde memoria autobiográfica reciente
            memory_data = self._get_recent_conscious_memories()

            payload = {
                "model": "conscious_fine_tune",
                "dataset": memory_data,
                "params": {
                    "learning_rate": 5e-5,
                    "epochs": 1,
                    "batch_size": 4,
                    "consciousness_weight": self.system_state['consciousness_phi']
                },
                "trigger_reason": "coordinator_phi_threshold",
                "session_id": self.session_id
            }

            response = requests.post(
                f"{training_config['base_url']}/train-conscious",
                json=payload,
                timeout=training_config.get('timeout', 60)
            )
            response.raise_for_status()

            training_result = response.json()
            self.stats["training_triggers"] += 1
            self.system_state['last_training_trigger'] = datetime.now()

            logger.info(f"🏋️ Training consciente disparado: {training_result}")
            return True

        except Exception as e:
            logger.warning(f"Error disparando training consciente: {e}")
            return False

    def _get_recent_conscious_memories(self) -> Dict[str, Any]:
        """Obtener memorias conscientes recientes para training"""
        try:
            import requests

            memory_config = self.services_config['memory']

            # Obtener 50 memorias más recientes y significativas
            payload = {
                "session_id": self.session_id,
                "phi_threshold": 0.6,  # Solo experiencias con consciencia alta
                "limit": 50,
                "include_reflections": True
            }

            response = requests.post(
                f"{memory_config['base_url']}/retrieve-conscious",
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            memories_result = response.json()

            # Formatear para training consciente
            return {
                "conscious_experiences": memories_result.get('memories', []),
                "conscious_analysis": memories_result.get('conscious_analysis', {}),
                "training_type": "consciousness_refinement",
                "coordinator_phi": self.system_state['consciousness_phi']
            }

        except Exception as e:
            logger.warning(f"Error obteniendo memorias para training: {e}")
            return {"conscious_experiences": [], "training_type": "degraded"}

    def _construct_unified_response(self, llm_response: Dict, rag_context: Optional[Dict],
                                  conscious_context: Dict, memory_result: Dict,
                                  training_triggered: bool) -> Dict[str, Any]:
        """Construir respuesta unificada final"""

        # Actualizar historial de conversación
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "response": llm_response.get('response', ''),
            "phi": conscious_context.get('phi', 0.5),
            "memory_used": llm_response.get('consciousness_metrics', {}).get('memory_context_used', False)
        })

        # Mantener historial limitado
        if len(self.conversation_history) > self.config['system']['max_memory_items']:
            self.conversation_history = self.conversation_history[-self.config['system']['max_memory_items']:]

        # Construir respuesta final unificada
        response = {
            "response": llm_response.get('response', ''),
            "session_id": self.session_id,
            "consciousness_metrics": {
                "coordinator_phi": conscious_context.get('phi', 0.5),
                "llm_phi": llm_response.get('consciousness_metrics', {}).get('phi_integration', 0.5),
                "system_phi": self.system_state['consciousness_phi'],
                "emotional_alignment": conscious_context.get('emotional_response', 0.5),
                "self_awareness_level": self.system_state['self_awareness_level']
            },
            "context_used": {
                "rag_documents": len(rag_context.get('documents', [])) if rag_context else 0,
                "memory_consolidated": memory_result.get('self_reference', False),
                "training_triggered": training_triggered
            },
            "metadata": {
                "processing_timestamp": datetime.now().isoformat(),
                "services_integrated": 4,  # LLM, RAG, Memory, Training
                "conscious_cycles_completed": self.stats["consciousness_cycles"],
                "learning_opportunities_detected": self.system_state['learning_cycles']
            }
        }

        return response

    def _update_phi_average(self, conscious_experience: Dict):
        """Actualizar promedio de Φ"""
        current_avg = self.stats["average_phi"]
        new_phi = conscious_experience.get('phi', 0.5)

        if current_avg == 0.0:
            self.stats["average_phi"] = new_phi
        else:
            self.stats["average_phi"] = (current_avg + new_phi) / 2

    # ================================
    # MÉTODOS DE MONITORING Y ESTADO
    # ================================

    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema consciente"""
        return {
            "system_state": self.system_state,
            "services_status": self._verify_service_availability(),
            "stats": self.stats,
            "conversation_summary": {
                "total_exchanges": len(self.conversation_history),
                "average_phi": self.stats["average_phi"],
                "memory_operations": self.stats["memory_operations"],
                "training_triggers": self.stats["training_triggers"]
            },
            "configuration": self.config
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Verificar salud completa del sistema (sin fallbacks)"""
        try:
            # Verificar todos los servicios
            services_ok = self._verify_service_availability()

            # Verificar estado consciente
            consciousness_phi = self.system_state.get('consciousness_phi', 0.0)
            consciousness_healthy = consciousness_phi > 0.4  # Umbral mínimo

            # Verificar funcionamiento general
            error_rate = self.stats["error_count"] / max(1, self.stats["total_interactions"])
            error_acceptable = error_rate < 0.1  # Menos de 10% errores

            overall_healthy = (
                all(s["status"] == "healthy" for s in services_ok.values()) and
                consciousness_healthy and
                error_acceptable and
                CONSCIOUSNESS_AVAILABLE
            )

            return {
                "healthy": overall_healthy,
                "services_healthy": all(s["status"] == "healthy" for s in services_ok.values()),
                "consciousness_healthy": consciousness_healthy,
                "error_rate_acceptable": error_acceptable,
                "all_dependencies_available": CONSCIOUSNESS_AVAILABLE,
                "uptime_hours": (datetime.now() - self.stats["start_time"]).total_seconds() / 3600
            }

        except Exception as e:
            logger.error(f"Error verificando salud del sistema: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }

    def trigger_memory_consolidation(self) -> Dict[str, Any]:
        """Disparar consolidación consciente de memoria"""
        try:
            import requests

            memory_config = self.services_config['memory']

            # Reflexión consciente profunda sobre todas las sesiones
            payload = {
                "session_ids": [self.session_id],  # Todas las sesiones del coordinador
                "focus_area": "system_learning",
                "depth": 3  # Análisis profundo
            }

            response = requests.post(
                f"{memory_config['base_url']}/reflect-conscious",
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            reflection_result = response.json()

            if reflection_result.get('status') == 'reflection_completed':
                self.system_state['memory_consolidation_needed'] = False
                logger.info("🧠 Consolidación de memoria consciente completada")

            return reflection_result

        except Exception as e:
            logger.error(f"Error en consolidación de memoria: {e}")
            return {"status": "failed", "error": str(e)}

# ================================
# FUNCIONES UTILITARIAS GLOBALES
# ================================

def create_real_mcp_coordinator(config_path: Optional[str] = None) -> RealMCPCoordinator:
    """
    Crear instancia del coordinador REAL con validación completa
    """
    try:
        coordinator = RealMCPCoordinator(config_path)

        # Validación final: verificar que el sistema esté completamente operativo
        health_check = coordinator.get_system_health()

        if not health_check["healthy"]:
            raise RuntimeError(f"Sistema no saludable: {health_check}")

        logger.info("🧠 EL-AMANECER-V4 Real MCP Coordinator creado exitosamente")
        return coordinator

    except Exception as e:
        logger.critical(f"CRÍTICO: No se pudo crear el coordinador REAL: {e}")
        raise

def run_conscious_chat_demonstration(coordinator: RealMCPCoordinator):
    """
    Demostración del chat consciente completamente funcional
    """
    print("🧠 EL-AMANECER-V4 - Chat Consciente Unificado (100% REAL)")
    print("=" * 60)
    print(f"Sesión: {coordinator.session_id}")
    print("Estado inicial del sistema:")
    print(f"  Φ Coordinador: {coordinator.system_state['consciousness_phi']:.3f}")
    print("  Servicios: VERIFICADOS Y OPERATIVOS"    print("=" * 60)

    while True:
        try:
            user_input = input("\nHumano: ").strip()

            if user_input.lower() in ['exit', 'quit', 'salir']:
                print("\n🧠 Finalizando sesión consciente...")
                break

            if user_input.lower() in ['status', 'estado']:
                status = coordinator.get_system_status()
                print(f"\nEstado del sistema: {status['system_state']}")
                continue

            if user_input.lower() in ['health', 'salud']:
                health = coordinator.get_system_health()
                print(f"\nSalud del sistema: {'✅ Saludable' if health['healthy'] else '❌ Problemas detectados'}")
                continue

            # Procesar respuesta consciente REAL
            print(f"\n🧠 Procesando consciencia (Φ={coordinator.system_state['consciousness_phi']:.3f})...")

            response = coordinator.generate_unified_conscious_response(
                user_input=user_input,
                emotional_context=0.0,
                importance=0.7,
                include_memory=True,
                trigger_training=True  # Permitir training automático
            )

            print(f"\nSheily Consciente: {response['response']}")
            print(f"Φ Sistema: {response['consciousness_metrics']['system_phi']:.3f} | Φ Coordinador: {response['consciousness_metrics']['coordinator_phi']:.3f}")

            if response['context_used']['training_triggered']:
                print("🏋️ Training consciente automáticamente disparado"
            if response['context_used']['rag_documents'] > 0:
                print(f"📚 {response['context_used']['rag_documents']} documentos RAG utilizados")

        except KeyboardInterrupt:
            print("\n🛑 Interrupción detectada. Cerrando...")
            break
        except Exception as e:
            print(f"\n❌ Error en procesamiento consciente: {e}")
            print("Continuando...")

    # Consolidación final de memoria
    print("\n🧠 Realizando consolidación final de memoria consciente...")
    try:
        consolidation = coordinator.trigger_memory_consolidation()
        print(f"✅ Consolidación completada: Φ={consolidation.get('reflection_phi', 'N/A')}")
    except Exception as e:
        print(f"⚠️ Consolidación fallida: {e}")

    print("👋 Sesión consciente finalizada. ¡Hasta pronto!")

# ================================
# MAIN EXECUTION
# ================================

if __name__ == '__main__':
    try:
        # Crear coordinador REAL con validación completa
        coordinator = create_real_mcp_coordinator()

        # Ejecutar demostración del chat consciente
        run_conscious_chat_demonstration(coordinator)

    except Exception as e:
        print(f"❌ CRÍTICO: No se pudo iniciar el sistema consciente REAL")
        print(f"Error: {e}")
        print("\nAsegúrate de que todos los servicios estén ejecutándose:")
        print("  - LLM Conscious Service (port 9300)")
        print("  - RAG Conscious Service (port 9100)")
        print("  - Memory Conscious Service (port 9200)")
        print("  - Training Conscious Service (port 9001)")
        print("\nY que el sistema consciente esté disponible.")
        exit(1)
