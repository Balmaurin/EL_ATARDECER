#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sheily Integration Manager V2 - Enhanced Integration System
=========================================================

Sistema mejorado de integración que conecta todos los componentes de Sheily:
- Chat engines real y fallback
- Servidores MCP individuales
- Sistema de memoria
- RAG system
- Métricas y monitoreo

Este módulo reemplaza los archivos faltantes de integración MCP.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Core imports
from sheily_core.config import get_config
from sheily_core.logger import get_logger
from sheily_core.safety import get_security_monitor

logger = logging.getLogger(__name__)


# Try imports for various components
COMPONENTS_AVAILABLE = {}

try:
    from sheily_core.chat import ChatEngine, ChatResponse, create_chat_engine

    COMPONENTS_AVAILABLE["chat_engine"] = True
except ImportError:
    COMPONENTS_AVAILABLE["chat_engine"] = False

try:
    from sheily_rag.rag_ranker import create_rag_system

    COMPONENTS_AVAILABLE["rag_system"] = True
except ImportError:
    COMPONENTS_AVAILABLE["rag_system"] = False

try:
    from sheily_core.memory import SheilyMemoryV2

    COMPONENTS_AVAILABLE["memory_system"] = True
except ImportError:
    COMPONENTS_AVAILABLE["memory_system"] = False

# Advanced integrations (previously MCP functionality)
try:
    from sheily_core.memory.human_mind_system import HumanMindMemorySystem

    COMPONENTS_AVAILABLE["human_memory"] = True
except ImportError:
    COMPONENTS_AVAILABLE["human_memory"] = False

try:
    from sheily_core.memory.continuous_learning_system import ContinuousLearningSystem

    COMPONENTS_AVAILABLE["continuous_learning"] = True
except ImportError:
    COMPONENTS_AVAILABLE["continuous_learning"] = False

try:
    from sheily_core.unified_systems.unified_consciousness_memory_system import (
        UnifiedConsciousnessMemorySystem,
    )

    COMPONENTS_AVAILABLE["emotional_intelligence"] = True
except ImportError:
    COMPONENTS_AVAILABLE["emotional_intelligence"] = False

try:
    from sheily_core.memory.memory_integrator import SheilyMemoryIntegrator

    COMPONENTS_AVAILABLE["file_processing"] = True
except ImportError:
    COMPONENTS_AVAILABLE["file_processing"] = False


@dataclass
class IntegrationConfig:
    """Configuración del sistema de integración"""

    enable_chat_engine: bool = True
    enable_rag_system: bool = True
    enable_memory_system: bool = True
    enable_metrics: bool = True

    # Advanced integrations (replacing MCP servers)
    enable_human_memory: bool = True
    enable_continuous_learning: bool = True
    enable_emotional_intelligence: bool = True
    enable_file_processing: bool = True

    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0
    memory_cache_size: int = 1000

    # Feature flags
    enable_enhanced_responses: bool = True
    enable_context_enrichment: bool = True
    enable_learning_feedback: bool = True


@dataclass
class IntegrationResponse:
    """Respuesta integrada del sistema"""

    query: str
    response: str
    branch: str
    confidence: float
    processing_time: float

    # Integration metadata
    components_used: List[str]
    enhancement_level: str  # 'basic', 'enhanced', 'advanced'
    context_sources: int
    memory_hits: int
    rag_sources: int

    # System info
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class SheilyIntegrationManager:
    """
    Manager principal de integración de Sheily

    Coordina todos los componentes del sistema para proporcionar
    respuestas mejoradas y inteligentes.
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.logger = get_logger("integration_manager")

        # Initialize components
        self.chat_engine = None
        self.rag_system = None
        self.memory_system = None
        self.security_monitor = None

        # Advanced integrated systems (replacing MCP servers)
        self.human_memory_system = None
        self.continuous_learning_system = None
        self.emotional_intelligence_system = None
        self.file_processing_system = None

        # Metrics
        self.metrics = {
            "requests_processed": 0,
            "total_processing_time": 0.0,
            "enhancement_stats": {"basic": 0, "enhanced": 0, "advanced": 0},
            "component_usage": {
                "chat_engine": 0,
                "rag_system": 0,
                "memory_system": 0,
                "human_memory": 0,
                "continuous_learning": 0,
                "emotional_intelligence": 0,
                "file_processing": 0,
            },
            "start_time": time.time(),
        }

        # Active sessions
        self.active_sessions = {}

        # Initialize system
        self._initialize_components()

    def _initialize_components(self):
        """Inicializar componentes disponibles"""
        self.logger.info("Initializing Sheily Integration Manager V2...")

        # Initialize security
        try:
            self.security_monitor = get_security_monitor()
            self.logger.info("Security monitor initialized")
        except Exception as e:
            self.logger.warning(f"Security monitor initialization failed: {e}")

        # Initialize chat engine
        if self.config.enable_chat_engine and COMPONENTS_AVAILABLE.get("chat_engine"):
            try:
                self.chat_engine = create_chat_engine()
                self.logger.info("Advanced ChatEngine initialized")
            except Exception as e:
                self.logger.error(f"ChatEngine initialization failed: {e}")
                self.chat_engine = self._create_fallback_chat_engine()
        else:
            self.chat_engine = self._create_fallback_chat_engine()

        # Initialize RAG system
        if self.config.enable_rag_system and COMPONENTS_AVAILABLE.get("rag_system"):
            try:
                self.rag_system = create_rag_system()
                self.logger.info("RAG system initialized")
            except Exception as e:
                self.logger.warning(f"RAG system initialization failed: {e}")

        # Initialize memory system
        if self.config.enable_memory_system and COMPONENTS_AVAILABLE.get(
            "memory_system"
        ):
            try:
                self.memory_system = SheilyMemoryV2()
                self.logger.info("Memory system initialized")
            except Exception as e:
                self.logger.warning(f"Memory system initialization failed: {e}")

        # Initialize advanced integrated systems (replacing MCP servers)
        self._initialize_integrated_systems()

        self.logger.info("Integration Manager initialized successfully")
        self.logger.info(f"Available components: {list(COMPONENTS_AVAILABLE.keys())}")

    def _initialize_integrated_systems(self):
        """Inicializar sistemas integrados que reemplazan MCP servers"""

        # Initialize human memory system (replaces human_memory_mcp_server)
        if self.config.enable_human_memory and COMPONENTS_AVAILABLE.get("human_memory"):
            try:
                self.human_memory_system = HumanMindMemorySystem("integrated_user")
                self.logger.info("Human memory system initialized")
            except Exception as e:
                self.logger.warning(f"Human memory system initialization failed: {e}")

        # Initialize continuous learning (replaces continuous_learning_mcp_server)
        if self.config.enable_continuous_learning and COMPONENTS_AVAILABLE.get(
            "continuous_learning"
        ):
            try:
                self.continuous_learning_system = ContinuousLearningSystem()
                self.logger.info("Continuous learning system initialized")
            except Exception as e:
                self.logger.warning(
                    f"Continuous learning system initialization failed: {e}"
                )

        # Initialize emotional intelligence (replaces emotional_intelligence_mcp_server)
        if self.config.enable_emotional_intelligence and COMPONENTS_AVAILABLE.get(
            "emotional_intelligence"
        ):
            try:
                self.emotional_intelligence_system = UnifiedConsciousnessMemorySystem()
                self.logger.info("Emotional intelligence system initialized")
            except Exception as e:
                self.logger.warning(
                    f"Emotional intelligence system initialization failed: {e}"
                )

        # Initialize file processing (replaces advanced_file_processing_mcp_server)
        if self.config.enable_file_processing and COMPONENTS_AVAILABLE.get(
            "file_processing"
        ):
            try:
                self.file_processing_system = SheilyMemoryIntegrator()
                self.logger.info("File processing system initialized")
            except Exception as e:
                self.logger.warning(
                    f"File processing system initialization failed: {e}"
                )

    def _create_fallback_chat_engine(self):
        """Crear motor de chat de respaldo"""

        class FallbackChatEngine:
            def __init__(self):
                self.responses = {
                    "saludo": "¡Hola! Soy Sheily AI. ¿En qué puedo ayudarte?",
                    "programación": "Puedo ayudarte con preguntas de programación y desarrollo.",
                    "ia": "La inteligencia artificial es mi especialidad. ¿Qué te gustaría saber?",
                    "general": "Estoy aquí para ayudarte. ¿Puedes ser más específico?",
                }

            def process_query(self, query: str, client_id: str = "unknown"):
                query_lower = query.lower()

                if any(word in query_lower for word in ["hola", "hello", "hi"]):
                    response = self.responses["saludo"]
                    branch = "saludo"
                    confidence = 0.9
                elif any(
                    word in query_lower for word in ["python", "código", "programar"]
                ):
                    response = self.responses["programación"]
                    branch = "programación"
                    confidence = 0.8
                elif any(word in query_lower for word in ["ia", "inteligencia", "ai"]):
                    response = self.responses["ia"]
                    branch = "inteligencia_artificial"
                    confidence = 0.8
                else:
                    response = self.responses["general"]
                    branch = "general"
                    confidence = 0.5

                return type(
                    "ChatResponse",
                    (),
                    {
                        "query": query,
                        "response": response,
                        "branch": branch,
                        "confidence": confidence,
                        "context_sources": 0,
                        "processing_time": 0.1,
                        "model_used": "fallback",
                        "error": None,
                        "metadata": {},
                    },
                )()

        return FallbackChatEngine()

    async def process_integrated_query(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> IntegrationResponse:
        """
        Procesar consulta con integración completa de componentes
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())
        context = context or {}

        components_used = []
        enhancement_level = "basic"
        memory_hits = 0
        rag_sources = 0

        try:
            # Security check
            if self.security_monitor:
                # Simplified security check
                security_result = await self.security_monitor.check_query(query, user_id)
                if not security_result.get("safe", True):
                    raise ValueError(f"Security violation: {security_result.get('reason', 'Unknown')}")

            # Memory lookup with integrated human memory
            memory_context = ""
            if self.memory_system or self.human_memory_system:
                try:
                    # Use integrated memory systems
                    if self.human_memory_system:
                        memory_result = self.process_memory_query(query, user_id)
                        if memory_result.get("success"):
                            memory_hits = memory_result.get("memories_found", 0)
                            components_used.append("human_memory")
                            enhancement_level = "enhanced"

                    # Basic memory system
                    if self.memory_system:
                        memory_results = []  # Could implement basic search
                        if memory_results:
                            memory_hits += len(memory_results)
                            components_used.append("memory_system")
                            self.metrics["component_usage"]["memory_system"] += 1
                except Exception as e:
                    self.logger.warning(f"Memory lookup failed: {e}")

            # Emotional analysis integration
            emotional_context = {}
            if self.emotional_intelligence_system:
                try:
                    emotional_result = self.analyze_emotional_context(query)
                    if emotional_result.get("success"):
                        emotional_context = emotional_result
                        components_used.append("emotional_intelligence")
                        enhancement_level = "advanced"
                except Exception as e:
                    self.logger.warning(f"Emotional analysis failed: {e}")

            # RAG context enrichment
            rag_context = ""
            if self.rag_system:
                try:
                    # Try to get RAG context
                    rag_results = []  # Simplified for now
                    if rag_results:
                        rag_sources = len(rag_results)
                        components_used.append("rag_system")
                        enhancement_level = "enhanced"
                except Exception as e:
                    self.logger.warning(f"RAG lookup failed: {e}")

            # Enhanced context preparation with emotional awareness
            enhanced_query = query
            if emotional_context.get("success"):
                dominant_emotion = emotional_context.get("dominant_emotion", "neutral")
                enhanced_query += f" [Contexto emocional: {dominant_emotion}]"
            if memory_context or rag_context:
                enhanced_query = (
                    f"Contexto: {memory_context} {rag_context}\nConsulta: {query}"
                )
                enhancement_level = (
                    "advanced" if memory_context and rag_context else "enhanced"
                )

            # Process with chat engine
            if self.chat_engine:
                chat_response = self.chat_engine.process_query(enhanced_query, user_id)
                components_used.append("chat_engine")
                self.metrics["component_usage"]["chat_engine"] += 1
            else:
                raise RuntimeError("No chat engine available")

            # Continuous learning integration (post-processing)
            if self.continuous_learning_system:
                try:
                    # Learn from this interaction
                    learning_content = (
                        f"Query: {query}\nResponse: {chat_response.response}"
                    )
                    learning_result = self.process_learning_interaction(
                        learning_content, "chat"
                    )
                    if learning_result.get("success"):
                        components_used.append("continuous_learning")
                except Exception as e:
                    self.logger.warning(f"Continuous learning failed: {e}")

            # Create integrated response
            processing_time = time.time() - start_time

            response = IntegrationResponse(
                query=query,
                response=chat_response.response,
                branch=chat_response.branch,
                confidence=chat_response.confidence,
                processing_time=processing_time,
                components_used=components_used,
                enhancement_level=enhancement_level,
                context_sources=chat_response.context_sources,
                memory_hits=memory_hits,
                rag_sources=rag_sources,
                session_id=session_id,
                user_id=user_id,
                metadata={
                    "original_processing_time": chat_response.processing_time,
                    "model_used": chat_response.model_used,
                    "integration_overhead": processing_time
                    - chat_response.processing_time,
                    "context_enhancement": enhancement_level != "basic",
                },
            )

            # Update metrics
            self._update_metrics(response)

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Integration processing error: {e}")

            return IntegrationResponse(
                query=query,
                response=f"Lo siento, hubo un error procesando tu consulta: {str(e)}",
                branch="error",
                confidence=0.0,
                processing_time=processing_time,
                components_used=components_used,
                enhancement_level="error",
                context_sources=0,
                memory_hits=0,
                rag_sources=0,
                session_id=session_id,
                user_id=user_id,
                error=str(e),
            )

    def _update_metrics(self, response: IntegrationResponse):
        """Actualizar métricas del sistema"""
        self.metrics["requests_processed"] += 1
        self.metrics["total_processing_time"] += response.processing_time
        self.metrics["enhancement_stats"][response.enhancement_level] += 1

    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema"""
        uptime = time.time() - self.metrics["start_time"]
        avg_processing_time = self.metrics["total_processing_time"] / max(
            self.metrics["requests_processed"], 1
        )

        return {
            "integration_manager": {
                "status": "active",
                "version": "2.0.0",
                "uptime_seconds": uptime,
                "config": {
                    "chat_engine_enabled": self.config.enable_chat_engine,
                    "rag_system_enabled": self.config.enable_rag_system,
                    "memory_system_enabled": self.config.enable_memory_system,
                    "integrated_systems_enabled": True,
                },
            },
            "components": {
                "chat_engine": "active" if self.chat_engine else "unavailable",
                "rag_system": "active" if self.rag_system else "unavailable",
                "memory_system": "active" if self.memory_system else "unavailable",
                "security_monitor": (
                    "active" if self.security_monitor else "unavailable"
                ),
            },
            "components_available": COMPONENTS_AVAILABLE,
            "metrics": {
                "requests_processed": self.metrics["requests_processed"],
                "average_processing_time": avg_processing_time,
                "enhancement_distribution": self.metrics["enhancement_stats"],
                "component_usage": self.metrics["component_usage"],
            },
            "active_sessions": len(self.active_sessions),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas detalladas de rendimiento"""
        uptime = time.time() - self.metrics["start_time"]
        requests_processed = self.metrics["requests_processed"]

        if requests_processed == 0:
            return {"message": "No requests processed yet"}

        return {
            "total_requests_processed": requests_processed,
            "average_processing_time": self.metrics["total_processing_time"]
            / requests_processed,
            "requests_per_minute": requests_processed / max(uptime / 60, 1),
            "enhancement_levels_distribution": self.metrics["enhancement_stats"],
            "component_usage_stats": self.metrics["component_usage"],
            "system_efficiency": {
                "uptime_hours": uptime / 3600,
                "total_processing_time": self.metrics["total_processing_time"],
                "system_utilization": (self.metrics["total_processing_time"] / uptime)
                * 100,
            },
        }

    # ========================================================================
    # INTEGRATED FUNCTIONALITY (replacing MCP servers)
    # ========================================================================

    def process_memory_query(
        self, query: str, user_id: str = "integrated_user"
    ) -> Dict[str, Any]:
        """Procesar consulta de memoria humana (reemplaza human_memory_mcp_server)"""
        if not self.human_memory_system:
            return {"error": "Human memory system not available", "success": False}

        try:
            # Search in human memory
            memories = self.human_memory_system.search_memories(query, limit=5)

            # Update metrics
            self.metrics["component_usage"]["human_memory"] += 1

            return {
                "success": True,
                "memories_found": len(memories),
                "memories": memories,
                "query": query,
                "user_id": user_id,
                "processing_time": 0.1,  # Placeholder
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def process_learning_interaction(
        self, content: str, interaction_type: str = "chat"
    ) -> Dict[str, Any]:
        """Procesar interacción de aprendizaje (reemplaza continuous_learning_mcp_server)"""
        if not self.continuous_learning_system:
            return {
                "error": "Continuous learning system not available",
                "success": False,
            }

        try:
            # Process learning from interaction
            learning_result = self.continuous_learning_system.process_interaction(
                content=content, interaction_type=interaction_type
            )

            # Update metrics
            self.metrics["component_usage"]["continuous_learning"] += 1

            # Calculate dynamic learning score based on content length and complexity
            base_score = 0.5
            complexity_bonus = min(0.4, len(content.split()) / 1000.0)
            learning_score = min(1.0, base_score + complexity_bonus)

            return {
                "success": True,
                "learning_applied": True,
                "content_processed": content,
                "learning_score": learning_result.get("learning_score", learning_score),
                "categories_identified": learning_result.get("categories", []),
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def analyze_emotional_context(self, text: str) -> Dict[str, Any]:
        """Analizar contexto emocional (reemplaza emotional_intelligence_mcp_server)"""
        if not self.emotional_intelligence_system:
            return {
                "error": "Emotional intelligence system not available",
                "success": False,
            }

        try:
            # Analyze emotions using unified consciousness system
            emotional_analysis = (
                self.emotional_intelligence_system._analyze_emotions_advanced(text)
            )
            emotional_valence = (
                self.emotional_intelligence_system._calculate_emotional_valence(text)
            )

            # Update metrics
            self.metrics["component_usage"]["emotional_intelligence"] += 1

            return {
                "success": True,
                "emotions_detected": emotional_analysis,
                "emotional_valence": emotional_valence,
                "text_analyzed": text,
                "dominant_emotion": (
                    max(emotional_analysis.items(), key=lambda x: x[1])[0]
                    if emotional_analysis
                    else "neutral"
                ),
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def process_file_for_memory(
        self, file_path: str, category: str = "general"
    ) -> Dict[str, Any]:
        """Procesar archivo para memoria (reemplaza advanced_file_processing_mcp_server)"""
        if not self.file_processing_system:
            return {"error": "File processing system not available", "success": False}

        try:
            # Process file using memory integrator
            result = self.file_processing_system.process_single_file(
                file_path, category
            )

            # Update metrics
            self.metrics["component_usage"]["file_processing"] += 1

            return {
                "success": True,
                "file_processed": file_path,
                "category": category,
                "chunks_created": result.get("chunks_created", 0),
                "memory_entries": result.get("memory_entries", 0),
                "processing_time": result.get("processing_time", 0.0),
            }
        except Exception as e:
            return {"error": str(e), "success": False}


# Factory functions for compatibility
def create_integration_manager(
    config: Optional[IntegrationConfig] = None,
) -> SheilyIntegrationManager:
    """Crear instancia del Integration Manager"""
    return SheilyIntegrationManager(config)


def get_integration_status() -> Dict[str, Any]:
    """Obtener estado de disponibilidad de componentes"""
    return {
        "integration_manager_available": True,
        "components_available": COMPONENTS_AVAILABLE,
        "sheily_core_available": True,
        "integration_status": "ready",
        "version": "2.0.0",
    }


# Singleton instance for global access
_global_integration_manager = None


def get_global_integration_manager() -> SheilyIntegrationManager:
    """Obtener instancia global del Integration Manager"""
    global _global_integration_manager
    if _global_integration_manager is None:
        _global_integration_manager = create_integration_manager()
    return _global_integration_manager


if __name__ == "__main__":
    # Test integration manager
    async def test_integration():
        manager = create_integration_manager()

        logger.info("🧪 Testing Sheily Integration Manager V2")
        logger.info("=" * 50)
        # Test queries
        test_queries = [
            "Hola, ¿cómo estás?",
            "¿Qué es Python?",
            "Explícame la inteligencia artificial",
            "¿Cómo funciona el machine learning?",
        ]

        for query in test_queries:
            logger.info(f"\n📝 Consulta: {query}")
            response = await manager.process_integrated_query(query, "test_user")
            logger.info(f"   Respuesta: {response.response}")
            logger.info(
                f"   Rama: {response.branch} | Confianza: {response.confidence:.2f}"
            )
            logger.info(f"   Componentes: {response.components_used}")
            logger.info(f"   Nivel: {response.enhancement_level}")
        logger.info("\n📊 Estado del sistema:")
        status = manager.get_system_status()
        logger.info(json.dumps(status, indent=2, ensure_ascii=False))

    asyncio.run(test_integration())
