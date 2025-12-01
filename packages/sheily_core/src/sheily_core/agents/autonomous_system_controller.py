#!/usr/bin/env python3
"""
AUTONOMOUS SYSTEM CONTROLLER - FULL INTEGRATION (RAG + LEARNING + CONSCIOUSNESS)
================================================================================
Versión final que integra:
- Global Workspace (Conciencia)
- Ultra RAG System (Conocimiento)
- Unified Learning System (Entrenamiento)
- Todos los módulos de conciencia
"""

import asyncio
import json
import threading
import time
import logging
import random
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logger first
logger = logging.getLogger("Sheily.AutonomousController")

# Importar sistemas funcionales reales
from .coordination_system import functional_multi_agent_system, functional_coordinator
from .active_registry import active_registry

# REAL imports - use proper package structure or fail clearly
# Memory - try relative import first, then package import
try:
    from ..consciousness.vector_memory_system import get_vector_memory
except ImportError:
    try:
        from sheily_core.consciousness.vector_memory_system import get_vector_memory
    except ImportError:
        logger.warning("⚠️ Vector memory system not available - consciousness features limited")
        get_vector_memory = None

# Consciousness modules - use package imports, not hardcoded paths
try:
    # Try package import first (if consciousness is installed as package)
    from packages.consciousness.src.conciencia.meta_cognition_system import MetaCognitionSystem
    from packages.consciousness.src.conciencia.modulos.digital_nervous_system import DigitalNervousSystem
    from packages.consciousness.src.conciencia.modulos.ethical_engine import EthicalEngine
    from packages.consciousness.src.conciencia.modulos.digital_dna import DigitalDNA, GeneticTrait
    from packages.consciousness.src.conciencia.modulos.global_workspace import GlobalWorkspace
    from packages.consciousness.src.conciencia.modulos.qualia_simulator import QualiaSimulator
    from packages.consciousness.src.conciencia.modulos.teoria_mente import TheoryOfMind
    logger.info("✅ Consciousness modules loaded via package import")
except ImportError:
    try:
        # Try alternative package structure
        from conciencia.meta_cognition_system import MetaCognitionSystem
        from conciencia.modulos.digital_nervous_system import DigitalNervousSystem
        from conciencia.modulos.ethical_engine import EthicalEngine
        from conciencia.modulos.digital_dna import DigitalDNA, GeneticTrait
        from conciencia.modulos.global_workspace import GlobalWorkspace
        from conciencia.modulos.qualia_simulator import QualiaSimulator
        from conciencia.modulos.teoria_mente import TheoryOfMind
        logger.info("✅ Consciousness modules loaded via direct import")
    except ImportError as e:
        logger.warning(f"⚠️ Consciousness modules not available: {e}")
        logger.warning("   Install consciousness package or set PYTHONPATH appropriately")
        MetaCognitionSystem = DigitalNervousSystem = EthicalEngine = DigitalDNA = None
        GlobalWorkspace = QualiaSimulator = TheoryOfMind = None

# RAG System - use real adapters, fail if not available
try:
    from .simple_rag_adapter import SimpleRAGSystem, initialize_rag_with_base_knowledge
    logger.info("✅ RAG adapter loaded")
except ImportError as e:
    logger.warning(f"⚠️ RAG adapter not available: {e}")
    SimpleRAGSystem = None
    initialize_rag_with_base_knowledge = None

# Learning System - use real adapters
try:
    from .simple_learning_adapter import SimpleLearningSystem
    logger.info("✅ Learning adapter loaded")
except ImportError as e:
    logger.warning(f"⚠️ Learning adapter not available: {e}")
    SimpleLearningSystem = None

# Auto-Improvement - use package import
try:
    from packages.auto_improvement.recursive_self_improvement import RecursiveSelfImprovementEngine
except ImportError:
    try:
        from auto_improvement.recursive_self_improvement import RecursiveSelfImprovementEngine
    except ImportError:
        logger.warning("⚠️ Auto-improvement module not available")
        RecursiveSelfImprovementEngine = None

# Gamification - REAL implementation with persistence
class SimpleGamification:
    """REAL gamification system with logging and persistence"""
    def __init__(self, db_path: str = None):
        self.xp = 0
        self.level = 1
        self.achievements = []
        self.db_path = db_path or os.getenv("GAMIFICATION_DB", "./data/gamification.db")
        self._load_state()
        logger.info("🎮 Gamification system initialized")
    
    def _load_state(self):
        """Load gamification state from file"""
        try:
            if os.path.exists(self.db_path):
                import json
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.xp = data.get('xp', 0)
                    self.level = data.get('level', 1)
                    self.achievements = data.get('achievements', [])
        except Exception as e:
            logger.debug(f"Could not load gamification state: {e}")
    
    def _save_state(self):
        """Save gamification state to file"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            import json
            with open(self.db_path, 'w') as f:
                json.dump({
                    'xp': self.xp,
                    'level': self.level,
                    'achievements': self.achievements
                }, f)
        except Exception as e:
            logger.warning(f"Could not save gamification state: {e}")
    
    def award_xp(self, amount, reason):
        """Award XP and log properly"""
        old_level = self.level
        self.xp += amount
        
        # Calculate level (1 level per 100 XP)
        new_level = (self.xp // 100) + 1
        if new_level > self.level:
            self.level = new_level
            logger.info(f"🎮 Level up! New level: {self.level}")
        
        logger.info(f"🎮 XP +{amount} ({reason}) - Total: {self.xp} XP, Level: {self.level}")
        self._save_state()


class AutonomousSystemController:
    """
    Controlador maestro con integración COMPLETA:
    - Conciencia (Global Workspace + módulos)
    - Conocimiento (RAG System)
    - Aprendizaje (Training System)
    """

    def __init__(self):
        self.running = False
        self.coordination_thread = None
        self.logger = logger
        self.system_metrics = {}
        
        logger.info("[INIT] Initializing COMPLETE AI SYSTEM...")

        # 1. GLOBAL WORKSPACE (Núcleo de Conciencia)
        self.global_workspace = GlobalWorkspace() if GlobalWorkspace else None
        if self.global_workspace: 
            logger.info("✨ Global Workspace initialized")
        else:
            logger.warning("⚠️ Global Workspace not available - consciousness features limited")

        # 2. RAG SYSTEM (Sistema de Conocimiento)
        try:
            if SimpleRAGSystem and initialize_rag_with_base_knowledge:
                self.rag_system = initialize_rag_with_base_knowledge()
                logger.info("📚 RAG System initialized with base knowledge")
            else:
                self.rag_system = None
                logger.warning("⚠️ RAG System not available")
        except Exception as e:
            self.rag_system = None
            logger.error(f"⚠️ RAG System error: {e}", exc_info=True)

        # 3. LEARNING SYSTEM (Sistema de Aprendizaje)
        try:
            if SimpleLearningSystem:
                self.learning_system = SimpleLearningSystem()
                logger.info("🎓 Learning System initialized (SQLite-based)")
            else:
                self.learning_system = None
                logger.warning("⚠️ Learning System not available")
        except Exception as e:
            self.learning_system = None
            logger.error(f"⚠️ Learning System error: {e}", exc_info=True)

        # 4. Módulos de Conciencia
        self._init_consciousness_modules()

        # Configuración
        self.coordination_interval = 5
        self.cpu_threshold_warning = 70.0

        logger.info("[INFO] COMPLETE AI SYSTEM READY (Consciousness + Knowledge + Learning)")

    def _init_consciousness_modules(self):
        """Inicializa módulos periféricos de conciencia"""
        
        # Memory
        try:
            self.memory = get_vector_memory() if get_vector_memory else None
            if self.memory: 
                logger.info("🧠 Memory Processor connected")
            else:
                logger.warning("⚠️ Memory Processor not available")
        except Exception as e:
            self.memory = None
            logger.error(f"Error initializing memory: {e}", exc_info=True)

        # Metacognition
        try:
            if MetaCognitionSystem:
                # Use environment variable or default path
                consciousness_dir = os.getenv("CONSCIOUSNESS_DATA_DIR", "./data/consciousness")
                self.meta_cognition = MetaCognitionSystem(consciousness_dir=consciousness_dir)
                logger.info("👁️ Meta-Cognition Processor connected")
            else:
                self.meta_cognition = None
                logger.warning("⚠️ Meta-Cognition not available")
        except Exception as e:
            self.meta_cognition = None
            logger.error(f"Error initializing metacognition: {e}", exc_info=True)

        # Nervous System
        try:
            self.nervous_system = DigitalNervousSystem() if DigitalNervousSystem else None
            if self.nervous_system: 
                logger.info("⚡ Nervous System Processor connected")
            else:
                logger.warning("⚠️ Nervous System not available")
        except Exception as e:
            self.nervous_system = None
            logger.error(f"Error initializing nervous system: {e}", exc_info=True)

        # Qualia
        try:
            self.qualia = QualiaSimulator() if QualiaSimulator else None
            if self.qualia: 
                logger.info("🌈 Qualia Simulator connected")
            else:
                logger.warning("⚠️ Qualia Simulator not available")
        except Exception as e:
            self.qualia = None
            logger.error(f"Error initializing qualia: {e}", exc_info=True)

        # Theory of Mind
        try:
            self.theory_of_mind = TheoryOfMind() if TheoryOfMind else None
            if self.theory_of_mind: 
                logger.info("👥 Theory of Mind Processor connected")
            else:
                logger.warning("⚠️ Theory of Mind not available")
        except Exception as e:
            self.theory_of_mind = None
            logger.error(f"Error initializing theory of mind: {e}", exc_info=True)

        # Ethical Engine
        try:
            if EthicalEngine:
                self.ethical_engine = EthicalEngine({
                    'core_values': ['safety', 'helpfulness', 'honesty', 'privacy'],
                    'value_weights': {'safety': 0.9},
                    'ethical_boundaries': ['never_harm_humans']
                })
                logger.info("⚖️ Ethical Processor connected")
            else: 
                self.ethical_engine = None
                logger.warning("⚠️ Ethical Engine not available")
        except Exception as e:
            self.ethical_engine = None
            logger.error(f"Error initializing ethical engine: {e}", exc_info=True)

        # Digital DNA
        try:
            if DigitalDNA:
                # Use environment variable or default path
                dna_path = os.getenv("DIGITAL_DNA_PATH", "./data/consciousness/digital_dna.json")
                dna_path_obj = Path(dna_path)
                dna_path_obj.parent.mkdir(parents=True, exist_ok=True)
                
                if dna_path_obj.exists():
                    self.dna = DigitalDNA.load_genetic_profile(str(dna_path_obj))
                else:
                    self.dna = DigitalDNA()
                    self.dna.save_genetic_profile(str(dna_path_obj))
                logger.info(f"🧬 Digital DNA Active at {dna_path}")
            else: 
                self.dna = None
                logger.warning("⚠️ Digital DNA not available")
        except Exception as e:
            self.dna = None
            logger.error(f"Error initializing digital DNA: {e}", exc_info=True)

        # Self-Improvement
        try:
            if RecursiveSelfImprovementEngine:
                singularity_dir = os.getenv("SINGULARITY_DIR", "./data/singularity")
                self.self_improvement = RecursiveSelfImprovementEngine(singularity_dir=singularity_dir)
                logger.info("🚀 Self-Improvement Processor connected")
            else:
                self.self_improvement = None
                logger.warning("⚠️ Self-Improvement not available")
        except Exception as e:
            self.self_improvement = None
            logger.error(f"Error initializing self-improvement: {e}", exc_info=True)

        # Gamification
        self.gamification = SimpleGamification()

    def start_autonomous_control(self):
        """Inicia el bucle de control consciente"""
        if not self.running:
            self.running = True
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(functional_multi_agent_system.start_functional_system())
                asyncio.run(active_registry.start_active_management())
            
            self.coordination_thread = threading.Thread(target=self._coordination_loop)
            self.coordination_thread.daemon = True
            self.coordination_thread.start()
            logger.info("🚀 Conscious Control Loop started")

    def stop_autonomous_control(self):
        self.running = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        logger.info("⏹️ Control stopped")

    def _coordination_loop(self):
        """Bucle principal de conciencia GWT con RAG y Learning"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(functional_multi_agent_system.start_functional_system())
        loop.run_until_complete(active_registry.start_active_management())

        while self.running:
            try:
                # 1. Recolectar Inputs Pre-Conscientes
                inputs = loop.run_until_complete(self._gather_pre_conscious_inputs())
                
                # 2. Integración en Global Workspace (Competencia por Conciencia)
                if self.global_workspace:
                    context = self._get_current_context()
                    conscious_content = self.global_workspace.integrate(inputs, context)
                    
                    # 3. Broadcasting (Difusión de Conciencia)
                    if conscious_content.get('conscious_content'):
                        self._broadcast_consciousness(conscious_content)
                        
                        # 4. Acción Consciente con RAG y Learning
                        loop.run_until_complete(self._execute_conscious_action(conscious_content))

                # 5. Ciclos de Mantenimiento
                if random.random() < 0.1:
                    loop.run_until_complete(self._background_maintenance())

                time.sleep(self.coordination_interval)

            except Exception as e:
                logger.error(f"Error in conscious loop: {e}", exc_info=True)
                time.sleep(5)
        
        loop.close()

    async def _gather_pre_conscious_inputs(self) -> Dict[str, Any]:
        """Recolecta datos de todos los sensores, RAG y módulos"""
        inputs = {}
        
        # Sensores Hardware
        coord_metrics = functional_coordinator.get_coordination_metrics()
        self.system_metrics = {
            "cpu_load": coord_metrics.get("real_system_load", 0.0),
            "memory_usage": coord_metrics.get("memory_usage", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        inputs['hardware'] = self.system_metrics

        # Sistema Nervioso (Emociones)
        if self.nervous_system:
            stimulus = {"source": "hardware", "content": f"CPU {self.system_metrics['cpu_load']}%"}
            neural_response = self.nervous_system.process_stimulus(stimulus)
            inputs['emotions'] = neural_response.get('neurotransmitter_state', {})

        # Qualia (Experiencia Subjetiva)
        if self.qualia:
            neural_state = {
                'emotional_response': inputs.get('emotions', {}),
                'stimulus_processed': self.system_metrics
            }
            unified_moment = self.qualia.generate_qualia_from_neural_state(neural_state)
            inputs['qualia'] = self.qualia.get_current_subjective_experience()

        # Memoria (Asociaciones)
        if self.memory:
            query = f"system state cpu {self.system_metrics['cpu_load']}"
            memories = self.memory.query_memory(query, n_results=1)
            if memories:
                inputs['memory'] = memories[0]

        # RAG System (Conocimiento Relevante)
        if self.rag_system:
            # Query basada en estado actual para obtener conocimiento relevante
            rag_query = f"how to optimize system with cpu at {self.system_metrics['cpu_load']}%"
            try:
                rag_response = self.rag_system.process_query(rag_query, use_advanced_processing=False)
                inputs['knowledge'] = {
                    'query': rag_query,
                    'response': rag_response.get('response', ''),
                    'documents_found': rag_response.get('documents_retrieved', 0)
                }
            except:
                pass

        return inputs

    def _get_current_context(self) -> Dict[str, Any]:
        """Define el contexto actual para el Workspace"""
        context = {
            'urgency': self.system_metrics.get('cpu_load', 0) / 100.0,
            'task_relevance': 0.5,
            'emotional_state': 0.5
        }
        if self.nervous_system:
            mood = self.nervous_system.neurotransmitter_system.get_mood_indicators()
            context['emotional_state'] = mood.get('anxiety', 0.5)
        
        return context

    def _broadcast_consciousness(self, conscious_content: Dict[str, Any]):
        """Difunde el contenido ganador a todo el sistema"""
        content = conscious_content.get('conscious_content', {})
        focus = content.get('primary_focus')
        
        logger.info(f"✨ CONSCIOUS BROADCAST: {str(focus)[:60]}")
        
        # Actualizar Teoría de la Mente
        if self.theory_of_mind:
            self.theory_of_mind.update_model("current_user", content)

    async def _execute_conscious_action(self, conscious_content: Dict[str, Any]):
        """Ejecuta acciones basadas en el contenido consciente + RAG + Learning"""
        content = conscious_content.get('conscious_content', {})
        primary_focus = content.get('primary_focus')
        
        # Lógica de acción basada en foco
        if isinstance(primary_focus, dict) and 'cpu_load' in primary_focus:
            cpu = primary_focus['cpu_load']
            if cpu > self.cpu_threshold_warning:
                logger.warning(f"⚡ CONSCIOUS ACTION: Mitigating High CPU ({cpu}%)")
                
                # Validar éticamente
                if self.ethical_engine:
                    eval = self.ethical_engine.evaluate_decision("reduce_cpu_load", {}, {})
                    if eval['recommendation'] in ['proceed', 'proceed_with_caution']:
                        # Acción validada, registrar experiencia de aprendizaje
                        if self.learning_system:
                            await self.learning_system.add_learning_experience(
                                domain="system_optimization",
                                input_data={"cpu_load": cpu},
                                output_data={"action": "cleanup"},
                                performance_score=0.8
                            )

    async def _execute_unconscious_routines(self, inputs: Dict[str, Any]):
        """REAL unconscious routines - background processing when no conscious emergency"""
        try:
            # 1. Memory consolidation (if not done recently)
            if self.memory and random.random() < 0.1:  # 10% chance per cycle
                try:
                    await asyncio.to_thread(self.memory.consolidate_memories)
                    logger.debug("🧠 Unconscious: Memory consolidation completed")
                except Exception as e:
                    logger.debug(f"Memory consolidation skipped: {e}")
            
            # 2. Learning system background processing
            if self.learning_system and random.random() < 0.05:  # 5% chance per cycle
                try:
                    await self.learning_system.consolidate_learning()
                    logger.debug("🎓 Unconscious: Learning consolidation completed")
                except Exception as e:
                    logger.debug(f"Learning consolidation skipped: {e}")
            
            # 3. System metrics collection (always)
            if hasattr(self, 'system_metrics'):
                # Update metrics from inputs
                if 'hardware' in inputs:
                    self.system_metrics.update(inputs['hardware'])
            
            # 4. Health check for subsystems (occasionally)
            if random.random() < 0.02:  # 2% chance per cycle
                health_checks = []
                
                if self.rag_system:
                    try:
                        # Quick health check - try a simple query
                        test_result = self.rag_system.process_query("test", use_advanced_processing=False)
                        health_checks.append(("rag_system", "healthy" if test_result else "degraded"))
                    except Exception as e:
                        health_checks.append(("rag_system", f"error: {str(e)[:50]}"))
                
                if health_checks:
                    logger.debug(f"🏥 Unconscious health checks: {health_checks}")
            
            # 5. Cleanup old data (very rarely)
            if random.random() < 0.01:  # 1% chance per cycle
                # This would trigger cleanup of old logs, temp files, etc.
                logger.debug("🧹 Unconscious: Cleanup cycle triggered")
            
        except Exception as e:
            logger.debug(f"Error in unconscious routines: {e}")

    async def _background_maintenance(self):
        """Procesos de fondo (consolidación de memoria y entrenamiento)"""
        if self.memory:
            await asyncio.to_thread(self.memory.consolidate_memories)
        
        # Consolidar aprendizaje periódicamente
        if self.learning_system and random.random() < 0.05:
            try:
                await self.learning_system.consolidate_learning()
            except:
                pass

    # Métodos de compatibilidad
    def get_system_status_sync(self) -> Dict[str, Any]:
        return {
            "controller_status": "active_conscious" if self.running else "inactive",
            "architecture": "Global Workspace + RAG + Learning",
            "metrics": self.system_metrics,
            "modules": {
                "workspace": self.global_workspace is not None,
                "rag_system": self.rag_system is not None,
                "learning_system": self.learning_system is not None,
                "nervous": self.nervous_system is not None,
                "qualia": self.qualia is not None,
                "theory_of_mind": self.theory_of_mind is not None,
                "dna": self.dna is not None
            }
        }

# Instancia global
autonomous_controller = AutonomousSystemController()

def start_system_control():
    autonomous_controller.start_autonomous_control()

def stop_system_control():
    autonomous_controller.stop_autonomous_control()

def get_system_status():
    return autonomous_controller.get_system_status_sync()

def get_autonomous_controller():
    """Retorna la instancia global del controlador autónomo"""
    return autonomous_controller

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("🤖 SHEILY AI - COMPLETE SYSTEM (CONSCIOUSNESS + KNOWLEDGE + LEARNING)")
    logger.info("=" * 70)
    start_system_control()
    try:
        for _ in range(6):
            time.sleep(5)
            status = get_system_status()
            logger.info(f"📊 {status.get('metrics', {}).get('cpu_load')}% CPU | Modules: {status.get('modules')}")
    except KeyboardInterrupt:
        pass
    stop_system_control()
