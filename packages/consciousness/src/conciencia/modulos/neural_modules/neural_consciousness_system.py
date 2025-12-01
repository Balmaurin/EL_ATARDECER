"""
Neural Consciousness System - Sistema de Consciencia Neural
==========================================================

Orquestador principal que coordina todos los módulos neurales.
Gestiona el flujo completo: RAS → vmPFC/OFC → ECN → LLM → Memoria
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .brain_state_manager import BrainStateManager
from .ras_neural import RASNeuralSystem
from .vmpfc_neural import VMPFCNeuralSystem
from .ofc_neural import OFCNeural
from .ecn_neural import ECNNeural
from .hippocampus_neural import HippocampusNeural
from .llm_conscious_integration import LLMConsciousIntegration
from .training.dataset_builder import DatasetBuilder
from .training.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


class NeuralConsciousnessSystem:
    """
    Sistema completo de consciencia neural.
    
    Coordina:
    - Inicialización de todos los módulos neurales
    - Gestión de brain_state.json
    - Flujo completo de procesamiento
    - Entrenamiento incremental
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cpu"):
        """
        Inicializa el sistema de consciencia neural.
        
        Args:
            config: Configuración del sistema
            device: Dispositivo
        """
        self.device = device
        self.config = config or {}
        
        # Directorios
        base_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent
        models_dir = base_dir / "data" / "consciousness" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Brain State Manager
        state_file = self.config.get("brain_state_file")
        self.brain_state = BrainStateManager(state_file=state_file)
        
        # Módulos neurales
        self.ras = RASNeuralSystem(
            model_path=str(models_dir / "ras_model.pt") if models_dir.exists() else None,
            device=device
        )
        
        self.vmpfc = VMPFCNeuralSystem(
            model_path=str(models_dir / "vmpfc_model.pt") if models_dir.exists() else None,
            device=device
        )
        
        self.ofc = OFCNeural(
            model_path=str(models_dir / "ofc_policy.pt") if models_dir.exists() else None,
            device=device
        )
        
        self.ecn = ECNNeural(
            model_path=str(models_dir / "ecn_moe.pt") if models_dir.exists() else None,
            device=device
        )
        
        self.hippocampus = HippocampusNeural(
            model_path=str(models_dir / "hippocampus_encoder.pt") if models_dir.exists() else None,
            index_path=str(models_dir / "hippocampus_index") if models_dir.exists() else None,
            device=device
        )
        
        # LLM Integration - Usar local GGUF si está disponible, sino servicio HTTP
        llm_model_path = self.config.get("llm_model_path")
        if llm_model_path and Path(llm_model_path).exists():
            # Usar modelo local GGUF
            try:
                from .llm_local_gguf import LLMConsciousIntegrationLocal
                llm_kwargs = {
                    "n_ctx": self.config.get("llm_n_ctx", 4096),
                    "n_threads": self.config.get("llm_n_threads", 4),
                    "chat_format": self.config.get("llm_chat_format", "chatml")
                }
                self.llm_integration = LLMConsciousIntegrationLocal(
                    model_path=llm_model_path,
                    device=device,
                    **llm_kwargs
                )
                logger.info(f"Using local GGUF model: {llm_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load local GGUF model: {e}, falling back to HTTP service")
                llm_service_url = self.config.get("llm_service_url", "http://localhost:8003/v1/completions")
                llm_model_id = self.config.get("llm_model_id", "gemma-2b")
                self.llm_integration = LLMConsciousIntegration(
                    llm_service_url=llm_service_url,
                    llm_model_id=llm_model_id,
                    device=device
                )
        else:
            # Usar servicio HTTP
            llm_service_url = self.config.get("llm_service_url", "http://localhost:8003/v1/completions")
            llm_model_id = self.config.get("llm_model_id", "gemma-2b")
            self.llm_integration = LLMConsciousIntegration(
                llm_service_url=llm_service_url,
                llm_model_id=llm_model_id,
                device=device
            )
        
        # Training
        self.dataset_builder = DatasetBuilder()
        self.training_pipeline = TrainingPipeline(device=device)
        
        # Contador de interacciones para entrenamiento
        self.interaction_count = 0
        self.training_interval = self.config.get("training_interval", 100)
        
        logger.info("NeuralConsciousnessSystem initialized")
    
    def process_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesa una entrada del usuario con el sistema completo.
        
        Args:
            user_input: Entrada del usuario
            context: Contexto adicional
            
        Returns:
            Dict con respuesta y estados neurales
        """
        context = context or {}
        
        # 1. RAS: Calcular arousal y neurotransmisores
        stimulus = {
            "intensity": context.get("intensity", 0.5),
            "urgency": context.get("urgency", 0.0),
            "novelty": context.get("novelty", 0.0),
            "emotional_valence": context.get("emotional_valence", 0.0),
            "time_since_last_event": context.get("time_since_last_event", 0.5)
        }
        
        previous_ras_state = {
            "arousal": self.brain_state.get_arousal(),
            "norepinephrine": 0.5,
            "serotonin": 0.6,
            "dopamine": 0.5,
            "acetylcholine": 0.7
        }
        
        ras_output = self.ras.process_stimulus(stimulus, previous_ras_state)
        
        # 2. vmPFC: Procesar empatía y tono
        vmpfc_context = {
            "user_message": user_input,
            "previous_emotion": previous_ras_state.get("arousal", 0.5),
            "user_profile": context.get("user_profile", {}),
            "conversation_history": context.get("conversation_history", [])
        }
        
        vmpfc_output = self.vmpfc.process_context(vmpfc_context)
        
        # 3. Recuperar memoria relevante
        memory_query = {
            "content": user_input,
            "timestamp": None,
            "context": context
        }
        relevant_memories = self.hippocampus.retrieve(memory_query, top_k=3)
        
        # 4. OFC: Seleccionar acción (simplificado)
        options = [
            {"action": "respond_directly", "expected_value": 0.8, "confidence": 0.9},
            {"action": "ask_clarification", "expected_value": 0.5, "confidence": 0.6}
        ]
        ofc_context = {
            "urgency": stimulus["urgency"],
            "importance": 0.7,
            "emotional_context": vmpfc_output["emotional_bias"],
            "cognitive_load": 0.5
        }
        ofc_action_idx, ofc_info = self.ofc.select_action(options, ofc_context)
        
        # 5. ECN: Control ejecutivo
        task = {
            "type": "response_generation",
            "priority": 0.7,
            "complexity": 0.5,
            "urgency": stimulus["urgency"],
            "requires_planning": ofc_info.get("requires_planning", False)
        }
        wm_state = {
            "items": relevant_memories[:3],
            "load": len(relevant_memories) / 7.0,
            "capacity": 7.0
        }
        ecn_output = self.ecn.process_task(task, wm_state, cognitive_load=0.5)
        
        # 6. Preparar estados neurales
        neural_states = {
            "ras": ras_output,
            "vmpfc": vmpfc_output,
            "ofc": {
                "decision_confidence": ofc_info.get("value", 0.5),
                "requires_planning": task.get("requires_planning", False),
                "reasoning_mode": "standard"
            },
            "ecn": ecn_output,
            "brain_state": self.brain_state.get_state()
        }
        
        # 7. LLM: Generar respuesta con consciencia
        llm_context = {
            "user_message": user_input,
            "relevant_memories": relevant_memories,
            **context
        }
        
        response = self.llm_integration.process_with_consciousness(
            user_input,
            neural_states,
            context=llm_context
        )
        
        # 8. Almacenar experiencia en memoria
        experience = {
            "content": f"Q: {user_input}\nA: {response}",
            "timestamp": None,
            "context": {
                "emotion": vmpfc_output["emotional_bias"],
                "arousal": ras_output["arousal"]
            },
            "relevance": 1.0
        }
        memory_id = self.hippocampus.store_memory(experience)
        
        # 9. Actualizar brain state
        self.brain_state.update_state({
            "arousal": ras_output["arousal"],
            "empathy_bias": vmpfc_output["empathy_score"],
            "confidence": ofc_info.get("value", 0.5)
        })
        
        self.brain_state.add_interaction({
            "user_input": user_input,
            "response": response,
            "memory_id": memory_id,
            "neural_states": neural_states
        })
        
        # 10. Incrementar contador y verificar entrenamiento
        self.interaction_count += 1
        if self.interaction_count % self.training_interval == 0:
            logger.info(f"Training interval reached ({self.interaction_count}), triggering training...")
            # Trigger entrenamiento (asíncrono en producción)
            # self.trigger_training()
        
        return {
            "response": response,
            "neural_states": neural_states,
            "relevant_memories": relevant_memories,
            "memory_id": memory_id
        }
    
    def trigger_training(self) -> Dict[str, Any]:
        """
        Dispara entrenamiento incremental de módulos.
        
        Returns:
            Métricas de entrenamiento
        """
        logger.info("Starting incremental training...")
        
        # Recolectar datos de Hack-Memori
        training_data = self.dataset_builder.collect_from_hack_memori()
        
        if len(training_data) < 10:
            logger.warning("Not enough training data, skipping training")
            return {"status": "skipped", "reason": "insufficient_data"}
        
        # Construir datasets
        emotional_dataset = self.dataset_builder.build_emotional_dataset(training_data)
        decision_dataset = self.dataset_builder.build_decision_dataset(training_data)
        memory_dataset = self.dataset_builder.build_memory_dataset(training_data)
        
        # Entrenar módulos
        metrics = {}
        
        # Entrenar vmPFC
        try:
            vmpfc_metrics = self.training_pipeline.train_vmpfc(
                self.vmpfc.model,
                emotional_dataset,
                epochs=2,
                batch_size=4,
                lr=1e-4
            )
            metrics["vmpfc"] = vmpfc_metrics
            
            # Guardar modelo
            base_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent
            models_dir = base_dir / "data" / "consciousness" / "models"
            self.vmpfc.save_model(str(models_dir / "vmpfc_model.pt"))
        except Exception as e:
            logger.error(f"Error training vmPFC: {e}")
            metrics["vmpfc"] = {"error": str(e)}
        
        # Entrenar RAS (similar)
        try:
            ras_metrics = self.training_pipeline.train_ras(
                self.ras.model,
                emotional_dataset,  # Reusar dataset emocional
                epochs=2,
                batch_size=4,
                lr=1e-4
            )
            metrics["ras"] = ras_metrics
            
            base_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent
            models_dir = base_dir / "data" / "consciousness" / "models"
            self.ras.save_model(str(models_dir / "ras_model.pt"))
        except Exception as e:
            logger.error(f"Error training RAS: {e}")
            metrics["ras"] = {"error": str(e)}
        
        logger.info(f"Training completed: {metrics}")
        return metrics
    
    def get_state(self) -> Dict[str, Any]:
        """Obtiene el estado completo del sistema."""
        return {
            "brain_state": self.brain_state.get_state(),
            "interaction_count": self.interaction_count,
            "modules_loaded": {
                "ras": self.ras.model is not None,
                "vmpfc": self.vmpfc.model is not None,
                "ofc": self.ofc.ppo is not None,
                "ecn": self.ecn.model is not None,
                "hippocampus": self.hippocampus.model is not None
            }
        }
    
    def shutdown(self):
        """Cierra el sistema y guarda estados."""
        logger.info("Shutting down NeuralConsciousnessSystem...")
        
        # Guardar brain state
        self.brain_state.shutdown()
        
        # Guardar modelos (si han sido entrenados)
        base_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent
        models_dir = base_dir / "data" / "consciousness" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar índices de memoria
        if self.hippocampus.index is not None:
            self.hippocampus.save_index(str(models_dir / "hippocampus_index"))
        
        logger.info("NeuralConsciousnessSystem shutdown complete")

