"""
Biological Consciousness Neural Wrapper
========================================

Wrapper que integra el sistema neural con BiologicalConsciousnessSystem.
Mantiene compatibilidad con el sistema existente mientras usa módulos neurales.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from .neural_modules.neural_consciousness_system import NeuralConsciousnessSystem
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    logger.warning("Neural modules not available, using original logic only")

logger = logging.getLogger(__name__)


class BiologicalConsciousnessNeuralWrapper:
    """
    Wrapper que integra módulos neurales con BiologicalConsciousnessSystem.
    
    Estrategia:
    - Si módulos neurales están disponibles y cargados → usar neurales
    - Si no → usar lógica original como fallback
    - Permite migración gradual
    """
    
    def __init__(self, biological_system, neural_config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el wrapper.
        
        Args:
            biological_system: Instancia de BiologicalConsciousnessSystem
            neural_config: Configuración para sistema neural
        """
        self.biological_system = biological_system
        self.neural_system = None
        self.use_neural = False
        
        if NEURAL_AVAILABLE:
            try:
                self.neural_system = NeuralConsciousnessSystem(
                    config=neural_config or {},
                    device="cpu"
                )
                self.use_neural = True
                logger.info("Neural modules loaded, using neural processing")
            except Exception as e:
                logger.warning(f"Failed to load neural modules: {e}, using original logic")
                self.use_neural = False
        else:
            logger.info("Neural modules not available, using original logic only")
    
    def process_experience(self, sensory_input: Dict[str, float], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa una experiencia usando neural o lógica original.
        
        Args:
            sensory_input: Entrada sensorial
            context: Contexto adicional
            
        Returns:
            Resultado del procesamiento
        """
        if self.use_neural and self.neural_system:
            # Usar sistema neural
            try:
                # Convertir sensory_input a formato de texto si es necesario
                user_input = context.get("user_message", "")
                if not user_input and sensory_input:
                    # Crear mensaje desde sensory_input
                    user_input = str(sensory_input)
                
                # Procesar con sistema neural
                result = self.neural_system.process_input(user_input, context)
                
                # Convertir a formato compatible con BiologicalConsciousnessSystem
                neural_result = {
                    "conscious_experience": {
                        "content": result["response"],
                        "intensity": result["neural_states"]["ras"]["arousal"],
                        "valence": result["neural_states"]["vmpfc"]["emotional_bias"],
                        "qualia": {
                            "empathy": result["neural_states"]["vmpfc"]["empathy_score"],
                            "confidence": result["neural_states"]["ofc"]["decision_confidence"]
                        }
                    },
                    "neural_states": result["neural_states"],
                    "memory_id": result.get("memory_id"),
                    "processing_mode": "neural"
                }
                
                return neural_result
                
            except Exception as e:
                logger.error(f"Error in neural processing: {e}, falling back to original")
                self.use_neural = False
        
        # Fallback a lógica original
        return self.biological_system.process_experience(sensory_input, context)
    
    def get_state(self) -> Dict[str, Any]:
        """Obtiene el estado del sistema (neural o original)."""
        if self.use_neural and self.neural_system:
            return {
                "mode": "neural",
                "state": self.neural_system.get_state()
            }
        else:
            return {
                "mode": "original",
                "state": getattr(self.biological_system, "get_state", lambda: {})()
            }
    
    def enable_neural(self, enable: bool = True):
        """
        Habilita o deshabilita el uso de módulos neurales.
        
        Args:
            enable: Si True, usar neurales; si False, usar lógica original
        """
        if enable and not self.use_neural:
            if self.neural_system:
                self.use_neural = True
                logger.info("Neural processing enabled")
            else:
                logger.warning("Cannot enable neural: neural system not available")
        elif not enable and self.use_neural:
            self.use_neural = False
            logger.info("Neural processing disabled, using original logic")
    
    def shutdown(self):
        """Cierra el sistema."""
        if self.neural_system:
            self.neural_system.shutdown()

