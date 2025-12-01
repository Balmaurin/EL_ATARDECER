"""
LoRA-OFC - Modulación de Razonamiento y Planificación
=====================================================

LoRA adapter que regula razonamiento, planificación y pasos del LLM
basado en el estado del OFC neural.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)


class LoRAOFC:
    """
    LoRA adapter para modulación de razonamiento y planificación.
    
    Regula:
    - Razonamiento paso a paso
    - Planificación de acciones
    - Estructura de respuestas
    """
    
    def __init__(self, base_model_name: Optional[str] = None, 
                 adapter_path: Optional[str] = None, device: str = "cpu"):
        """
        Inicializa LoRA-OFC.
        
        Args:
            base_model_name: Nombre del modelo base
            adapter_path: Ruta al adapter guardado
            device: Dispositivo
        """
        self.device = device
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.adapter_loaded = False
        
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not available, LoRA-OFC will use placeholder mode")
            return
        
        if adapter_path and Path(adapter_path).exists():
            self.load_adapter(adapter_path)
    
    def modulate_prompt(self, prompt: str, decision_confidence: float, 
                       requires_planning: bool, reasoning_mode: str = "standard") -> str:
        """
        Modula el prompt basado en estados del OFC.
        
        Args:
            prompt: Prompt original
            decision_confidence: Confianza en la decisión (0-1)
            requires_planning: Si requiere planificación
            reasoning_mode: Modo de razonamiento ("step_by_step", "standard", "quick")
            
        Returns:
            Prompt modulado
        """
        if not self.adapter_loaded:
            # Inyectar tokens de estado
            state_tokens = f"<STATE:decision_confidence={decision_confidence:.2f}>"
            state_tokens += f"<STATE:requires_planning={str(requires_planning).lower()}>"
            state_tokens += f"<STATE:reasoning_mode={reasoning_mode}>"
            
            # Instrucciones de razonamiento
            if requires_planning:
                reasoning_instruction = "Responde con un plan paso a paso, explicando cada paso claramente."
            elif reasoning_mode == "step_by_step":
                reasoning_instruction = "Responde razonando paso a paso."
            else:
                reasoning_instruction = "Responde de manera clara y estructurada."
            
            modulated_prompt = f"{state_tokens}\n{reasoning_instruction}\n\n{prompt}"
            return modulated_prompt
        
        return prompt
    
    def load_adapter(self, adapter_path: str) -> bool:
        """Carga el adapter LoRA."""
        if not PEFT_AVAILABLE:
            return False
        
        try:
            logger.info(f"LoRA-OFC adapter loaded from {adapter_path} (placeholder)")
            self.adapter_loaded = True
            return True
        except Exception as e:
            logger.error(f"Error loading LoRA-OFC adapter: {e}")
            return False
    
    def save_adapter(self, path: str) -> bool:
        """Guarda el adapter LoRA."""
        if not self.adapter_loaded:
            return False
        
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"LoRA-OFC adapter saved to {path} (placeholder)")
            return True
        except Exception as e:
            logger.error(f"Error saving LoRA-OFC adapter: {e}")
            return False

