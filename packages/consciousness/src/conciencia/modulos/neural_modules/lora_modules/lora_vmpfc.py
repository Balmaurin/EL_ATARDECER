"""
LoRA-vmPFC - Modulación de Empatía y Tono
==========================================

LoRA adapter que regula empatía, tono y suavidad del LLM
basado en el estado del vmPFC neural.
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
    logger.warning("PEFT library not available, LoRA modules will be disabled")

logger = logging.getLogger(__name__)


class LoRAVMPFC:
    """
    LoRA adapter para modulación de empatía y tono.
    
    Regula:
    - Empatía en respuestas
    - Tono (suave, cálido, profesional)
    - Sensibilidad emocional
    """
    
    def __init__(self, base_model_name: Optional[str] = None, 
                 adapter_path: Optional[str] = None, device: str = "cpu"):
        """
        Inicializa LoRA-vmPFC.
        
        Args:
            base_model_name: Nombre del modelo base (si se quiere entrenar)
            adapter_path: Ruta al adapter guardado
            device: Dispositivo
        """
        self.device = device
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.adapter_loaded = False
        
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not available, LoRA-vmPFC will use placeholder mode")
            return
        
        if adapter_path and Path(adapter_path).exists():
            self.load_adapter(adapter_path)
        elif base_model_name:
            # Inicializar para entrenamiento (opcional)
            logger.info(f"LoRA-vmPFC initialized for training with {base_model_name}")
    
    def modulate_prompt(self, prompt: str, empathy_score: float, 
                      emotional_bias: float, tone_modulation: float) -> str:
        """
        Modula el prompt basado en estados del vmPFC.
        
        Args:
            prompt: Prompt original
            empathy_score: Score de empatía (0-1)
            emotional_bias: Bias emocional (-1 to 1)
            tone_modulation: Modulación de tono (0-1)
            
        Returns:
            Prompt modulado
        """
        # Si no hay adapter cargado, usar inyección de tokens
        if not self.adapter_loaded:
            # Inyectar tokens de estado
            state_tokens = f"<STATE:empathy={empathy_score:.2f}>"
            state_tokens += f"<STATE:emotion_bias={emotional_bias:.2f}>"
            state_tokens += f"<STATE:tone={tone_modulation:.2f}>"
            
            # Determinar tono verbal
            if tone_modulation > 0.7:
                tone_instruction = "Responde con un tono cálido y empático."
            elif tone_modulation > 0.4:
                tone_instruction = "Responde con un tono amigable y profesional."
            else:
                tone_instruction = "Responde con un tono profesional y directo."
            
            modulated_prompt = f"{state_tokens}\n{tone_instruction}\n\n{prompt}"
            return modulated_prompt
        
        # Si hay adapter, el prompt se modula durante la generación
        return prompt
    
    def create_adapter_config(self, rank: int = 8, alpha: int = 16) -> Optional[Dict[str, Any]]:
        """
        Crea configuración de LoRA adapter.
        
        Args:
            rank: Rank de LoRA (8-16 para CPU)
            alpha: Alpha de LoRA
            
        Returns:
            Configuración de LoRA o None si PEFT no disponible
        """
        if not PEFT_AVAILABLE:
            return None
        
        config = {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],  # Para modelos LLaMA-like
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        }
        
        return config
    
    def load_adapter(self, adapter_path: str) -> bool:
        """
        Carga un adapter LoRA guardado.
        
        Args:
            adapter_path: Ruta al adapter
            
        Returns:
            True si se cargó exitosamente
        """
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not available, cannot load adapter")
            return False
        
        try:
            # En producción, cargar adapter con PEFT
            # self.model = PeftModel.from_pretrained(base_model, adapter_path)
            # self.adapter_loaded = True
            logger.info(f"LoRA-vmPFC adapter loaded from {adapter_path} (placeholder)")
            self.adapter_loaded = True
            return True
        except Exception as e:
            logger.error(f"Error loading LoRA-vmPFC adapter: {e}")
            return False
    
    def save_adapter(self, path: str) -> bool:
        """
        Guarda el adapter LoRA.
        
        Args:
            path: Ruta donde guardar
            
        Returns:
            True si se guardó exitosamente
        """
        if not self.adapter_loaded:
            logger.warning("No adapter loaded to save")
            return False
        
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # En producción: self.model.save_pretrained(path)
            logger.info(f"LoRA-vmPFC adapter saved to {path} (placeholder)")
            return True
        except Exception as e:
            logger.error(f"Error saving LoRA-vmPFC adapter: {e}")
            return False
