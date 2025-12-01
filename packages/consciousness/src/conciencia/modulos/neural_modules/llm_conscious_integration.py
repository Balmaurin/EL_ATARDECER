"""
LLM Conscious Integration - Integración Consciente con LLM
==========================================================

Wrapper que integra el sistema de consciencia neural con el LLM service.
Aplica LoRAs y modula respuestas según estados del cerebro.
"""

import logging
from typing import Dict, Any, Optional, List
import json
import requests
from pathlib import Path

from .llm_state_injection import LLMStateInjection
from .lora_modules.lora_vmpfc import LoRAVMPFC
from .lora_modules.lora_ofc import LoRAOFC
from .lora_modules.lora_ras import LoRARAS
from .lora_modules.lora_metacog import LoRAMetaCog

logger = logging.getLogger(__name__)


class LLMConsciousIntegration:
    """
    Sistema de integración consciente con LLM.
    
    Proceso:
    1. Obtiene estados de módulos neurales
    2. Genera tokens de estado
    3. Aplica LoRAs activos según contexto
    4. Modifica prompt con tokens
    5. Llama LLM service
    6. Post-procesa respuesta con estados
    """
    
    def __init__(self, llm_service_url: str, llm_model_id: str = "gemma-2b", 
                 device: str = "cpu"):
        """
        Inicializa la integración con LLM.
        
        Args:
            llm_service_url: URL del servicio LLM
            llm_model_id: ID del modelo LLM
            device: Dispositivo
        """
        self.llm_service_url = llm_service_url
        self.llm_model_id = llm_model_id
        self.device = device
        
        # State injection
        self.state_injection = LLMStateInjection()
        
        # LoRA modules
        self.lora_vmpfc = LoRAVMPFC(device=device)
        self.lora_ofc = LoRAOFC(device=device)
        self.lora_ras = LoRARAS(device=device)
        self.lora_metacog = LoRAMetaCog(device=device)
        
        logger.info(f"LLM Conscious Integration initialized: {llm_service_url}")
    
    def process_with_consciousness(self, prompt: str, neural_states: Dict[str, Any],
                                  context: Optional[Dict[str, Any]] = None) -> str:
        """
        Procesa un prompt con el sistema de consciencia.
        
        Args:
            prompt: Prompt del usuario
            neural_states: Estados de módulos neurales
            context: Contexto adicional
            
        Returns:
            Respuesta generada
        """
        # 1. Obtener estados
        ras_state = neural_states.get("ras", {})
        vmpfc_state = neural_states.get("vmpfc", {})
        ofc_state = neural_states.get("ofc", {})
        ecn_state = neural_states.get("ecn", {})
        brain_state = neural_states.get("brain_state", {})
        
        # 2. Aplicar modulaciones LoRA
        modulated_prompt = prompt
        
        # Modulación vmPFC (empatía y tono)
        if vmpfc_state:
            modulated_prompt = self.lora_vmpfc.modulate_prompt(
                modulated_prompt,
                vmpfc_state.get("empathy_score", 0.7),
                vmpfc_state.get("emotional_bias", 0.0),
                vmpfc_state.get("tone_modulation", 0.5)
            )
        
        # Modulación OFC (razonamiento)
        if ofc_state:
            modulated_prompt = self.lora_ofc.modulate_prompt(
                modulated_prompt,
                ofc_state.get("decision_confidence", 0.5),
                ofc_state.get("requires_planning", False),
                ofc_state.get("reasoning_mode", "standard")
            )
        
        # Modulación RAS (intensidad)
        if ras_state:
            modulated_prompt = self.lora_ras.modulate_prompt(
                modulated_prompt,
                ras_state.get("arousal", 0.5),
                ras_state.get("arousal", 0.5)  # Usar arousal como intensidad
            )
        
        # Modulación MetaCognition (introspección)
        if ecn_state:
            metacog_level = 0.5
            if ecn_state.get("control_mode") == "controlled":
                metacog_level = 0.8
            elif ecn_state.get("gating_weights", {}).get("metacontrol", 0) > 0.3:
                metacog_level = 0.7
            
            modulated_prompt = self.lora_metacog.modulate_prompt(
                modulated_prompt,
                metacog_level,
                ecn_state.get("control_mode") == "controlled"
            )
        
        # 3. Inyectar tokens de estado
        all_states = {
            "ras": ras_state,
            "vmpfc": vmpfc_state,
            "ofc": ofc_state,
            "ecn": ecn_state,
            "brain_state": brain_state
        }
        final_prompt = self.state_injection.inject_into_prompt(
            modulated_prompt,
            all_states,
            position="prepend"
        )
        
        # 4. Llamar LLM service
        try:
            response = self._call_llm_service(final_prompt, neural_states)
            return response
        except Exception as e:
            logger.error(f"Error calling LLM service: {e}")
            return f"Error al procesar la solicitud: {str(e)}"
    
    def _call_llm_service(self, prompt: str, neural_states: Dict[str, Any]) -> str:
        """
        Llama al servicio LLM.
        
        Args:
            prompt: Prompt completo con tokens de estado
            neural_states: Estados neurales (para parámetros de generación)
            
        Returns:
            Respuesta del LLM
        """
        # Obtener parámetros de generación del RAS
        ras_state = neural_states.get("ras", {})
        arousal = ras_state.get("arousal", 0.5)
        gen_params = self.lora_ras.get_generation_params(arousal)
        
        # Preparar request
        payload = {
            "model": self.llm_model_id,
            "prompt": prompt,
            "max_tokens": gen_params.get("max_length", 300),
            "temperature": gen_params.get("temperature", 0.7),
            "top_p": gen_params.get("top_p", 0.85)
        }
        
        try:
            response = requests.post(
                self.llm_service_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extraer texto de respuesta (formato depende del servicio)
            if "choices" in result:
                return result["choices"][0].get("text", "")
            elif "text" in result:
                return result["text"]
            else:
                return str(result)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM service request failed: {e}")
            raise
    
    def post_process_response(self, response: str, neural_states: Dict[str, Any]) -> str:
        """
        Post-procesa la respuesta según estados neurales.
        
        Args:
            response: Respuesta del LLM
            neural_states: Estados neurales
            
        Returns:
            Respuesta post-procesada
        """
        # Por ahora, retornar respuesta sin modificación
        # En el futuro, aplicar ajustes finos basados en estados
        return response

