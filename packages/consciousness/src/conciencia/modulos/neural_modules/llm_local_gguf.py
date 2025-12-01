"""
LLM Local GGUF Integration - Integración con Modelo Local GGUF
==============================================================

Wrapper para usar modelos GGUF locales con llama.cpp en lugar de servicio HTTP.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import subprocess
import json
import tempfile
import os

logger = logging.getLogger(__name__)

# Verificar si llama-cpp-python está disponible
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available, local GGUF models cannot be used")


class LLMLocalGGUF:
    """
    Integración con modelo GGUF local usando llama.cpp.
    
    Soporta modelos como Gemma, LLaMA, Mistral, etc.
    """
    
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 4, 
                 chat_format: str = "chatml"):
        """
        Inicializa el modelo GGUF local.
        
        Args:
            model_path: Ruta al archivo .gguf
            n_ctx: Tamaño del contexto
            n_threads: Número de threads para CPU
            chat_format: Formato de chat (chatml, llama-2, etc.)
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.chat_format = chat_format
        self.model = None
        
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required for local GGUF models. Install with: pip install llama-cpp-python")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo GGUF."""
        try:
            logger.info(f"Loading GGUF model from {self.model_path}")
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            logger.info("GGUF model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading GGUF model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.7, 
                 top_p: float = 0.85, **kwargs) -> str:
        """
        Genera texto usando el modelo local.
        
        Args:
            prompt: Prompt de entrada
            max_tokens: Máximo número de tokens
            temperature: Temperatura para sampling
            top_p: Top-p sampling
            **kwargs: Parámetros adicionales
            
        Returns:
            Texto generado
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Generar con el modelo
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["</s>", "\n\n\n"],  # Stop sequences
                echo=False,
                **kwargs
            )
            
            # Extraer texto
            if isinstance(response, dict):
                text = response.get("choices", [{}])[0].get("text", "")
            elif isinstance(response, str):
                text = response
            else:
                text = str(response)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def generate_chat(self, messages: list, max_tokens: int = 300, 
                     temperature: float = 0.7, top_p: float = 0.85) -> str:
        """
        Genera respuesta en formato chat.
        
        Args:
            messages: Lista de mensajes [{"role": "user", "content": "..."}]
            max_tokens: Máximo número de tokens
            temperature: Temperatura
            top_p: Top-p
            
        Returns:
            Respuesta del asistente
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Construir prompt en formato ChatML para Gemma
        if self.chat_format == "chatml":
            prompt = self._format_chatml(messages)
        else:
            # Formato simple
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            prompt += "\nassistant: "
        
        return self.generate(prompt, max_tokens, temperature, top_p)
    
    def _format_chatml(self, messages: list) -> str:
        """
        Formatea mensajes en formato ChatML (usado por Gemma).
        
        Args:
            messages: Lista de mensajes
            
        Returns:
            Prompt formateado
        """
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"
        
        formatted += "<|assistant|>\n"
        return formatted


class LLMConsciousIntegrationLocal:
    """
    Versión local de LLMConsciousIntegration que usa modelo GGUF.
    """
    
    def __init__(self, model_path: str, device: str = "cpu", **llm_kwargs):
        """
        Inicializa integración con modelo local.
        
        Args:
            model_path: Ruta al modelo GGUF
            device: Dispositivo (siempre "cpu" para GGUF)
            **llm_kwargs: Parámetros adicionales para el modelo
        """
        from .llm_state_injection import LLMStateInjection
        from .lora_modules.lora_vmpfc import LoRAVMPFC
        from .lora_modules.lora_ofc import LoRAOFC
        from .lora_modules.lora_ras import LoRARAS
        from .lora_modules.lora_metacog import LoRAMetaCog
        
        # Cargar modelo local
        self.llm = LLMLocalGGUF(model_path, **llm_kwargs)
        
        # State injection
        self.state_injection = LLMStateInjection()
        
        # LoRA modules (placeholder, no se pueden usar con GGUF directamente)
        self.lora_vmpfc = LoRAVMPFC(device=device)
        self.lora_ofc = LoRAOFC(device=device)
        self.lora_ras = LoRARAS(device=device)
        self.lora_metacog = LoRAMetaCog(device=device)
        
        logger.info(f"LLM Local GGUF Integration initialized: {model_path}")
    
    def process_with_consciousness(self, prompt: str, neural_states: Dict[str, Any],
                                  context: Optional[Dict[str, Any]] = None) -> str:
        """
        Procesa un prompt con el sistema de consciencia usando modelo local.
        
        Args:
            prompt: Prompt del usuario
            neural_states: Estados de módulos neurales
            context: Contexto adicional
            
        Returns:
            Respuesta generada
        """
        # Obtener estados
        ras_state = neural_states.get("ras", {})
        vmpfc_state = neural_states.get("vmpfc", {})
        ofc_state = neural_states.get("ofc", {})
        ecn_state = neural_states.get("ecn", {})
        brain_state = neural_states.get("brain_state", {})
        
        # Aplicar modulaciones LoRA (inyección de tokens)
        modulated_prompt = prompt
        
        # Modulación vmPFC
        if vmpfc_state:
            modulated_prompt = self.lora_vmpfc.modulate_prompt(
                modulated_prompt,
                vmpfc_state.get("empathy_score", 0.7),
                vmpfc_state.get("emotional_bias", 0.0),
                vmpfc_state.get("tone_modulation", 0.5)
            )
        
        # Modulación OFC
        if ofc_state:
            modulated_prompt = self.lora_ofc.modulate_prompt(
                modulated_prompt,
                ofc_state.get("decision_confidence", 0.5),
                ofc_state.get("requires_planning", False),
                ofc_state.get("reasoning_mode", "standard")
            )
        
        # Modulación RAS
        if ras_state:
            modulated_prompt = self.lora_ras.modulate_prompt(
                modulated_prompt,
                ras_state.get("arousal", 0.5),
                ras_state.get("arousal", 0.5)
            )
        
        # Modulación MetaCognition
        if ecn_state:
            metacog_level = 0.5
            if ecn_state.get("control_mode") == "controlled":
                metacog_level = 0.8
            
            modulated_prompt = self.lora_metacog.modulate_prompt(
                modulated_prompt,
                metacog_level,
                ecn_state.get("control_mode") == "controlled"
            )
        
        # Inyectar tokens de estado
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
        
        # Obtener parámetros de generación del RAS
        arousal = ras_state.get("arousal", 0.5)
        if arousal > 0.7:
            gen_params = {"temperature": 0.8, "max_tokens": 500, "top_p": 0.9}
        elif arousal > 0.4:
            gen_params = {"temperature": 0.7, "max_tokens": 300, "top_p": 0.85}
        else:
            gen_params = {"temperature": 0.6, "max_tokens": 200, "top_p": 0.8}
        
        # Generar con modelo local
        try:
            response = self.llm.generate(
                final_prompt,
                max_tokens=gen_params["max_tokens"],
                temperature=gen_params["temperature"],
                top_p=gen_params["top_p"]
            )
            return response
        except Exception as e:
            logger.error(f"Error generating with local model: {e}")
            return f"Error al procesar la solicitud: {str(e)}"

