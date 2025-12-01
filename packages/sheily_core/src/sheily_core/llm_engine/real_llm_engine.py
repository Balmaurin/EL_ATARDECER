#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sheily LLM Engine - Real Model Integration
=========================================

Integración real con modelos GGUF usando llama.cpp para reemplazar los fallbacks.
Soporte para Llama 3.2 1B Q4 y otros modelos compatibles.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sheily_core.config import get_config
from sheily_core.logger import get_logger

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuración del motor LLM"""

    model_path: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    context_length: int = 2048
    threads: int = 4
    verbose: bool = False


class RealLLMEngine:
    """
    Motor LLM real usando llama-cpp-python para modelos GGUF
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.logger = get_logger("llm_engine")
        self.config = config or self._create_default_config()
        self.model = None
        self.model_loaded = False

        # Validate configuration
        self._validate_config()

        # Try to initialize model
        self._initialize_model()

    def _create_default_config(self) -> LLMConfig:
        """Crear configuración por defecto"""
        # Try to find model
        model_path = self._find_model()

        return LLMConfig(
            model_path=model_path,
            max_tokens=256,
            temperature=0.7,
            context_length=2048,
            threads=min(4, os.cpu_count() or 4),
        )

    def _find_model(self) -> str:
        """Buscar modelo GGUF disponible (priorizando Gemma2)"""
        # Priorizar Gemma2
        model_locations = [
            # User's specific model location
            "modelsLLM/gemma-2-2b-it-Q4_K_M.gguf",
            str(Path(os.getcwd()) / "modelsLLM" / "gemma-2-2b-it-Q4_K_M.gguf"),
            # Docker paths
            "/app/models/gemma-2-9b-it-Q4_K_M.gguf",
            "/app/models/gemma-2-9b-it.gguf",
            # Local paths
            "./models/gemma-2-9b-it-Q4_K_M.gguf",
            "./models/gemma-2-9b-it.gguf",
            "models/gemma-2-9b-it-Q4_K_M.gguf",
            "models/gemma-2-9b-it.gguf",
            # Buscar cualquier Gemma en models/
            str(Path(os.getcwd()) / "models" / "gemma-2-9b-it-Q4_K_M.gguf"),
            str(Path(os.getcwd()) / "models" / "gemma-2-9b-it.gguf"),
            # Fallback a otros modelos
            "/app/models/llama-3.2-1b-instruct.gguf",
            "/app/models/llama-3.2-3b-instruct.gguf",
            "./models/llama-3.2-1b-instruct.gguf",
            "./models/llama-3.2-3b-instruct.gguf",
            # Buscar cualquier .gguf en models/
            os.path.expanduser("~/models/gemma-2-9b-it-Q4_K_M.gguf"),
        ]

        # Buscar dinámicamente en modelsLLM/ primero (usuario específico)
        modelsllm_dir = Path("modelsLLM")
        if modelsllm_dir.exists():
            for pattern in ["gemma-2-2b*", "gemma-2*", "*.gguf"]:
                for gguf_file in modelsllm_dir.glob(pattern):
                    if gguf_file.is_file():
                        model_path = str(gguf_file.absolute())
                        # Priorizar el modelo específico del usuario
                        if "gemma-2-2b-it-Q4_K_M" in model_path:
                            model_locations.insert(0, model_path)
                        elif "gemma" in model_path.lower():
                            model_locations.insert(1, model_path)
                        else:
                            model_locations.insert(2, model_path)

        # Buscar dinámicamente en models/ si existe
        models_dir = Path("models")
        if models_dir.exists():
            # Priorizar Gemma2
            for pattern in ["gemma-2-9b*", "gemma-2*", "*.gguf"]:
                for gguf_file in models_dir.glob(pattern):
                    if gguf_file.is_file():
                        model_path = str(gguf_file.absolute())
                        if "gemma" in model_path.lower():
                            # Priorizar Gemma2
                            model_locations.insert(0, model_path)
                        else:
                            model_locations.append(model_path)

        for location in model_locations:
            path = Path(location)
            if path.exists() and path.is_file():
                self.logger.info(f"Found model at: {path.absolute()}")
                return str(path.absolute())

        self.logger.warning("No GGUF model found in standard locations")
        return ""

    def _validate_config(self):
        """Validar configuración"""
        if not self.config.model_path:
            raise ValueError("Model path is required")

        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")

        # Check model size
        model_size = Path(self.config.model_path).stat().st_size / (1024 * 1024)  # MB
        self.logger.info(f"Model size: {model_size:.1f} MB")

        if model_size < 100:
            self.logger.warning("Model file seems too small, might be corrupted")

        self.logger.info("LLM configuration validated successfully")

    def _initialize_model(self):
        """Inicializar modelo"""
        try:
            self.logger.info(f"Initializing model: {self.config.model_path}")

            # Import llama-cpp-python here to avoid import errors if not installed
            from llama_cpp import Llama

            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.context_length,
                n_threads=self.config.threads,
                verbose=self.config.verbose,
            )

            self.model_loaded = True
            self.logger.info("Model initialized successfully with llama-cpp-python")

        except ImportError:
            self.logger.error(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )
            self.model_loaded = False
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            self.model_loaded = False

    def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generar respuesta usando el modelo real
        """
        if not self.model_loaded or not self.model:
            raise RuntimeError("Modelo no cargado. Initialize primero.")

        start_time = time.time()

        # Merge kwargs with config
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)

        try:
            # Prepare enhanced prompt with Sheily personality
            enhanced_prompt = f"""Eres Sheily, una asistente de IA inteligente y útil. Respondes de manera clara, precisa y amigable.

Usuario: {prompt}
Sheily:"""

            # Generate response
            output = self.model(
                enhanced_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                echo=False,
            )

            response = output["choices"][0]["text"].strip()

            # Clean up response
            if "Usuario:" in response:
                response = response.split("Usuario:")[-1].strip()
            if "Sheily:" in response:
                response = response.replace("Sheily:", "").strip()

            # Basic cleanup
            response = response.replace("\\n", "\n").strip()

            if not response:
                response = "Lo siento, no pude generar una respuesta clara."

            processing_time = time.time() - start_time

            return {
                "response": response,
                "processing_time": processing_time,
                "model_used": "gemma-2-9b-gguf",
                "model_info": {
                    "path": self.config.model_path,
                    "size_mb": Path(self.config.model_path).stat().st_size
                    / (1024 * 1024),
                },
                "success": True,
                "method": "llama_cpp_python",
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Generation failed: {e}")

            return {
                "response": "Lo siento, ocurrió un error al procesar tu solicitud.",
                "processing_time": processing_time,
                "model_used": "gemma-2-9b-gguf",
                "success": False,
                "error": str(e),
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.config.model_path,
            "model_info": {
                "size_mb": (
                    Path(self.config.model_path).stat().st_size / (1024 * 1024)
                    if self.config.model_path
                    else 0
                ),
            },
            "config": {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "context_length": self.config.context_length,
                "threads": self.config.threads,
            },
        }

    def is_available(self) -> bool:
        """Verificar si el motor está disponible"""
        return self.model_loaded and self.model is not None


# Factory function
def create_real_llm_engine(config: Optional[LLMConfig] = None) -> RealLLMEngine:
    """Crear instancia del motor LLM real"""
    return RealLLMEngine(config)


# Test function
def test_llm_engine():
    """Test del motor LLM"""
    logger.info("🧪 Testing Real LLM Engine")
    logger.info("=" * 40)
    try:
        engine = create_real_llm_engine()

        logger.info(f"Model loaded: {engine.model_loaded}")
        logger.info(f"Model info: {engine.get_model_info()}")
        if engine.is_available():
            test_prompts = [
                "Hola, ¿cómo estás?",
                "¿Qué es Python?",
                "Explica la ley de Ohm",
            ]

            for prompt in test_prompts:
                logger.info(f"\n📝 Prompt: {prompt}")
                result = engine.generate_response(prompt)
                logger.info(f"   Response: {result['response'][:100]}...")
                logger.info(f"   Time: {result['processing_time']:.2f}s")
                logger.info(f"   Method: {result.get('method', 'unknown')}")
        else:
            logger.error("❌ Engine not available")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_llm_engine()
