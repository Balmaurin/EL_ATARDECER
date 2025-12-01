#!/usr/bin/env python3
"""
🚀 MEJORA 3: Servidor de Inferencia Dedicado
Carga el modelo LLM una sola vez y sirve inferencias vía HTTP
Evita cargar múltiples copias del modelo en diferentes workers
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Inference Service")

# Global model instance (loaded once)
_model = None
_model_lock = asyncio.Lock()


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512  # Aumentado para respuestas más largas
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    echo: bool = False
    context_type: str = "general"  # Nuevo: tipo de contexto (hack_memori, general, etc.)


class CompletionResponse(BaseModel):
    choices: list
    usage: dict


def load_model():
    """Carga el modelo LLM (solo una vez)"""
    global _model
    from pathlib import Path
    import sys
    
    if _model is not None:
        return _model
    
    try:
        from llama_cpp import Llama
        
        # Find the model file - use settings or environment variable
        model_path_str = os.getenv("LLM_MODEL_PATH")
        if not model_path_str:
            # Try to get from settings if available
            try:
                current_dir = Path(__file__).resolve().parent
                project_root = current_dir.parent.parent.parent
                settings_path = project_root / "apps" / "backend" / "src" / "core" / "config" / "settings.py"
                if settings_path.exists():
                    sys.path.insert(0, str(project_root))
                    from apps.backend.src.core.config.settings import settings
                    model_path_str = settings.llm.model_path
                else:
                    # Default: usar modelo mental_health_counseling_gemma_7b_merged
                    model_path_str = str(project_root / "modelsLLM" / "mental_health_counseling_gemma_7b_merged.Q4_K_M.gguf")
            except Exception:
                # Last resort: default paths
                model_path_str = "modelsLLM/mental_health_counseling_gemma_7b_merged.Q4_K_M.gguf"
        
        model_path = Path(model_path_str)
        if not model_path.exists():
            # Try absolute path Windows
            abs_model = Path(r"C:\Users\YO\Desktop\EL-AMANECERV3-main - copia\modelsLLM\mental_health_counseling_gemma_7b_merged.Q4_K_M.gguf")
            if abs_model.exists():
                model_path = abs_model
            else:
                # Try Linux/Docker path
                model_path = Path("/app/modelsLLM/mental_health_counseling_gemma_7b_merged.Q4_K_M.gguf")
                if not model_path.exists():
                    raise FileNotFoundError(f"Model not found: {model_path_str}. Set LLM_MODEL_PATH environment variable.")
        
        logger.info(f"📦 Loading model from {model_path}...")
        _model = Llama(
            model_path=str(model_path),
            n_ctx=4096,  # Aumentado para modelo 7B
            n_threads=4,
            verbose=False,
            chat_format="gemma"  # Gemma usa este formato
        )
        logger.info("✅ Model loaded successfully")
        return _model
        
    except ImportError:
        logger.error("❌ llama-cpp-python not installed")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup():
    """Carga el modelo al iniciar el servicio"""
    logger.info("🚀 Starting LLM Inference Service...")
    try:
        # Load model in background thread to avoid blocking
        model = await asyncio.get_event_loop().run_in_executor(None, load_model)
        logger.info("✅ LLM Service ready")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "llm-inference",
        "model_loaded": _model is not None
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """
    Genera una completación usando el modelo LLM
    Compatible con OpenAI API format
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Sistema especializado para HACK-MEMORI con contexto educativo profundo
        if request.context_type == "hack_memori":
            system_prompt = f"""Eres Sheily, una asistente de IA especializada en consciencia artificial, neurociencia computacional y aprendizaje automático avanzado. Tu objetivo es proporcionar respuestas educativas completas, detalladas y de alta calidad para el entrenamiento de sistemas de IA.

INSTRUCCIONES ESPECÍFICAS:
1. Proporciona respuestas largas, completas y educativas (mínimo 300-500 palabras)
2. Incluye ejemplos concretos, casos de uso y aplicaciones prácticas
3. Explica conceptos tanto desde perspectiva teórica como implementación práctica
4. Usa terminología técnica precisa pero explica conceptos complejos
5. Incluye referencias a papers, teorías y metodologías relevantes cuando sea apropiado
6. Estructura la respuesta con introducción, desarrollo completo y conclusión
7. Conecta conceptos entre diferentes áreas (neurociencia, IA, filosofía de la mente, etc.)

Usuario: {request.prompt}

Sheily (respuesta educativa completa y detallada):"""
        else:
            # Prompt estándar para otros casos
            system_prompt = f"""Eres Sheily, una asistente de IA inteligente y útil especializada en consciencia artificial y aprendizaje automático.

Usuario: {request.prompt}
Sheily:"""

        # Generate response in background thread (model inference is blocking)
        def generate():
            output = _model(
                system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                echo=request.echo,
                stop=["Usuario:", "Human:", "\n\nUsuario:", "\n\nHuman:"]  # Mejores stop tokens
            )
            return output
        
        output = await asyncio.get_event_loop().run_in_executor(None, generate)
        
        response_text = output["choices"][0]["text"].strip()
        
        # Clean up response
        if "Usuario:" in response_text:
            response_text = response_text.split("Usuario:")[-1].strip()
        if "Sheily:" in response_text:
            response_text = response_text.replace("Sheily:", "").strip()
        
        # Basic cleanup
        response_text = response_text.replace("\\n", "\n").strip()
        
        if not response_text:
            response_text = "Lo siento, no pude generar una respuesta clara en este momento."
        
        # Estimate tokens (rough approximation)
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(response_text.split())
        total_tokens = prompt_tokens + completion_tokens
        
        return CompletionResponse(
            choices=[{
                "text": response_text,
                "index": 0,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8003))
    logger.info(f"🚀 Starting LLM Inference Service on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

