"""
Neural Consciousness API - Endpoints para Sistema Neural
========================================================

API endpoints para interactuar con el sistema de consciencia neural.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

try:
    from apps.backend.src.core.security.manager import get_current_user
    from apps.backend.src.models.base import User
except ImportError:
    # Fallback if imports fail
    get_current_user = None
    User = None

logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/api/v1/consciousness", tags=["consciousness"])


class ProcessInputRequest(BaseModel):
    """Request para procesar input."""
    user_input: str
    context: Optional[Dict[str, Any]] = None


class ProcessInputResponse(BaseModel):
    """Response de procesamiento."""
    response: str
    neural_states: Dict[str, Any]
    memory_id: Optional[int] = None
    processing_mode: str


class StateResponse(BaseModel):
    """Response de estado."""
    brain_state: Dict[str, Any]
    interaction_count: int
    modules_loaded: Dict[str, bool]


# Instancia global del sistema (se inicializará desde gateway)
_neural_system = None


def set_neural_system(system):
    """Establece la instancia del sistema neural desde el gateway."""
    global _neural_system
    _neural_system = system


def get_neural_system():
    """Obtiene la instancia del sistema neural."""
    global _neural_system
    if _neural_system is None:
        # Intentar obtener desde el contexto de FastAPI si está disponible
        try:
            from fastapi import Request
            # Esto se manejará mejor con dependency injection
            pass
        except:
            pass
        
        # Si no está disponible, intentar inicializar
        if _neural_system is None:
            try:
                from packages.consciousness.src.conciencia.modulos.neural_modules.neural_consciousness_system import NeuralConsciousnessSystem
                from apps.backend.src.core.config.settings import settings
                from pathlib import Path
                
                base_dir = Path(__file__).parent.parent.parent
                config = {
                    "llm_service_url": settings.llm_service_url,
                    "llm_model_id": settings.llm_model_id,
                    "training_interval": 100,
                    "brain_state_file": str(base_dir / "data" / "consciousness" / "brain_state.json")
                }
                _neural_system = NeuralConsciousnessSystem(config=config, device="cpu")
                logger.info("Neural consciousness system initialized (fallback)")
            except Exception as e:
                logger.error(f"Failed to initialize neural system: {e}")
                raise HTTPException(status_code=503, detail="Neural consciousness system not available")
    
    return _neural_system


@router.post("/process", response_model=ProcessInputResponse)
async def process_input(
    request: ProcessInputRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Procesa un input del usuario con el sistema de consciencia neural.
    
    Args:
        request: Request con input y contexto
        current_user: Usuario autenticado
        
    Returns:
        Response con respuesta y estados neurales
    """
    try:
        neural_system = get_neural_system()
        
        context = request.context or {}
        context["user_id"] = str(current_user.id)
        context["user_profile"] = {
            "empathy_level": 0.7,
            "openness": 0.5
        }
        
        result = neural_system.process_input(request.user_input, context)
        
        return ProcessInputResponse(
            response=result["response"],
            neural_states=result["neural_states"],
            memory_id=result.get("memory_id"),
            processing_mode="neural"
        )
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state", response_model=StateResponse)
async def get_state(
    current_user: User = Depends(get_current_user)
):
    """
    Obtiene el estado actual del sistema de consciencia.
    
    Args:
        current_user: Usuario autenticado
        
    Returns:
        Estado del sistema
    """
    try:
        neural_system = get_neural_system()
        state = neural_system.get_state()
        
        return StateResponse(
            brain_state=state["brain_state"],
            interaction_count=state["interaction_count"],
            modules_loaded=state["modules_loaded"]
        )
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def trigger_training(
    current_user: User = Depends(get_current_user)
):
    """
    Dispara entrenamiento manual del sistema neural.
    
    Args:
        current_user: Usuario autenticado
        
    Returns:
        Métricas de entrenamiento
    """
    try:
        neural_system = get_neural_system()
        metrics = neural_system.trigger_training()
        
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lora/status")
async def get_lora_status(
    current_user: User = Depends(get_current_user)
):
    """
    Obtiene el estado de los módulos LoRA.
    
    Args:
        current_user: Usuario autenticado
        
    Returns:
        Estado de LoRAs
    """
    try:
        neural_system = get_neural_system()
        llm_integration = neural_system.llm_integration
        
        return {
            "lora_vmpfc": {
                "loaded": llm_integration.lora_vmpfc.adapter_loaded,
                "path": llm_integration.lora_vmpfc.adapter_path
            },
            "lora_ofc": {
                "loaded": llm_integration.lora_ofc.adapter_loaded,
                "path": llm_integration.lora_ofc.adapter_path
            },
            "lora_ras": {
                "loaded": llm_integration.lora_ras.adapter_loaded,
                "path": llm_integration.lora_ras.adapter_path
            },
            "lora_metacog": {
                "loaded": llm_integration.lora_metacog.adapter_loaded,
                "path": llm_integration.lora_metacog.adapter_path
            }
        }
    except Exception as e:
        logger.error(f"Error getting LoRA status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

