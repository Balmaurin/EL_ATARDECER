"""
⏰ MEJORA 2: Temporal.io Workflows para Hack-Memori
Reemplaza los loops asyncio en memoria por workflows persistentes
Garantiza que los entrenamientos continúen aunque el servidor se reinicie
"""

import logging
from datetime import timedelta
from temporalio import workflow, activity
from temporalio.common import RetryPolicy
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@activity.defn
async def generate_question_activity(session_id: str) -> Dict[str, Any]:
    """Activity para generar una pregunta en una sesión"""
    try:
        from apps.backend.hack_memori_service import HackMemoriService
        
        service = HackMemoriService()
        session = service.get_session(session_id)
        
        if not session or session.get("status") != "RUNNING":
            return {"status": "stopped", "session_id": session_id}
        
        # Generate question
        question_text = service._generate_question(session_id)
        question = service.add_question(session_id, question_text, origin="temporal")
        
        return {
            "status": "success",
            "session_id": session_id,
            "question_id": question["id"],
            "question_text": question_text
        }
    except Exception as e:
        logger.error(f"Error in generate_question_activity: {e}")
        return {"status": "error", "error": str(e)}


@activity.defn
async def generate_response_activity(session_id: str, question_id: str, question_text: str) -> Dict[str, Any]:
    """Activity para generar una respuesta usando el servidor de inferencia"""
    try:
        from apps.backend.hack_memori_service import HackMemoriService
        import httpx
        
        service = HackMemoriService()
        
        # MEJORA 3: Llamar al servidor de inferencia dedicado
        from apps.backend.src.core.config.settings import settings
        llm_service_url = settings.llm_service_url
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                llm_service_url,
                json={
                    "prompt": question_text,
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40
                }
            )
            response.raise_for_status()
            result = response.json()
            
            response_text = result.get("choices", [{}])[0].get("text", "").strip()
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
        
        # Add response
        from apps.backend.src.core.config.settings import settings
        service.add_response(
            question_id, session_id, settings.llm_model_id,
            question_text, response_text, tokens_used
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "question_id": question_id,
            "response_text": response_text,
            "tokens_used": tokens_used
        }
    except Exception as e:
        logger.error(f"Error in generate_response_activity: {e}")
        return {"status": "error", "error": str(e)}


@activity.defn
async def check_session_status_activity(session_id: str) -> Dict[str, Any]:
    """Activity para verificar el estado de una sesión"""
    try:
        from apps.backend.hack_memori_service import HackMemoriService
        
        service = HackMemoriService()
        session = service.get_session(session_id)
        
        if not session:
            return {"status": "not_found", "session_id": session_id}
        
        return {
            "status": "found",
            "session_id": session_id,
            "session_status": session.get("status"),
            "is_running": session.get("status") == "RUNNING"
        }
    except Exception as e:
        logger.error(f"Error in check_session_status_activity: {e}")
        return {"status": "error", "error": str(e)}


@workflow.defn
class TrainingSessionWorkflow:
    """
    Workflow de Temporal para sesiones de entrenamiento Hack-Memori
    Persiste el estado y puede reanudarse después de reinicios
    """
    
    def __init__(self) -> None:
        self.session_id: Optional[str] = None
        self.iteration_count: int = 0
    
    @workflow.run
    async def run(self, session_id: str) -> Dict[str, Any]:
        """
        Ejecuta el loop de entrenamiento de forma persistente
        
        Si el servidor se cae, Temporal reanuda exactamente donde se quedó
        """
        self.session_id = session_id
        logger.info(f"[TEMPORAL] Starting workflow for session: {session_id}")
        
        # Retry policy para actividades
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=60),
            maximum_attempts=5
        )
        
        while True:
            try:
                # Verificar estado de la sesión (persistente)
                status_result = await workflow.execute_activity(
                    check_session_status_activity,
                    session_id,
                    start_to_close_timeout=timedelta(seconds=10),
                    retry_policy=retry_policy
                )
                
                if not status_result.get("is_running"):
                    logger.info(f"[TEMPORAL] Session {session_id} stopped, ending workflow")
                    break
                
                # Generar pregunta (persistente)
                question_result = await workflow.execute_activity(
                    generate_question_activity,
                    session_id,
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=retry_policy
                )
                
                if question_result.get("status") != "success":
                    logger.warning(f"[TEMPORAL] Failed to generate question: {question_result}")
                    await workflow.sleep(timedelta(seconds=30))  # Sleep persistente
                    continue
                
                question_id = question_result["question_id"]
                question_text = question_result["question_text"]
                
                # Generar respuesta (persistente)
                response_result = await workflow.execute_activity(
                    generate_response_activity,
                    session_id,
                    question_id,
                    question_text,
                    start_to_close_timeout=timedelta(seconds=120),
                    retry_policy=retry_policy
                )
                
                if response_result.get("status") == "success":
                    self.iteration_count += 1
                    logger.info(
                        f"[TEMPORAL] Iteration {self.iteration_count} complete for session {session_id}"
                    )
                
                # Obtener delay de configuración
                from apps.backend.hack_memori_service import HackMemoriService
                service = HackMemoriService()
                session = service.get_session(session_id)
                config = session.get("config", {}) if session else {}
                delay_seconds = config.get("frequency", 30)
                
                # Sleep persistente - si el servidor se cae aquí, Temporal reanuda después del sleep
                await workflow.sleep(timedelta(seconds=delay_seconds))
                
            except Exception as e:
                logger.error(f"[TEMPORAL] Error in workflow loop: {e}")
                # Sleep antes de reintentar
                await workflow.sleep(timedelta(seconds=30))
        
        return {
            "status": "completed",
            "session_id": session_id,
            "iterations": self.iteration_count
        }

