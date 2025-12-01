#!/usr/bin/env python3
"""
Script de Recuperación de Entrenamientos Perdidos - Hack-Memori
================================================================

Este script detecta sesiones con 100+ Q&A que no fueron entrenadas
y ejecuta los entrenamientos retroactivamente.

Uso:
    python scripts/recover_hack_memori_training.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.backend.hack_memori_service import HackMemoriService
from packages.sheily_core.src.sheily_core.training.integral_trainer import ComponentTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def detect_unprocessed_sessions() -> List[Dict[str, Any]]:
    """
    Detectar sesiones con 100+ Q&A que no tienen reportes de entrenamiento
    """
    service = HackMemoriService()
    sessions = service.get_sessions()
    unprocessed = []
    
    # Obtener sesiones ya procesadas (con reportes de entrenamiento)
    reports_dir = Path("data/hack_memori/training_reports")
    processed_sessions = set()
    if reports_dir.exists():
        for report_file in reports_dir.glob("training_report_*.json"):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    session_id = report_data.get("session_id")
                    if session_id:
                        processed_sessions.add(session_id)
            except Exception as e:
                logger.warning(f"Error leyendo reporte {report_file}: {e}")
    
    logger.info(f"📊 Sesiones procesadas encontradas: {len(processed_sessions)}")
    
    # Verificar cada sesión
    for session in sessions:
        session_id = session.get("session_id") or session.get("id")
        if not session_id:
            continue
        
        # Saltar si ya fue procesada
        if session_id in processed_sessions:
            logger.info(f"✅ Sesión {session_id} ya fue entrenada")
            continue
        
        # Contar Q&A en la sesión
        qa_count = service._get_session_qa_count(session_id)
        
        if qa_count >= 100:
            unprocessed.append({
                "session_id": session_id,
                "session_name": session.get("name", "Unknown"),
                "qa_count": qa_count,
                "created_at": session.get("created_at", ""),
                "status": session.get("status", "unknown")
            })
            logger.info(f"🔍 Sesión {session_id}: {qa_count} Q&A (NO procesada)")
        else:
            logger.debug(f"⏭️ Sesión {session_id}: {qa_count} Q&A (insuficiente)")
    
    return unprocessed


async def recover_training_for_session(session_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecutar entrenamiento para una sesión específica
    """
    session_id = session_info["session_id"]
    qa_count = session_info["qa_count"]
    
    logger.info("=" * 80)
    logger.info(f"🔄 RECUPERANDO ENTRENAMIENTO PARA SESIÓN: {session_id}")
    logger.info(f"📊 Q&A disponibles: {qa_count}")
    logger.info("=" * 80)
    
    try:
        # Inicializar entrenador
        trainer = ComponentTrainer(base_path="data/hack_memori")
        
        # Ejecutar entrenamiento
        training_result = await trainer.train_all_components(trigger_threshold=100)
        
        # Verificar resultado
        if training_result.get("status") == "insufficient_data":
            logger.warning(f"⚠️ Datos insuficientes para sesión {session_id}")
            return {
                "session_id": session_id,
                "success": False,
                "error": "Datos insuficientes",
                "result": training_result
            }
        
        # Guardar reporte
        service = HackMemoriService()
        user_id = 1  # Default user ID para recuperación
        service._save_training_report(session_id, training_result, 0)
        
        logger.info("=" * 80)
        logger.info(f"✅ ENTRENAMIENTO RECUPERADO PARA SESIÓN: {session_id}")
        logger.info(f"📊 Componentes entrenados: {training_result.get('components_trained', 0)}")
        logger.info(f"📊 Componentes mejorados: {training_result.get('components_improved', 0)}")
        logger.info(f"📊 Éxito general: {training_result.get('overall_success', False)}")
        logger.info("=" * 80)
        
        return {
            "session_id": session_id,
            "success": True,
            "result": training_result,
            "components_trained": training_result.get("components_trained", 0),
            "components_improved": training_result.get("components_improved", 0),
            "overall_success": training_result.get("overall_success", False)
        }
        
    except Exception as e:
        logger.error(f"❌ Error recuperando entrenamiento para sesión {session_id}: {e}", exc_info=True)
        return {
            "session_id": session_id,
            "success": False,
            "error": str(e)
        }


async def main():
    """
    Función principal: detectar y recuperar entrenamientos perdidos
    """
    logger.info("🚀 INICIANDO RECUPERACIÓN DE ENTRENAMIENTOS PERDIDOS")
    logger.info("=" * 80)
    
    # 1. Detectar sesiones no procesadas
    logger.info("🔍 Detectando sesiones con 100+ Q&A no procesadas...")
    unprocessed_sessions = await detect_unprocessed_sessions()
    
    if not unprocessed_sessions:
        logger.info("✅ No hay sesiones pendientes de entrenamiento")
        return
    
    logger.info(f"📋 Sesiones pendientes encontradas: {len(unprocessed_sessions)}")
    for session in unprocessed_sessions:
        logger.info(f"  - {session['session_id']}: {session['qa_count']} Q&A")
    
    # 2. Ejecutar entrenamientos para cada sesión
    logger.info("=" * 80)
    logger.info("🔄 Ejecutando entrenamientos recuperados...")
    logger.info("=" * 80)
    
    results = []
    for i, session_info in enumerate(unprocessed_sessions, 1):
        logger.info(f"\n📦 Procesando sesión {i}/{len(unprocessed_sessions)}")
        result = await recover_training_for_session(session_info)
        results.append(result)
        
        # Pequeña pausa entre entrenamientos
        if i < len(unprocessed_sessions):
            await asyncio.sleep(2)
    
    # 3. Resumen final
    logger.info("\n" + "=" * 80)
    logger.info("📊 RESUMEN DE RECUPERACIÓN")
    logger.info("=" * 80)
    
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful
    
    logger.info(f"✅ Entrenamientos exitosos: {successful}/{len(results)}")
    logger.info(f"❌ Entrenamientos fallidos: {failed}/{len(results)}")
    
    if successful > 0:
        total_components_trained = sum(r.get("components_trained", 0) for r in results if r.get("success"))
        total_components_improved = sum(r.get("components_improved", 0) for r in results if r.get("success"))
        logger.info(f"🧠 Total componentes entrenados: {total_components_trained}")
        logger.info(f"📈 Total componentes mejorados: {total_components_improved}")
    
    # Guardar resumen
    summary_file = Path("data/hack_memori/training_reports/recovery_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "recovery_timestamp": datetime.now().isoformat(),
        "total_sessions_processed": len(results),
        "successful": successful,
        "failed": failed,
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Resumen guardado en: {summary_file}")
    logger.info("=" * 80)
    logger.info("✅ RECUPERACIÓN COMPLETADA")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Recuperación interrumpida por el usuario")
    except Exception as e:
        logger.error(f"❌ Error fatal en recuperación: {e}", exc_info=True)
        sys.exit(1)

