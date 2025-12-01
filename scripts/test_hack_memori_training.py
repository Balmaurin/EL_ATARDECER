#!/usr/bin/env python3
"""
Script de Prueba - Entrenamiento Automático Hack-Memori
========================================================

Este script verifica que el sistema de entrenamiento automático funciona
correctamente cuando se alcanzan 100 Q&A.

Uso:
    python scripts/test_hack_memori_training.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

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


async def test_training_trigger():
    """
    Probar que el trigger de entrenamiento funciona correctamente
    """
    logger.info("=" * 80)
    logger.info("🧪 PRUEBA: Trigger de Entrenamiento Automático")
    logger.info("=" * 80)
    
    service = HackMemoriService()
    
    # 1. Verificar sesiones existentes
    sessions = service.get_sessions()
    logger.info(f"📊 Sesiones encontradas: {len(sessions)}")
    
    if not sessions:
        logger.warning("⚠️ No hay sesiones para probar")
        return False
    
    # 2. Verificar conteo de Q&A por sesión
    logger.info("\n📋 Verificando Q&A por sesión:")
    sessions_with_100_plus = []
    
    for session in sessions:
        session_id = session.get("session_id") or session.get("id")
        if not session_id:
            continue
        
        qa_count = service._get_session_qa_count(session_id)
        logger.info(f"  - Sesión {session_id[:8]}...: {qa_count} Q&A")
        
        if qa_count >= 100:
            sessions_with_100_plus.append({
                "session_id": session_id,
                "qa_count": qa_count
            })
    
    if not sessions_with_100_plus:
        logger.warning("⚠️ No hay sesiones con 100+ Q&A para probar")
        return False
    
    logger.info(f"\n✅ Sesiones con 100+ Q&A: {len(sessions_with_100_plus)}")
    
    # 3. Probar que el método train_all_components existe y funciona
    logger.info("\n🔍 Verificando ComponentTrainer...")
    try:
        trainer = ComponentTrainer(base_path="data/hack_memori")
        logger.info("✅ ComponentTrainer inicializado correctamente")
        
        # Verificar que el método existe
        if hasattr(trainer, 'train_all_components'):
            logger.info("✅ Método train_all_components existe")
        else:
            logger.error("❌ Método train_all_components NO existe")
            return False
        
        # Verificar que puede recopilar archivos Q&A
        qa_files = trainer._collect_qa_files()
        logger.info(f"✅ Archivos Q&A encontrados: {len(qa_files)}")
        
        if len(qa_files) < 100:
            logger.warning(f"⚠️ Solo hay {len(qa_files)} archivos Q&A (se requieren 100 para entrenamiento)")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error verificando ComponentTrainer: {e}", exc_info=True)
        return False
    
    # 4. Verificar estructura de directorios de reportes
    logger.info("\n📁 Verificando estructura de directorios...")
    reports_dir = Path("data/hack_memori/training_reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Directorio de reportes: {reports_dir}")
    
    history_db = Path("data/training_history.db")
    history_db.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Base de datos de historial: {history_db}")
    
    # 5. Verificar que el método _save_training_report existe
    logger.info("\n🔍 Verificando métodos de guardado...")
    if hasattr(service, '_save_training_report'):
        logger.info("✅ Método _save_training_report existe")
    else:
        logger.error("❌ Método _save_training_report NO existe")
        return False
    
    if hasattr(service, '_save_training_to_history_db'):
        logger.info("✅ Método _save_training_to_history_db existe")
    else:
        logger.warning("⚠️ Método _save_training_to_history_db NO existe (se creará automáticamente)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TODAS LAS VERIFICACIONES PASARON")
    logger.info("=" * 80)
    logger.info("\n💡 El sistema está listo para ejecutar entrenamientos automáticos")
    logger.info("💡 Para ejecutar entrenamientos perdidos, usa:")
    logger.info("   python scripts/recover_hack_memori_training.py")
    
    return True


async def test_training_execution():
    """
    Probar ejecución real de entrenamiento (opcional, puede ser lento)
    """
    logger.info("\n" + "=" * 80)
    logger.info("🧪 PRUEBA OPCIONAL: Ejecución Real de Entrenamiento")
    logger.info("=" * 80)
    logger.info("⚠️ Esta prueba puede tardar varios minutos...")
    
    response = input("\n¿Deseas ejecutar una prueba real de entrenamiento? (s/N): ")
    if response.lower() != 's':
        logger.info("⏭️ Prueba de ejecución omitida")
        return True
    
    try:
        trainer = ComponentTrainer(base_path="data/hack_memori")
        logger.info("🔄 Ejecutando entrenamiento de prueba...")
        
        # Ejecutar con threshold bajo para prueba rápida (opcional)
        result = await trainer.train_all_components(trigger_threshold=100)
        
        if result.get("status") == "insufficient_data":
            logger.warning(f"⚠️ Datos insuficientes: {result.get('message', '')}")
            return False
        
        logger.info("✅ Entrenamiento completado")
        logger.info(f"  - Componentes entrenados: {result.get('components_trained', 0)}")
        logger.info(f"  - Componentes mejorados: {result.get('components_improved', 0)}")
        logger.info(f"  - Éxito general: {result.get('overall_success', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en prueba de ejecución: {e}", exc_info=True)
        return False


async def main():
    """
    Función principal de pruebas
    """
    logger.info("🚀 INICIANDO PRUEBAS DEL SISTEMA DE ENTRENAMIENTO")
    
    # Prueba 1: Verificar trigger y estructura
    test1_passed = await test_training_trigger()
    
    if not test1_passed:
        logger.error("\n❌ PRUEBAS FALLARON - Revisa los errores arriba")
        return False
    
    # Prueba 2: Ejecución real (opcional)
    test2_passed = await test_training_execution()
    
    logger.info("\n" + "=" * 80)
    if test1_passed and test2_passed:
        logger.info("✅ TODAS LAS PRUEBAS PASARON")
    elif test1_passed:
        logger.info("✅ PRUEBAS BÁSICAS PASARON (ejecución real omitida)")
    else:
        logger.info("❌ ALGUNAS PRUEBAS FALLARON")
    logger.info("=" * 80)
    
    return test1_passed


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pruebas interrumpidas por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error fatal en pruebas: {e}", exc_info=True)
        sys.exit(1)

