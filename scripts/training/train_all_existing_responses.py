"""
Script para entrenar TODOS los archivos JSON existentes en data/hack_memori/responses
Ejecuta entrenamiento completo del sistema con todos los Q&A disponibles
"""
import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def train_all_existing_responses():
    """
    Entrenar con TODOS los archivos JSON existentes en responses/
    """
    logger.info("=" * 80)
    logger.info("🚀 ENTRENAMIENTO COMPLETO CON TODOS LOS ARCHIVOS EXISTENTES")
    logger.info("=" * 80)
    
    try:
        # 1. Verificar que existe la carpeta de responses
        responses_dir = Path("data/hack_memori/responses")
        if not responses_dir.exists():
            logger.error(f"❌ No existe el directorio: {responses_dir}")
            return
        
        # 2. Contar archivos JSON
        response_files = list(responses_dir.glob("*.json"))
        total_files = len(response_files)
        
        if total_files == 0:
            logger.warning("⚠️ No se encontraron archivos JSON en responses/")
            return
        
        logger.info(f"📊 Archivos JSON encontrados: {total_files}")
        
        # 3. Inicializar ComponentTrainer
        from packages.sheily_core.src.sheily_core.training.integral_trainer import ComponentTrainer
        
        logger.info("🧠 Inicializando ComponentTrainer...")
        trainer = ComponentTrainer(base_path="data/hack_memori")
        
        # 4. Ejecutar entrenamiento con TODOS los archivos
        # incremental=False para usar TODOS los archivos, no solo los nuevos
        logger.info("=" * 80)
        logger.info(f"🔄 INICIANDO ENTRENAMIENTO CON {total_files} ARCHIVOS")
        logger.info("   Modo: NO incremental (usa TODOS los archivos)")
        logger.info("=" * 80)
        
        training_result = await trainer.train_all_components(
            trigger_threshold=total_files,  # Usar total de archivos como threshold
            incremental=False  # IMPORTANTE: False para entrenar TODOS, no solo nuevos
        )
        
        # 5. Mostrar resultados
        logger.info("=" * 80)
        logger.info("✅ ENTRENAMIENTO COMPLETADO")
        logger.info("=" * 80)
        logger.info(f"Estado: {training_result.get('status', 'unknown')}")
        logger.info(f"Componentes entrenados: {training_result.get('components_trained', 0)}")
        logger.info(f"Componentes mejorados: {training_result.get('components_improved', 0)}")
        logger.info(f"Total componentes: {training_result.get('total_components', 0)}")
        logger.info(f"Archivos Q&A usados: {training_result.get('qa_count', 0)}")
        logger.info(f"Éxito general: {training_result.get('overall_success', False)}")
        
        if training_result.get('training_id'):
            logger.info(f"Training ID: {training_result.get('training_id')}")
        
        # Mostrar métricas por componente si están disponibles
        if 'component_metrics' in training_result:
            logger.info("\n📊 Métricas por componente:")
            for component, metrics in training_result['component_metrics'].items():
                logger.info(f"  - {component}: {metrics}")
        
        # Mostrar validaciones si están disponibles
        if 'validation_results' in training_result:
            validations = training_result['validation_results']
            logger.info(f"\n✅ Validaciones:")
            logger.info(f"  - Componentes validados: {validations.get('components_validated', 0)}")
            logger.info(f"  - Componentes mejorados: {validations.get('components_improved', 0)}")
            logger.info(f"  - Componentes degradados: {validations.get('components_degraded', 0)}")
            logger.info(f"  - Mejora general: {validations.get('overall_improvement', False)}")
        
        logger.info("=" * 80)
        
        # 6. Guardar reporte final
        report_path = Path(f"data/training_reports/training_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(training_result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 Reporte guardado en: {report_path}")
        
        return training_result
        
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        result = asyncio.run(train_all_existing_responses())
        if result and result.get('overall_success'):
            print("\n✅ Entrenamiento completado exitosamente")
            sys.exit(0)
        else:
            print("\n⚠️ Entrenamiento completado con advertencias")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por el usuario")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)





