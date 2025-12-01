"""
Script para entrenar 100 archivos JSON con ENTRENAMIENTO COMPLETO
Versión completa con fine-tuning profundo, todas las épocas, todos los componentes
Tiempo estimado: 2-3 horas (solo 1 batch de 100 archivos)
"""
import asyncio
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configurar encoding UTF-8 para Windows
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def train_100_files_complete():
    """
    Entrenar 100 archivos JSON con ENTRENAMIENTO COMPLETO
    - Fine-tuning profundo de embeddings (3 épocas)
    - Entrenamiento completo de LLM (3 épocas)
    - Todos los componentes
    - Tiempo estimado: 2-3 horas
    """
    logger.info("=" * 80)
    logger.info("🚀 ENTRENAMIENTO COMPLETO - 100 ARCHIVOS")
    logger.info("=" * 80)
    logger.info("📊 Modo: COMPLETO (con fine-tuning profundo)")
    logger.info("   - Fine-tuning de embeddings: ACTIVADO (3 épocas)")
    logger.info("   - Entrenamiento LLM: ACTIVADO (3 épocas)")
    logger.info("   - Todos los componentes: ACTIVADOS")
    logger.info("   - Tiempo estimado: 2-3 horas")
    logger.info("=" * 80)
    
    try:
        # 1. Verificar que existe la carpeta de responses
        responses_dir = Path("data/hack_memori/responses")
        if not responses_dir.exists():
            logger.error(f"[ERROR] No existe el directorio: {responses_dir}")
            return
        
        # 2. Obtener todos los archivos JSON
        all_response_files = sorted(list(responses_dir.glob("*.json")))
        total_files = len(all_response_files)
        
        if total_files == 0:
            logger.warning("[WARNING] No se encontraron archivos JSON en responses/")
            return
        
        logger.info(f"[INFO] Total de archivos JSON encontrados: {total_files}")
        
        # 3. Filtrar archivos válidos
        valid_files = []
        stats = {
            "with_id": 0,
            "with_response": 0,
            "with_quality_score": 0,
            "with_accepted": 0,
            "quality_high": 0,
            "quality_low": 0,
            "accepted_true": 0,
            "accepted_false": 0
        }
        
        for file_path in all_response_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Estadísticas
                if data.get("id"):
                    stats["with_id"] += 1
                if data.get("response"):
                    stats["with_response"] += 1
                if "quality_score" in data:
                    stats["with_quality_score"] += 1
                    if data.get("quality_score", 0) >= 0.6:
                        stats["quality_high"] += 1
                    else:
                        stats["quality_low"] += 1
                if "accepted_for_training" in data:
                    stats["with_accepted"] += 1
                    if data.get("accepted_for_training"):
                        stats["accepted_true"] += 1
                    else:
                        stats["accepted_false"] += 1
                
                # Verificar que tiene los campos necesarios
                qa_id = data.get("id")
                response = data.get("response", "")
                
                if not qa_id or not response:
                    continue
                
                # Calcular quality_score si no existe
                quality_score = data.get("quality_score")
                if quality_score is None:
                    word_count = len(response.split())
                    if word_count >= 50:
                        quality_score = 0.7
                    elif word_count >= 20:
                        quality_score = 0.6
                    else:
                        quality_score = 0.4
                
                # Verificar accepted_for_training
                accepted = data.get("accepted_for_training")
                if accepted is None:
                    accepted = quality_score >= 0.6
                
                # Criterio: aceptar si tiene id, response y calidad mínima
                min_quality = 0.5
                if qa_id and response and quality_score >= min_quality:
                    if accepted is None and quality_score >= 0.6:
                        accepted = True
                    valid_files.append(file_path)
                    
            except Exception as e:
                logger.debug(f"Error leyendo {file_path}: {e}")
        
        # Mostrar estadísticas
        logger.info("=" * 80)
        logger.info("[INFO] ESTADÍSTICAS DE ARCHIVOS")
        logger.info("=" * 80)
        logger.info(f"  - Con ID: {stats['with_id']}/{total_files}")
        logger.info(f"  - Con response: {stats['with_response']}/{total_files}")
        logger.info(f"  - Con quality_score: {stats['with_quality_score']}/{total_files}")
        logger.info(f"    * Alta calidad (>=0.6): {stats['quality_high']}")
        logger.info(f"    * Baja calidad (<0.6): {stats['quality_low']}")
        logger.info(f"  - Con accepted_for_training: {stats['with_accepted']}/{total_files}")
        logger.info(f"    * Aceptados: {stats['accepted_true']}")
        logger.info(f"    * Rechazados: {stats['accepted_false']}")
        logger.info("=" * 80)
        logger.info(f"[INFO] Archivos válidos: {len(valid_files)}")
        
        # 4. Verificar Q&A ya usados
        from apps.backend.training_monitor import training_monitor
        
        all_qa_ids = []
        file_to_id = {}
        for file_path in valid_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                qa_id = data.get("id")
                if qa_id:
                    all_qa_ids.append(qa_id)
                    file_to_id[str(file_path)] = qa_id
            except Exception:
                pass
        
        unused_qa_ids = set(training_monitor.get_unused_qa_ids(all_qa_ids))
        unused_files = [
            Path(f) for f in file_to_id.keys() 
            if file_to_id[f] in unused_qa_ids
        ]
        
        logger.info(f"[INFO] Archivos nuevos (no usados): {len(unused_files)}")
        
        if len(unused_files) == 0:
            logger.warning("[WARNING] No hay archivos nuevos para entrenar")
            return
        
        # 5. Tomar solo los primeros 100 archivos
        files_to_train = unused_files[:100]
        logger.info(f"[INFO] Entrenando con los primeros 100 archivos nuevos")
        logger.info(f"[INFO] Archivos seleccionados: {len(files_to_train)}")
        
        # 6. Inicializar ComponentTrainer
        from packages.sheily_core.src.sheily_core.training.integral_trainer import ComponentTrainer
        
        logger.info("[INFO] Inicializando ComponentTrainer...")
        trainer = ComponentTrainer(base_path="data/hack_memori")
        
        # 7. Entrenar con ENTRENAMIENTO COMPLETO
        logger.info("=" * 80)
        logger.info("[INFO] INICIANDO ENTRENAMIENTO COMPLETO")
        logger.info("=" * 80)
        logger.info("[INFO] Este proceso puede tardar 2-3 horas")
        logger.info("[INFO] Incluye:")
        logger.info("   - Fine-tuning profundo de embeddings (3 épocas)")
        logger.info("   - Entrenamiento completo de LLM (3 épocas)")
        logger.info("   - Entrenamiento de todos los componentes")
        logger.info("=" * 80)
        
        try:
            # Asegurar que NO está en modo rápido
            import os
            if 'SKIP_EMBEDDING_FINETUNING' in os.environ:
                del os.environ['SKIP_EMBEDDING_FINETUNING']
            if 'REDUCED_EPOCHS' in os.environ:
                del os.environ['REDUCED_EPOCHS']
            
            # Entrenar con modo completo (incremental=True para tracking)
            training_result = await trainer.train_all_components(
                trigger_threshold=len(files_to_train),
                incremental=True  # Solo entrenar Q&A nuevos
            )
            
            logger.info("=" * 80)
            logger.info("[OK] ENTRENAMIENTO COMPLETO FINALIZADO")
            logger.info("=" * 80)
            logger.info(f"Estado: {training_result.get('status', 'unknown')}")
            logger.info(f"Componentes entrenados: {training_result.get('components_trained', 0)}")
            logger.info(f"Componentes mejorados: {training_result.get('components_improved', 0)}")
            logger.info(f"Total componentes: {training_result.get('total_components', 0)}")
            logger.info(f"Archivos Q&A usados: {training_result.get('qa_count', 0)}")
            logger.info(f"Éxito general: {training_result.get('overall_success', False)}")
            
            # Mostrar métricas por componente si están disponibles
            if 'component_metrics' in training_result:
                logger.info("\n[INFO] Métricas por componente:")
                for component, metrics in training_result['component_metrics'].items():
                    logger.info(f"  - {component}: {metrics}")
            
            # Mostrar validaciones si están disponibles
            if 'validation_results' in training_result:
                validations = training_result['validation_results']
                logger.info(f"\n[INFO] Validaciones:")
                logger.info(f"  - Componentes validados: {validations.get('components_validated', 0)}")
                logger.info(f"  - Componentes mejorados: {validations.get('components_improved', 0)}")
                logger.info(f"  - Componentes degradados: {validations.get('components_degraded', 0)}")
                logger.info(f"  - Mejora general: {validations.get('overall_improvement', False)}")
            
            logger.info("=" * 80)
            
            # Guardar reporte final
            report_path = Path(f"data/training_reports/training_100_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(training_result, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"[INFO] Reporte guardado en: {report_path}")
            logger.info("=" * 80)
            
            return training_result
            
        except Exception as e:
            logger.error(f"[ERROR] Error en entrenamiento: {e}", exc_info=True)
            raise
        
    except Exception as e:
        logger.error(f"[ERROR] Error en entrenamiento: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        logger.info("[INFO] Iniciando entrenamiento completo de 100 archivos...")
        logger.info("[INFO] Tiempo estimado: 2-3 horas")
        logger.info("[INFO] Presiona Ctrl+C para cancelar")
        logger.info("")
        
        result = asyncio.run(train_100_files_complete())
        
        if result and result.get('overall_success'):
            logger.info("\n[OK] Entrenamiento completado exitosamente")
            sys.exit(0)
        else:
            logger.info("\n[WARNING] Entrenamiento completado con advertencias")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n[INFO] Entrenamiento interrumpido por el usuario")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n[ERROR] Error: {e}")
        sys.exit(1)

