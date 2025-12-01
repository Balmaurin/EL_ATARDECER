"""
Script OPTIMIZADO para entrenar 600 archivos JSON en 6 ciclos de 100 archivos
VERSIÓN RÁPIDA: Omite fine-tuning profundo, reduce épocas, optimiza procesos
Tiempo estimado: 30-60 minutos (vs 8-12 horas de la versión completa)
"""
import asyncio
import sys
import logging
import json
import os
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

# MODO RÁPIDO: Variables de entorno para optimización
os.environ['TRAINING_FAST_MODE'] = 'true'
os.environ['SKIP_EMBEDDING_FINETUNING'] = 'true'  # Omitir fine-tuning de embeddings (más lento)
os.environ['REDUCED_EPOCHS'] = '1'  # 1 época en lugar de 3
os.environ['INCREASED_BATCH_SIZE'] = '32'  # Batch más grande = más rápido


async def train_in_batches_fast(batch_size: int = 100, max_batches: int = 6):
    """
    Entrenar archivos JSON en batches - VERSIÓN RÁPIDA
    
    Optimizaciones:
    - Omite fine-tuning profundo de embeddings
    - Reduce épocas a 1
    - Aumenta batch_size
    - Solo entrena componentes críticos
    """
    logger.info("=" * 80)
    logger.info("🚀 ENTRENAMIENTO RÁPIDO POR BATCHES (MODO OPTIMIZADO)")
    logger.info("=" * 80)
    logger.info("⚡ Optimizaciones activas:")
    logger.info("   - Fine-tuning de embeddings: DESACTIVADO")
    logger.info("   - Épocas: 1 (en lugar de 3)")
    logger.info("   - Batch size aumentado: 32")
    logger.info("   - Solo componentes críticos")
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
        
        # 3. Filtrar archivos válidos (mismo proceso que antes)
        valid_files = []
        for file_path in all_response_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
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
                
                accepted = data.get("accepted_for_training")
                if accepted is None:
                    accepted = quality_score >= 0.6
                
                if qa_id and response and quality_score >= 0.5:
                    valid_files.append(file_path)
            except Exception as e:
                logger.debug(f"Error leyendo {file_path}: {e}")
        
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
        
        # 5. Dividir en batches
        num_batches = min((len(unused_files) + batch_size - 1) // batch_size, max_batches)
        logger.info(f"[INFO] Dividiendo en {num_batches} batches de {batch_size} archivos")
        
        # 6. Inicializar ComponentTrainer
        from packages.sheily_core.src.sheily_core.training.integral_trainer import ComponentTrainer
        
        logger.info("[INFO] Inicializando ComponentTrainer...")
        trainer = ComponentTrainer(base_path="data/hack_memori")
        
        # 7. Entrenar cada batch
        all_results = []
        for batch_num in range(1, num_batches + 1):
            start_idx = (batch_num - 1) * batch_size
            end_idx = min(start_idx + batch_size, len(unused_files))
            batch_files = unused_files[start_idx:end_idx]
            
            logger.info("=" * 80)
            logger.info(f"[CICLO {batch_num}/{num_batches}]")
            logger.info(f"   Archivos: {len(batch_files)}")
            logger.info(f"   Rango: {start_idx + 1} - {end_idx} de {len(unused_files)}")
            logger.info("=" * 80)
            
            try:
                # Entrenar con modo rápido
                training_result = await trainer.train_all_components(
                    trigger_threshold=len(batch_files),
                    incremental=True
                )
                
                if training_result.get("status") == "insufficient_data":
                    logger.warning(f"[WARNING] Batch {batch_num}: Insuficientes Q&A válidos")
                    continue
                
                all_results.append({
                    "batch": batch_num,
                    "files_count": len(batch_files),
                    "result": training_result
                })
                
                logger.info("=" * 80)
                logger.info(f"[OK] CICLO {batch_num}/{num_batches} COMPLETADO")
                logger.info(f"   Componentes entrenados: {training_result.get('components_trained', 0)}")
                logger.info(f"   Archivos usados: {training_result.get('qa_count', 0)}")
                logger.info("=" * 80)
                
                # Esperar un poco entre batches
                if batch_num < num_batches:
                    logger.info("[INFO] Esperando 3 segundos antes del siguiente batch...")
                    await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"[ERROR] Error en batch {batch_num}: {e}", exc_info=True)
                all_results.append({
                    "batch": batch_num,
                    "files_count": len(batch_files),
                    "error": str(e)
                })
        
        # 8. Resumen final
        logger.info("=" * 80)
        logger.info("[RESUMEN FINAL]")
        logger.info("=" * 80)
        logger.info(f"Total batches procesados: {len(all_results)}")
        logger.info(f"Total archivos procesados: {sum(r.get('files_count', 0) for r in all_results)}")
        
        successful_batches = sum(1 for r in all_results if r.get('result', {}).get('overall_success', False))
        logger.info(f"Batches exitosos: {successful_batches}/{len(all_results)}")
        
        # 9. Guardar reporte
        report_path = Path(f"data/training_reports/training_batches_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_report = {
            "training_completed": datetime.now().isoformat(),
            "mode": "fast",
            "total_files": total_files,
            "valid_files": len(valid_files),
            "unused_files": len(unused_files),
            "batches_processed": len(all_results),
            "batch_size": batch_size,
            "results": all_results
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"[INFO] Reporte guardado en: {report_path}")
        logger.info("=" * 80)
        
        return final_report
        
    except Exception as e:
        logger.error(f"[ERROR] Error en entrenamiento: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        batch_size = 100
        max_batches = 6
        
        if len(sys.argv) > 1:
            try:
                batch_size = int(sys.argv[1])
            except ValueError:
                logger.warning(f"[WARNING] Argumento inválido para batch_size, usando default: 100")
        
        if len(sys.argv) > 2:
            try:
                max_batches = int(sys.argv[2])
            except ValueError:
                logger.warning(f"[WARNING] Argumento inválido para max_batches, usando default: 6")
        
        logger.info(f"[INFO] Configuración: batch_size={batch_size}, max_batches={max_batches}")
        
        result = asyncio.run(train_in_batches_fast(batch_size=batch_size, max_batches=max_batches))
        
        if result:
            successful = sum(1 for r in result.get('results', []) if r.get('result', {}).get('overall_success', False))
            total = len(result.get('results', []))
            if successful == total and total > 0:
                logger.info("\n[OK] Entrenamiento completado exitosamente")
                sys.exit(0)
            else:
                logger.info(f"\n[WARNING] Entrenamiento completado: {successful}/{total} batches exitosos")
                sys.exit(1)
        else:
            logger.warning("\n[WARNING] No se procesaron batches")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n[INFO] Entrenamiento interrumpido por el usuario")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n[ERROR] Error: {e}")
        sys.exit(1)

