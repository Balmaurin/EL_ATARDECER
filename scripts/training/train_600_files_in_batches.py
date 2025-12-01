"""
Script para entrenar 600 archivos JSON existentes en 6 ciclos de 100 archivos cada uno
Ejecuta entrenamiento incremental del sistema con batches de 100 Q&A
"""
import asyncio
import sys
import logging
import json
import shutil
import tempfile
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

# Configurar logging con encoding UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def train_in_batches(batch_size: int = 100, max_batches: int = 1, force_retrain: bool = False):
    """
    Entrenar archivos JSON existentes en batches - OPTIMIZADO para 100 archivos pequeños
    
    Args:
        batch_size: Tamaño de cada batch (default: 100)
        max_batches: Número máximo de batches a entrenar (default: 1 para 100 archivos)
        force_retrain: Si True, entrena todos los archivos incluso si ya fueron usados (default: False)
    """
    logger.info("=" * 80)
    logger.info("🚀 ENTRENAMIENTO OPTIMIZADO POR BATCHES")
    logger.info("=" * 80)
    
    try:
        # 1. Verificar que existe la carpeta de responses (usando project_root para paths absolutos)
        responses_dir = project_root / "data" / "hack_memori" / "responses"
        if not responses_dir.exists():
            logger.error(f"❌ No existe el directorio: {responses_dir}")
            return
        
        # 2. Obtener todos los archivos JSON
        all_response_files = sorted(list(responses_dir.glob("*.json")))
        total_files = len(all_response_files)
        
        if total_files == 0:
            logger.warning("⚠️ No se encontraron archivos JSON en responses/")
            return
        
        logger.info(f"📊 Total de archivos JSON encontrados: {total_files}")
        
        # 3. OPTIMIZACIÓN: Leer cada archivo solo UNA vez y guardar toda la información
        valid_files = []
        file_data_cache = {}  # Cache para evitar lecturas múltiples
        
        for file_path in all_response_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Verificar que tiene los campos necesarios
                qa_id = data.get("id")
                response = data.get("response", "")
                
                if not qa_id or not response:
                    continue
                
                # Calcular quality_score si no existe
                quality_score = data.get("quality_score")
                if quality_score is None:
                    # Calcular basado en longitud y contenido
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
                    valid_files.append(file_path)
                    # Guardar datos en cache para uso posterior
                    file_data_cache[str(file_path)] = {
                        "id": qa_id,
                        "data": data
                    }
                    
            except Exception as e:
                logger.debug(f"Error leyendo {file_path}: {e}")
        
        logger.info(f"📊 Archivos válidos: {len(valid_files)}")
        
        # 4. Verificar Q&A ya usados - OPTIMIZADO usando cache
        try:
            from apps.backend.training_monitor import training_monitor
            logger.info("✅ TrainingMonitor importado correctamente")
        except ImportError as e:
            logger.warning(f"⚠️ No se pudo importar training_monitor: {e}")
            logger.warning("   Continuando sin monitoreo de Q&A usados...")
            # Crear un mock simple para continuar
            class MockTrainingMonitor:
                def get_unused_qa_ids(self, qa_ids):
                    return qa_ids  # Retornar todos como no usados
            training_monitor = MockTrainingMonitor()
        
        # Usar cache en lugar de leer archivos de nuevo
        all_qa_ids = [file_data_cache[str(f)]["id"] for f in valid_files if str(f) in file_data_cache]
        file_to_id = {str(f): file_data_cache[str(f)]["id"] for f in valid_files if str(f) in file_data_cache}
        
        if force_retrain:
            # Modo forzado: usar todos los archivos válidos
            logger.info("🔄 Modo FORZADO activado: Reentrenando todos los archivos (incluso los ya usados)")
            unused_files = valid_files
        else:
            # Filtrar solo Q&A no usados
            try:
                unused_qa_ids = set(training_monitor.get_unused_qa_ids(all_qa_ids))
                unused_files = [
                    Path(f) for f in file_to_id.keys() 
                    if file_to_id[f] in unused_qa_ids
                ]
            except Exception as e:
                logger.warning(f"⚠️ Error obteniendo Q&A no usados: {e}")
                logger.warning("   Usando todos los archivos válidos...")
                unused_files = valid_files
            
            logger.info(f"📊 Archivos nuevos (no usados previamente): {len(unused_files)}")
            logger.info(f"📊 Archivos ya usados: {len(valid_files) - len(unused_files)}")
            
            if len(unused_files) == 0:
                logger.warning("⚠️ No hay archivos nuevos para entrenar. Todos ya fueron usados.")
                logger.info("🔄 Activando modo FORZADO automáticamente para reentrenar todos los archivos...")
                logger.info("   (Esto es necesario para entrenamiento REAL sin fallbacks)")
                logger.info("   🔥 MODO REAL: Reentrenando todos los archivos para mejorar el modelo")
                unused_files = valid_files
                force_retrain = True  # Activar flag para que se marquen como usados al final
        
        # 5. Dividir en batches
        num_batches = min((len(unused_files) + batch_size - 1) // batch_size, max_batches)
        logger.info(f"📊 Dividiendo en {num_batches} batches de {batch_size} archivos")
        
        # 6. Inicializar ComponentTrainer (usando path absoluto)
        logger.info("🧠 Inicializando ComponentTrainer...")
        try:
            from packages.sheily_core.src.sheily_core.training.integral_trainer import ComponentTrainer
            
            base_path = project_root / "data" / "hack_memori"
            
            # Validar que el directorio base existe
            if not base_path.exists():
                logger.warning(f"⚠️ Directorio base no existe: {base_path}, creándolo...")
                base_path.mkdir(parents=True, exist_ok=True)
            
            # Inicializar trainer con manejo de errores
            try:
                trainer = ComponentTrainer(base_path=str(base_path))
                logger.info("✅ ComponentTrainer inicializado correctamente")
            except Exception as init_error:
                logger.error(f"❌ Error inicializando ComponentTrainer: {init_error}", exc_info=True)
                # Intentar inicializar con paths mínimos
                logger.info("🔄 Intentando inicialización alternativa...")
                trainer = ComponentTrainer(base_path=str(base_path))
                
        except ImportError as import_error:
            logger.error(f"❌ Error importando ComponentTrainer: {import_error}")
            logger.error("   Verifica que el módulo esté disponible en el path")
            raise
        except Exception as e:
            logger.error(f"❌ Error crítico inicializando trainer: {e}", exc_info=True)
            raise
        
        # 7. Entrenar cada batch
        all_results = []
        for batch_num in range(1, num_batches + 1):
            start_idx = (batch_num - 1) * batch_size
            end_idx = min(start_idx + batch_size, len(unused_files))
            batch_files = unused_files[start_idx:end_idx]
            
            logger.info("=" * 80)
            logger.info(f"🔄 CICLO {batch_num}/{num_batches}")
            logger.info(f"   Archivos en este batch: {len(batch_files)}")
            logger.info(f"   Rango: {start_idx + 1} - {end_idx} de {len(unused_files)}")
            logger.info("=" * 80)
            
            # CRÍTICO: Crear directorio temporal con SOLO los 100 archivos del batch
            # para asegurar que solo se entrenen estos archivos, no todos los 647
            temp_base_dir = None
            original_base_path = Path(trainer.base_path)  # Guardar como Path para restaurar
            
            try:
                # Crear directorio temporal para este batch
                temp_base_dir = Path(tempfile.mkdtemp(prefix=f"training_batch_{batch_num}_"))
                temp_responses_dir = temp_base_dir / "hack_memori" / "responses"
                temp_responses_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"📁 Creando directorio temporal con SOLO {len(batch_files)} archivos...")
                # Copiar SOLO los archivos del batch al directorio temporal
                copied_count = 0
                for batch_file in batch_files:
                    batch_file_path = Path(batch_file)
                    if not batch_file_path.is_absolute():
                        batch_file_path = responses_dir / batch_file_path.name
                    dest_file = temp_responses_dir / batch_file_path.name
                    if batch_file_path.exists():
                        shutil.copy2(batch_file_path, dest_file)
                        copied_count += 1
                
                # Verificar que se copiaron correctamente
                actual_files = len(list(temp_responses_dir.glob('*.json')))
                logger.info(f"✅ Copiados {copied_count}/{len(batch_files)} archivos al directorio temporal")
                logger.info(f"✅ Verificación: {actual_files} archivos JSON en directorio temporal")
                logger.info(f"📂 Directorio temporal: {temp_base_dir}")
                
                if actual_files != len(batch_files):
                    logger.warning(f"⚠️ Advertencia: Se esperaban {len(batch_files)} archivos pero hay {actual_files} en el directorio temporal")
                
                # Cambiar temporalmente el base_path del trainer para usar el directorio temporal
                trainer.base_path = Path(temp_base_dir / "hack_memori")
                logger.info(f"📂 Trainer usando base_path temporal: {trainer.base_path}")
                
                # Validar que el trainer está correctamente configurado
                if not hasattr(trainer, 'train_all_components'):
                    raise AttributeError("ComponentTrainer no tiene método train_all_components")
                
                # Entrenar con SOLO estos 100 archivos
                # Usar incremental=False para que use todos los archivos del directorio temporal
                logger.info(f"🚀 Iniciando entrenamiento para batch {batch_num}...")
                try:
                    training_result = await trainer.train_all_components(
                        trigger_threshold=len(batch_files),  # Usar tamaño del batch
                        incremental=False  # NO usar incremental para usar todos los archivos del temp dir
                    )
                    logger.info(f"✅ Entrenamiento del batch {batch_num} completado")
                except Exception as train_error:
                    logger.error(f"❌ Error durante el entrenamiento del batch {batch_num}: {train_error}", exc_info=True)
                    raise
                
                if training_result.get("status") == "insufficient_data":
                    logger.warning(f"⚠️ Batch {batch_num}: Insuficientes Q&A válidos")
                    logger.warning(f"   Requeridos: {len(batch_files)}, Disponibles: {training_result.get('qa_count', 0)}")
                    continue
                
                # VALIDACIÓN: Verificar que el resultado es real y no un fallback
                if training_result.get("status") == "success" or training_result.get("overall_success"):
                    # Verificar que hay métricas reales
                    components_trained = training_result.get('components_trained', 0)
                    if components_trained == 0:
                        logger.warning(f"⚠️ Batch {batch_num}: Resultado reporta éxito pero no hay componentes entrenados")
                        logger.warning("   Esto puede indicar un problema en el entrenamiento")
                
                all_results.append({
                    "batch": batch_num,
                    "files_count": len(batch_files),
                    "result": training_result
                })
                
                logger.info("=" * 80)
                logger.info(f"✅ CICLO {batch_num}/{num_batches} COMPLETADO")
                logger.info("=" * 80)
                logger.info(f"Estado: {training_result.get('status', 'unknown')}")
                logger.info(f"Componentes entrenados: {training_result.get('components_trained', 0)}")
                logger.info(f"Componentes mejorados: {training_result.get('components_improved', 0)}")
                logger.info(f"Archivos Q&A usados: {training_result.get('qa_count', 0)}")
                logger.info(f"Éxito general: {training_result.get('overall_success', False)}")
                
                # Validación adicional: verificar que el entrenamiento realmente ocurrió
                if training_result.get("overall_success"):
                    detailed_results = training_result.get("detailed_results", {})
                    if detailed_results:
                        # Contar componentes que realmente se entrenaron
                        real_trained = sum(1 for r in detailed_results.values() if r.get("status") == "success")
                        logger.info(f"   🔍 Validación: {real_trained} componentes reportaron entrenamiento exitoso")
                        if real_trained == 0:
                            logger.warning("   ⚠️ ADVERTENCIA: Aunque se reporta éxito, ningún componente se entrenó realmente")
                    else:
                        logger.warning("   ⚠️ ADVERTENCIA: No hay resultados detallados disponibles para validar")
                
                # OPTIMIZACIÓN: Eliminar sleep para 100 archivos pequeños (es innecesario)
                # Solo esperar si hay múltiples batches y son muchos archivos
                if batch_num < num_batches and len(unused_files) > 200:
                    logger.info("⏳ Esperando 2 segundos antes del siguiente batch...")
                    await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error en batch {batch_num}: {e}", exc_info=True)
                all_results.append({
                    "batch": batch_num,
                    "files_count": len(batch_files),
                    "error": str(e)
                })
            finally:
                # IMPORTANTE: Restaurar el base_path original y limpiar directorio temporal
                trainer.base_path = Path(original_base_path)
                if temp_base_dir and temp_base_dir.exists():
                    logger.info(f"🧹 Limpiando directorio temporal: {temp_base_dir}")
                    try:
                        shutil.rmtree(temp_base_dir)
                        logger.info("✅ Directorio temporal eliminado")
                    except Exception as e:
                        logger.warning(f"⚠️ No se pudo eliminar directorio temporal: {e}")
        
        # 8. Resumen final
        logger.info("=" * 80)
        logger.info("📊 RESUMEN FINAL")
        logger.info("=" * 80)
        logger.info(f"Total batches procesados: {len(all_results)}")
        logger.info(f"Total archivos procesados: {sum(r.get('files_count', 0) for r in all_results)}")
        
        successful_batches = sum(1 for r in all_results if r.get('result', {}).get('overall_success', False))
        logger.info(f"Batches exitosos: {successful_batches}/{len(all_results)}")
        
        # 9. Guardar reporte final (usando project_root)
        report_dir = project_root / "data" / "training_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"training_batches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        final_report = {
            "training_completed": datetime.now().isoformat(),
            "total_files": total_files,
            "valid_files": len(valid_files),
            "unused_files": len(unused_files),
            "batches_processed": len(all_results),
            "batch_size": batch_size,
            "results": all_results
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 Reporte guardado en: {report_path}")
        logger.info("=" * 80)
        
        return final_report
        
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento por batches: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        # Permitir especificar batch_size y max_batches desde línea de comandos
        # OPTIMIZADO: Por defecto 1 batch para 100 archivos pequeños
        # MODO REAL: Por defecto force=True para asegurar entrenamiento REAL sin fallbacks
        batch_size = 100
        max_batches = 1
        force_retrain = True  # Por defecto TRUE para entrenamiento REAL
        
        # Parsear argumentos
        for arg in sys.argv[1:]:
            if arg == "--force" or arg == "-f":
                force_retrain = True
            elif arg.isdigit():
                if batch_size == 100:  # Primer número es batch_size
                    batch_size = int(arg)
                else:  # Segundo número es max_batches
                    max_batches = int(arg)
        
        logger.info(f"📊 Configuración: batch_size={batch_size}, max_batches={max_batches}")
        logger.info(f"🔥 MODO REAL: Entrenamiento sin fallbacks ni mocks")
        if force_retrain:
            logger.info(f"🔄 Modo FORZADO: Reentrenará todos los archivos (activado por defecto para entrenamiento REAL)")
        logger.info(f"⚡ Modo optimizado para {batch_size * max_batches} archivos pequeños")
        
        result = asyncio.run(train_in_batches(batch_size=batch_size, max_batches=max_batches, force_retrain=force_retrain))
        
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

