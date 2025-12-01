#!/usr/bin/env python3
"""
SMART TRAINING TRIGGER - Activador automático del sistema de entrenamiento inteligente
Se ejecuta automáticamente cada 100 archivos Q&A en responses/
Basado en verificación periódica del conteo de archivos

USO:
- Automático: Se ejecuta desde cron/scheduler cada hora
- Manual: python scripts/training/smart_training_trigger.py

CONFIGURACIÓN:
- TRIGGER_THRESHOLD: 100 archivos (configurable)
- CHECK_INTERVAL: 3600 segundos (1 hora)
- AUTO_CLEANUP: True (limpia logs antiguos)
"""

import asyncio
import sys
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import time
import os

# Configuración del trigger
TRIGGER_THRESHOLD = int(os.environ.get('SMART_TRAINING_THRESHOLD', '100'))
CHECK_INTERVAL = int(os.environ.get('CHECK_INTERVAL', '3600'))  # 1 hora por defecto
AUTO_CLEANUP = os.environ.get('AUTO_CLEANUP', 'true').lower() == 'true'

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configurar encoding UTF-8 para Windows (ANTES de logging)
import io
if sys.platform == 'win32':
    # Configurar stdout y stderr para UTF-8
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # Configurar variable de entorno para forzar UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configurar logging con handler seguro para UTF-8
class SafeStreamHandler(logging.StreamHandler):
    """Handler que maneja encoding UTF-8 de forma segura en Windows"""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Asegurar que el mensaje se puede escribir
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Si falla, reemplazar caracteres problemáticos
            try:
                msg = self.format(record)
                # Reemplazar emojis problemáticos con texto simple
                msg = msg.encode('ascii', errors='replace').decode('ascii', errors='replace')
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)

# Configurar logging con rotación automática
from logging.handlers import RotatingFileHandler

log_file = Path("logs/smart_training_trigger.log")
log_file.parent.mkdir(parents=True, exist_ok=True)

# Configurar handlers
handlers = [
    RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'),  # 10MB, 5 backups, UTF-8
    SafeStreamHandler(sys.stdout)  # Handler seguro para stdout
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers,
    force=True  # Forzar reconfiguración
)
logger = logging.getLogger(__name__)

# Estado del trigger
TRIGGER_STATE_FILE = Path("data/training/smart_trigger_state.json")
TRIGGER_HISTORY_FILE = Path("data/training/smart_trigger_history.json")


class SmartTrainingTrigger:
    """
    Trigger inteligente que monitorea archivos Q&A y activa entrenamiento automático
    """

    def __init__(self):
        self.responses_dir = Path("data/hack_memori/responses")
        self.state = self._load_state()
        self.history = self._load_history()
        self._force_mode = False  # Flag para modo forzado

        logger.info("✅ SmartTrainingTrigger inicializado")
        logger.info(f"   📊 Threshold: {TRIGGER_THRESHOLD} archivos")
        logger.info(f"   ⏰ Check interval: {CHECK_INTERVAL} segundos")

    def _load_state(self) -> Dict:
        """Cargar estado del trigger"""
        if TRIGGER_STATE_FILE.exists():
            try:
                with open(TRIGGER_STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando estado del trigger: {e}")

        # Estado inicial
        return {
            "last_check": None,
            "last_trigger": None,
            "files_last_check": 0,
            "total_triggers": 0,
            "consecutive_failures": 0,
            "created_at": datetime.now().isoformat()
        }

    def _save_state(self):
        """Guardar estado del trigger"""
        TRIGGER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRIGGER_STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def _load_history(self) -> List[Dict]:
        """Cargar historial de triggers"""
        if TRIGGER_HISTORY_FILE.exists():
            try:
                with open(TRIGGER_HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando historial: {e}")
        return []

    def _save_history(self):
        """Guardar historial de triggers"""
        TRIGGER_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRIGGER_HISTORY_FILE, 'w') as f:
            json.dump(self.history[-100:], f, indent=2, default=str)  # Últimos 100

    async def check_and_trigger(self, force: bool = False) -> Dict[str, Any]:
        """
        Verificar si hay suficientes archivos y activar entrenamiento si es necesario

        Args:
            force: Si True, ignora verificaciones de tiempo y ejecuta siempre

        Returns:
            Resultado de la verificación y posible activación
        """
        now = datetime.now()
        result = {
            "timestamp": now.isoformat(),
            "files_found": 0,
            "should_trigger": False,
            "training_triggered": False,
            "reason": "",
            "training_result": None
        }

        try:
            # Contar archivos válidos
            valid_files = self._count_valid_files()
            result["files_found"] = valid_files

            logger.info(f"📊 Verificación: {valid_files} archivos Q&A encontrados")

            # Verificar si debe activarse (force se pasa como parámetro o desde _force_mode)
            force_mode = getattr(self, '_force_mode', False) or force
            should_trigger, reason = self._should_trigger_training(valid_files, force=force_mode)
            result["should_trigger"] = should_trigger
            result["reason"] = reason

            if should_trigger:
                logger.info(f"🚀 Activando entrenamiento inteligente - {reason}")

                # Ejecutar entrenamiento inteligente
                training_result = await self._execute_smart_training(valid_files)

                result["training_triggered"] = True
                result["training_result"] = training_result

                # Actualizar estado
                self.state["last_trigger"] = now.isoformat()
                self.state["total_triggers"] += 1
                self.state["consecutive_failures"] = 0

                # Registrar en historial
                self.history.append({
                    "timestamp": now.isoformat(),
                    "files_available": valid_files,
                    "trigger_reason": reason,
                    "training_result": training_result,
                    "status": "success" if training_result.get("status") != "failed" else "failed"
                })

                logger.info(f"✅ Entrenamiento completado exitosamente")
                logger.info(f"   📊 Componentes entrenados: {training_result.get('components_trained', 0)}")

            else:
                logger.info(f"⏸️ Entrenamiento no necesario - {reason}")

            # Actualizar estado
            self.state["last_check"] = now.isoformat()
            self.state["files_last_check"] = valid_files

        except Exception as e:
            logger.error(f"❌ Error en trigger: {e}")
            result["error"] = str(e)
            self.state["consecutive_failures"] += 1

        # Guardar estado e historial
        self._save_state()
        self._save_history()

        # Limpieza automática si está habilitada
        if AUTO_CLEANUP:
            await self._auto_cleanup()

        return result

    def _count_valid_files(self) -> int:
        """Contar archivos Q&A válidos (todos los que tienen id y response, no solo accepted_for_training)"""
        if not self.responses_dir.exists():
            logger.warning(f"Directorio de responses no existe: {self.responses_dir}")
            return 0

        valid_count = 0
        accepted_count = 0
        for file_path in self.responses_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Verificar criterios básicos de validez (id y response)
                if data.get("id") and data.get("response", "").strip():
                    valid_count += 1
                    # Contar también los aceptados para logging
                    if data.get("accepted_for_training", False):
                        accepted_count += 1

            except Exception as e:
                logger.debug(f"Error leyendo {file_path}: {e}")
                continue

        # Log detallado
        logger.debug(f"   📊 Archivos válidos: {valid_count} (de los cuales {accepted_count} aceptados para entrenamiento)")
        
        return valid_count

    def _should_trigger_training(self, file_count: int, force: bool = False) -> tuple[bool, str]:
        """
        Determinar si se debe activar el entrenamiento

        Args:
            file_count: Número de archivos disponibles
            force: Si True, ignora verificaciones de tiempo y ejecuta siempre

        Returns:
            (should_trigger, reason)
        """
        # Verificar threshold mínimo (a menos que sea forzado)
        if not force and file_count < TRIGGER_THRESHOLD:
            return False, f"Insuficientes archivos ({file_count}/{TRIGGER_THRESHOLD})"

        # Si es forzado, ejecutar siempre (si hay archivos)
        if force:
            if file_count >= TRIGGER_THRESHOLD:
                return True, f"Ejecución forzada con {file_count} archivos"
            else:
                return True, f"Ejecución forzada (archivos: {file_count}, threshold: {TRIGGER_THRESHOLD})"

        # Verificar tiempo desde último trigger (evitar triggers muy frecuentes)
        last_trigger = self.state.get("last_trigger")
        if last_trigger:
            try:
                last_trigger_dt = datetime.fromisoformat(last_trigger)
                hours_since_last = (datetime.now() - last_trigger_dt).total_seconds() / 3600

                # Mínimo 4 horas entre triggers
                if hours_since_last < 4:
                    return False, f"Demasiado reciente (último trigger hace {hours_since_last:.1f} horas)"
            except:
                pass

        # Verificar fallos consecutivos (máximo 3)
        if self.state.get("consecutive_failures", 0) >= 3:
            return False, f"Demasiados fallos consecutivos ({self.state['consecutive_failures']})"

        # Calcular archivos nuevos desde último trigger
        files_since_last = file_count
        if self.state.get("last_trigger"):
            last_count = self.state.get("files_last_check", 0)
            files_since_last = file_count - last_count

            if files_since_last < TRIGGER_THRESHOLD:
                return False, f"Insuficientes archivos nuevos ({files_since_last}) desde último trigger"

        return True, f"Threshold alcanzado: {file_count} archivos totales, {files_since_last} nuevos"

    async def _execute_smart_training(self, file_count: int) -> Dict[str, Any]:
        """Ejecutar el sistema de entrenamiento inteligente"""
        try:
            # Importamos y ejecutamos el sistema de entrenamiento inteligente
            import importlib
            smart_system_module = importlib.import_module('packages.sheily_core.src.sheily_core.training.smart_training_system')
            run_smart_training = getattr(smart_system_module, 'run_smart_training')

            logger.info(f"🧠 Ejecutando entrenamiento inteligente con {file_count} archivos...")

            # Ejecutar entrenamiento inteligente
            result = await run_smart_training(trigger_files=file_count)

            return result

        except Exception as e:
            logger.error(f"Error ejecutando entrenamiento inteligente: {e}")
            return {"status": "failed", "error": str(e)}

    async def _auto_cleanup(self):
        """Limpieza automática de archivos antiguos"""
        try:
            # Limpiar logs antiguos (más de 30 días)
            log_dir = Path("logs")
            if log_dir.exists():
                cutoff_date = datetime.now() - timedelta(days=30)
                for log_file in log_dir.glob("smart_training_*.log*"):
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        logger.debug(f"Limpiado log antiguo: {log_file}")

            # Limpiar reportes antiguos (mantener últimos 10)
            reports_dir = Path("data/training_reports")
            if reports_dir.exists():
                report_files = sorted(
                    reports_dir.glob("smart_training_*.json"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )

                if len(report_files) > 10:
                    for old_file in report_files[10:]:
                        old_file.unlink()
                        logger.debug(f"Limpiado reporte antiguo: {old_file}")

        except Exception as e:
            logger.debug(f"Error en limpieza automática: {e}")

    async def run_continuous_monitoring(self):
        """
        Ejecutar monitoreo continuo (para servicios/cron)
        """
        logger.info("🔄 Iniciando monitoreo continuo del sistema de entrenamiento inteligente")
        logger.info(f"⏰ Intervalo de verificación: {CHECK_INTERVAL} segundos")

        try:
            while True:
                logger.info("🔍 Verificando necesidad de entrenamiento...")

                result = await self.check_and_trigger()

                if result.get("training_triggered"):
                    logger.info("✅ Ciclo de entrenamiento completado")
                    # Esperar un poco después de un trigger para evitar solapamientos
                    await asyncio.sleep(300)  # 5 minutos
                else:
                    logger.debug(f"⏸️ Sin necesidad de entrenamiento: {result.get('reason', 'Unknown')}")

                # Esperar hasta siguiente verificación
                logger.debug(f"⏰ Esperando {CHECK_INTERVAL} segundos hasta siguiente verificación...")
                await asyncio.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("🛑 Monitoreo continuo detenido por usuario")
        except Exception as e:
            logger.error(f"❌ Error en monitoreo continuo: {e}")


async def run_trigger_check(force: bool = False) -> Dict[str, Any]:
    """
    Función principal para ejecutar una verificación única del trigger
    
    Args:
        force: Si True, ignora verificaciones de tiempo y ejecuta siempre
    """
    trigger = SmartTrainingTrigger()
    trigger._force_mode = force  # Activar modo forzado
    return await trigger.check_and_trigger(force=force)


async def run_continuous_trigger():
    """
    Función para ejecutar monitoreo continuo (para servicios)
    """
    trigger = SmartTrainingTrigger()
    await trigger.run_continuous_monitoring()


def main():
    """Función principal para CLI"""
    import argparse

    parser = argparse.ArgumentParser(description="Smart Training Trigger - Activador automático de entrenamiento inteligente")
    parser.add_argument("--continuous", action="store_true",
                       help="Ejecutar monitoreo continuo (para servicios/cron)")
    # Declarar global antes de usarlo
    global TRIGGER_THRESHOLD
    
    parser.add_argument("--force", action="store_true",
                       help="Forzar activación del entrenamiento (ignorar verificaciones)")
    parser.add_argument("--threshold", type=int, default=TRIGGER_THRESHOLD,
                       help=f"Threshold de archivos (default: {TRIGGER_THRESHOLD})")

    args = parser.parse_args()

    # Configurar threshold si se especifica
    TRIGGER_THRESHOLD = args.threshold

    async def run_with_options():
        if args.force:
            logger.info("⚡ Modo FORZADO activado - ejecutando entrenamiento sin verificaciones")
            # Importar dinámicamente para evitar problemas de importación
            import importlib
            smart_system_module = importlib.import_module('packages.sheily_core.src.sheily_core.training.smart_training_system')
            run_smart_training_func = getattr(smart_system_module, 'run_smart_training')

            # Contar archivos actuales
            responses_dir = Path("data/hack_memori/responses")
            file_count = len(list(responses_dir.glob("*.json"))) if responses_dir.exists() else 0

            logger.info(f"📊 Ejecutando entrenamiento forzado con {file_count} archivos")
            result = await run_smart_training_func(trigger_files=file_count)

            print(json.dumps(result, indent=2, default=str))
            return

        if args.continuous:
            logger.info("🔄 Ejecutando monitoreo continuo...")
            await run_continuous_trigger()
        else:
            logger.info("🔍 Ejecutando verificación única...")
            # Si se usa --force, pasar el flag a run_trigger_check
            result = await run_trigger_check(force=args.force)

            # Imprimir resultado en formato legible
            print("\n" + "="*60)
            print("RESULTADO DE VERIFICACIÓN DEL TRIGGER")
            print("="*60)
            print(f"📅 Timestamp: {result['timestamp']}")
            print(f"📊 Archivos encontrados: {result['files_found']}")
            print(f"🎯 Debería activar: {result['should_trigger']}")
            print(f"🚀 Entrenamiento activado: {result['training_triggered']}")
            print(f"💬 Razón: {result['reason']}")
            print("="*60)

            if result['training_triggered']:
                training = result.get('training_result', {})
                if training.get('status') != 'failed':
                    components = training.get('components_trained', 0)
                    successful = len(training.get('successful_components', []))
                    print("✅ ENTRENAMIENTO EXITOSO")
                    print(f"   📊 Componentes analizados: {training.get('components_analyzed', 0)}")
                    print(f"   🎯 Componentes entrenados: {components}")
                    print(f"   ✅ Componentes exitosos: {successful}")
                    print(f"   📈 Mejora estimada: {training.get('total_estimated_improvement', 0):.1%}")
                else:
                    print("❌ ENTRENAMIENTO FALLIDO")
                    print(f"   Error: {training.get('error', 'Unknown')}")
            else:
                print("⏸️ ENTRENAMIENTO NO NECESARIO")
                print(f"   {result['reason']}")

    # Ejecutar
    try:
        asyncio.run(run_with_options())
    except KeyboardInterrupt:
        logger.info("🛑 Ejecución detenida por usuario")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Error en ejecución: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
