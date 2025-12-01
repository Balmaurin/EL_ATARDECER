"""
Sistema de monitoreo y tracking de entrenamiento - REAL, sin mocks
"""
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingProgress:
    """Estado de progreso de entrenamiento en tiempo real"""
    training_id: str
    status: str  # "running", "completed", "failed", "pending"
    progress_percent: float
    current_component: Optional[str]
    components_completed: int
    total_components: int
    started_at: str
    estimated_completion: Optional[str]
    current_metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


@dataclass
class ComponentTrainingStatus:
    """Estado de entrenamiento de un componente específico"""
    component_name: str
    status: str
    progress: float
    metrics: Dict[str, Any]
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


@dataclass
class ValidationResult:
    """Resultado de validación post-entrenamiento"""
    component_name: str
    validation_passed: bool
    improvement_score: float
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    test_results: Dict[str, Any]
    validation_time: str


class TrainingMonitor:
    """Monitor de entrenamiento con tracking real de Q&A usados"""
    
    def __init__(self, db_path: str = "data/training_monitor.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Estado activo de entrenamientos
        self.active_trainings: Dict[str, TrainingProgress] = {}
        
        # Q&A ya usados en entrenamiento (para entrenamiento incremental)
        self.used_qa_ids: Set[str] = set()
        self._load_used_qa_ids()
    
    def _init_database(self):
        """Inicializar base de datos SQLite para tracking"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Tabla de entrenamientos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    training_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress_percent REAL DEFAULT 0.0,
                    components_completed INTEGER DEFAULT 0,
                    total_components INTEGER DEFAULT 0,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    result TEXT,
                    qa_count INTEGER DEFAULT 0,
                    qa_ids TEXT,
                    current_component TEXT
                )
            """)
            
            # Agregar columna current_component si no existe (para bases de datos existentes)
            try:
                cursor.execute("ALTER TABLE training_runs ADD COLUMN current_component TEXT")
            except sqlite3.OperationalError:
                pass  # Columna ya existe
            
            # Tabla de componentes entrenados
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS component_trainings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    training_id TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    metrics TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    FOREIGN KEY (training_id) REFERENCES training_runs(training_id)
                )
            """)
            
            # Tabla de Q&A usados (para entrenamiento incremental)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS used_qa (
                    qa_id TEXT PRIMARY KEY,
                    training_id TEXT NOT NULL,
                    component_name TEXT,
                    used_at TEXT NOT NULL,
                    quality_score REAL,
                    FOREIGN KEY (training_id) REFERENCES training_runs(training_id)
                )
            """)
            
            # Tabla de validaciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    training_id TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    validation_passed BOOLEAN NOT NULL,
                    improvement_score REAL,
                    before_metrics TEXT,
                    after_metrics TEXT,
                    test_results TEXT,
                    validation_time TEXT NOT NULL,
                    FOREIGN KEY (training_id) REFERENCES training_runs(training_id)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("✅ Base de datos de monitoreo inicializada")
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}", exc_info=True)
    
    def _load_used_qa_ids(self):
        """Cargar IDs de Q&A ya usados desde la base de datos"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT qa_id FROM used_qa")
            rows = cursor.fetchall()
            self.used_qa_ids = {row[0] for row in rows}
            conn.close()
            logger.info(f"✅ Cargados {len(self.used_qa_ids)} Q&A usados previamente")
        except Exception as e:
            logger.error(f"Error cargando Q&A usados: {e}")
            self.used_qa_ids = set()
    
    def get_unused_qa_ids(self, all_qa_ids: List[str]) -> List[str]:
        """Obtener IDs de Q&A que NO han sido usados en entrenamiento"""
        return [qa_id for qa_id in all_qa_ids if qa_id not in self.used_qa_ids]
    
    def mark_qa_as_used(self, training_id: str, qa_ids: List[str], component_name: Optional[str] = None):
        """Marcar Q&A como usados en entrenamiento"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            for qa_id in qa_ids:
                # Obtener quality_score del Q&A si está disponible
                quality_score = 0.5  # Default
                try:
                    from apps.backend.hack_memori_service import HackMemoriService
                    hack_memori = HackMemoriService()
                    # Buscar el response file
                    response_files = list(hack_memori.responses_dir.glob("*.json"))
                    for r_file in response_files:
                        with open(r_file, 'r', encoding='utf-8') as f:
                            r_data = json.load(f)
                            if r_data.get("id") == qa_id:
                                quality_score = r_data.get("quality_score", 0.5)
                                break
                except Exception:
                    pass
                
                cursor.execute("""
                    INSERT OR REPLACE INTO used_qa (qa_id, training_id, component_name, used_at, quality_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (qa_id, training_id, component_name, datetime.now().isoformat(), quality_score))
                
                self.used_qa_ids.add(qa_id)
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Marcados {len(qa_ids)} Q&A como usados")
        except Exception as e:
            logger.error(f"Error marcando Q&A como usados: {e}", exc_info=True)
    
    def start_training(self, training_id: str, total_components: int, qa_count: int) -> TrainingProgress:
        """Iniciar tracking de un entrenamiento"""
        progress = TrainingProgress(
            training_id=training_id,
            status="running",
            progress_percent=0.0,
            current_component=None,
            components_completed=0,
            total_components=total_components,
            started_at=datetime.now().isoformat(),
            estimated_completion=None,
            current_metrics={},
            errors=[],
            warnings=[]
        )
        
        self.active_trainings[training_id] = progress
        
        # Guardar en base de datos
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO training_runs 
                (training_id, status, progress_percent, components_completed, total_components, started_at, qa_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                training_id, "running", 0.0, 0, total_components, 
                progress.started_at, qa_count
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error guardando entrenamiento: {e}")
        
        return progress
    
    def update_progress(self, training_id: str, component_name: str, progress: float, 
                       metrics: Optional[Dict[str, Any]] = None):
        """Actualizar progreso de entrenamiento"""
        if training_id not in self.active_trainings:
            return
        
        training = self.active_trainings[training_id]
        training.current_component = component_name
        training.progress_percent = progress
        training.current_metrics = metrics or {}
        
        # Actualizar en base de datos
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE training_runs 
                SET progress_percent = ?, current_component = ?
                WHERE training_id = ?
            """, (progress, component_name, training_id))
            
            # Guardar estado del componente
            cursor.execute("""
                INSERT OR REPLACE INTO component_trainings 
                (training_id, component_name, status, progress, metrics, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                training_id, component_name, "running", progress,
                json.dumps(metrics or {}), datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error actualizando progreso: {e}")
    
    def complete_component(self, training_id: str, component_name: str, 
                          metrics: Optional[Dict[str, Any]] = None):
        """Marcar componente como completado"""
        if training_id not in self.active_trainings:
            return
        
        training = self.active_trainings[training_id]
        training.components_completed += 1
        training.progress_percent = (training.components_completed / training.total_components) * 100
        
        # Actualizar en base de datos
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE component_trainings 
                SET status = ?, progress = 100.0, metrics = ?, completed_at = ?
                WHERE training_id = ? AND component_name = ?
            """, (
                "completed", json.dumps(metrics or {}), 
                datetime.now().isoformat(), training_id, component_name
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error completando componente: {e}")
    
    def complete_training(self, training_id: str, result: Dict[str, Any]):
        """Completar entrenamiento"""
        if training_id not in self.active_trainings:
            return
        
        training = self.active_trainings[training_id]
        training.status = "completed"
        training.progress_percent = 100.0
        
        # Guardar en base de datos
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE training_runs 
                SET status = ?, progress_percent = 100.0, completed_at = ?, result = ?
                WHERE training_id = ?
            """, (
                "completed", datetime.now().isoformat(), 
                json.dumps(result), training_id
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error completando entrenamiento: {e}")
        
        # Mantener en active_trainings por 1 hora para consultas
        asyncio.create_task(self._remove_after_delay(training_id, 3600))
    
    def fail_training(self, training_id: str, error: str):
        """Marcar entrenamiento como fallido"""
        if training_id not in self.active_trainings:
            return
        
        training = self.active_trainings[training_id]
        training.status = "failed"
        training.errors.append(error)
        
        # Guardar en base de datos
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE training_runs 
                SET status = ?, completed_at = ?, result = ?
                WHERE training_id = ?
            """, (
                "failed", datetime.now().isoformat(), 
                json.dumps({"error": error}), training_id
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error marcando entrenamiento como fallido: {e}")
    
    def save_validation(self, training_id: str, validation: ValidationResult):
        """Guardar resultado de validación"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO validations 
                (training_id, component_name, validation_passed, improvement_score,
                 before_metrics, after_metrics, test_results, validation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                training_id, validation.component_name, validation.validation_passed,
                validation.improvement_score, json.dumps(validation.before_metrics),
                json.dumps(validation.after_metrics), json.dumps(validation.test_results),
                validation.validation_time
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error guardando validación: {e}")
    
    def get_training_status(self, training_id: str) -> Optional[TrainingProgress]:
        """Obtener estado de entrenamiento"""
        # Primero buscar en activos
        if training_id in self.active_trainings:
            return self.active_trainings[training_id]
        
        # Si no está activo, buscar en base de datos
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM training_runs WHERE training_id = ?
            """, (training_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return TrainingProgress(
                    training_id=row[0],
                    status=row[1],
                    progress_percent=row[2],
                    current_component=None,
                    components_completed=row[3],
                    total_components=row[4],
                    started_at=row[5],
                    estimated_completion=row[6],
                    current_metrics=json.loads(row[7]) if row[7] else {},
                    errors=[],
                    warnings=[]
                )
        except Exception as e:
            logger.error(f"Error obteniendo estado: {e}")
        
        return None
    
    def get_latest_training(self) -> Optional[TrainingProgress]:
        """Obtener el último entrenamiento"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM training_runs 
                ORDER BY started_at DESC 
                LIMIT 1
            """)
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return TrainingProgress(
                    training_id=row[0],
                    status=row[1],
                    progress_percent=row[2],
                    current_component=None,
                    components_completed=row[3],
                    total_components=row[4],
                    started_at=row[5],
                    estimated_completion=row[6],
                    current_metrics=json.loads(row[7]) if row[7] else {},
                    errors=[],
                    warnings=[]
                )
        except Exception as e:
            logger.error(f"Error obteniendo último entrenamiento: {e}")
        
        return None
    
    async def _remove_after_delay(self, training_id: str, delay: int):
        """Remover entrenamiento de activos después de un delay"""
        await asyncio.sleep(delay)
        if training_id in self.active_trainings:
            del self.active_trainings[training_id]


# Instancia global del monitor
training_monitor = TrainingMonitor()

