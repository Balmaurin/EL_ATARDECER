#!/usr/bin/env python3
"""
Auto Training System - Sistema Real de Entrenamiento Automático
Sistema automático de entrenamiento para Sheily AI con integración real
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingOpportunity:
    """Oportunidad de entrenamiento detectada"""
    opportunity_id: str
    type: str  # fine_tuning, full_training, incremental
    dataset_size: int
    priority: str  # low, medium, high, critical
    confidence: float
    detected_at: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, queued, running, completed, failed


@dataclass
class TrainingJob:
    """Job de entrenamiento real"""
    job_id: str
    opportunity_id: str
    training_type: str
    dataset_path: str
    config: Dict[str, Any]
    status: str = "queued"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class AutoTrainingSystem:
    """
    Sistema de entrenamiento automático y mejora continua.
    Integrado con sistemas reales de entrenamiento del proyecto.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "min_feedback_samples": 50,
            "min_negative_feedback_ratio": 0.3,
            "training_threshold_confidence": 0.7,
            "monitoring_interval_seconds": 300,  # 5 minutos
            "max_concurrent_jobs": 2,
        }
        self.is_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Base de datos para persistencia
        self.db_path = Path("data/auto_training.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Sistema de entrenamiento real
        self.training_system = None
        self._init_training_system()
        
        # Estado
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.pending_opportunities: List[TrainingOpportunity] = []
        self.feedback_buffer: List[Dict[str, Any]] = []
        
        logger.info("✅ AutoTrainingSystem initialized with real training integration")

    def _init_database(self):
        """Inicializar base de datos SQLite para persistencia"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Tabla de feedback
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_data TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                processed BOOLEAN DEFAULT 0
            )
        """)
        
        # Tabla de oportunidades
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS opportunities (
                opportunity_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                dataset_size INTEGER,
                priority TEXT,
                confidence REAL,
                detected_at TEXT,
                status TEXT,
                metrics TEXT
            )
        """)
        
        # Tabla de jobs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id TEXT PRIMARY KEY,
                opportunity_id TEXT,
                training_type TEXT,
                dataset_path TEXT,
                config TEXT,
                status TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                metrics TEXT,
                error TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized: {self.db_path}")

    def _init_training_system(self):
        """Inicializar sistema de entrenamiento real"""
        try:
            # Intentar cargar UnifiedLearningTrainingSystem (sistema principal)
            from sheily_core.unified_systems.unified_learning_training_system import (
                UnifiedLearningTrainingSystem,
                TrainingMode
            )
            self.training_system = UnifiedLearningTrainingSystem()
            self.TrainingMode = TrainingMode
            logger.info("✅ Connected to UnifiedLearningTrainingSystem")
        except ImportError:
            try:
                # Fallback a RealTrainingSystem
                from sheily_core.training.real_training_system import RealTrainingSystem
                self.training_system = RealTrainingSystem()
                logger.info("✅ Connected to RealTrainingSystem")
            except ImportError:
                logger.warning("⚠️ No training system available - will queue jobs only")
                self.training_system = None

    async def start_monitoring(self):
        """Inicia el monitoreo de oportunidades de entrenamiento."""
        if self.is_active:
            logger.warning("Monitoring already active")
            return
            
        self.is_active = True
        logger.info("🚀 AutoTrainingSystem monitoring started")
        
        # Iniciar loop de monitoreo en background
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Detiene el monitoreo."""
        self.is_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("⏹️ AutoTrainingSystem monitoring stopped")
        
    async def process_feedback(self, feedback_data: Dict[str, Any]):
        """
        Procesa feedback para entrenamiento futuro.
        
        Args:
            feedback_data: Dict con feedback. Puede ser:
                - Lista de items: [{"query": "...", "response": "...", "rating": 1-5}, ...]
                - Dict con estadísticas: {"total": 100, "negative": 30, "patterns": [...]}
        """
        if not self.is_active:
            logger.warning("AutoTrainingSystem is not active")
            return {"status": "ignored", "reason": "inactive"}
        
        # Normalizar feedback_data
        if isinstance(feedback_data, list):
            feedback_items = feedback_data
        elif isinstance(feedback_data, dict) and "items" in feedback_data:
            feedback_items = feedback_data["items"]
        else:
            feedback_items = [feedback_data]
        
        logger.info(f"📥 Feedback received: {len(feedback_items)} items")
        
        # Guardar en buffer y base de datos
        timestamp = datetime.now().isoformat()
        for item in feedback_items:
            self.feedback_buffer.append({
                **item,
                "timestamp": timestamp
            })
        
        # Guardar en BD
        await self._save_feedback_to_db(feedback_items)
        
        # Analizar oportunidad de entrenamiento
        opportunity = await self._analyze_training_opportunity(feedback_items)
        
        if opportunity:
            job_id = await self._trigger_training_job(opportunity)
            return {
                "status": "processed",
                "action": "training_triggered",
                "job_id": job_id,
                "opportunity": asdict(opportunity)
            }
            
        return {"status": "processed", "action": "queued_for_analysis"}

    async def _save_feedback_to_db(self, feedback_items: List[Dict[str, Any]]):
        """Guardar feedback en base de datos"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for item in feedback_items:
            cursor.execute("""
                INSERT INTO feedback (feedback_data, timestamp, processed)
                VALUES (?, ?, ?)
            """, (json.dumps(item), datetime.now().isoformat(), False))
        
        conn.commit()
        conn.close()

    async def _analyze_training_opportunity(
        self, feedback_items: List[Dict[str, Any]]
    ) -> Optional[TrainingOpportunity]:
        """
        Analiza si el feedback justifica un nuevo job de entrenamiento.
        Usa heurísticas reales basadas en calidad y cantidad de feedback.
        """
        if not feedback_items:
            return None
        
        # Calcular métricas reales del feedback
        total_items = len(feedback_items)
        negative_count = 0
        positive_count = 0
        ratings = []
        patterns = {}
        
        for item in feedback_items:
            # Detectar feedback negativo
            rating = item.get("rating", item.get("score", 3))
            if isinstance(rating, (int, float)):
                ratings.append(rating)
                if rating < 3:
                    negative_count += 1
                elif rating >= 4:
                    positive_count += 1
            
            # Detectar patrones de error
            if item.get("error") or item.get("incorrect"):
                error_type = item.get("error_type", "unknown")
                patterns[error_type] = patterns.get(error_type, 0) + 1
        
        # Calcular ratios
        negative_ratio = negative_count / total_items if total_items > 0 else 0
        avg_rating = sum(ratings) / len(ratings) if ratings else 3.0
        
        # Heurísticas para detectar oportunidad
        min_samples = self.config["min_feedback_samples"]
        min_negative_ratio = self.config["min_negative_feedback_ratio"]
        
        should_train = False
        priority = "low"
        confidence = 0.0
        training_type = "fine_tuning"
        
        # Heurística 1: Suficiente feedback negativo
        if total_items >= min_samples and negative_ratio >= min_negative_ratio:
            should_train = True
            priority = "high" if negative_ratio >= 0.5 else "medium"
            confidence = min(1.0, negative_ratio * 1.5)
            training_type = "fine_tuning"
        
        # Heurística 2: Mucho feedback nuevo (incremental learning)
        elif total_items >= min_samples * 2:
            should_train = True
            priority = "medium"
            confidence = 0.6
            training_type = "incremental"
        
        # Heurística 3: Patrones de error consistentes
        elif patterns and max(patterns.values()) >= min_samples * 0.3:
            should_train = True
            priority = "high"
            confidence = 0.8
            training_type = "fine_tuning"
        
        # Heurística 4: Rating promedio muy bajo
        elif avg_rating < 2.0 and total_items >= min_samples:
            should_train = True
            priority = "critical"
            confidence = 0.9
            training_type = "full_training"
        
        if should_train and confidence >= self.config["training_threshold_confidence"]:
            opportunity = TrainingOpportunity(
                opportunity_id=str(uuid.uuid4()),
                type=training_type,
                dataset_size=total_items,
                priority=priority,
                confidence=confidence,
                metrics={
                    "total_items": total_items,
                    "negative_count": negative_count,
                    "negative_ratio": negative_ratio,
                    "avg_rating": avg_rating,
                    "patterns": patterns,
                    "positive_count": positive_count
                }
            )
            
            # Guardar oportunidad en BD
            await self._save_opportunity(opportunity)
            
            logger.info(
                f"🎯 Training opportunity detected: {training_type} "
                f"(priority: {priority}, confidence: {confidence:.2f})"
            )
            
            return opportunity
        
        return None

    async def _save_opportunity(self, opportunity: TrainingOpportunity):
        """Guardar oportunidad en base de datos"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO opportunities 
            (opportunity_id, type, dataset_size, priority, confidence, detected_at, status, metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            opportunity.opportunity_id,
            opportunity.type,
            opportunity.dataset_size,
            opportunity.priority,
            opportunity.confidence,
            opportunity.detected_at.isoformat(),
            opportunity.status,
            json.dumps(opportunity.metrics)
        ))
        
        conn.commit()
        conn.close()

    async def _trigger_training_job(self, opportunity: TrainingOpportunity) -> str:
        """
        Dispara un job de entrenamiento real usando el sistema de entrenamiento.
        """
        job_id = str(uuid.uuid4())
        
        # Preparar dataset desde feedback buffer
        dataset_path = await self._prepare_training_dataset(opportunity)
        
        # Crear job
        job = TrainingJob(
            job_id=job_id,
            opportunity_id=opportunity.opportunity_id,
            training_type=opportunity.type,
            dataset_path=str(dataset_path),
            config={
                "priority": opportunity.priority,
                "dataset_size": opportunity.dataset_size,
                "mode": opportunity.type
            }
        )
        
        # Guardar job
        await self._save_job(job)
        
        # Ejecutar job si hay sistema de entrenamiento disponible
        if self.training_system:
            asyncio.create_task(self._execute_training_job(job))
        else:
            logger.warning(f"⚠️ Training system not available - job {job_id} queued")
            job.status = "queued"
            self.active_jobs[job_id] = job
        
        logger.info(f"🚀 Training job {job_id} triggered: {opportunity.type}")
        return job_id

    async def _prepare_training_dataset(self, opportunity: TrainingOpportunity) -> Path:
        """Preparar dataset de entrenamiento desde feedback buffer"""
        dataset_dir = Path("data/training_datasets")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Filtrar feedback relevante
        relevant_feedback = self.feedback_buffer[-opportunity.dataset_size:]
        
        # Crear dataset en formato JSONL
        dataset_file = dataset_dir / f"auto_training_{opportunity.opportunity_id}.jsonl"
        
        with open(dataset_file, "w", encoding="utf-8") as f:
            for item in relevant_feedback:
                # Formatear para entrenamiento
                training_item = {
                    "instruction": item.get("query", item.get("input", "")),
                    "output": item.get("response", item.get("output", "")),
                    "rating": item.get("rating", item.get("score", 3))
                }
                f.write(json.dumps(training_item, ensure_ascii=False) + "\n")
        
        logger.info(f"📊 Dataset prepared: {dataset_file} ({len(relevant_feedback)} samples)")
        return dataset_file

    async def _execute_training_job(self, job: TrainingJob):
        """Ejecutar job de entrenamiento real"""
        job.status = "running"
        job.started_at = datetime.now()
        await self._update_job(job)
        
        try:
            if not self.training_system:
                raise RuntimeError("Training system not available")
            
            # Determinar modo de entrenamiento
            if hasattr(self, 'TrainingMode'):
                if job.training_type == "fine_tuning":
                    mode = self.TrainingMode.FINE_TUNE
                elif job.training_type == "full_training":
                    mode = self.TrainingMode.FULL_TRAINING
                else:
                    mode = self.TrainingMode.INCREMENTAL
            else:
                mode = job.training_type
            
            # Iniciar sesión de entrenamiento
            if hasattr(self.training_system, 'start_training_session'):
                session_id = await self.training_system.start_training_session(
                    model_name="gemma-2-2b-it",  # Modelo base configurable
                    dataset_path=job.dataset_path,
                    training_mode=mode,
                    config=job.config
                )
                
                job.metrics["session_id"] = session_id
                logger.info(f"✅ Training session started: {session_id}")
            else:
                # Fallback: usar método directo si existe
                logger.warning("⚠️ start_training_session not available, using fallback")
                job.metrics["status"] = "started_manually"
            
            job.status = "completed"
            job.completed_at = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Training job {job.job_id} failed: {e}")
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now()
        
        finally:
            await self._update_job(job)
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

    async def _save_job(self, job: TrainingJob):
        """Guardar job en base de datos"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO training_jobs
            (job_id, opportunity_id, training_type, dataset_path, config, status,
             created_at, started_at, completed_at, metrics, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.job_id,
            job.opportunity_id,
            job.training_type,
            job.dataset_path,
            json.dumps(job.config),
            job.status,
            job.created_at.isoformat(),
            job.started_at.isoformat() if job.started_at else None,
            job.completed_at.isoformat() if job.completed_at else None,
            json.dumps(job.metrics),
            job.error
        ))
        
        conn.commit()
        conn.close()

    async def _update_job(self, job: TrainingJob):
        """Actualizar job en base de datos"""
        await self._save_job(job)

    async def _monitoring_loop(self):
        """Loop de monitoreo continuo"""
        interval = self.config["monitoring_interval_seconds"]
        
        while self.is_active:
            try:
                # Procesar feedback acumulado
                if len(self.feedback_buffer) >= self.config["min_feedback_samples"]:
                    opportunity = await self._analyze_training_opportunity(self.feedback_buffer)
                    if opportunity:
                        await self._trigger_training_job(opportunity)
                        # Limpiar buffer procesado
                        self.feedback_buffer = []
                
                # Verificar jobs activos
                await self._check_active_jobs()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)

    async def _check_active_jobs(self):
        """Verificar estado de jobs activos"""
        for job_id, job in list(self.active_jobs.items()):
            # Verificar si job completó (si hay sistema de entrenamiento)
            if self.training_system and hasattr(self.training_system, 'get_session_status'):
                try:
                    session_id = job.metrics.get("session_id")
                    if session_id:
                        status = await self.training_system.get_session_status(session_id)
                        if status.get("status") == "completed":
                            job.status = "completed"
                            job.completed_at = datetime.now()
                            await self._update_job(job)
                            del self.active_jobs[job_id]
                except Exception as e:
                    logger.warning(f"Error checking job {job_id}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del sistema"""
        return {
            "active": self.is_active,
            "active_jobs": len(self.active_jobs),
            "pending_opportunities": len(self.pending_opportunities),
            "feedback_buffer_size": len(self.feedback_buffer),
            "training_system_available": self.training_system is not None
        }

    def get_job_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener historial de jobs"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM training_jobs
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        jobs = []
        for row in cursor.fetchall():
            jobs.append({
                "job_id": row[0],
                "status": row[5],
                "training_type": row[2],
                "created_at": row[6],
                "completed_at": row[8]
            })
        
        conn.close()
        return jobs


# Alias for compatibility
AutoTrainer = AutoTrainingSystem
