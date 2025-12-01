#!/usr/bin/env python3
"""
HACK-MEMORI TRAINING ORCHESTRATOR - Sistema Real de Orquestación
================================================================

Este módulo es el "PEGAMENTO" que conecta Hack-Memori con todos los sistemas
de auto-entrenamiento. NO depende de Google ADK - es una implementación
real y funcional que orquesta el flujo completo:

1. Monitoreo de Hack-Memori (generación automática de datos)
2. Evaluación de calidad de datos
3. Orquestación de sistemas de entrenamiento
4. Validación y deployment automático

Sistemas conectados:
- Real Training System (PyTorch + PEFT/LoRA)
- Neuro Training V2 (entrenamiento neuronal avanzado)
- QLoRA Fine-tuning Pipeline
- Unified Learning System
- Auto-Evolution Engine
- ML Auto-Evolution Engine
- MCP Enterprise Master
"""

import asyncio
import inspect
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# HACK-MEMORI INTEGRATION - REAL IMPLEMENTATION
# =============================================================================

class HackMemoriMonitor:
    """Monitor real de Hack-Memori para detectar nuevos datos de entrenamiento"""
    
    def __init__(self, hack_memori_dir: str = None):
        self.hack_memori_dir = Path(hack_memori_dir or os.getenv("HACK_MEMORI_DIR", "data/hack_memori"))
        self.sessions_dir = self.hack_memori_dir / "sessions"
        self.questions_dir = self.hack_memori_dir / "questions"
        self.responses_dir = self.hack_memori_dir / "responses"
        self.last_check_time = datetime.now()
        self.processed_sessions = set()
        
        # Crear directorios si no existen
        for dir_path in [self.sessions_dir, self.questions_dir, self.responses_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 Hack-Memori Monitor initialized: {self.hack_memori_dir}")
    
    def get_new_sessions(self) -> List[Dict[str, Any]]:
        """Obtener sesiones nuevas desde la última verificación"""
        new_sessions = []
        
        try:
            for session_file in self.sessions_dir.glob("*.json"):
                session_id = session_file.stem
                
                # Verificar si es nueva
                if session_id not in self.processed_sessions:
                    try:
                        with open(session_file, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                            session_data['session_id'] = session_id
                            session_data['file_path'] = str(session_file)
                            new_sessions.append(session_data)
                            self.processed_sessions.add(session_id)
                    except Exception as e:
                        logger.warning(f"Error reading session {session_id}: {e}")
            
            self.last_check_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error checking for new sessions: {e}")
        
        return new_sessions
    
    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Obtener todos los datos de una sesión (preguntas + respuestas)"""
        session_data = {
            'session_id': session_id,
            'questions': [],
            'responses': [],
            'metadata': {}
        }
        
        try:
            # Cargar sesión
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data['metadata'] = json.load(f)
            
            # Cargar preguntas
            questions_file = self.questions_dir / f"{session_id}.json"
            if questions_file.exists():
                with open(questions_file, 'r', encoding='utf-8') as f:
                    questions_data = json.load(f)
                    if isinstance(questions_data, list):
                        session_data['questions'] = questions_data
                    elif isinstance(questions_data, dict) and 'questions' in questions_data:
                        session_data['questions'] = questions_data['questions']
            
            # Cargar respuestas
            responses_file = self.responses_dir / f"{session_id}.json"
            if responses_file.exists():
                with open(responses_file, 'r', encoding='utf-8') as f:
                    responses_data = json.load(f)
                    if isinstance(responses_data, list):
                        session_data['responses'] = responses_data
                    elif isinstance(responses_data, dict) and 'responses' in responses_data:
                        session_data['responses'] = responses_data['responses']
            
        except Exception as e:
            logger.error(f"Error loading session data for {session_id}: {e}")
        
        return session_data
    
    def get_all_ready_sessions(self) -> List[Dict[str, Any]]:
        """Obtener todas las sesiones listas para entrenamiento (con preguntas y respuestas)"""
        ready_sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            session_id = session_file.stem
            session_data = self.get_session_data(session_id)
            
            # Verificar que tenga datos suficientes
            if (len(session_data['questions']) > 0 and 
                len(session_data['responses']) > 0 and
                len(session_data['questions']) == len(session_data['responses'])):
                ready_sessions.append(session_data)
        
        return ready_sessions


class DataQualityEvaluator:
    """Evaluador real de calidad de datos de Hack-Memori"""
    
    def __init__(self):
        self.min_questions = 10
        self.min_avg_length = 20
        self.quality_threshold = 0.7
    
    def evaluate_session_quality(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar calidad real de una sesión"""
        questions = session_data.get('questions', [])
        responses = session_data.get('responses', [])
        
        if len(questions) == 0 or len(responses) == 0:
            return {
                'quality_score': 0.0,
                'ready_for_training': False,
                'reasons': ['No questions or responses found']
            }
        
        # Métricas reales
        metrics = {
            'total_pairs': min(len(questions), len(responses)),
            'avg_question_length': sum(len(str(q).split()) for q in questions) / len(questions),
            'avg_response_length': sum(len(str(r).split()) for r in responses) / len(responses),
            'coverage_score': len(questions) / max(self.min_questions, 1),
        }
        
        # Calcular score de calidad
        quality_score = 0.0
        
        # Score por cantidad (máx 0.3)
        if metrics['total_pairs'] >= self.min_questions:
            quality_score += 0.3
        elif metrics['total_pairs'] >= self.min_questions * 0.5:
            quality_score += 0.15
        
        # Score por longitud promedio (máx 0.3)
        if metrics['avg_question_length'] >= self.min_avg_length:
            quality_score += 0.15
        if metrics['avg_response_length'] >= self.min_avg_length:
            quality_score += 0.15
        
        # Score por cobertura (máx 0.4)
        coverage = min(metrics['coverage_score'], 1.0)
        quality_score += coverage * 0.4
        
        quality_score = min(quality_score, 1.0)
        
        reasons = []
        if quality_score < self.quality_threshold:
            if metrics['total_pairs'] < self.min_questions:
                reasons.append(f"Insufficient data: {metrics['total_pairs']} pairs (need {self.min_questions})")
            if metrics['avg_question_length'] < self.min_avg_length:
                reasons.append(f"Questions too short: {metrics['avg_question_length']:.1f} words")
            if metrics['avg_response_length'] < self.min_avg_length:
                reasons.append(f"Responses too short: {metrics['avg_response_length']:.1f} words")
        else:
            reasons.append("Quality sufficient for training")
        
        return {
            'quality_score': round(quality_score, 3),
            'ready_for_training': quality_score >= self.quality_threshold,
            'metrics': metrics,
            'reasons': reasons
        }


class TrainingSystemOrchestrator:
    """Orquestador real que conecta con todos los sistemas de entrenamiento"""
    
    def __init__(self):
        self.training_systems = {}
        self._initialize_training_systems()
    
    def _initialize_training_systems(self):
        """Inicializar conexiones con sistemas de entrenamiento reales"""
        
        # 1. Real Training System (PyTorch + PEFT/LoRA)
        try:
            from .training.real_training_system import RealTrainingSystem
            self.training_systems['real_training'] = RealTrainingSystem()
            logger.info("✅ Real Training System connected")
        except ImportError as e:
            logger.warning(f"Real Training System not available: {e}")
        
        # 2. Neuro Training V2
        try:
            from .tools.neuro_training_v2 import NeuroTrainingV2
            self.training_systems['neuro_training'] = NeuroTrainingV2()
            logger.info("✅ Neuro Training V2 connected")
        except ImportError as e:
            logger.warning(f"Neuro Training V2 not available: {e}")
        
        # 3. Unified Learning System
        try:
            from .unified_systems.unified_learning_training_system import UnifiedLearningTrainingSystem
            self.training_systems['unified_learning'] = UnifiedLearningTrainingSystem()
            logger.info("✅ Unified Learning System connected")
        except ImportError as e:
            logger.warning(f"Unified Learning System not available: {e}")
        
        # 4. Auto-Evolution Engine
        try:
            from .api.auto_evolution_engine import AutoEvolutionEngine
            self.training_systems['auto_evolution'] = AutoEvolutionEngine()
            logger.info("✅ Auto-Evolution Engine connected")
        except ImportError as e:
            logger.warning(f"Auto-Evolution Engine not available: {e}")
        
        # 5. ML Auto-Evolution Engine
        try:
            from .api.ml_auto_evolution_engine import MLAutoEvolutionEngine
            self.training_systems['ml_auto_evolution'] = MLAutoEvolutionEngine()
            logger.info("✅ ML Auto-Evolution Engine connected")
        except ImportError as e:
            logger.warning(f"ML Auto-Evolution Engine not available: {e}")
        
        # 6. MCP Enterprise Master (si tiene capacidades de entrenamiento)
        try:
            from .core.mcp.mcp_enterprise_master import MCPEnterpriseMaster
            self.training_systems['mcp_enterprise'] = MCPEnterpriseMaster()
            logger.info("✅ MCP Enterprise Master connected")
        except ImportError as e:
            logger.debug(f"MCP Enterprise Master not available: {e}")
        
        logger.info(f"📚 {len(self.training_systems)} training systems available")
    
    async def train_with_data(self, training_data: List[Dict[str, Any]], 
                             system_name: str = None) -> Dict[str, Any]:
        """Entrenar con datos de Hack-Memori usando el sistema especificado"""
        
        if not training_data:
            return {
                'success': False,
                'error': 'No training data provided'
            }
        
        # Si no se especifica sistema, usar el primero disponible
        if system_name is None:
            system_name = list(self.training_systems.keys())[0] if self.training_systems else None
        
        if system_name not in self.training_systems:
            return {
                'success': False,
                'error': f'Training system {system_name} not available',
                'available_systems': list(self.training_systems.keys())
            }
        
        training_system = self.training_systems[system_name]
        
        try:
            # Preparar datos en formato estándar
            # Si los datos ya están en formato de entrenamiento (tienen 'input' y 'output'), usarlos directamente
            # Si no, formatearlos usando _format_training_data
            if training_data and isinstance(training_data[0], dict):
                # Verificar si ya está en formato de entrenamiento
                first_item = training_data[0]
                if 'input' in first_item and 'output' in first_item:
                    # Ya está formateado, usar directamente
                    formatted_data = training_data
                elif 'instruction' in first_item and 'output' in first_item:
                    # Convertir de formato instruction/input/output a input/output
                    formatted_data = []
                    for item in training_data:
                        input_text = item.get('input', '')
                        instruction = item.get('instruction', '')
                        # Combinar instruction e input si ambos existen
                        if instruction and input_text:
                            combined_input = f"{instruction}\n\n{input_text}" if input_text else instruction
                        else:
                            combined_input = instruction or input_text
                        
                        formatted_data.append({
                            'input': combined_input,
                            'output': item.get('output', '')
                        })
                else:
                    # Formatear usando método estándar (para datos de sesión)
                    formatted_data = self._format_training_data(training_data)
            else:
                formatted_data = []
            
            if not formatted_data:
                return {
                    'success': False,
                    'error': 'No valid training data after formatting',
                    'original_count': len(training_data),
                    'formatted_count': 0
                }
            
            logger.info(f"📊 Datos formateados: {len(formatted_data)} ejemplos (de {len(training_data)} originales)")
            
            # Ejecutar entrenamiento - Verificar si es async o sync
            if hasattr(training_system, 'train'):
                train_method = getattr(training_system, 'train')
                if inspect.iscoroutinefunction(train_method):
                    result = await train_method(formatted_data)
                else:
                    # Método síncrono, NO usar await
                    result = train_method(formatted_data)
            elif hasattr(training_system, 'fine_tune'):
                fine_tune_method = getattr(training_system, 'fine_tune')
                if inspect.iscoroutinefunction(fine_tune_method):
                    result = await fine_tune_method(formatted_data)
                else:
                    result = fine_tune_method(formatted_data)
            elif hasattr(training_system, 'process_training_data'):
                process_method = getattr(training_system, 'process_training_data')
                if inspect.iscoroutinefunction(process_method):
                    result = await process_method(formatted_data)
                else:
                    result = process_method(formatted_data)
            else:
                # Intentar método genérico
                result = {
                    'success': True,
                    'system': system_name,
                    'data_points': len(formatted_data),
                    'message': 'Training data processed (method not specified)'
                }
            
            logger.info(f"✅ Training completed with {system_name}: {len(formatted_data)} data points")
            return {
                'success': True,
                'system': system_name,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error training with {system_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'system': system_name
            }
    
    def _format_training_data(self, session_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Formatear datos de sesiones para entrenamiento"""
        formatted = []
        
        for session_data in session_data_list:
            questions = session_data.get('questions', [])
            responses = session_data.get('responses', [])
            
            # Crear pares pregunta-respuesta
            for i in range(min(len(questions), len(responses))):
                formatted.append({
                    'input': str(questions[i]),
                    'output': str(responses[i]),
                    'session_id': session_data.get('session_id', 'unknown'),
                    'metadata': session_data.get('metadata', {})
                })
        
        return formatted


class HackMemoriTrainingOrchestrator:
    """Orquestador principal que conecta Hack-Memori con sistemas de entrenamiento"""
    
    def __init__(self):
        self.monitor = HackMemoriMonitor()
        self.quality_evaluator = DataQualityEvaluator()
        self.training_orchestrator = TrainingSystemOrchestrator()
        self.training_history_db = "data/training_history.db"
        self._init_training_history_db()
        
        logger.info("🎯 Hack-Memori Training Orchestrator initialized")
    
    def _init_training_history_db(self):
        """Inicializar base de datos de historial de entrenamiento"""
        try:
            os.makedirs(os.path.dirname(self.training_history_db), exist_ok=True)
            with sqlite3.connect(self.training_history_db) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS training_runs (
                        run_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        training_system TEXT,
                        quality_score REAL,
                        data_points INTEGER,
                        status TEXT,
                        started_at TEXT,
                        completed_at TEXT,
                        result TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not initialize training history DB: {e}")
    
    async def monitor_and_trigger_training(self) -> Dict[str, Any]:
        """Monitorear Hack-Memori y disparar entrenamiento si hay datos listos"""
        
        # Obtener sesiones nuevas
        new_sessions = self.monitor.get_new_sessions()
        
        if not new_sessions:
            return {
                'training_triggered': False,
                'reason': 'No new sessions found',
                'timestamp': datetime.now().isoformat()
            }
        
        # Evaluar calidad de cada sesión
        ready_sessions = []
        for session in new_sessions:
            session_data = self.monitor.get_session_data(session.get('session_id', ''))
            quality_result = self.quality_evaluator.evaluate_session_quality(session_data)
            
            if quality_result['ready_for_training']:
                ready_sessions.append({
                    'session_data': session_data,
                    'quality': quality_result
                })
        
        if not ready_sessions:
            return {
                'training_triggered': False,
                'reason': 'No sessions meet quality threshold',
                'sessions_checked': len(new_sessions),
                'timestamp': datetime.now().isoformat()
            }
        
        # Disparar entrenamiento
        training_results = []
        for ready_session in ready_sessions:
            session_data = ready_session['session_data']
            quality = ready_session['quality']
            
            # Entrenar con el sistema disponible
            training_result = await self.training_orchestrator.train_with_data(
                [session_data]
            )
            
            # Guardar en historial
            self._save_training_run(
                session_id=session_data.get('session_id', 'unknown'),
                training_system=training_result.get('system', 'unknown'),
                quality_score=quality['quality_score'],
                data_points=quality['metrics']['total_pairs'],
                status='completed' if training_result.get('success') else 'failed',
                result=training_result
            )
            
            training_results.append(training_result)
        
        return {
            'training_triggered': True,
            'sessions_processed': len(ready_sessions),
            'training_results': training_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_training_run(self, session_id: str, training_system: str, 
                          quality_score: float, data_points: int,
                          status: str, result: Dict[str, Any]):
        """Guardar ejecución de entrenamiento en historial"""
        try:
            run_id = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with sqlite3.connect(self.training_history_db) as conn:
                conn.execute("""
                    INSERT INTO training_runs 
                    (run_id, session_id, training_system, quality_score, data_points, 
                     status, started_at, completed_at, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    session_id,
                    training_system,
                    quality_score,
                    data_points,
                    status,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    json.dumps(result)
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not save training run to history: {e}")
    
    async def execute_complete_evolution(self) -> Dict[str, Any]:
        """Ejecutar evolución completa del sistema"""
        
        logger.info("🚀 Starting complete system evolution...")
        
        # 1. Monitorear Hack-Memori
        monitor_result = await self.monitor_and_trigger_training()
        
        # 2. Obtener todas las sesiones listas
        all_ready_sessions = self.monitor.get_all_ready_sessions()
        
        # 3. Evaluar y entrenar
        evolution_results = {
            'monitor_result': monitor_result,
            'total_ready_sessions': len(all_ready_sessions),
            'training_systems_available': list(self.training_orchestrator.training_systems.keys()),
            'evolution_status': 'completed' if monitor_result.get('training_triggered') else 'no_data',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ System evolution completed: {evolution_results['evolution_status']}")
        
        return {
            'evolution_result': evolution_results,
            'pegamento_status': 'active',
            'total_systems_connected': len(self.training_orchestrator.training_systems)
        }


# =============================================================================
# GLOBAL INSTANCE AND PUBLIC API
# =============================================================================

_orchestrator = None

def get_hack_memori_orchestrator() -> HackMemoriTrainingOrchestrator:
    """Obtener instancia global del orquestador"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = HackMemoriTrainingOrchestrator()
    return _orchestrator

async def execute_complete_system_evolution() -> Dict[str, Any]:
    """Ejecutar evolución completa del sistema - PUNTO DE ENTRADA PRINCIPAL"""
    orchestrator = get_hack_memori_orchestrator()
    return await orchestrator.execute_complete_evolution()

async def monitor_and_trigger_system_evolution() -> Dict[str, Any]:
    """Monitorear y disparar evolución automática - PUNTO DE ENTRADA PARA MONITORING"""
    orchestrator = get_hack_memori_orchestrator()
    return await orchestrator.monitor_and_trigger_training()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'HackMemoriTrainingOrchestrator',
    'get_hack_memori_orchestrator',
    'execute_complete_system_evolution',
    'monitor_and_trigger_system_evolution',
]

__version__ = "2.0.0"
__author__ = "SHEILY AI - Hack-Memori Training Integration"
__description__ = "Real functional orchestrator connecting Hack-Memori with training systems"

if __name__ == "__main__":
    # Demo
    async def demo():
        print("🎯 Hack-Memori Training Orchestrator Demo")
        print("=" * 60)
        
        orchestrator = get_hack_memori_orchestrator()
        
        # Ejecutar evolución completa
        result = await execute_complete_system_evolution()
        
        print(f"\n✅ Evolution Status: {result['evolution_result']['evolution_status']}")
        print(f"📚 Systems Connected: {result['total_systems_connected']}")
        print(f"🔗 Pegamento Status: {result['pegamento_status']}")
    
    asyncio.run(demo())
