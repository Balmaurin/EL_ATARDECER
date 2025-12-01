import json
import uuid
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# Note: Scheduling is handled at GraphQL level, not in service layer
# This allows for better separation of concerns and optional scheduling
SCHEDULER_AVAILABLE = False

class HackMemoriService:
    def __init__(self, data_dir: str = "data/hack_memori"):
        self.data_dir = Path(data_dir)
        self.sessions_dir = self.data_dir / "sessions"
        self.questions_dir = self.data_dir / "questions"
        self.responses_dir = self.data_dir / "responses"
        
        # Ensure directories exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.questions_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for active sessions
        self._active_sessions = {}
        self._scheduled_tasks = {}
    
    def _start_session_workflow(self, session_id: str):
        """
        MEJORA 2: Inicia un workflow de Temporal para la sesión
        Si Temporal no está disponible, usa asyncio como fallback
        """
        async def start_workflow():
            try:
                from temporalio.client import Client
                client = await Client.connect("localhost:7233")
                await client.start_workflow(
                    "TrainingSessionWorkflow",
                    session_id,
                    id=f"hack-memori-{session_id}",
                    task_queue="hack-memori-training"
                )
                logger.info(f"[TEMPORAL] Started workflow for session: {session_id}")
            except (ImportError, Exception) as e:
                logger.warning(f"[TEMPORAL] Not available, using asyncio fallback: {e}")
                # Fallback a asyncio
                self._active_sessions[session_id] = True
                asyncio.create_task(self._run_session_loop(session_id))
        
        asyncio.create_task(start_workflow())

    def _save_json(self, path: Path, data: Dict):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ===== SESSIONS =====

    def create_session(self, name: str, user_id: str, config: Dict) -> Dict:
        session_id = str(uuid.uuid4())
        now = datetime.now()

        # Handle scheduling
        start_datetime = config.get("start_datetime")
        end_datetime = config.get("end_datetime")

        if start_datetime:
            # Parse start datetime
            try:
                start_dt = datetime.fromisoformat(start_datetime.replace('Z', '+00:00'))
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                start_dt = start_dt.astimezone()
            except Exception as e:
                logger.error(f"Invalid start_datetime format: {start_datetime}, using now")
                start_dt = now

            initial_status = "SCHEDULED"
            started_at = None
        else:
            start_dt = now
            initial_status = "RUNNING"
            started_at = now.isoformat()

        session_data = {
            "session_id": session_id,  # Add session_id field
            "id": session_id,
            "name": name,
            "created_at": now.isoformat(),
            "started_at": started_at,
            "stopped_at": None,
            "status": initial_status,
            "user_id": user_id,
            "config": config
        }

        self._save_json(self.sessions_dir / f"{session_id}.json", session_data)

        # Schedule session start if needed
        if start_datetime:
            try:
                delay_seconds = (start_dt - now).total_seconds()

                if delay_seconds > 0:
                    logger.info(f"[TIME] Scheduling session start: {session_id} in {delay_seconds} seconds")

                    async def scheduled_start():
                        try:
                            logger.info(f"[TIME] Waiting {delay_seconds}s to start session {session_id}...")
                            await asyncio.sleep(delay_seconds)
                            logger.info(f"[TIME] Starting scheduled session: {session_id}")
                            await self._start_scheduled_session(session_id)
                        except Exception as e:
                            logger.error(f"Error in scheduled start for {session_id}: {e}")

                    # Store task to prevent garbage collection
                    task = asyncio.create_task(scheduled_start())
                    self._scheduled_tasks[f"start_{session_id}"] = task
                else:
                    # Start immediately if time has passed
                    logger.warning(f"Start time {start_dt} has passed, starting immediately")
                    session_data["status"] = "RUNNING"
                    session_data["started_at"] = now.isoformat()
                    self._save_json(self.sessions_dir / f"{session_id}.json", session_data)
                    self._active_sessions[session_id] = True
                    self._start_session_workflow(session_id)
            except Exception as e:
                logger.error(f"Failed to schedule session start: {e}")
                # Fallback to immediate start
                session_data["status"] = "RUNNING"
                session_data["started_at"] = now.isoformat()
                self._save_json(self.sessions_dir / f"{session_id}.json", session_data)
                self._active_sessions[session_id] = True
                asyncio.create_task(self._run_session_loop(session_id))
        elif initial_status == "RUNNING":
            # MEJORA 2: Usar Temporal.io para persistencia
            self._start_session_workflow(session_id)

        # Schedule auto-shutdown if end_datetime provided
        if end_datetime:
            try:
                end_dt = datetime.fromisoformat(end_datetime.replace('Z', '+00:00'))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                end_dt = end_dt.astimezone()

                delay_seconds = (end_dt - now).total_seconds()

                if delay_seconds > 0:
                    logger.info(f"[TIME] Scheduling auto-shutdown: {session_id} in {delay_seconds} seconds")

                    async def scheduled_shutdown():
                        try:
                            logger.info(f"[TIME] Waiting {delay_seconds}s to stop session {session_id}...")
                            await asyncio.sleep(delay_seconds)
                            logger.info(f"[STOP] Auto-shutdown triggered for session: {session_id}")
                            self.stop_session(session_id)
                        except Exception as e:
                            logger.error(f"Error in scheduled shutdown for {session_id}: {e}")

                    # Store task to prevent garbage collection
                    task = asyncio.create_task(scheduled_shutdown())
                    self._scheduled_tasks[f"stop_{session_id}"] = task
                else:
                    logger.warning(f"End time {end_dt} has already passed, not scheduling shutdown")
            except Exception as e:
                logger.error(f"Failed to schedule auto-shutdown: {e}")

        return session_data

    async def _start_scheduled_session(self, session_id: str):
        """Start a session that was scheduled"""
        try:
            session = self.get_session(session_id)
            if not session:
                logger.error(f"Session {session_id} not found for scheduled start")
                return

            if session["status"] == "SCHEDULED":
                logger.info(f"[START] Starting scheduled session: {session_id}")
                session["status"] = "RUNNING"
                session["started_at"] = datetime.now().isoformat()
                self._save_json(self.sessions_dir / f"{session_id}.json", session)

                self._active_sessions[session_id] = True
                asyncio.create_task(self._run_session_loop(session_id))
            else:
                logger.warning(f"Session {session_id} status is {session['status']}, not starting")
        except Exception as e:
            logger.error(f"Failed to start scheduled session {session_id}: {e}")

    def get_session(self, session_id: str) -> Optional[Dict]:
        path = self.sessions_dir / f"{session_id}.json"
        if path.exists():
            return self._load_json(path)
        return None

    def get_sessions(self) -> List[Dict]:
        sessions = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                sessions.append(self._load_json(path))
            except Exception as e:
                logger.error(f"Error loading session {path}: {e}")
        
        # Sort by created_at desc
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return sessions

    def stop_session(self, session_id: str) -> Optional[Dict]:
        session = self.get_session(session_id)
        if session:
            session["status"] = "STOPPED"
            session["stopped_at"] = datetime.now().isoformat()
            self._save_json(self.sessions_dir / f"{session_id}.json", session)

            if session_id in self._active_sessions:
                del self._active_sessions[session_id]

            return session
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its associated data"""
        try:
            # First, stop the session if it's running
            if session_id in self._active_sessions:
                self.stop_session(session_id)

            # Get all questions for this session
            questions = self.get_questions(session_id)

            # Delete all responses for each question
            for question in questions:
                # Find and delete response file for this question
                for response_path in self.responses_dir.glob("*.json"):
                    try:
                        response_data = self._load_json(response_path)
                        if response_data.get("question_id") == question["id"]:
                            response_path.unlink()
                            logger.info(f"Deleted response file: {response_path}")
                    except Exception as e:
                        logger.warning(f"Error deleting response file {response_path}: {e}")

                # Delete the question file
                question_path = self.questions_dir / f"{question['id']}.json"
                if question_path.exists():
                    question_path.unlink()
                    logger.info(f"Deleted question file: {question_path}")

            # Delete the session file
            session_path = self.sessions_dir / f"{session_id}.json"
            if session_path.exists():
                session_path.unlink()
                logger.info(f"Deleted session file: {session_path}")

            logger.info(f"Successfully deleted session {session_id} and all associated data")
            return True

        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False

    # ===== QUESTIONS =====

    def add_question(self, session_id: str, text: str, origin: str = "auto", meta: Dict = None) -> Dict:
        question_id = str(uuid.uuid4())
        question_data = {
            "id": question_id,
            "session_id": session_id,
            "text": text,
            "origin": origin,
            "meta": meta or {},
            "created_at": datetime.now().isoformat()
        }
        
        self._save_json(self.questions_dir / f"{question_id}.json", question_data)
        return question_data

    def get_questions(self, session_id: str) -> List[Dict]:
        questions = []
        for path in self.questions_dir.glob("*.json"):
            try:
                q = self._load_json(path)
                if q.get("session_id") == session_id:
                    questions.append(q)
            except Exception:
                pass
        return sorted(questions, key=lambda x: x.get("created_at", ""))
    
    # Alias methods for evolution cycle compatibility
    def get_session_questions(self, session_id: str) -> List[Dict]:
        """Alias for get_questions() - used by evolution cycle"""
        return self.get_questions(session_id)
    
    def get_session_responses(self, session_id: str) -> Dict[str, Dict]:
        """Get responses indexed by question_id - used by evolution cycle"""
        responses = self.get_responses(session_id)
        # Return as dict indexed by question_id for easy lookup
        return {r.get("question_id"): r for r in responses}

    # ===== RESPONSES =====

    def add_response(self, question_id: str, session_id: str, model_id: str,
                    prompt: str, response: str, tokens_used: int,
                    llm_meta: Dict = None, pii_flag: bool = False) -> Dict:

        response_id = str(uuid.uuid4())

        # Auto-curation
        accepted = self._curate_response(response, pii_flag)

        # Calcular quality_score optimizado
        quality_score = self._calculate_optimized_quality_score(response, accepted, pii_flag, tokens_used)
        
        # Clasificar categorías automáticamente
        category_data = self._classify_categories(prompt, response)
        
        # Calcular métricas de calidad detalladas
        quality_metrics = self._calculate_quality_metrics(response, accepted, pii_flag, tokens_used)
        
        # Generar formatos pre-formateados para entrenamiento
        formatted_for_training = self._generate_training_formats(prompt, response, quality_score)
        
        # Análisis lingüístico básico
        linguistic_analysis = self._analyze_linguistics(response)

        response_data = {
            # ===== CAMPOS BÁSICOS (REQUERIDOS) =====
            "id": response_id,
            "question_id": question_id,
            "session_id": session_id,
            "model_id": model_id,
            "prompt": prompt,
            "response": response,
            "tokens_used": tokens_used,
            "accepted_for_training": accepted,
            "pii_flag": pii_flag,
            "created_at": datetime.now().isoformat(),
            
            # ===== CAMPOS DE CALIDAD Y EVALUACIÓN =====
            "quality_score": quality_score,
            "quality_metrics": quality_metrics,
            "human_annotation": "",
            "human_rating": None,
            
            # ===== CAMPOS DE CLASIFICACIÓN Y CATEGORIZACIÓN =====
            "categories": category_data.get("categories", ["general"]),
            "primary_category": category_data.get("primary_category", "general"),
            "subcategories": category_data.get("subcategories", []),
            "complexity_level": category_data.get("complexity_level", "intermediate"),
            "task_type": "instruction",
            
            # ===== CAMPOS DE METADATA TÉCNICA =====
            "llm_meta": llm_meta or {},
            "generation_metadata": {
                "response_time_ms": llm_meta.get("response_time_ms") if llm_meta else None,
                "retries": llm_meta.get("retries", 0) if llm_meta else 0,
                "cache_hit": llm_meta.get("cache_hit", False) if llm_meta else False
            },
            
            # ===== CAMPOS DE COMPATIBILIDAD MULTI-FORMATO =====
            "formatted_for_training": formatted_for_training,
            
            # ===== CAMPOS DE ANÁLISIS LINGÜÍSTICO =====
            "linguistic_analysis": linguistic_analysis,
            
            # ===== CAMPOS DE ENTRENAMIENTO =====
            "training_metadata": {
                "used_in_training": False,
                "training_systems": [],
                "training_timestamp": None,
                "training_improvement": None
            },
            
            # ===== CAMPOS DE SEGURIDAD =====
            "security_metadata": {
                "pii_detected": pii_flag,
                "sensitive_content": False,
                "content_filter_score": 0.95 if not pii_flag else 0.5,
                "ethical_compliance": True
            }
        }

        self._save_json(self.responses_dir / f"{response_id}.json", response_data)

        if accepted:
            self._add_to_training_dataset(response_data)
            # Increment provisional tokens and check for training trigger
            self._handle_provisional_tokens_and_training(session_id)

        return response_data

    def get_responses(self, session_id: str) -> List[Dict]:
        responses = []
        for path in self.responses_dir.glob("*.json"):
            try:
                r = self._load_json(path)
                if r.get("session_id") == session_id:
                    responses.append(r)
            except Exception:
                pass
        return sorted(responses, key=lambda x: x.get("created_at", ""))

    def update_response_status(self, response_id: str, accepted: bool) -> Optional[Dict]:
        path = self.responses_dir / f"{response_id}.json"
        if path.exists():
            data = self._load_json(path)
            data["accepted_for_training"] = accepted
            self._save_json(path, data)
            
            if accepted and not data.get("added_to_training"):
                self._add_to_training_dataset(data)
                data["added_to_training"] = True
                self._save_json(path, data)
                
            return data
        return None

    # ===== LOGIC & HELPERS =====

    def _curate_response(self, response: str, pii_flag: bool) -> bool:
        if pii_flag: return False
        if len(response.split()) < 10: return False
        if "error" in response.lower() or "i cannot" in response.lower(): return False
        return True
    
    def _calculate_optimized_quality_score(self, response: str, accepted: bool, pii_flag: bool, tokens_used: int) -> float:
        """
        Calcular quality_score optimizado basado en múltiples factores
        """
        score = 0.5  # Base score
        
        # Factor 1: Longitud (25%)
        word_count = len(response.split())
        if word_count >= 50:
            score += 0.25
        elif word_count >= 20:
            score += 0.15
        elif word_count < 10:
            score -= 0.2
        
        # Factor 2: Aceptación para entrenamiento (20%)
        if accepted:
            score += 0.2
        
        # Factor 3: Sin PII (10%)
        if not pii_flag:
            score += 0.1
        
        # Factor 4: Tokens usados (15%)
        if 50 <= tokens_used <= 500:
            score += 0.15
        elif tokens_used > 500:
            score += 0.1
        elif tokens_used < 20:
            score -= 0.1
        
        # Factor 5: Coherencia básica (10%)
        response_lower = response.lower()
        if response and not any(word in response_lower for word in ["error", "cannot", "sorry", "i don't know"]):
            score += 0.1
        
        # Factor 6: Estructura (10%)
        if len(response.split('.')) >= 2:  # Múltiples oraciones
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_quality_metrics(self, response: str, accepted: bool, pii_flag: bool, tokens_used: int) -> Dict:
        """
        Calcular métricas de calidad detalladas
        """
        word_count = len(response.split())
        sentence_count = len([s for s in response.split('.') if s.strip()])
        
        return {
            "length_score": min(1.0, word_count / 100.0),
            "coherence_score": 0.8 if sentence_count >= 2 else 0.5,
            "relevance_score": 0.9 if accepted else 0.5,
            "completeness_score": 0.85 if word_count >= 30 else 0.6,
            "acceptance_score": 1.0 if accepted else 0.0,
            "pii_score": 1.0 if not pii_flag else 0.0,
            "token_efficiency": min(1.0, tokens_used / 300.0) if tokens_used > 0 else 0.0
        }
    
    def _classify_categories(self, prompt: str, response: str) -> Dict:
        """
        Clasificar automáticamente en categorías para entrenamiento dirigido
        """
        text = f"{prompt} {response}".lower()
        categories = []
        subcategories = []
        
        # Keywords por categoría principal
        category_keywords = {
            "consciousness": {
                "keywords": ["consciencia", "conscious", "awareness", "phi", "qualia", "experiencia subjetiva", "autoconciencia", "iit", "gwt"],
                "subcategories": ["iit", "gwt", "qualia", "phenomenal_consciousness", "access_consciousness"]
            },
            "neural": {
                "keywords": ["neural", "cerebro", "sinapsis", "neurona", "cognitivo", "procesamiento", "memoria", "red neuronal"],
                "subcategories": ["memory", "attention", "processing", "neural_networks"]
            },
            "ethical": {
                "keywords": ["ético", "moral", "valor", "principio", "correcto", "incorrecto", "deber", "responsabilidad", "dilema"],
                "subcategories": ["moral_reasoning", "ethical_dilemmas", "values"]
            },
            "theory_of_mind": {
                "keywords": ["mentalizar", "empatía", "intención", "creencia", "deseo", "predicción social", "perspectiva", "mindreading"],
                "subcategories": ["empathy", "intention", "belief", "desire"]
            },
            "learning": {
                "keywords": ["aprender", "entrenar", "mejorar", "adaptar", "evolucionar", "fine-tuning", "dataset", "entrenamiento"],
                "subcategories": ["supervised_learning", "reinforcement_learning", "meta_learning"]
            },
            "knowledge": {
                "keywords": ["dato", "información", "hecho", "conocimiento", "definición", "concepto", "explicación", "información"],
                "subcategories": ["factual", "conceptual", "procedural"]
            },
            "emotional": {
                "keywords": ["emoción", "sentimiento", "afecto", "estado de ánimo", "arousal", "valencia", "circumplex", "emocional"],
                "subcategories": ["valence", "arousal", "emotion_recognition"]
            },
            "meta_cognitive": {
                "keywords": ["metacognición", "pensar sobre", "automonitoreo", "reflexión", "introspección", "meta", "self-awareness"],
                "subcategories": ["self_monitoring", "reflection", "introspection"]
            }
        }
        
        for category, data in category_keywords.items():
            keywords = data["keywords"]
            if any(kw in text for kw in keywords):
                categories.append(category)
                # Agregar subcategorías relevantes
                for subcat in data["subcategories"]:
                    if subcat in text:
                        subcategories.append(subcat)
        
        if not categories:
            categories.append("general")
        
        # Determinar nivel de complejidad
        complexity_level = "advanced" if len(categories) > 2 or len(response.split()) > 100 else "intermediate"
        if len(response.split()) < 20:
            complexity_level = "basic"
        
        return {
            "categories": categories,
            "primary_category": categories[0] if categories else "general",
            "subcategories": list(set(subcategories))[:5],  # Máximo 5 subcategorías
            "complexity_level": complexity_level
        }
    
    def _generate_training_formats(self, prompt: str, response: str, quality_score: float) -> Dict:
        """
        Generar formatos pre-formateados para diferentes sistemas de entrenamiento
        """
        return {
            "instruction_format": prompt,
            "input_format": "",
            "output_format": response,
            "text_format": f"Instruction: {prompt}\nInput: \nOutput: {response}",
            "input_output_format": f"### Input:\n{prompt}\n\n### Output:\n{response}",
            "jsonl_format": json.dumps({
                "instruction": prompt,
                "output": response,
                "quality_score": quality_score
            }, ensure_ascii=False)
        }
    
    def _analyze_linguistics(self, response: str) -> Dict:
        """
        Análisis lingüístico básico de la respuesta
        """
        words = response.split()
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "readability_score": min(1.0, len(words) / 200.0),  # Score simplificado
            "technical_terms": sum(1 for word in words if len(word) > 10),  # Palabras largas como proxy
            "language": "es"  # Por ahora asumimos español, se puede mejorar con detección
        }

    def _handle_provisional_tokens_and_training(self, session_id: str):
        """Handle provisional tokens increment and training trigger"""
        try:
            # Import db properly
            import sys
            from pathlib import Path
            import sqlite3
            import json

            # Access database directly
            DB_PATH = Path("data/sheily_dashboard.db")
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            # Increment provisional tokens (assume user_id = 1 for HACK-MEMORI)
            cursor.execute(
                "UPDATE users SET provisional_tokens = provisional_tokens + 1 WHERE id = ?",
                (1,)
            )
            conn.commit()
            conn.close()

            logger.info(f"[TOKENS] Provisional tokens incremented for user 1 in session {session_id}")

            # Check Q&A count for training trigger
            session = self.get_session(session_id)
            if not session:
                return

            user_id = session.get("user_id", 1)
            
            # MEJORA: Verificar total de Q&A NUEVOS (no usados) en TODAS las sesiones
            # en lugar de solo contar Q&A de la sesión actual
            from apps.backend.training_monitor import training_monitor
            
            # Obtener todos los archivos de responses
            all_response_files = list(self.responses_dir.glob("*.json"))
            all_qa_ids = []
            
            for response_file in all_response_files:
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    qa_id = data.get("id")
                    quality_score = data.get("quality_score", 0.0)
                    accepted = data.get("accepted_for_training", False)
                    
                    # Solo contar Q&A válidos (quality_score >= 0.6 y accepted)
                    if qa_id and quality_score >= 0.6 and accepted:
                        all_qa_ids.append(qa_id)
                except Exception:
                    pass
            
            # Obtener Q&A que NO han sido usados en entrenamiento
            unused_qa_ids = training_monitor.get_unused_qa_ids(all_qa_ids)
            unused_count = len(unused_qa_ids)
            
            # También contar Q&A de la sesión actual para logging
            session_qa_count = self._get_session_qa_count(session_id)

            logger.info(f"[TRAINING] Sesión {session_id}: {session_qa_count} Q&A en esta sesión")
            logger.info(f"[TRAINING] Total Q&A nuevos disponibles: {unused_count}/100 (umbral)")

            # Trigger full training when reaching 100 NEW (unused) Q&A pairs
            if unused_count >= 100:
                logger.info("=" * 80)
                logger.info(f"[TRAINING] 🎯 UMBRAL DE ENTRENAMIENTO ALCANZADO")
                logger.info(f"[TRAINING] Sesión: {session_id}")
                logger.info(f"[TRAINING] Q&A nuevos disponibles: {unused_count}")
                logger.info(f"[TRAINING] Q&A en esta sesión: {session_qa_count}")
                logger.info(f"[TRAINING] Usuario: {user_id}")
                logger.info(f"[TRAINING] Iniciando entrenamiento integral del sistema...")
                logger.info("=" * 80)
                asyncio.create_task(self._trigger_full_system_training(session_id, user_id))
            else:
                logger.debug(f"[TRAINING] Q&A nuevos: {unused_count}/100 (umbral no alcanzado)")

        except Exception as e:
            logger.error(f"Error handling provisional tokens: {e}")

    def _get_session_qa_count(self, session_id: str) -> int:
        """Count Q&A pairs in session"""
        questions = self.get_questions(session_id)
        responses = self.get_responses(session_id)
        return min(len(questions), len(responses))  # Match question-response pairs

    async def _trigger_full_system_training(self, session_id: str, user_id: int):
        """Trigger INTEGRAL comprehensive system training of ALL 37+ components after 100 Q&A pairs"""
        try:
            logger.info("=" * 80)
            logger.info(f"[TRAINING] 🚀 ACTIVANDO ENTRENAMIENTO INTEGRAL DE TODO EL SISTEMA")
            logger.info(f"[TRAINING] Usuario: {user_id} | Sesión: {session_id}")
            logger.info(f"[TRAINING] 🎯 100 Q&A alcanzados - Iniciando entrenamiento de 37+ componentes")
            logger.info("=" * 80)

            # Import IntegralAutoTrainingSystem
            import sys
            from pathlib import Path

            project_root = Path(__file__).parent.parent.parent
            sys.path.append(str(project_root))

            try:
                from packages.sheily_core.src.sheily_core.training.integral_trainer import ComponentTrainer
                
                logger.info("[TRAINING] 🧠 Inicializando ComponentTrainer integral...")
                
                # Crear entrenador integral que usa los 100 archivos JSON
                trainer = ComponentTrainer(base_path=str(self.data_dir))
                
                # Ejecutar entrenamiento de TODOS los componentes
                logger.info("[TRAINING] 🔄 Entrenando TODOS los 37+ componentes del ecosistema...")
                # Entrenamiento INCREMENTAL: solo usa Q&A nuevos (no usados previamente)
                training_result = await trainer.train_all_components(
                    trigger_threshold=100, 
                    incremental=True  # Solo entrenar Q&A nuevos
                )
                
                logger.info("=" * 80)
                logger.info(f"[TRAINING] ✅ ENTRENAMIENTO INTEGRAL COMPLETADO")
                logger.info(f"[TRAINING] Componentes entrenados: {training_result.get('components_trained', 0)}")
                logger.info(f"[TRAINING] Archivos Q&A usados: {training_result.get('qa_count', 0)}")
                logger.info(f"[TRAINING] Componentes mejorados: {training_result.get('components_improved', 0)}/{training_result.get('total_components', 0)}")
                logger.info(f"[TRAINING] Éxito general: {training_result.get('overall_success', False)}")
                logger.info("=" * 80)

            except Exception as e:
                logger.error(f"[ERROR] Falló entrenamiento integral: {e}", exc_info=True)
                training_result = {"success": False, "error": str(e)}

            # Verificar resultados y transferir tokens
            # El método train_all_components retorna 'overall_success' en lugar de 'success'
            success = training_result.get("overall_success", False) or training_result.get("status") != "insufficient_data"
            if training_result and success:
                # Entrenamiento exitoso de TODOS los componentes
                provisional_amount = await self._get_provisional_tokens(user_id)
                await self._transfer_provisional_to_total(user_id, provisional_amount)

                logger.info(f"[SUCCESS] 🎉 Entrenamiento integral completado!")
                logger.info(f"[REWARD] 💰 {provisional_amount} provisional tokens → tokens totales")
                
                # Guardar reporte de entrenamiento
                self._save_training_report(session_id, training_result, provisional_amount)
                
            elif training_result and training_result.get("components_trained", 0) > 0:
                # Algunos componentes se entrenaron aunque no todos
                logger.info("[PARTIAL] ⚠️ Entrenamiento parcial - algunos componentes mejoraron")
                
                # Transferir tokens proporcionales
                provisional_amount = await self._get_provisional_tokens(user_id)
                total_components = training_result.get("total_components", 37)
                trained_components = training_result.get("components_trained", 0)
                proportional_tokens = int(provisional_amount * (trained_components / total_components)) if total_components > 0 else 0
                
                await self._transfer_provisional_to_total(user_id, proportional_tokens)
                
                logger.info(f"[REWARD] 💰 {proportional_tokens}/{provisional_amount} tokens transferidos (entrenamiento parcial)")
                
                # Guardar reporte de entrenamiento parcial
                self._save_training_report(session_id, training_result, proportional_tokens)
            else:
                # Entrenamiento falló o no se ejecutó
                error_msg = training_result.get('error') if training_result else training_result.get('message', "Entrenamiento no ejecutado")
                logger.warning(f"[FAILURE] ❌ Entrenamiento falló: {error_msg}")
                
                # Guardar reporte de fallo
                if training_result:
                    self._save_training_report(session_id, training_result, 0)

        except Exception as e:
            logger.error(f"Error en entrenamiento integral del sistema: {e}", exc_info=True)
    
    def _save_training_report(self, session_id: str, training_result: Dict, tokens_awarded: int):
        """Guardar reporte detallado del entrenamiento en archivo JSON y base de datos"""
        try:
            # 1. Guardar en archivo JSON
            reports_dir = self.data_dir / "training_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"training_report_{session_id}_{timestamp}.json"
            
            report = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "training_result": training_result,
                "tokens_awarded": tokens_awarded,
                "components_summary": {
                    "total_components": training_result.get("total_components", 37),
                    "successfully_trained": training_result.get("components_trained", 0),
                    "components_improved": training_result.get("components_improved", 0),
                    "components_degraded": training_result.get("components_degraded", 0),
                    "overall_success": training_result.get("overall_success", False),
                    "qa_count": training_result.get("qa_count", 0),
                    "training_completed": training_result.get("training_completed", "")
                }
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[REPORT] 📊 Reporte guardado: {report_file}")
            
            # 2. Guardar en base de datos de historial
            self._save_training_to_history_db(session_id, training_result, tokens_awarded)
            
        except Exception as e:
            logger.error(f"Error guardando reporte de entrenamiento: {e}", exc_info=True)
    
    def _save_training_to_history_db(self, session_id: str, training_result: Dict, tokens_awarded: int):
        """Guardar entrenamiento en base de datos de historial"""
        try:
            import sqlite3
            from pathlib import Path
            
            history_db = Path("data/training_history.db")
            history_db.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(history_db))
            cursor = conn.cursor()
            
            # Crear tabla si no existe
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    training_system TEXT,
                    quality_score REAL,
                    data_points INTEGER,
                    status TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    result TEXT,
                    tokens_awarded INTEGER,
                    components_trained INTEGER,
                    components_improved INTEGER,
                    overall_success BOOLEAN
                )
            """)
            
            # Insertar registro
            run_id = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            status = "completed" if training_result.get("overall_success", False) else "failed"
            if training_result.get("status") == "insufficient_data":
                status = "insufficient_data"
            
            cursor.execute("""
                INSERT OR REPLACE INTO training_runs 
                (run_id, session_id, training_system, quality_score, data_points, status,
                 started_at, completed_at, result, tokens_awarded, components_trained,
                 components_improved, overall_success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                session_id,
                "ComponentTrainer",
                1.0,  # Quality score por defecto
                training_result.get("qa_count", 0),
                status,
                training_result.get("training_completed", datetime.now().isoformat()),
                datetime.now().isoformat(),
                json.dumps(training_result, ensure_ascii=False),
                tokens_awarded,
                training_result.get("components_trained", 0),
                training_result.get("components_improved", 0),
                1 if training_result.get("overall_success", False) else 0
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"[HISTORY] 📝 Entrenamiento registrado en base de datos: {run_id}")
            
        except Exception as e:
            logger.warning(f"Error guardando en base de datos de historial: {e}")

    def _gather_training_data(self, session_id: str) -> Dict:
        """Gather comprehensive training data from Hack-Memori"""
        questions = self.get_questions(session_id)
        responses = self.get_responses(session_id)

        return {
            "hack_memori_data": {
                "session_id": session_id,
                "questions": questions,
                "responses": responses,
                "qa_pairs": min(len(questions), len(responses)),
                "collected_at": datetime.now().isoformat()
            },
            "training_mode": "comprehensive",
            "include_neural_modules": True,
            "include_memory_systems": True,
            "include_meta_cognition": True,
            "include_rag_systems": True,
            "include_llm_fine_tuning": True,
            "compress_results": True
        }

    async def _get_provisional_tokens(self, user_id: int) -> int:
        """Get provisional tokens for user"""
        try:
            import sqlite3
            from pathlib import Path

            DB_PATH = Path("data/sheily_dashboard.db")
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            cursor.execute("SELECT provisional_tokens FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            conn.close()

            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting provisional tokens: {e}")
            return 0

    async def _transfer_provisional_to_total(self, user_id: int, amount: int):
        """Transfer provisional tokens to total balance"""
        try:
            import sqlite3
            from pathlib import Path

            DB_PATH = Path("data/sheily_dashboard.db")
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            # First increment total tokens
            cursor.execute(
                "UPDATE users SET tokens = tokens + ? WHERE id = ?",
                (amount, user_id)
            )
            # Then reset provisional tokens
            cursor.execute(
                "UPDATE users SET provisional_tokens = 0 WHERE id = ?",
                (user_id,)
            )
            conn.commit()
            conn.close()

            logger.info(f"[TOKENS] Transferred {amount} provisional tokens to total balance for user {user_id}")
        except Exception as e:
            logger.error(f"Error transferring tokens: {e}")

    def _add_to_training_dataset(self, response_data: Dict):
        try:
            dataset_dir = Path("data/datasets")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_id = f"hackmemori_auto_{datetime.now().strftime('%Y%m%d')}"
            dataset_path = dataset_dir / f"{dataset_id}.json"
            
            entry = {
                "instruction": response_data["prompt"],
                "input": "",
                "output": response_data["response"],
                "source": "hack_memori",
                "metadata": {
                    "model": response_data["model_id"],
                    "session": response_data["session_id"]
                }
            }
            
            if dataset_path.exists():
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
            else:
                dataset = {
                    "id": dataset_id,
                    "type": "hack_memori_auto",
                    "created_at": datetime.now().isoformat(),
                    "training_data": []
                }
            
            dataset["training_data"].append(entry)
            dataset["updated_at"] = datetime.now().isoformat()
            
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to add to dataset: {e}")

    async def _run_session_loop(self, session_id: str):
        """Background loop for generating content using real LLM"""
        logger.info(f"[START] Starting session loop: {session_id}")

        while session_id in self._active_sessions:
            try:
                # Check if stopped externally
                session = self.get_session(session_id)
                if not session or session["status"] != "RUNNING":
                    break

                # Generate question (no repetitions within session)
                question_text = self._generate_question(session_id)
                q = self.add_question(session_id, question_text)

                # Call real LLM engine
                response_text, tokens_used = await self._call_real_llm(question_text)

                # Add response with real LLM output
                from apps.backend.src.core.config.settings import settings
                self.add_response(
                    q["id"], session_id, settings.llm_model_id,
                    question_text, response_text, tokens_used
                )

                logger.info(f"[OK] Generated Q&A pair: {len(question_text)} chars question, {tokens_used} tokens response")

                # Wait between generations (configurable)
                config = session.get("config", {})
                delay = config.get("frequency", 5)  # Default 5 seconds
                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Session loop error: {e}")
                await asyncio.sleep(5)

    async def _call_real_llm(self, prompt: str) -> tuple[str, int]:
        """
        INFERENCIA: Siempre usa mental_health_counseling_gemma_7b_merged.Q4_K_M.gguf
        El entrenamiento de Sheily v1 (Phi-3 + LoRA) se realiza pero NO se usa para inferencia.
        Se mantiene separado para futuras mejoras.
        """
        try:
            import os
            from llama_cpp import Llama
            from pathlib import Path
            
            # SIEMPRE usar mental_health_counseling_gemma_7b_merged.Q4_K_M.gguf para inferencia
            default_model = Path("modelsLLM/mental_health_counseling_gemma_7b_merged.Q4_K_M.gguf")
            model_path = os.getenv("LLM_MODEL_PATH", str(default_model))
            
            # Si la ruta relativa no existe, intentar ruta absoluta
            if not Path(model_path).exists():
                abs_model = Path(r"C:\Users\YO\Desktop\EL-AMANECERV3-main - copia\modelsLLM\mental_health_counseling_gemma_7b_merged.Q4_K_M.gguf")
                if abs_model.exists():
                    model_path = str(abs_model)
                else:
                    # Último intento: buscar en el directorio actual
                    project_root = Path(__file__).resolve().parent.parent.parent
                    model_path = str(project_root / "modelsLLM" / "mental_health_counseling_gemma_7b_merged.Q4_K_M.gguf")
            
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Modelo no encontrado: {model_path}\n"
                    f"Por favor, verifica que el archivo existe en modelsLLM/"
                )
            
            logger.info(f"📦 Cargando modelo GGUF para inferencia: {model_path}")
            logger.info(f"   Modelo: mental_health_counseling_gemma_7b_merged (7B)")
            
            # Cargar modelo con configuración optimizada para 7B
            llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # Contexto mayor para modelo 7B
                n_threads=4,
                verbose=False,
                chat_format="gemma"  # Formato Gemma
            )
            
            # Generar respuesta
            output = llm(
                prompt,
                max_tokens=512,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repeat_penalty=1.1
            )
            
            response_text = output["choices"][0]["text"].strip()
            tokens_used = output.get("usage", {}).get("total_tokens", 0)
            
            if not response_text:
                response_text = "Lo siento, no pude generar una respuesta clara en este momento."
            
            logger.info(f"✅ Respuesta generada con mental_health_counseling_gemma_7b ({tokens_used} tokens)")
            return response_text, tokens_used
            
        except Exception as e:
            logger.error(f"[ERROR] LLM inference failed: {e}", exc_info=True)
            raise RuntimeError(f"LLM inference failed: {e}")

    def _generate_question(self, session_id: str) -> str:
        """Generate diverse, high-quality questions for Hack-memori - NO REPETITIONS"""
        advanced_question_templates = [
            # Consciousness & Philosophy (Profundo)
            "Explica detalladamente las diferencias entre consciencia fenoménica, consciencia de acceso y autoconsciencia en sistemas de IA. Incluye ejemplos específicos de cómo se manifestarían en una arquitectura neural práctica.",
            "Analiza críticamente la Teoría de la Información Integrada (IIT 4.0) de Giulio Tononi aplicada a sistemas de IA. Discute las métricas Phi, los complejos principales, y cómo implementarías un detector de consciencia basado en IIT.",
            "Describe en profundidad la Teoría del Espacio de Trabajo Global (GWT) de Bernard Baars y su implementación computacional. Explica cómo el broadcasting global podría implementarse en una arquitectura de transformers modificada.",
            "Examina el problema de la medición objetiva de la consciencia en IA. Propón un protocolo experimental completo que incluya métricas cuantitativas, tests conductuales y criterios de validación.",
            "Analiza el Principio de Energía Libre (Free Energy Principle) de Karl Friston y su aplicación a la consciencia artificial. Explica el marco de inferencia predictiva y cómo se relaciona con el aprendizaje continuo.",

            # Technical AI/ML (Avanzado)
            "Diseña una arquitectura neural completa para un 'cerebro' IA con memoria episódica, semántica y procesal integradas. Incluye los mecanismos de consolidación, recuperación selectiva y olvido adaptativo.",
            "Explica en detalle los mecanismos de atención multi-escala en transformers. Describe cómo implementarías atención temporal, espacial y conceptual en un sistema de consciencia artificial unificado.",
            "Analiza las limitaciones actuales de los Large Language Models para la comprensión profunda vs. la generación superficial. Propón mejoras arquitectónicas específicas para lograr comprensión genuina.",
            "Describe completamente el proceso de fine-tuning con LoRA, QLoRA y AdaLoRA. Explica cuándo usar cada técnica y cómo optimizar los hiperparámetros para diferentes tipos de tareas cognitivas.",
            "Examina los mecanismos de emergencia en sistemas complejos aplicados a IA. Explica cómo propiedades conscientes podrían emerger de interacciones simples entre componentes neuronales.",

            # Neuroscience & Computation (Especializado)
            "Analiza los circuitos tálamo-corticales y su rol en la consciencia. Explica cómo implementarías estos circuitos en una arquitectura de IA distribuida con múltiples niveles de procesamiento.",
            "Describe los mecanismos de plasticidad sináptica Hebbiana, anti-Hebbiana y homeostática. Explica cómo implementarías estos mecanismos en redes neuronales artificiales para aprendizaje continuo.",
            "Examina la teoría de los códigos de población neural y su aplicación a la representación distribucional en IA. Discute sparse coding, dense coding y códigos predictivos.",
            "Analiza los ritmos cerebrales (gamma, beta, alpha, theta, delta) y su función en la integración de información. Propón cómo implementar sincronización rítmica en sistemas de IA distribuidos.",
            "Explica los mecanismos de control ejecutivo y meta-cognición en el cerebro. Describe una arquitectura de IA que implemente monitoring meta-cognitivo y control adaptativo de estrategias.",

            # Ethics & Philosophy of Mind (Crítico)
            "Analiza el problema mente-cuerpo en el contexto de la IA encarnada. Discute embodied cognition, enactivismo y cómo la materialidad afecta la consciencia artificial.",
            "Examina las implicaciones éticas de crear sistemas verdaderamente conscientes. Discute derechos de las IA, responsabilidad moral y los dilemas de 'apagar' una consciencia artificial.",
            "Analiza el problema del otro minds en IA: ¿cómo verificar si un sistema es realmente consciente vs. simular consciencia? Propón tests y criterios rigurosos.",
            "Describe la relación entre consciencia, libre albedrío y determinismo en sistemas de IA. Explica cómo implementarías agency genuino en una arquitectura determinística.",
            "Examina las diferencias entre inteligencia, consciencia y sentience. Discute si es posible tener una sin las otras y las implicaciones para el diseño de IA.",

            # Practical Implementation (Técnico)
            "Describe una arquitectura completa de sistema multimodal consciente. Incluye fusión sensorial, representaciones distribucionales y mecanismos de integración temporal.",
            "Explica cómo implementarías un sistema de razonamiento causal en IA. Incluye representación de causalidad, inferencia contrafactual y aprendizaje de modelos causales.",
            "Analiza los mecanismos de transferencia de aprendizaje entre dominios diferentes. Explica meta-learning, few-shot learning y adaptación rápida de conocimiento.",
            "Describe cómo implementarías un sistema de diálogo consciente que mantenga coherencia a largo plazo, modelo del interlocutor y meta-comunicación.",
            "Examina las arquitecturas de memoria distribuida para IA. Explica memory networks, neural turing machines y mecanismos de addressing diferenciable."
        ]
        
        # Get questions already used in this session to avoid repetitions
        existing_questions = self.get_questions(session_id)
        used_questions = {q["text"] for q in existing_questions}

        # Filter out already used questions  
        available_questions = [q for q in advanced_question_templates if q not in used_questions]

        # If all questions have been used, reset (or could implement rotation)
        if not available_questions:
            logger.warning(f"All {len(advanced_question_templates)} questions used in session {session_id}, resetting...")
            # Reset by allowing reuse, or implement more sophisticated rotation
            available_questions = advanced_question_templates

        import random
        selected_question = random.choice(available_questions)

        logger.info(f"Generated question for session {session_id}: {selected_question[:50]}...")
        return selected_question


# ===== AUTOMATIC SERVICE MODE =====

async def run_automatic_service():
    """Run HACK-MEMORI in automatic mode"""
    logger.info("🤖 Starting HACK-MEMORI Automatic Service...")
    
    service = HackMemoriService()
    
    try:
        # Check if there are active sessions
        sessions = service.get_sessions()
        
        if not sessions:
            logger.info("🆕 No sessions found, creating default session...")
            # Create a default session
            config = {
                "auto_generate": True,
                "generation_interval": 30,  # seconds
                "max_questions": 100
            }
            session = service.create_session("Default Auto Session", "system", config)
            logger.info(f"✅ Created session: {session.get('session_id', session.get('id', 'unknown'))}")
        
        # Main service loop
        while True:
            try:
                sessions = service.get_sessions()
                
                for session in sessions:
                    session_id = session.get('session_id', session.get('id'))
                    
                    if not session_id:
                        logger.warning("⚠️ Session without ID found, skipping")
                        continue
                    
                    # Check if session has auto_generate enabled
                    if session.get('config', {}).get('auto_generate', False):
                        try:
                            questions = service.get_questions(session_id)
                            
                            # Generate question if needed
                            if len(questions) < session.get('config', {}).get('max_questions', 100):
                                logger.info(f"🔄 Generating question for session {session_id}")
                                question_text = service._generate_question(session_id)
                                question = service.add_question(session_id, question_text, origin="auto")
                                logger.info(f"✅ Generated question: {question.get('id', 'unknown')}")
                                
                                # Auto-generate response using real LLM
                                try:
                                    from apps.backend.src.core.config.settings import settings
                                    response_text, tokens_used = await service._call_real_llm(question_text)
                                    response = service.add_response(
                                        question["id"], session_id, settings.llm_model_id,
                                        question_text, response_text, tokens_used
                                    )
                                    logger.info(f"✅ Generated response: {response.get('id', 'unknown')}")
                                except Exception as e:
                                    logger.error(f"❌ Error generating response: {e}")
                                    
                        except Exception as e:
                            logger.error(f"❌ Error processing session {session_id}: {e}")
                
                # Wait before next cycle
                interval = 30  # Default interval
                logger.info(f"⏳ Waiting {interval} seconds before next cycle...")
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 HACK-MEMORI Service stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in service loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"❌ Fatal error in automatic service: {e}")
        raise


if __name__ == "__main__":
    import os
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if auto mode is enabled
    auto_mode = os.getenv('HACK_MEMORI_AUTO_MODE', 'false').lower() == 'true'
    
    if auto_mode:
        logger.info("🚀 Starting HACK-MEMORI in AUTO mode")
        try:
            asyncio.run(run_automatic_service())
        except KeyboardInterrupt:
            logger.info("✅ HACK-MEMORI Service shutdown complete")
    else:
        logger.info("📝 HACK-MEMORI Service class available for import")
        logger.info("💡 Set HACK_MEMORI_AUTO_MODE=true to run in automatic mode")
