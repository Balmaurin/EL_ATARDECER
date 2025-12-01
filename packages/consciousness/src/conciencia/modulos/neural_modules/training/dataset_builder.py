"""
Dataset Builder - Constructor de Datasets
==========================================

Recolecta datos de Hack-Memori y construye datasets para entrenamiento
de módulos neurales.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Constructor de datasets para entrenamiento.
    
    Recolecta de Hack-Memori:
    - Estímulos del usuario (preguntas)
    - Estados emocionales (inferidos o labels)
    - Decisiones tomadas (respuestas generadas)
    - Aciertos/errores (feedback implícito/explícito)
    - Cambios en memoria (nuevas experiencias)
    """
    
    def __init__(self, hack_memori_data_path: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Inicializa el constructor de datasets.
        
        Args:
            hack_memori_data_path: Ruta a datos de Hack-Memori
            output_dir: Directorio de salida para datasets
        """
        if hack_memori_data_path:
            self.hack_memori_path = Path(hack_memori_data_path)
        else:
            # Buscar en ubicación por defecto
            base_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent
            self.hack_memori_path = base_dir / "data" / "hack_memori"
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            base_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent
            self.output_dir = base_dir / "data" / "consciousness" / "training_data"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DatasetBuilder initialized: output={self.output_dir}")
    
    def collect_from_hack_memori(self, session_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Recolecta datos de Hack-Memori.
        
        Args:
            session_ids: IDs de sesiones específicas (None = todas)
            
        Returns:
            Lista de datos recolectados
        """
        collected_data = []
        
        # Buscar archivos de sesiones
        if not self.hack_memori_path.exists():
            logger.warning(f"Hack-Memori data path not found: {self.hack_memori_path}")
            return collected_data
        
        # Buscar archivos JSON de sesiones
        session_files = list(self.hack_memori_path.glob("*.json"))
        
        for session_file in session_files:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # Si se especificaron session_ids, filtrar
                if session_ids and session_data.get("session_id") not in session_ids:
                    continue
                
                # Extraer datos relevantes
                questions = session_data.get("questions", [])
                responses = session_data.get("responses", [])
                
                for q_idx, question in enumerate(questions):
                    # Buscar respuesta correspondiente
                    response = None
                    for resp in responses:
                        if resp.get("question_id") == question.get("id"):
                            response = resp
                            break
                    
                    # Construir entrada de dataset
                    entry = {
                        "timestamp": question.get("timestamp", datetime.now().isoformat()),
                        "user_message": question.get("question_text", ""),
                        "response": response.get("response_text", "") if response else "",
                        "tokens_used": response.get("tokens_used", 0) if response else 0,
                        "model_id": response.get("model_id", "") if response else "",
                        "session_id": session_data.get("session_id", ""),
                        "question_id": question.get("id", ""),
                        "auto_generated": question.get("auto_generated", False)
                    }
                    
                    # Inferir estados emocionales (simplificado)
                    entry["inferred_emotion"] = self._infer_emotion(entry["user_message"])
                    entry["inferred_urgency"] = self._infer_urgency(entry["user_message"])
                    
                    collected_data.append(entry)
            
            except Exception as e:
                logger.error(f"Error reading session file {session_file}: {e}")
                continue
        
        logger.info(f"Collected {len(collected_data)} entries from Hack-Memori")
        return collected_data
    
    def _infer_emotion(self, text: str) -> float:
        """
        Infiere emoción del texto (simplificado).
        
        Args:
            text: Texto a analizar
            
        Returns:
            Score emocional (-1 a 1)
        """
        text_lower = text.lower()
        
        # Palabras positivas
        positive_words = ["gracias", "excelente", "bueno", "ayuda", "perfecto", "genial"]
        positive_count = sum(1 for w in positive_words if w in text_lower)
        
        # Palabras negativas
        negative_words = ["mal", "error", "problema", "no funciona", "incorrecto"]
        negative_count = sum(1 for w in negative_words if w in text_lower)
        
        # Calcular score
        if positive_count > negative_count:
            return min(1.0, 0.3 + positive_count * 0.2)
        elif negative_count > positive_count:
            return max(-1.0, -0.3 - negative_count * 0.2)
        else:
            return 0.0
    
    def _infer_urgency(self, text: str) -> float:
        """
        Infiere urgencia del texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Score de urgencia (0-1)
        """
        text_lower = text.lower()
        
        urgent_words = ["urgente", "rápido", "inmediato", "ahora", "ya", "rápidamente"]
        urgent_count = sum(1 for w in urgent_words if w in text_lower)
        
        return min(1.0, urgent_count * 0.3)
    
    def build_emotional_dataset(self, data: List[Dict[str, Any]], 
                                output_file: Optional[str] = None) -> str:
        """
        Construye dataset para entrenamiento de vmPFC.
        
        Args:
            data: Datos recolectados
            output_file: Archivo de salida (opcional)
            
        Returns:
            Ruta al archivo guardado
        """
        if output_file is None:
            output_file_path = self.output_dir / "emotional_dataset.jsonl"
        else:
            output_file_path = Path(output_file)
        
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = str(output_file_path)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                # Construir ejemplo para vmPFC
                example = {
                    "context": {
                        "user_message": entry["user_message"],
                        "previous_emotion": entry.get("inferred_emotion", 0.0),
                        "user_profile": {
                            "empathy_level": 0.7,
                            "openness": 0.5
                        },
                        "conversation_history": []
                    },
                    "target": {
                        "empathy_score": max(0.5, min(1.0, 0.7 + entry.get("inferred_emotion", 0.0) * 0.3)),
                        "emotional_bias": entry.get("inferred_emotion", 0.0),
                        "tone_modulation": 0.7 if entry.get("inferred_emotion", 0.0) > 0 else 0.5
                    }
                }
                
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        logger.info(f"Emotional dataset saved to {output_file_path}")
        return str(output_file_path)
    
    def build_decision_dataset(self, data: List[Dict[str, Any]], 
                               output_file: Optional[str] = None) -> str:
        """
        Construye dataset para entrenamiento de OFC.
        
        Args:
            data: Datos recolectados
            output_file: Archivo de salida (opcional)
            
        Returns:
            Ruta al archivo guardado
        """
        if output_file is None:
            output_file_path = self.output_dir / "decision_dataset.jsonl"
        else:
            output_file_path = Path(output_file)
        
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(str(output_file_path), 'w', encoding='utf-8') as f:
            for entry in data:
                # Construir ejemplo para OFC
                example = {
                    "options": [
                        {
                            "action": "respond_directly",
                            "expected_value": 0.7,
                            "confidence": 0.8
                        },
                        {
                            "action": "ask_clarification",
                            "expected_value": 0.5,
                            "confidence": 0.6
                        }
                    ],
                    "context": {
                        "urgency": entry.get("inferred_urgency", 0.0),
                        "importance": 0.7,
                        "emotional_context": entry.get("inferred_emotion", 0.0),
                        "cognitive_load": 0.5
                    },
                    "target": {
                        "selected_action": 0,  # Índice de acción seleccionada
                        "reward": 0.3  # Reward estimado
                    }
                }
                
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        logger.info(f"Decision dataset saved to {output_file_path}")
        return str(output_file_path)
    
    def build_memory_dataset(self, data: List[Dict[str, Any]], 
                            output_file: Optional[str] = None) -> str:
        """
        Construye dataset para entrenamiento de Hipocampo.
        
        Args:
            data: Datos recolectados
            output_file: Archivo de salida (opcional)
            
        Returns:
            Ruta al archivo guardado
        """
        if output_file is None:
            output_file_path = self.output_dir / "memory_dataset.jsonl"
        else:
            output_file_path = Path(output_file)
        
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(str(output_file_path), 'w', encoding='utf-8') as f:
            for entry in data:
                # Construir ejemplo para Hipocampo
                experience = {
                    "content": f"Q: {entry['user_message']}\nA: {entry['response']}",
                    "timestamp": entry["timestamp"],
                    "context": {
                        "session_id": entry.get("session_id", ""),
                        "emotion": entry.get("inferred_emotion", 0.0)
                    },
                    "relevance": 1.0
                }
                
                f.write(json.dumps(experience, ensure_ascii=False) + "\n")
        
        logger.info(f"Memory dataset saved to {output_file_path}")
        return str(output_file_path)

