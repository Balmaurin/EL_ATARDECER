"""
Brain State Manager - Estados Internos Persistentes
==================================================

Gestiona el estado cognitivo global persistente del sistema de consciencia.
Actualiza y mantiene brain_state.json con información de todos los módulos neurales.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import threading

logger = logging.getLogger(__name__)


class BrainStateManager:
    """
    Gestor de estados internos persistentes del cerebro artificial.
    
    Mantiene un archivo JSON con el estado cognitivo global que se actualiza
    tras cada interacción, proporcionando continuidad temporal al sistema.
    """
    
    def __init__(self, state_file: Optional[str] = None, auto_save_interval: int = 10):
        """
        Inicializa el gestor de estados.
        
        Args:
            state_file: Ruta al archivo brain_state.json (default: data/consciousness/brain_state.json)
            auto_save_interval: Guardar automáticamente cada N interacciones
        """
        if state_file is None:
            base_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent
            state_file = str(base_dir / "data" / "consciousness" / "brain_state.json")
        
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.auto_save_interval = auto_save_interval
        self.interaction_count = 0
        self._lock = threading.Lock()
        
        # Inicializar estado por defecto
        self.state = self._default_state()
        
        # Cargar estado existente si existe
        self.load_state()
        
        logger.info(f"BrainStateManager initialized: {self.state_file}")
    
    def _default_state(self) -> Dict[str, Any]:
        """Estado por defecto del cerebro."""
        return {
            "arousal": 0.5,
            "mood": "neutral",
            "empathy_bias": 0.7,
            "recent_memory_vector": [],
            "goals": ["ayudar", "aprender"],
            "confidence": 0.5,
            "energy_level": 0.8,
            "stress_level": 0.2,
            "social_need_satisfaction": 0.6,
            "curiosity_drive": 0.7,
            "homeostatic_balance": 0.85,
            "last_update": datetime.now().isoformat(),
            "creation_time": datetime.now().isoformat(),
            "total_interactions": 0,
            "recent_interactions": []
        }
    
    def load_state(self) -> Dict[str, Any]:
        """
        Carga el estado desde el archivo JSON.
        
        Returns:
            Estado cargado o estado por defecto si no existe
        """
        if not self.state_file.exists():
            logger.info(f"State file not found, using default state: {self.state_file}")
            return self.state
        
        try:
            with self._lock:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    
                # Merge con estado por defecto para asegurar todas las claves
                self.state = {**self._default_state(), **loaded}
                
                # Asegurar tipos correctos
                self.state["recent_memory_vector"] = loaded.get("recent_memory_vector", [])
                self.state["goals"] = loaded.get("goals", ["ayudar", "aprender"])
                self.state["recent_interactions"] = loaded.get("recent_interactions", [])
                
                logger.info(f"State loaded from {self.state_file}")
                return self.state
                
        except Exception as e:
            logger.error(f"Error loading state: {e}, using default")
            return self.state
    
    def save_state(self) -> bool:
        """
        Guarda el estado actual al archivo JSON.
        
        Returns:
            True si se guardó exitosamente
        """
        try:
            self.state["last_update"] = datetime.now().isoformat()
            
            with self._lock:
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    json.dump(self.state, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"State saved to {self.state_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False
    
    def update_state(self, updates: Dict[str, Any], auto_save: bool = True) -> None:
        """
        Actualiza el estado con nuevos valores.
        
        Args:
            updates: Diccionario con valores a actualizar
            auto_save: Si True, guarda automáticamente si pasó auto_save_interval
        """
        with self._lock:
            # Actualizar valores
            for key, value in updates.items():
                if key in self.state:
                    # Para valores numéricos, hacer transición suave
                    if isinstance(self.state[key], (int, float)) and isinstance(value, (int, float)):
                        # Smooth transition (evita saltos bruscos)
                        alpha = 0.3  # Factor de suavizado
                        self.state[key] = alpha * value + (1 - alpha) * self.state[key]
                    else:
                        self.state[key] = value
                else:
                    logger.warning(f"Unknown state key: {key}")
            
            self.state["last_update"] = datetime.now().isoformat()
            self.interaction_count += 1
            self.state["total_interactions"] = self.interaction_count
        
        # Auto-save si es necesario
        if auto_save and self.interaction_count % self.auto_save_interval == 0:
            self.save_state()
    
    def add_interaction(self, interaction_data: Dict[str, Any], max_recent: int = 100) -> None:
        """
        Añade una interacción reciente al historial.
        
        Args:
            interaction_data: Datos de la interacción
            max_recent: Máximo número de interacciones recientes a mantener
        """
        with self._lock:
            if "recent_interactions" not in self.state:
                self.state["recent_interactions"] = []
            
            interaction = {
                "timestamp": datetime.now().isoformat(),
                **interaction_data
            }
            
            self.state["recent_interactions"].append(interaction)
            
            # Mantener solo las últimas N
            if len(self.state["recent_interactions"]) > max_recent:
                self.state["recent_interactions"] = self.state["recent_interactions"][-max_recent:]
    
    def get_state(self) -> Dict[str, Any]:
        """Obtiene una copia del estado actual."""
        with self._lock:
            return self.state.copy()
    
    def get_arousal(self) -> float:
        """Obtiene el nivel de arousal actual."""
        return self.state.get("arousal", 0.5)
    
    def get_mood(self) -> str:
        """Obtiene el mood actual."""
        return self.state.get("mood", "neutral")
    
    def get_empathy_bias(self) -> float:
        """Obtiene el bias de empatía actual."""
        return self.state.get("empathy_bias", 0.7)
    
    def get_goals(self) -> List[str]:
        """Obtiene los objetivos actuales."""
        return self.state.get("goals", ["ayudar", "aprender"])
    
    def update_memory_vector(self, vector: List[float], max_length: int = 50) -> None:
        """
        Actualiza el vector de memoria reciente.
        
        Args:
            vector: Nuevo vector de memoria
            max_length: Longitud máxima del vector
        """
        with self._lock:
            if len(vector) > max_length:
                vector = vector[-max_length:]
            self.state["recent_memory_vector"] = vector
    
    def shutdown(self) -> None:
        """Guarda el estado al cerrar el sistema."""
        logger.info("Saving brain state on shutdown...")
        self.save_state()
