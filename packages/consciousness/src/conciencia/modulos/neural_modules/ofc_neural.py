"""
OFC Neural - Orbitofrontal Cortex Neural
=========================================

Policy Network pequeño con PPO para toma de decisiones.
Aprende políticas de decisión mediante Reinforcement Learning.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import numpy as np

from .rl.ppo_light import PPOLight
from .rl.reward_system import RewardSystem

logger = logging.getLogger(__name__)


class OFCNeural:
    """
    Sistema completo OFC Neural con RL.
    
    Usa PPO para aprender políticas de decisión basadas en:
    - Opciones de acción disponibles
    - Contexto de la situación
    - Valores estimados
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Inicializa el sistema OFC Neural.
        
        Args:
            model_path: Ruta al modelo guardado
            device: Dispositivo
        """
        self.device = device
        
        # PPO para aprendizaje de políticas
        state_dim = 64  # Dimensión del estado (contexto + opciones)
        action_dim = 10  # Número máximo de acciones posibles
        
        self.ppo = PPOLight(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            lr=3e-4,
            gamma=0.99,
            eps_clip=0.2,
            device=device
        )
        
        # Reward system
        self.reward_system = RewardSystem()
        
        self.model_path = model_path
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.info("OFC Neural initialized with random policy")
    
    def encode_state(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> np.ndarray:
        """
        Codifica el estado (opciones + contexto) en un vector.
        
        Args:
            options: Lista de opciones de acción disponibles
            context: Contexto de la situación
            
        Returns:
            Vector de estado [state_dim]
        """
        state_dim = 64
        features = []
        
        # Features de opciones (primeras 5 opciones)
        for i, option in enumerate(options[:5]):
            value = option.get("expected_value", 0.5)
            confidence = option.get("confidence", 0.5)
            features.extend([value, confidence])
        
        # Padding si hay menos de 5 opciones
        while len(features) < 10:
            features.extend([0.0, 0.0])
        
        # Features de contexto
        urgency = context.get("urgency", 0.0)
        importance = context.get("importance", 0.5)
        emotional_context = context.get("emotional_context", 0.0)
        cognitive_load = context.get("cognitive_load", 0.5)
        
        features.extend([urgency, importance, emotional_context, cognitive_load])
        
        # Features adicionales (goal alignment, etc.)
        goal_alignment = context.get("goal_alignment", 0.5)
        resource_availability = context.get("resource_availability", 0.8)
        time_pressure = context.get("time_pressure", 0.0)
        
        features.extend([goal_alignment, resource_availability, time_pressure])
        
        # Padding o truncado a state_dim
        if len(features) < state_dim:
            # Repetir features o usar padding
            while len(features) < state_dim:
                features.extend(features[:min(len(features), state_dim - len(features))])
        else:
            features = features[:state_dim]
        
        return np.array(features, dtype=np.float32)
    
    def select_action(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Selecciona una acción usando la política aprendida.
        
        Args:
            options: Lista de opciones de acción
            context: Contexto de la situación
            
        Returns:
            (action_index, action_info)
        """
        if len(options) == 0:
            return -1, {"error": "No options available"}
        
        # Codificar estado
        state = self.encode_state(options, context)
        
        # Seleccionar acción usando PPO
        action_idx, log_prob, value = self.ppo.select_action(state)
        
        # Asegurar que el índice está en rango
        action_idx = min(action_idx, len(options) - 1)
        
        action_info = {
            "action_index": action_idx,
            "selected_option": options[action_idx],
            "log_prob": log_prob.item(),
            "value": value.item(),
            "all_options": options
        }
        
        return action_idx, action_info
    
    def update_policy(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool, log_prob: torch.Tensor, 
                      value: torch.Tensor):
        """
        Almacena una transición para actualización de política.
        
        Args:
            state: Estado
            action: Acción tomada
            reward: Reward recibido
            next_state: Siguiente estado
            done: Si terminó el episodio
            log_prob: Log probabilidad
            value: Valor estimado
        """
        self.ppo.store_transition(state, action, reward, next_state, done, log_prob, value)
    
    def train_step(self, batch_size: int = 8) -> Dict[str, float]:
        """
        Entrena la política con transiciones almacenadas.
        
        Args:
            batch_size: Tamaño del batch
            
        Returns:
            Métricas de entrenamiento
        """
        return self.ppo.update(batch_size=batch_size, n_epochs=4)
    
    def evaluate_response(self, response: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evalúa una respuesta y calcula rewards.
        
        Args:
            response: Respuesta generada
            context: Contexto
            
        Returns:
            Dict con rewards
        """
        return self.reward_system.calculate_reward(response, context)
    
    def save_model(self, path: str) -> bool:
        """Guarda el modelo."""
        return self.ppo.save_model(path)
    
    def load_model(self, path: str) -> bool:
        """Carga el modelo."""
        return self.ppo.load_model(path)
