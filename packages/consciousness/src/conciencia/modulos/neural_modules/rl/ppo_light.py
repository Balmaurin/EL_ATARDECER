"""
PPO Light - Proximal Policy Optimization CPU-Friendly
=====================================================

Implementación ligera de PPO optimizada para CPU.
Actor-Critic pequeño con updates incrementales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """
    Actor-Critic pequeño para PPO.
    
    Arquitectura:
    - Shared encoder
    - Actor head (policy)
    - Critic head (value)
    """
    
    def __init__(self, state_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        Inicializa Actor-Critic.
        
        Args:
            state_dim: Dimensión del estado
            action_dim: Dimensión de acciones
            hidden_dim: Tamaño de capas ocultas
        """
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: Estado [batch_size, state_dim]
            
        Returns:
            (action_probs, value)
        """
        encoded = self.encoder(state)
        action_probs = self.actor(encoded)
        value = self.critic(encoded)
        return action_probs, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Muestra una acción de la política.
        
        Args:
            state: Estado [state_dim]
            
        Returns:
            (action, log_prob, value)
        """
        state = state.unsqueeze(0)  # [1, state_dim]
        action_probs, value = self.forward(state)
        
        # Muestrear acción
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()


class PPOLight:
    """
    PPO ligero optimizado para CPU.
    
    Características:
    - Updates incrementales (no full rollouts)
    - Mini-batches pequeños (4-8)
    - Clipping conservador
    """
    
    def __init__(self, state_dim: int = 64, action_dim: int = 10, 
                 hidden_dim: int = 64, lr: float = 3e-4, 
                 gamma: float = 0.99, eps_clip: float = 0.2, device: str = "cpu"):
        """
        Inicializa PPO Light.
        
        Args:
            state_dim: Dimensión del estado
            action_dim: Dimensión de acciones
            hidden_dim: Tamaño de capas ocultas
            lr: Learning rate
            gamma: Discount factor
            eps_clip: Clipping epsilon
            device: Dispositivo
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        # Modelo
        self.model = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Buffer para updates incrementales
        self.buffer = deque(maxlen=100)  # Mantener últimas 100 transiciones
        
        logger.info(f"PPO Light initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Selecciona una acción dado un estado.
        
        Args:
            state: Estado como numpy array
            
        Returns:
            (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        return self.model.get_action(state_tensor)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool, log_prob: torch.Tensor, 
                        value: torch.Tensor):
        """
        Almacena una transición en el buffer.
        
        Args:
            state: Estado
            action: Acción tomada
            reward: Reward recibido
            next_state: Siguiente estado
            done: Si el episodio terminó
            log_prob: Log probabilidad de la acción
            value: Valor estimado
        """
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "log_prob": log_prob.detach(),
            "value": value.detach()
        }
        self.buffer.append(transition)
    
    def update(self, batch_size: int = 8, n_epochs: int = 4) -> Dict[str, float]:
        """
        Actualiza la política usando PPO.
        
        Args:
            batch_size: Tamaño del mini-batch
            n_epochs: Número de épocas de actualización
            
        Returns:
            Dict con métricas de entrenamiento
        """
        if len(self.buffer) < batch_size:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        
        # Calcular returns y advantages
        states = []
        actions = []
        old_log_probs = []
        returns = []
        advantages = []
        
        # Calcular returns (simplificado, sin GAE para CPU efficiency)
        episode_rewards = []
        for trans in self.buffer:
            episode_rewards.append(trans["reward"])
            if trans["done"] or len(episode_rewards) == len(self.buffer):
                # Calcular return
                G = 0
                for r in reversed(episode_rewards):
                    G = r + self.gamma * G
                    returns.append(G)
                episode_rewards = []
        
        # Si no hay episodios completos, usar rewards simples
        if len(returns) == 0:
            returns = [trans["reward"] for trans in self.buffer]
        
        # Normalizar returns
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Preparar datos
        for i, trans in enumerate(self.buffer):
            states.append(trans["state"])
            actions.append(trans["action"])
            old_log_probs.append(trans["log_prob"].cpu().item())
            if i < len(returns):
                advantages.append(returns[i] - trans["value"].cpu().item())
            else:
                advantages.append(trans["reward"] - trans["value"].cpu().item())
        
        # Normalizar advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convertir a tensores
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns[:len(states)]).to(self.device)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # Entrenar por n_epochs
        for epoch in range(n_epochs):
            # Mini-batches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Forward
                action_probs, values = self.model(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Policy loss (PPO clipped)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy  # Entropy bonus
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss
                
                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Grad clipping
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        # Limpiar buffer después de update
        self.buffer.clear()
        
        metrics = {
            "loss": (total_policy_loss + total_value_loss) / (n_epochs * (len(states) // batch_size + 1)),
            "policy_loss": total_policy_loss / (n_epochs * (len(states) // batch_size + 1)),
            "value_loss": total_value_loss / (n_epochs * (len(states) // batch_size + 1))
        }
        
        return metrics
    
    def save_model(self, path: str) -> bool:
        """Guarda el modelo."""
        try:
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, path)
            logger.info(f"PPO model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving PPO model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Carga el modelo."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"PPO model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading PPO model: {e}")
            return False
