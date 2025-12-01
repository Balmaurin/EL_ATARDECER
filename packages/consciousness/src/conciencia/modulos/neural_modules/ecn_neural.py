"""
ECN Neural - Executive Control Network Neural
=============================================

Mixture of Experts (MoE) ligero para control ejecutivo.
Gating network que selecciona expertos según la tarea.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Expert(nn.Module):
    """
    Experto individual en el MoE.
    Cada experto se especializa en un aspecto del control ejecutivo.
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, output_dim: int = 32):
        """
        Inicializa un experto.
        
        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Tamaño de capa oculta
            output_dim: Dimensión de salida
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del experto."""
        return self.network(x)


class MoEGating(nn.Module):
    """
    Red de gating para MoE.
    Decide qué expertos activar y con qué peso.
    """
    
    def __init__(self, input_dim: int = 64, num_experts: int = 4, hidden_dim: int = 32):
        """
        Inicializa la red de gating.
        
        Args:
            input_dim: Dimensión de entrada
            num_experts: Número de expertos
            hidden_dim: Tamaño de capa oculta
        """
        super().__init__()
        
        self.num_experts = num_experts
        
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula pesos de gating.
        
        Args:
            x: Entrada [batch_size, input_dim]
            
        Returns:
            Pesos de gating [batch_size, num_experts]
        """
        return self.gating_network(x)


class ECNMoE(nn.Module):
    """
    Mixture of Experts para control ejecutivo.
    
    Expertos:
    - Planning: Planificación de tareas
    - Inhibition: Control inhibitorio
    - Attention: Control de atención
    - MetaControl: Meta-control y ajuste de políticas
    """
    
    def __init__(self, input_dim: int = 64, num_experts: int = 4, 
                 expert_hidden: int = 32, expert_output: int = 32):
        """
        Inicializa el MoE.
        
        Args:
            input_dim: Dimensión de entrada
            num_experts: Número de expertos (4)
            expert_hidden: Tamaño de capas ocultas de expertos
            expert_output: Dimensión de salida de expertos
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.expert_output = expert_output
        
        # Gating network
        self.gating = MoEGating(input_dim, num_experts, hidden_dim=32)
        
        # Expertos
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden, expert_output)
            for _ in range(num_experts)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(expert_output, expert_output)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Entrada [batch_size, input_dim]
            
        Returns:
            (output, gating_weights)
        """
        # Calcular pesos de gating
        gating_weights = self.gating(x)  # [batch_size, num_experts]
        
        # Ejecutar expertos
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [batch_size, expert_output]
            expert_outputs.append(expert_out)
        
        # Stack expertos
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, expert_output]
        
        # Weighted combination
        gating_weights_expanded = gating_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        output = (expert_outputs * gating_weights_expanded).sum(dim=1)  # [batch_size, expert_output]
        
        # Output projection
        output = self.output_proj(output)
        
        return output, gating_weights


class ECNNeural:
    """
    Sistema completo ECN Neural con MoE.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Inicializa el sistema ECN Neural.
        
        Args:
            model_path: Ruta al modelo guardado
            device: Dispositivo
        """
        self.device = torch.device(device)
        self.model = ECNMoE().to(self.device)
        self.model_path = model_path
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.info("ECN Neural initialized with random weights")
    
    def encode_task(self, task: Dict[str, Any], wm_state: Dict[str, Any], 
                   cognitive_load: float) -> np.ndarray:
        """
        Codifica la tarea y estado en un vector.
        
        Args:
            task: Información de la tarea
            wm_state: Estado de working memory
            cognitive_load: Carga cognitiva actual
            
        Returns:
            Vector de entrada [input_dim]
        """
        input_dim = 64
        features = []
        
        # Features de tarea
        task_type = task.get("type", "general")
        task_priority = task.get("priority", 0.5)
        task_complexity = task.get("complexity", 0.5)
        task_urgency = task.get("urgency", 0.0)
        
        # Codificar tipo de tarea (one-hot simplificado)
        task_types = ["general", "planning", "decision", "inhibition", "attention"]
        task_type_vec = [0.0] * len(task_types)
        if task_type in task_types:
            task_type_vec[task_types.index(task_type)] = 1.0
        
        features.extend([task_priority, task_complexity, task_urgency])
        features.extend(task_type_vec)
        
        # Features de working memory
        wm_items_count = len(wm_state.get("items", []))
        wm_load = wm_state.get("load", 0.0)
        wm_capacity = wm_state.get("capacity", 7.0)
        wm_utilization = wm_items_count / max(wm_capacity, 1.0)
        
        features.extend([wm_load, wm_utilization])
        
        # Cognitive load
        features.append(cognitive_load)
        
        # Features adicionales
        has_conflict = task.get("conflict", False)
        is_novel = task.get("novel", False)
        requires_planning = task.get("requires_planning", False)
        
        features.extend([float(has_conflict), float(is_novel), float(requires_planning)])
        
        # Padding o truncado
        if len(features) < input_dim:
            while len(features) < input_dim:
                features.extend(features[:min(len(features), input_dim - len(features))])
        else:
            features = features[:input_dim]
        
        return np.array(features, dtype=np.float32)
    
    def process_task(self, task: Dict[str, Any], wm_state: Dict[str, Any], 
                    cognitive_load: float) -> Dict[str, Any]:
        """
        Procesa una tarea y retorna control mode y gating weights.
        
        Args:
            task: Información de la tarea
            wm_state: Estado de working memory
            cognitive_load: Carga cognitiva
            
        Returns:
            Dict con control_mode, gating_weights, expert_outputs
        """
        self.model.eval()
        
        with torch.no_grad():
            # Codificar entrada
            input_vec = self.encode_task(task, wm_state, cognitive_load)
            x = torch.FloatTensor(input_vec).unsqueeze(0).to(self.device)  # [1, input_dim]
            
            # Forward pass
            output, gating_weights = self.model(x)
            output = output.squeeze(0).cpu().numpy()
            gating_weights = gating_weights.squeeze(0).cpu().numpy()
            
            # Determinar control mode basado en gating weights
            # Si planning o metacontrol dominan → controlled mode
            planning_weight = gating_weights[0]  # Asumiendo experto 0 es planning
            metacontrol_weight = gating_weights[3]  # Asumiendo experto 3 es metacontrol
            
            if planning_weight + metacontrol_weight > 0.5:
                control_mode = "controlled"
            else:
                control_mode = "automatic"
            
            result = {
                "control_mode": control_mode,
                "gating_weights": {
                    "planning": float(gating_weights[0]),
                    "inhibition": float(gating_weights[1]),
                    "attention": float(gating_weights[2]),
                    "metacontrol": float(gating_weights[3])
                },
                "expert_output": output.tolist(),
                "cognitive_load": cognitive_load
            }
            
            return result
    
    def save_model(self, path: str) -> bool:
        """Guarda el modelo."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict()
            }, path)
            logger.info(f"ECN model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving ECN model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Carga el modelo."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info(f"ECN model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading ECN model: {e}")
            return False
