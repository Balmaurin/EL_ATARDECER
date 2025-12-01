"""
RAS Neural - Reticular Activating System Neural
================================================

Red neuronal pequeña para modulación de arousal global.
Optimizada para CPU con inference rápida.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RASNeural(nn.Module):
    """
    Red neuronal pequeña para calcular arousal y niveles de neurotransmisores.
    
    Arquitectura:
    - Input: Estímulos, estado previo, tiempo desde último evento
    - Hidden: 2 capas pequeñas (32-64 unidades)
    - Output: Arousal scalar + 5 niveles de neurotransmisores
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 6):
        """
        Inicializa la red RAS.
        
        Args:
            input_dim: Dimensión de entrada (estímulos + estado + tiempo)
            hidden_dim: Tamaño de capas ocultas (64 para CPU)
            output_dim: Salidas (arousal + 5 neurotransmisores)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Red pequeña: 2 capas ocultas
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Normalización para estabilidad
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Dropout ligero
        self.dropout = nn.Dropout(0.1)
        
        # Inicialización
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa pesos con Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, input_dim]
            
        Returns:
            Tensor de salida [batch_size, output_dim]
            [arousal, norepinephrine, serotonin, dopamine, acetylcholine, histamine]
        """
        # Capa 1
        x = self.fc1(x)
        if x.dim() > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Capa 2
        x = self.fc2(x)
        if x.dim() > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Salida
        x = self.fc3(x)
        
        # Aplicar activaciones apropiadas
        # Arousal: sigmoid (0-1)
        # Neurotransmisores: sigmoid (0-1)
        x = torch.sigmoid(x)
        
        return x
    
    def predict(self, stimulus: Dict[str, Any], previous_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Predice arousal y neurotransmisores dado un estímulo.
        
        Args:
            stimulus: Dict con información del estímulo
                - intensity: float (0-1)
                - urgency: float (0-1)
                - novelty: float (0-1)
                - emotional_valence: float (-1 to 1)
            previous_state: Estado previo (arousal, etc.)
            
        Returns:
            Dict con arousal y niveles de neurotransmisores
        """
        self.eval()
        
        with torch.no_grad():
            # Construir vector de entrada
            input_vector = self._build_input_vector(stimulus, previous_state)
            
            # Convertir a tensor
            x = torch.FloatTensor(input_vector).unsqueeze(0)  # [1, input_dim]
            
            # Forward pass
            output = self.forward(x)
            output = output.squeeze(0).numpy()
            
            # Parsear salida
            result = {
                "arousal": float(output[0]),
                "norepinephrine": float(output[1]),
                "serotonin": float(output[2]),
                "dopamine": float(output[3]),
                "acetylcholine": float(output[4]),
                "histamine": float(output[5])
            }
            
            return result
    
    def _build_input_vector(self, stimulus: Dict[str, Any], previous_state: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Construye el vector de entrada a partir del estímulo y estado previo.
        
        Args:
            stimulus: Información del estímulo
            previous_state: Estado previo
            
        Returns:
            Vector de entrada [input_dim]
        """
        # Features del estímulo
        intensity = stimulus.get("intensity", 0.5)
        urgency = stimulus.get("urgency", 0.0)
        novelty = stimulus.get("novelty", 0.0)
        emotional_valence = stimulus.get("emotional_valence", 0.0)
        
        # Normalizar valence a 0-1
        emotional_valence_norm = (emotional_valence + 1.0) / 2.0
        
        # Estado previo
        if previous_state:
            prev_arousal = previous_state.get("arousal", 0.5)
            prev_norepinephrine = previous_state.get("norepinephrine", 0.5)
            prev_serotonin = previous_state.get("serotonin", 0.6)
            prev_dopamine = previous_state.get("dopamine", 0.5)
            prev_acetylcholine = previous_state.get("acetylcholine", 0.7)
        else:
            prev_arousal = 0.5
            prev_norepinephrine = 0.5
            prev_serotonin = 0.6
            prev_dopamine = 0.5
            prev_acetylcholine = 0.7
        
        # Tiempo desde último evento (normalizado, asumimos 0.5 si no disponible)
        time_since_last = stimulus.get("time_since_last_event", 0.5)
        
        # Construir vector
        input_vector = np.array([
            intensity,
            urgency,
            novelty,
            emotional_valence_norm,
            prev_arousal,
            prev_norepinephrine,
            prev_serotonin,
            prev_dopamine,
            prev_acetylcholine,
            time_since_last
        ], dtype=np.float32)
        
        return input_vector


class RASNeuralSystem:
    """
    Sistema completo RAS Neural con carga/guardado de modelos.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Inicializa el sistema RAS Neural.
        
        Args:
            model_path: Ruta al modelo guardado (opcional)
            device: Dispositivo ("cpu" o "cuda")
        """
        self.device = torch.device(device)
        self.model = RASNeural().to(self.device)
        self.model_path = model_path
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.info("RAS Neural initialized with random weights")
    
    def process_stimulus(self, stimulus: Dict[str, Any], previous_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Procesa un estímulo y retorna arousal y neurotransmisores.
        
        Args:
            stimulus: Información del estímulo
            previous_state: Estado previo del sistema
            
        Returns:
            Dict con arousal y niveles de neurotransmisores
        """
        return self.model.predict(stimulus, previous_state)
    
    def save_model(self, path: str) -> bool:
        """
        Guarda el modelo entrenado.
        
        Args:
            path: Ruta donde guardar
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim
            }, path)
            logger.info(f"RAS model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving RAS model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Carga un modelo entrenado.
        
        Args:
            path: Ruta al modelo
            
        Returns:
            True si se cargó exitosamente
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info(f"RAS model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading RAS model: {e}")
            return False
    
    def train_step(self, batch_stimuli: list, batch_targets: list, optimizer, criterion) -> float:
        """
        Un paso de entrenamiento.
        
        Args:
            batch_stimuli: Lista de estímulos
            batch_targets: Lista de targets (arousal + neurotransmisores)
            optimizer: Optimizador PyTorch
            criterion: Función de pérdida
            
        Returns:
            Loss promedio del batch
        """
        self.model.train()
        
        # Construir batch
        batch_inputs = []
        batch_outputs = []
        
        for stimulus, target in zip(batch_stimuli, batch_targets):
            input_vec = self.model._build_input_vector(stimulus, None)
            batch_inputs.append(input_vec)
            
            # Target: [arousal, norepinephrine, serotonin, dopamine, acetylcholine, histamine]
            target_vec = np.array([
                target.get("arousal", 0.5),
                target.get("norepinephrine", 0.5),
                target.get("serotonin", 0.6),
                target.get("dopamine", 0.5),
                target.get("acetylcholine", 0.7),
                target.get("histamine", 0.4)
            ], dtype=np.float32)
            batch_outputs.append(target_vec)
        
        # Convertir a tensores
        inputs = torch.FloatTensor(batch_inputs).to(self.device)
        targets = torch.FloatTensor(batch_outputs).to(self.device)
        
        # Forward
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        return loss.item()
