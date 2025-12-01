"""
vmPFC Neural - Ventromedial Prefrontal Cortex Neural
====================================================

MLP pequeño para procesamiento de empatía y afecto.
Entrenado con datos de Hack-Memori para aprender respuestas emocionales apropiadas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VMPFCNeural(nn.Module):
    """
    MLP pequeño para empatía y procesamiento emocional.
    
    Arquitectura:
    - Input: Contexto conversacional, estado emocional previo, user profile
    - Hidden: 2-3 capas (64-128 unidades)
    - Output: Empathy score, emotional bias, tone modulation
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 3):
        """
        Inicializa la red vmPFC.
        
        Args:
            input_dim: Dimensión de entrada (embedding de contexto)
            hidden_dim: Tamaño de capas ocultas (128 para CPU)
            output_dim: Salidas (empathy_score, emotional_bias, tone_modulation)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # MLP: 3 capas
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Normalización
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
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
            [empathy_score, emotional_bias, tone_modulation]
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
        
        # Capa 3
        x = self.fc3(x)
        if x.dim() > 1:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Salida
        x = self.fc4(x)
        
        # Aplicar activaciones
        # Empathy score: sigmoid (0-1)
        # Emotional bias: tanh (-1 to 1)
        # Tone modulation: sigmoid (0-1)
        empathy = torch.sigmoid(x[:, 0:1])
        bias = torch.tanh(x[:, 1:2])
        tone = torch.sigmoid(x[:, 2:3])
        
        return torch.cat([empathy, bias, tone], dim=1)
    
    def encode_context(self, context: Dict[str, Any], embedding_dim: int = 128) -> np.ndarray:
        """
        Codifica el contexto conversacional en un vector.
        
        Args:
            context: Dict con información del contexto
                - user_message: str
                - previous_emotion: float
                - user_profile: dict
                - conversation_history: list
            embedding_dim: Dimensión del embedding
            
        Returns:
            Vector de embedding [embedding_dim]
        """
        # Features simples (en producción usar sentence transformer)
        features = []
        
        # Longitud del mensaje (normalizada)
        user_message = context.get("user_message", "")
        msg_length = len(user_message) / 500.0  # Normalizar a ~0-1
        features.append(min(1.0, msg_length))
        
        # Emoción previa
        prev_emotion = context.get("previous_emotion", 0.0)
        features.append((prev_emotion + 1.0) / 2.0)  # Normalizar a 0-1
        
        # User profile features
        user_profile = context.get("user_profile", {})
        empathy_level = user_profile.get("empathy_level", 0.7)
        features.append(empathy_level)
        
        openness = user_profile.get("openness", 0.5)
        features.append(openness)
        
        # Conversation history length
        history = context.get("conversation_history", [])
        history_length = len(history) / 20.0  # Normalizar
        features.append(min(1.0, history_length))
        
        # Sentiment del mensaje (simplificado)
        positive_words = ["gracias", "excelente", "bueno", "ayuda", "perfecto"]
        negative_words = ["mal", "error", "problema", "no funciona"]
        
        msg_lower = user_message.lower()
        positive_count = sum(1 for w in positive_words if w in msg_lower)
        negative_count = sum(1 for w in negative_words if w in msg_lower)
        
        sentiment = (positive_count - negative_count) / 5.0  # Normalizar
        features.append((sentiment + 1.0) / 2.0)  # 0-1
        
        # Padding o truncado a embedding_dim
        if len(features) < embedding_dim:
            # Repetir features o usar padding
            while len(features) < embedding_dim:
                features.extend(features[:min(len(features), embedding_dim - len(features))])
        else:
            features = features[:embedding_dim]
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Predice empatía, bias emocional y modulación de tono.
        
        Args:
            context: Contexto conversacional
            
        Returns:
            Dict con empathy_score, emotional_bias, tone_modulation
        """
        self.eval()
        
        with torch.no_grad():
            # Codificar contexto
            context_vec = self.encode_context(context, self.input_dim)
            
            # Convertir a tensor
            x = torch.FloatTensor(context_vec).unsqueeze(0)  # [1, input_dim]
            
            # Forward pass
            output = self.forward(x)
            output = output.squeeze(0).numpy()
            
            # Parsear salida
            result = {
                "empathy_score": float(output[0]),
                "emotional_bias": float(output[1]),
                "tone_modulation": float(output[2])
            }
            
            return result


class VMPFCNeuralSystem:
    """
    Sistema completo vmPFC Neural con carga/guardado de modelos.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Inicializa el sistema vmPFC Neural.
        
        Args:
            model_path: Ruta al modelo guardado (opcional)
            device: Dispositivo ("cpu" o "cuda")
        """
        self.device = torch.device(device)
        self.model = VMPFCNeural().to(self.device)
        self.model_path = model_path
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.info("vmPFC Neural initialized with random weights")
    
    def process_context(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Procesa el contexto y retorna empatía, bias y tono.
        
        Args:
            context: Contexto conversacional
            
        Returns:
            Dict con empathy_score, emotional_bias, tone_modulation
        """
        return self.model.predict(context)
    
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
            logger.info(f"vmPFC model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving vmPFC model: {e}")
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
            logger.info(f"vmPFC model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading vmPFC model: {e}")
            return False
    
    def train_step(self, batch_contexts: list, batch_targets: list, optimizer, criterion) -> float:
        """
        Un paso de entrenamiento.
        
        Args:
            batch_contexts: Lista de contextos
            batch_targets: Lista de targets (empathy, bias, tone)
            optimizer: Optimizador PyTorch
            criterion: Función de pérdida
            
        Returns:
            Loss promedio del batch
        """
        self.model.train()
        
        # Construir batch
        batch_inputs = []
        batch_outputs = []
        
        for context, target in zip(batch_contexts, batch_targets):
            input_vec = self.model.encode_context(context, self.model.input_dim)
            batch_inputs.append(input_vec)
            
            # Target: [empathy_score, emotional_bias, tone_modulation]
            target_vec = np.array([
                target.get("empathy_score", 0.7),
                target.get("emotional_bias", 0.0),
                target.get("tone_modulation", 0.5)
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
