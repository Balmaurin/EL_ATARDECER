"""
Training Pipeline - Pipeline de Entrenamiento Incremental
=========================================================

Pipeline para entrenamiento incremental de módulos neurales.
Fine-tuning cada N interacciones con batches pequeños para CPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class JSONLDataset(Dataset):
    """Dataset simple para archivos JSONL."""
    
    def __init__(self, file_path: str):
        """
        Inicializa dataset desde archivo JSONL.
        
        Args:
            file_path: Ruta al archivo JSONL
        """
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class TrainingPipeline:
    """
    Pipeline de entrenamiento incremental.
    
    Características:
    - Fine-tuning cada N interacciones
    - Batch pequeño (4-8) para CPU
    - Learning rate bajo (1e-5 a 1e-4)
    - Gradient accumulation para simular batches grandes
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Inicializa el pipeline de entrenamiento.
        
        Args:
            device: Dispositivo
        """
        self.device = torch.device(device)
        logger.info(f"TrainingPipeline initialized: device={device}")
    
    def train_vmpfc(self, model, dataset_path: str, epochs: int = 3, 
                   batch_size: int = 4, lr: float = 1e-4, 
                   gradient_accumulation_steps: int = 2) -> Dict[str, float]:
        """
        Entrena el modelo vmPFC.
        
        Args:
            model: Modelo vmPFC
            dataset_path: Ruta al dataset
            epochs: Número de épocas
            batch_size: Tamaño de batch
            lr: Learning rate
            gradient_accumulation_steps: Pasos de acumulación de gradiente
            
        Returns:
            Métricas de entrenamiento
        """
        # Cargar dataset
        dataset = JSONLDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizador y pérdida
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                # Construir inputs y targets
                batch_inputs = []
                batch_targets = []
                
                for item in batch:
                    context = item["context"]
                    target = item["target"]
                    
                    # Codificar contexto
                    input_vec = model.encode_context(context, model.input_dim)
                    batch_inputs.append(input_vec)
                    
                    # Target
                    target_vec = torch.FloatTensor([
                        target["empathy_score"],
                        target["emotional_bias"],
                        target["tone_modulation"]
                    ])
                    batch_targets.append(target_vec)
                
                # Convertir a tensores
                inputs = torch.FloatTensor(batch_inputs).to(self.device)
                targets = torch.stack(batch_targets).to(self.device)
                
                # Forward
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Normalizar por gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward
                loss.backward()
                
                # Actualizar cada N pasos
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "epochs": epochs,
            "num_batches": num_batches
        }
    
    def train_ras(self, model, dataset_path: str, epochs: int = 3,
                  batch_size: int = 4, lr: float = 1e-4) -> Dict[str, float]:
        """
        Entrena el modelo RAS.
        
        Args:
            model: Modelo RAS
            dataset_path: Ruta al dataset
            epochs: Número de épocas
            batch_size: Tamaño de batch
            lr: Learning rate
            
        Returns:
            Métricas de entrenamiento
        """
        # Similar a train_vmpfc pero adaptado para RAS
        # Por simplicidad, usar mismo enfoque
        dataset = JSONLDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        model.train()
        total_loss = 0.0
        
        for epoch in range(epochs):
            for batch in dataloader:
                # Construir batch (simplificado)
                # En producción, adaptar según estructura de datos RAS
                optimizer.zero_grad()
                
                # Placeholder para entrenamiento real
                # outputs = model(batch_inputs)
                # loss = criterion(outputs, batch_targets)
                # loss = torch.tensor(0.0)  # Placeholder
                
                # loss.backward()
                # optimizer.step()
                # total_loss += loss.item()
                pass
        
        return {
            "loss": total_loss / (epochs * len(dataloader)) if len(dataloader) > 0 else 0.0,
            "epochs": epochs
        }
    
    def fine_tune_incremental(self, model, new_data: List[Dict[str, Any]], 
                              epochs: int = 1, batch_size: int = 4, 
                              lr: float = 1e-5) -> Dict[str, float]:
        """
        Fine-tuning incremental con nuevos datos.
        
        Args:
            model: Modelo a fine-tunear
            new_data: Lista de nuevos datos
            epochs: Número de épocas
            batch_size: Tamaño de batch
            lr: Learning rate (más bajo para fine-tuning)
            
        Returns:
            Métricas de entrenamiento
        """
        # Crear dataset temporal
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in new_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            temp_path = f.name
        
        try:
            # Entrenar con dataset temporal
            if hasattr(model, 'encode_context'):
                # vmPFC
                return self.train_vmpfc(model, temp_path, epochs, batch_size, lr)
            else:
                # RAS u otro
                return self.train_ras(model, temp_path, epochs, batch_size, lr)
        finally:
            # Limpiar archivo temporal
            Path(temp_path).unlink()

