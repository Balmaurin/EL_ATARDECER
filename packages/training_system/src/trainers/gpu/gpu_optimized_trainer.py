#!/usr/bin/env python3
"""
ENTRENADOR DE PESOS GPU-OPTIMIZADO
==================================
Entrenamiento real de redes neuronales optimizado para GPU/CPU
con datos reales del proyecto Sheily
"""

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GPUNeuralTrainer")


class SheilyDataset(Dataset):
    """Dataset personalizado para datos del proyecto Sheily"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AdvancedSheilyNetwork(nn.Module):
    """Red neuronal avanzada para análisis del proyecto Sheily"""

    def __init__(
        self, input_size: int = 512, num_classes: int = 12, dropout: float = 0.3
    ):
        super().__init__()

        # Encoder con attention
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=dropout, batch_first=True
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Inicialización de pesos
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Inicialización avanzada de pesos"""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)

        # Self-attention (requiere dimensión [batch, seq, features])
        # Para datos tabulares, seq_len = 1
        encoded_seq = encoded.unsqueeze(1)  # [batch, 1, 256]

        attended, _ = self.attention(encoded_seq, encoded_seq, encoded_seq)
        attended = attended.squeeze(1)  # [batch, 256]

        # Residual connection
        attended = attended + encoded

        # Decoding
        output = self.decoder(attended)

        return output


class GPUOptimizedTrainer:
    """Entrenador optimizado para GPU con técnicas avanzadas"""

    def __init__(self):
        # Detectar dispositivo
        self.device = self._setup_device()

        # Configuración de entrenamiento
        self.use_mixed_precision = torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_mixed_precision else None

        # Directorios
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "extracted_project_data"
        self.models_dir = self.project_root / "gpu_trained_models"
        self.models_dir.mkdir(exist_ok=True)

        # Métricas
        self.training_metrics = {
            "epoch_times": [],
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "learning_rates": [],
        }

        logger.info("🚀 Entrenador GPU-Optimizado inicializado")

    def _setup_device(self):
        """Configurar dispositivo óptimo"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            logger.info(f"🔥 GPU detectada: {gpu_name}")
            logger.info(f"💾 VRAM: {memory_gb:.1f} GB")
            logger.info("✅ Entrenamiento con GPU habilitado")
            logger.info("⚡ Precisión mixta habilitada (FP16)")

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("🍎 Metal Performance Shaders (Apple) detectado")

        else:
            device = torch.device("cpu")
            logger.warning("⚠️ Solo CPU disponible - entrenamiento será más lento")

        return device

    def load_project_data(self) -> Dict:
        """Cargar datos del proyecto"""
        logger.info("📂 Cargando datos del proyecto Sheily...")

        analysis_files = list(self.data_dir.glob("project_analysis_*.json"))
        if not analysis_files:
            raise FileNotFoundError("❌ Ejecuta extract_project_files.py primero")

        latest_file = max(analysis_files, key=lambda f: f.stat().st_mtime)

        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"✅ Datos cargados: {data['totals']['files_processed']} archivos")
        return data

    def create_advanced_features(
        self, project_data: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crear características avanzadas para entrenamiento"""
        logger.info("🔧 Generando características avanzadas...")

        weights_data = project_data["weights_dataset"]
        neural_patterns = weights_data["neural_patterns"]
        complexity_weights = weights_data["complexity_weights"]

        features = []
        labels = []

        # Categorías expandidas
        categories = {
            "mcp_orchestrator": 0,
            "agent_system": 1,
            "security_crypto": 2,
            "rag_retrieval": 3,
            "blockchain_solana": 4,
            "education_learning": 5,
            "consciousness_ai": 6,
            "api_endpoint": 7,
            "data_processing": 8,
            "neural_network": 9,
            "federated_learning": 10,
            "quantum_computing": 11,
        }

        for pattern_name, weight_value in neural_patterns.items():
            # Características base
            feature_vector = [
                weight_value,  # Peso original
                abs(weight_value),  # Magnitud
                weight_value**2,  # Cuadrático
                math.log1p(abs(weight_value)),  # Logarítmico
                math.tanh(weight_value),  # Tanh normalizado
            ]

            # Análisis del nombre
            name_lower = pattern_name.lower()
            name_parts = name_lower.split("_")

            # One-hot encoding para palabras clave
            keywords = [
                "mcp",
                "agent",
                "neural",
                "api",
                "system",
                "manager",
                "core",
                "advanced",
                "security",
                "crypto",
                "rag",
                "retrieval",
                "blockchain",
                "education",
                "consciousness",
                "quantum",
                "federated",
                "learning",
            ]

            for keyword in keywords:
                feature_vector.append(
                    1.0 if any(keyword in part for part in name_parts) else 0.0
                )

            # Características morfológicas del nombre
            feature_vector.extend(
                [
                    len(pattern_name) / 100.0,  # Longitud normalizada
                    len(name_parts) / 20.0,  # Número de partes
                    sum(len(part) for part in name_parts)
                    / 100.0,  # Longitud total de partes
                    len([p for p in name_parts if len(p) > 3])
                    / 10.0,  # Partes significativas
                ]
            )

            # Complejidad del archivo asociado
            file_complexity = [0.5, 0.5, 0.5, 0.5]  # Valores por defecto

            for file_path, complexity in complexity_weights.items():
                if any(part in file_path.lower() for part in name_parts[:2]):
                    if isinstance(complexity, dict):
                        file_complexity = [
                            complexity.get("line_complexity", 0.5),
                            complexity.get("function_density", 0.5),
                            complexity.get("class_density", 0.5),
                            complexity.get("avg_function_complexity", 0.5),
                        ]
                    break

            feature_vector.extend(file_complexity)

            # Estadísticas posicionales
            pattern_hash = hash(pattern_name) % 1000
            feature_vector.extend(
                [
                    pattern_hash / 1000.0,  # Hash normalizado
                    (pattern_hash % 100) / 100.0,  # Características residuales
                    (pattern_hash % 10) / 10.0,
                ]
            )

            # Padding para llegar a 512 características
            while len(feature_vector) < 512:
                # Agregar ruido gaussiano pequeño para diversidad
                feature_vector.append(np.random.normal(0, 0.01))

            # Truncar a exactamente 512
            feature_vector = feature_vector[:512]

            # Clasificación mejorada
            label = self._classify_pattern_advanced(name_lower, categories)

            features.append(feature_vector)
            labels.append(label)

        # Convertir a numpy y normalizar
        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        # Normalización Z-score
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8  # Evitar división por cero
        X = (X - X_mean) / X_std

        logger.info(
            f"✅ Dataset creado: {X.shape[0]} muestras, {X.shape[1]} características"
        )
        logger.info(f"📊 Distribución de clases: {np.bincount(y)}")

        return X, y

    def _classify_pattern_advanced(self, pattern_name: str, categories: Dict) -> int:
        """Clasificación avanzada de patrones"""
        # Mapeo específico por palabras clave
        classification_rules = {
            "mcp_orchestrator": ["mcp", "orchestrator", "coordinator", "master"],
            "agent_system": ["agent", "multi_agent", "autonomous"],
            "security_crypto": ["security", "crypto", "auth", "guard", "safe"],
            "rag_retrieval": ["rag", "retrieval", "search", "index", "semantic"],
            "blockchain_solana": ["blockchain", "solana", "token", "wallet", "spl"],
            "education_learning": [
                "education",
                "learning",
                "student",
                "teacher",
                "course",
            ],
            "consciousness_ai": ["consciousness", "memory", "emotional", "human"],
            "api_endpoint": ["api", "endpoint", "server", "client", "http"],
            "data_processing": ["data", "processor", "cache", "database", "storage"],
            "neural_network": ["neural", "network", "model", "training", "brain"],
            "federated_learning": ["federated", "distributed", "fl", "coordination"],
            "quantum_computing": ["quantum", "multiverse", "temporal"],
        }

        # Buscar coincidencias exactas
        for category_name, keywords in classification_rules.items():
            if any(keyword in pattern_name for keyword in keywords):
                return categories[category_name]

        # Fallback: clasificación por heurísticas
        if any(word in pattern_name for word in ["system", "core", "main"]):
            return categories["agent_system"]
        elif any(word in pattern_name for word in ["advanced", "engine", "optimizer"]):
            return categories["neural_network"]
        else:
            return categories["data_processing"]  # Default

    def create_data_loaders(
        self, X: np.ndarray, y: np.ndarray, batch_size: int = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Crear data loaders optimizados"""
        if batch_size is None:
            # Batch size adaptativo según dispositivo
            if torch.cuda.is_available():
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                batch_size = min(128, max(16, int(memory_gb * 8)))  # Adaptativo
            else:
                batch_size = 32

        logger.info(f"📦 Batch size: {batch_size}")

        # Crear dataset
        dataset = SheilyDataset(X, y)

        # Dividir dataset
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Crear data loaders con optimizaciones
        num_workers = (
            min(4, torch.get_num_threads()) if torch.cuda.is_available() else 0
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

        logger.info(
            f"📊 Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
        )

        return train_loader, val_loader, test_loader

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
    ) -> Tuple[float, float]:
        """Entrenar una época con optimizaciones GPU"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Barra de progreso simple
        num_batches = len(train_loader)

        for batch_idx, (data, target) in enumerate(train_loader):
            # Mover a dispositivo
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            if self.use_mixed_precision and self.scaler:
                # Entrenamiento con precisión mixta
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Entrenamiento estándar
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Estadísticas
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Log cada 20% del entrenamiento
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                progress = 100.0 * (batch_idx + 1) / num_batches
                logger.info(
                    f"  Época {epoch}, Progreso: {progress:.1f}%, "
                    f"Pérdida: {loss.item():.6f}"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate_epoch(
        self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validar modelo"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                if self.use_mixed_precision:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train_model(self, epochs: int = 50, learning_rate: float = 0.001) -> Dict:
        """Ejecutar entrenamiento completo"""
        logger.info("🚀 INICIANDO ENTRENAMIENTO GPU-OPTIMIZADO")
        logger.info("=" * 80)

        start_time = time.time()

        # 1. Cargar y preparar datos
        project_data = self.load_project_data()
        X, y = self.create_advanced_features(project_data)
        train_loader, val_loader, test_loader = self.create_data_loaders(X, y)

        # 2. Crear modelo
        num_classes = len(np.unique(y))
        model = AdvancedSheilyNetwork(
            input_size=X.shape[1], num_classes=num_classes, dropout=0.3
        ).to(self.device)

        # 3. Configurar optimización
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )

        # Criterio con ponderación de clases
        class_counts = np.bincount(y)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights).to(self.device)
        )

        # Log info del modelo
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"🧠 Modelo: {model.__class__.__name__}")
        logger.info(f"🔢 Parámetros totales: {total_params:,}")
        logger.info(f"🎯 Parámetros entrenables: {trainable_params:,}")
        logger.info(f"💾 Tamaño estimado: {total_params * 4 / 1e6:.1f} MB")
        logger.info(f"📊 Clases: {num_classes}")
        logger.info(f"⚡ Precisión mixta: {'✅' if self.use_mixed_precision else '❌'}")

        # 4. Entrenamiento
        best_val_accuracy = 0.0
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            epoch_start = time.time()

            # Entrenar
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch + 1
            )

            # Validar
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)

            # Scheduler step
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            # Métricas
            epoch_time = time.time() - epoch_start
            self.training_metrics["epoch_times"].append(epoch_time)
            self.training_metrics["train_losses"].append(train_loss)
            self.training_metrics["train_accuracies"].append(train_acc)
            self.training_metrics["val_losses"].append(val_loss)
            self.training_metrics["val_accuracies"].append(val_acc)
            self.training_metrics["learning_rates"].append(current_lr)

            # Log progreso
            logger.info(f"Época {epoch+1}/{epochs} ({epoch_time:.2f}s):")
            logger.info(f"  📈 Train - Loss: {train_loss:.6f}, Acc: {train_acc:.2f}%")
            logger.info(f"  📊 Val   - Loss: {val_loss:.6f}, Acc: {val_acc:.2f}%")
            logger.info(f"  🎚️ LR: {current_lr:.2e}")

            # Early stopping y guardado del mejor modelo
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0

                # Guardar mejor modelo
                self.save_model(model, optimizer, epoch, val_acc, train_loss)
                logger.info(f"  💾 Mejor modelo guardado (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(
                    f"⏹️ Early stopping activado después de {patience} épocas sin mejora"
                )
                break

            # Log memoria GPU si está disponible
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_cached = torch.cuda.memory_reserved() / 1e9
                if epoch % 5 == 0:  # Cada 5 épocas
                    logger.info(
                        f"  🔥 GPU Mem: {memory_used:.1f}GB usada, {memory_cached:.1f}GB reservada"
                    )

        # 5. Evaluación final en test set
        test_loss, test_acc = self.validate_epoch(model, test_loader, criterion)

        total_time = time.time() - start_time

        # 6. Reporte final
        final_report = {
            "training_completed": True,
            "device_used": str(self.device),
            "mixed_precision": self.use_mixed_precision,
            "total_epochs": epoch + 1,
            "total_time_seconds": total_time,
            "total_time_formatted": f"{total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s",
            "best_val_accuracy": best_val_accuracy,
            "final_test_accuracy": test_acc,
            "final_test_loss": test_loss,
            "total_parameters": total_params,
            "dataset_size": len(X),
            "num_classes": num_classes,
            "training_metrics": self.training_metrics,
            "model_architecture": "AdvancedSheilyNetwork",
            "timestamp": datetime.now().isoformat(),
        }

        # Guardar reporte
        report_path = (
            self.models_dir
            / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        logger.info("🎉 ENTRENAMIENTO COMPLETADO!")
        logger.info(f"⏱️ Tiempo total: {final_report['total_time_formatted']}")
        logger.info(f"🏆 Mejor precisión validación: {best_val_accuracy:.2f}%")
        logger.info(f"🧪 Precisión test final: {test_acc:.2f}%")
        logger.info(f"📋 Reporte guardado: {report_path}")

        return final_report

    def save_model(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        accuracy: float,
        loss: float,
    ):
        """Guardar modelo entrenado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"sheily_gpu_model_{timestamp}.pth"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": accuracy,
            "loss": loss,
            "device": str(self.device),
            "model_class": model.__class__.__name__,
            "training_metrics": self.training_metrics,
            "timestamp": timestamp,
        }

        torch.save(checkpoint, model_path)
        logger.info(f"💾 Modelo guardado: {model_path}")


def main():
    """Función principal"""
    trainer = GPUOptimizedTrainer()

    try:
        report = trainer.train_model(epochs=30, learning_rate=0.001)

        print("\n" + "=" * 80)
        print("🎯 ENTRENAMIENTO GPU-OPTIMIZADO COMPLETADO")
        print("=" * 80)
        print(f"🔥 Dispositivo utilizado: {report['device_used']}")
        print(f"⚡ Precisión mixta: {'✅' if report['mixed_precision'] else '❌'}")
        print(f"🧠 Parámetros entrenados: {report['total_parameters']:,}")
        print(f"⏱️ Tiempo de entrenamiento: {report['total_time_formatted']}")
        print(f"🏆 Mejor precisión: {report['best_val_accuracy']:.2f}%")
        print(f"🧪 Precisión test: {report['final_test_accuracy']:.2f}%")
        print("\n✅ PESOS NEURONALES REALES ENTRENADOS EXITOSAMENTE")

    except Exception as e:
        logger.error(f"❌ Error durante entrenamiento: {e}")
        raise


if __name__ == "__main__":
    main()
