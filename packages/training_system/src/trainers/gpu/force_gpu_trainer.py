#!/usr/bin/env python3
"""
🚀 ENTRENADOR FORZADO CON GPU - SHEILY SYSTEM
==============================================
Sistema de entrenamiento neuronal optimizado con detección automática de GPU
y fallback inteligente. Diseñado para entrenar pesos neuronales reales del
proyecto Sheily con máximo rendimiento.

Autor: Sistema Sheily
Versión: 2.0.0
Fecha: 2025-11-19
"""

import json
import logging
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split

warnings.filterwarnings("ignore")

# Configuración de logging sin emojis para Windows
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gpu_training.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class AdvancedSheilyDataset(Dataset):
    """Dataset avanzado para datos del proyecto Sheily"""

    def __init__(self, features: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.augment and torch.rand(1) > 0.5:
            # Aplicar augmentación suave
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            x = torch.clamp(x, 0, 1)

        return x, y


class GPUOptimizedNetwork(nn.Module):
    """Red neuronal optimizada para GPU con arquitectura avanzada"""

    def __init__(self, input_size: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()

        # Capas de codificación con normalización
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate // 2),
        )

        # Mecanismo de atención
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=dropout_rate, batch_first=True
        )

        # Capas de clasificación
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(128, num_classes),
        )

        # Inicialización de pesos
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicialización Xavier/He para optimización GPU"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)

        # Atención (reshape para multihead attention)
        attn_input = encoded.unsqueeze(1)  # [B, 1, 256]
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.squeeze(1)  # [B, 256]

        # Conexión residual
        features = encoded + attn_output

        # Clasificación
        output = self.classifier(features)
        return output


class ForceGPUTrainer:
    """Entrenador forzado con GPU y optimizaciones avanzadas"""

    def __init__(self, force_cpu: bool = False):
        self.device = self._setup_device(force_cpu)
        self.scaler = GradScaler() if self.device.type == "cuda" else None
        self.mixed_precision = self.device.type == "cuda"

        # Crear directorio para modelos
        self.model_dir = Path("force_gpu_models")
        self.model_dir.mkdir(exist_ok=True)

        logger.info(f"ROCKET Entrenador GPU forzado inicializado")
        logger.info(f"FIRE Dispositivo: {self.device}")
        logger.info(
            f"LIGHTNING Precision mixta: {'SI' if self.mixed_precision else 'NO'}"
        )

    def _setup_device(self, force_cpu: bool) -> torch.device:
        """Configuración inteligente de dispositivo"""
        if force_cpu:
            return torch.device("cpu")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"TARGET GPU detectada: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"CHART Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
            return device
        else:
            logger.warning("WARNING No se detectó GPU CUDA, usando CPU")
            return torch.device("cpu")

    def _load_project_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Carga datos del proyecto con validación"""
        try:
            with open("extracted_project_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"FOLDER Datos cargados: {len(data['files'])} archivos")

            # Generar características avanzadas
            features_list = []
            labels = []

            # Mapeo de tipos de archivo a etiquetas
            file_type_map = {
                ".py": 0,
                ".js": 1,
                ".ts": 2,
                ".html": 3,
                ".css": 4,
                ".json": 5,
                ".md": 6,
                ".yml": 7,
                ".yaml": 7,
                ".txt": 8,
                ".sh": 9,
                ".ps1": 10,
                "other": 11,
            }

            for file_data in data["files"]:
                # Características estructurales
                features = [
                    file_data.get("lines_of_code", 0) / 1000.0,  # Normalizado
                    file_data.get("complexity_score", 0) / 100.0,
                    file_data.get("function_count", 0) / 50.0,
                    file_data.get("class_count", 0) / 10.0,
                    len(file_data.get("imports", [])) / 20.0,
                ]

                # Características semánticas (basadas en patrones de código)
                content = file_data.get("content", "")
                semantic_features = [
                    content.count("def ") / 100.0,
                    content.count("class ") / 10.0,
                    content.count("import ") / 20.0,
                    content.count("async ") / 5.0,
                    content.count("await ") / 10.0,
                    len(content.split()) / 10000.0,  # Palabras totales
                ]

                # Características de complejidad avanzada
                complexity_features = [
                    content.count("if ") / 50.0,
                    content.count("for ") / 30.0,
                    content.count("while ") / 10.0,
                    content.count("try:") / 5.0,
                    content.count("except") / 5.0,
                ]

                # Extender con características aleatorias basadas en hash del contenido
                import hashlib

                hash_obj = hashlib.md5(content.encode())
                hash_int = int(hash_obj.hexdigest()[:8], 16)
                np.random.seed(hash_int % (2**31))
                random_features = np.random.random(
                    500
                ).tolist()  # 500 características adicionales

                # Combinar todas las características
                all_features = (
                    features + semantic_features + complexity_features + random_features
                )
                features_list.append(
                    all_features[:512]
                )  # Tomar exactamente 512 características

                # Determinar etiqueta
                file_ext = file_data.get("extension", "other")
                label = file_type_map.get(file_ext, 11)
                labels.append(label)

            features_array = np.array(features_list, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int64)

            # Normalización
            features_array = (features_array - features_array.mean()) / (
                features_array.std() + 1e-8
            )
            features_array = np.clip(features_array, -3, 3)  # Clip outliers

            logger.info(
                f"CHECK Dataset creado: {features_array.shape[0]} muestras, {features_array.shape[1]} características"
            )
            logger.info(f"CHART Distribucion de clases: {np.bincount(labels_array)}")

            return features_array, labels_array

        except Exception as e:
            logger.error(f"X Error cargando datos: {e}")
            raise

    def train(
        self, epochs: int = 50, batch_size: int = 64, learning_rate: float = 0.001
    ):
        """Entrenamiento principal con optimizaciones GPU"""

        logger.info("ROCKET INICIANDO ENTRENAMIENTO FORZADO CON GPU")
        logger.info("=" * 80)

        # Cargar datos
        features, labels = self._load_project_data()

        # Crear datasets
        dataset = AdvancedSheilyDataset(features, labels, augment=True)

        # División de datos
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Optimización dinámica de batch size para GPU
        if self.device.type == "cuda":
            # Usar batch size más grande para GPU
            batch_size = min(batch_size * 2, 128)

        # DataLoaders con optimizaciones
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 if self.device.type == "cuda" else 0,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.device.type == "cuda" and True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=4 if self.device.type == "cuda" else 0,
            pin_memory=self.device.type == "cuda",
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=4 if self.device.type == "cuda" else 0,
            pin_memory=self.device.type == "cuda",
        )

        logger.info(f"PACKAGE Batch size optimizado: {batch_size}")
        logger.info(
            f"CHART Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
        )

        # Crear modelo
        model = GPUOptimizedNetwork(
            input_size=features.shape[1],
            num_classes=len(np.unique(labels)),
            dropout_rate=0.2 if self.device.type == "cuda" else 0.3,
        ).to(self.device)

        # Información del modelo
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"BRAIN Modelo: GPUOptimizedNetwork")
        logger.info(f"NUMBERS Parámetros totales: {total_params:,}")
        logger.info(f"TARGET Parámetros entrenables: {trainable_params:,}")
        logger.info(f"DISK Tamaño estimado: {total_params * 4 / 1e6:.1f} MB")
        logger.info(f"CHART Clases: {len(np.unique(labels))}")
        logger.info(
            f"LIGHTNING Precisión mixta: {'SI' if self.mixed_precision else 'NO'}"
        )

        # Optimizador y scheduler
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999)
        )

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        # Función de pérdida con pesos de clase
        class_weights = torch.FloatTensor(1.0 / np.bincount(labels)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Variables de seguimiento
        best_val_acc = 0
        training_start = time.time()
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        # Loop de entrenamiento
        for epoch in range(epochs):
            epoch_start = time.time()

            # Entrenamiento
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()

                if self.mixed_precision:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                scheduler.step()

                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)

                # Progreso cada 20% del epoch
                if batch_idx % (len(train_loader) // 5) == 0 and batch_idx > 0:
                    progress = 100.0 * batch_idx / len(train_loader)
                    logger.info(
                        f"  Epoca {epoch+1}, Progreso: {progress:.1f}%, Perdida: {loss.item():.6f}"
                    )

            # Validación
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)

                    if self.mixed_precision:
                        with autocast():
                            output = model(data)
                            loss = criterion(output, target)
                    else:
                        output = model(data)
                        loss = criterion(output, target)

                    val_loss += loss.item()
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)

            # Métricas del epoch
            epoch_time = time.time() - epoch_start
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            current_lr = optimizer.param_groups[0]["lr"]

            train_losses.append(train_loss_avg)
            val_losses.append(val_loss_avg)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            logger.info(f"Epoca {epoch+1}/{epochs} ({epoch_time:.2f}s):")
            logger.info(
                f"  CHART Train - Loss: {train_loss_avg:.6f}, Acc: {train_acc:.2f}%"
            )
            logger.info(f"  BAR Val   - Loss: {val_loss_avg:.6f}, Acc: {val_acc:.2f}%")
            logger.info(f"  SLIDER LR: {current_lr:.2e}")

            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = self.model_dir / f"force_gpu_model_{timestamp}.pth"

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_acc,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss_avg,
                        "train_loss": train_loss_avg,
                    },
                    model_path,
                )

                logger.info(f"DISK Modelo guardado: {model_path}")
                logger.info(f"  DISK Mejor modelo guardado (Val Acc: {val_acc:.2f}%)")

        # Evaluación final en test
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                if self.mixed_precision:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)

                test_loss += loss.item()
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)

        test_acc = 100.0 * test_correct / test_total
        test_loss_avg = test_loss / len(test_loader)

        # Guardar reporte final
        total_time = time.time() - training_start
        time_formatted = str(timedelta(seconds=int(total_time)))

        report = {
            "training_completed": True,
            "device_used": str(self.device),
            "mixed_precision": self.mixed_precision,
            "total_epochs": epochs,
            "total_time_seconds": total_time,
            "total_time_formatted": time_formatted,
            "best_val_accuracy": best_val_acc,
            "final_test_accuracy": test_acc,
            "final_test_loss": test_loss_avg,
            "total_parameters": total_params,
            "dataset_size": len(dataset),
            "num_classes": len(np.unique(labels)),
            "training_metrics": {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accuracies": train_accs,
                "val_accuracies": val_accs,
            },
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.model_dir / f"force_gpu_report_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("PARTY ENTRENAMIENTO COMPLETADO!")
        logger.info(f"CLOCK Tiempo total: {time_formatted}")
        logger.info(f"TROPHY Mejor precision validacion: {best_val_acc:.2f}%")
        logger.info(f"LAB Precision test final: {test_acc:.2f}%")
        logger.info(f"CLIPBOARD Reporte guardado: {report_path}")

        return model, report


def main():
    """Función principal de entrenamiento"""
    print("=" * 80)
    print("TARGET ENTRENADOR FORZADO CON GPU - SHEILY SYSTEM")
    print("=" * 80)

    # Verificar disponibilidad de GPU
    if torch.cuda.is_available():
        print(f"FIRE GPU detectada: {torch.cuda.get_device_name(0)}")
        print(
            f"CHART Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        force_cpu = False
    else:
        print("WARNING GPU no disponible, entrenando en CPU")
        print("BULB Para obtener GPU: instala drivers NVIDIA y CUDA Toolkit")
        force_cpu = True

    # Crear entrenador
    trainer = ForceGPUTrainer(force_cpu=force_cpu)

    # Entrenar modelo
    model, report = trainer.train(epochs=50, batch_size=64, learning_rate=0.002)

    print("\n" + "=" * 80)
    print("TARGET ENTRENAMIENTO FORZADO COMPLETADO")
    print("=" * 80)
    print(f"FIRE Dispositivo utilizado: {report['device_used']}")
    print(f"LIGHTNING Precision mixta: {'SI' if report['mixed_precision'] else 'NO'}")
    print(f"BRAIN Parametros entrenados: {report['total_parameters']:,}")
    print(f"CLOCK Tiempo de entrenamiento: {report['total_time_formatted']}")
    print(f"TROPHY Mejor precision: {report['best_val_accuracy']:.2f}%")
    print(f"LAB Precision test: {report['final_test_accuracy']:.2f}%")
    print("")
    print("CHECK PESOS NEURONALES REALES ENTRENADOS EXITOSAMENTE")


if __name__ == "__main__":
    main()
