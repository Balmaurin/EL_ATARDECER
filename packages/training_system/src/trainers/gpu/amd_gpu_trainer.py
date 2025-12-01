#!/usr/bin/env python3
"""
🚀 ENTRENADOR AMD GPU OPTIMIZADO - SHEILY SYSTEM
================================================
Sistema de entrenamiento neuronal optimizado para GPU AMD Radeon con ROCm.
Detecta automáticamente hardware disponible y optimiza para máximo rendimiento.

Autor: Sistema Sheily
Versión: 3.0.0
Fecha: 2025-11-19
"""

import json
import logging
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

warnings.filterwarnings("ignore")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("amd_gpu_training.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class OptimizedSheilyDataset(Dataset):
    """Dataset optimizado para GPU AMD"""

    def __init__(self, features: np.ndarray, labels: np.ndarray, device: torch.device):
        # Pre-cargar datos en GPU si está disponible
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

        if device.type != "cpu":
            self.features = self.features.pin_memory()
            self.labels = self.labels.pin_memory()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AdvancedAMDNetwork(nn.Module):
    """Red neuronal optimizada para GPU AMD"""

    def __init__(self, input_size: int, num_classes: int, device_type: str = "cpu"):
        super().__init__()

        # Ajustar arquitectura según el dispositivo
        if device_type != "cpu":
            # Configuración optimizada para GPU
            hidden_sizes = [1024, 512, 256, 128]
            dropout_rate = 0.2
        else:
            # Configuración para CPU
            hidden_sizes = [512, 256, 128]
            dropout_rate = 0.3

        layers = []

        # Primera capa
        layers.extend(
            [
                nn.Linear(input_size, hidden_sizes[0]),
                nn.BatchNorm1d(hidden_sizes[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
        )

        # Capas ocultas
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.BatchNorm1d(hidden_sizes[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate * 0.8),  # Reducir dropout gradualmente
                ]
            )

        # Capa de salida
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.network = nn.Sequential(*layers)

        # Inicialización optimizada para AMD GPU
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicialización optimizada de pesos"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


class AMDGPUTrainer:
    """Entrenador optimizado para GPU AMD"""

    def __init__(self):
        self.device = self._detect_best_device()
        self.model_dir = Path("amd_gpu_models")
        self.model_dir.mkdir(exist_ok=True)

        logger.info("ROCKET Entrenador AMD GPU iniciado")
        logger.info(f"DEVICE Dispositivo: {self.device}")

        # Configuraciones específicas para GPU AMD
        self._configure_amd_optimizations()

    def _detect_best_device(self) -> torch.device:
        """Detecta el mejor dispositivo disponible"""

        # Verificar ROCm/HIP para GPU AMD
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU GPU detectada: {gpu_name}")

            # Verificar memoria GPU
            if hasattr(torch.cuda, "get_device_properties"):
                props = torch.cuda.get_device_properties(0)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"MEMORY Memoria GPU: {memory_gb:.1f} GB")

            return device
        else:
            logger.warning("WARNING No se detectó GPU compatible, usando CPU")
            logger.info(
                "TIP Instala ROCm para usar GPU AMD: https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html"
            )
            return torch.device("cpu")

    def _configure_amd_optimizations(self):
        """Configuraciones específicas para optimizar rendimiento AMD"""

        # Habilitar optimizaciones de PyTorch
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Configuraciones para mejor rendimiento en AMD
        if self.device.type != "cpu":
            # Configurar memoria para mejor rendimiento
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        logger.info("SETTINGS Optimizaciones AMD configuradas")

    def _generate_enhanced_data(self):
        """Genera datos sintéticos mejorados basados en el proyecto"""

        logger.info("GEAR Generando datos sintéticos avanzados...")

        # Parámetros del dataset
        n_samples = 8000
        n_features = 512
        n_classes = 12

        # Configurar semillas para reproducibilidad
        np.random.seed(42)
        torch.manual_seed(42)

        # Generar datos más realistas
        features_list = []
        labels = []

        # Crear patrones más complejos basados en tipos de archivos
        for class_id in range(n_classes):
            samples_per_class = n_samples // n_classes

            for _ in range(samples_per_class):
                # Crear características base
                base_features = np.random.randn(n_features).astype(np.float32)

                # Añadir patrones específicos por clase
                if class_id < 3:  # Lenguajes de programación
                    base_features[:100] *= 1.5  # Enfatizar características de código
                elif class_id < 6:  # Archivos de configuración
                    base_features[100:200] *= 1.2  # Patrones de configuración
                else:  # Otros tipos
                    base_features[200:300] *= 0.8

                # Añadir ruido controlado
                noise = np.random.normal(0, 0.1, n_features).astype(np.float32)
                features = base_features + noise

                features_list.append(features)
                labels.append(class_id)

        # Mezclar datos
        combined = list(zip(features_list, labels))
        np.random.shuffle(combined)
        features_list, labels = zip(*combined)

        features_array = np.array(features_list, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int64)

        # Normalización avanzada
        features_array = (features_array - features_array.mean(axis=0)) / (
            features_array.std(axis=0) + 1e-8
        )
        features_array = np.clip(features_array, -3, 3)

        logger.info(
            f"CHECK Dataset generado: {features_array.shape[0]} muestras, {features_array.shape[1]} características"
        )
        logger.info(f"CHART Distribución equilibrada: {np.bincount(labels_array)}")

        return features_array, labels_array

    def train(
        self, epochs: int = 50, batch_size: int = 128, learning_rate: float = 0.002
    ):
        """Entrenamiento principal optimizado"""

        logger.info("ROCKET INICIANDO ENTRENAMIENTO AMD GPU OPTIMIZADO")
        logger.info("=" * 70)

        # Generar datos
        features, labels = self._generate_enhanced_data()

        # Crear dataset optimizado
        dataset = OptimizedSheilyDataset(features, labels, self.device)

        # División de datos
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Optimizar batch size según dispositivo
        if self.device.type != "cpu":
            batch_size = min(batch_size * 2, 256)  # Aumentar para GPU
            num_workers = 4
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        # DataLoaders optimizados
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        logger.info(f"PACKAGE Batch size optimizado: {batch_size}")
        logger.info(
            f"CHART Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
        )

        # Crear modelo optimizado
        model = AdvancedAMDNetwork(
            input_size=features.shape[1],
            num_classes=len(np.unique(labels)),
            device_type=self.device.type,
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"BRAIN Modelo: AdvancedAMDNetwork")
        logger.info(f"NUMBERS Parámetros totales: {total_params:,}")
        logger.info(f"TARGET Parámetros entrenables: {trainable_params:,}")
        logger.info(f"DISK Tamaño estimado: {total_params * 4 / 1e6:.1f} MB")

        # Optimizador avanzado
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Scheduler avanzado
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        # Función de pérdida con pesos de clase
        class_counts = np.bincount(labels)
        class_weights = len(labels) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # Variables de seguimiento
        best_val_acc = 0
        training_start = time.time()
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        logger.info(f"FIRE Optimizador: AdamW con OneCycleLR")
        logger.info(f"BALANCE Pesos de clase aplicados")

        # Loop de entrenamiento
        for epoch in range(epochs):
            epoch_start = time.time()

            # Entrenamiento
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device, non_blocking=True), target.to(
                    self.device, non_blocking=True
                )

                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)

                # Progreso cada 20%
                if batch_idx % max(1, len(train_loader) // 5) == 0 and batch_idx > 0:
                    progress = 100.0 * batch_idx / len(train_loader)
                    current_lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"  Epoca {epoch+1}, Progreso: {progress:.1f}%, Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                    )

            # Validación
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device, non_blocking=True), target.to(
                        self.device, non_blocking=True
                    )
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
                f"  CHART Train - Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}%"
            )
            logger.info(
                f"  BAR   Val   - Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%"
            )
            logger.info(f"  SLIDER LR: {current_lr:.2e}")

            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = self.model_dir / f"amd_gpu_model_{timestamp}.pth"

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_accuracy": val_acc,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss_avg,
                        "train_loss": train_loss_avg,
                        "device": str(self.device),
                        "total_params": total_params,
                    },
                    model_path,
                )

                logger.info(
                    f"DISK Mejor modelo guardado: {model_path} (Val Acc: {val_acc:.2f}%)"
                )

        # Evaluación final en test
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(
                    self.device, non_blocking=True
                )
                output = model(data)
                loss = criterion(output, target)

                test_loss += loss.item()
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)

        test_acc = 100.0 * test_correct / test_total
        test_loss_avg = test_loss / len(test_loader)
        total_time = time.time() - training_start

        # Reporte final
        report = {
            "training_completed": True,
            "device_used": str(self.device),
            "total_epochs": epochs,
            "total_time_seconds": total_time,
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
            "hardware_info": {
                "device_type": self.device.type,
                "batch_size_used": batch_size,
                "num_workers": num_workers,
            },
        }

        # Guardar reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.model_dir / f"amd_training_report_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("PARTY ENTRENAMIENTO AMD COMPLETADO!")
        logger.info(f"CLOCK Tiempo total: {timedelta(seconds=int(total_time))}")
        logger.info(f"TROPHY Mejor precisión validación: {best_val_acc:.2f}%")
        logger.info(f"LAB Precisión test final: {test_acc:.2f}%")
        logger.info(f"CLIPBOARD Reporte guardado: {report_path}")

        return model, report


def main():
    """Función principal optimizada"""
    print("=" * 80)
    print("TARGET ENTRENADOR AMD GPU OPTIMIZADO - SHEILY SYSTEM")
    print("=" * 80)

    # Información del sistema
    print(f"PYTHON PyTorch versión: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"GPU GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Versión CUDA: {torch.version.cuda}")
    else:
        print("CPU Entrenamiento en CPU (considere instalar ROCm para GPU AMD)")

    # Crear entrenador
    trainer = AMDGPUTrainer()

    # Entrenar modelo
    model, report = trainer.train(epochs=50, batch_size=128, learning_rate=0.002)

    print("\n" + "=" * 80)
    print("TARGET ENTRENAMIENTO AMD GPU COMPLETADO")
    print("=" * 80)
    print(f"DEVICE Dispositivo utilizado: {report['device_used']}")
    print(f"BRAIN Parámetros entrenados: {report['total_parameters']:,}")
    print(
        f"CLOCK Tiempo de entrenamiento: {timedelta(seconds=int(report['total_time_seconds']))}"
    )
    print(f"TROPHY Mejor precisión: {report['best_val_accuracy']:.2f}%")
    print(f"LAB Precisión test: {report['final_test_accuracy']:.2f}%")
    print(
        f"SPEED Throughput: {report['dataset_size'] * report['total_epochs'] / report['total_time_seconds']:.0f} samples/sec"
    )
    print("")
    print("CHECK PESOS NEURONALES OPTIMIZADOS ENTRENADOS EXITOSAMENTE")
    print("")
    print("NEXT Para máximo rendimiento, instala ROCm:")
    print("LINK https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html")


if __name__ == "__main__":
    main()
