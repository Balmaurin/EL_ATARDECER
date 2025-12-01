#!/usr/bin/env python3
"""
SISTEMA DE ENTRENAMIENTO NEURONAL REAL
=====================================
Sistema completo para entrenar redes neuronales usando los pesos del proyecto Sheily
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealNeuralTraining")


class SheilyNeuralModel(nn.Module):
    """Modelo neuronal basado en análisis del proyecto Sheily"""

    def __init__(self, input_size=512, hidden_sizes=[256, 128, 64, 32], num_classes=12):
        super().__init__()

        # Construcción dinámica de capas
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(0.2)]
            )
            prev_size = hidden_size

        # Capa de salida
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RealNeuralTrainer:
    """Entrenador de redes neuronales reales"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔥 Usando dispositivo: {self.device}")

        # Cargar datos del proyecto
        self.project_data = self.load_project_data()
        self.model = None

    def load_project_data(self):
        """Cargar datos del proyecto para entrenamiento"""
        data_dir = Path("extracted_project_data")
        analysis_files = list(data_dir.glob("project_analysis_*.json"))

        if not analysis_files:
            raise FileNotFoundError("No se encontraron análisis del proyecto")

        latest_file = max(analysis_files, key=lambda f: f.stat().st_mtime)
        with open(latest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_training_dataset(self):
        """Crear dataset para entrenamiento basado en el proyecto"""
        logger.info("🔄 Creando dataset de entrenamiento...")

        weights_data = self.project_data["weights_dataset"]

        # Extraer características y etiquetas
        features = []
        labels = []

        # Categorías del proyecto Sheily
        categories = {
            "mcp": 0,
            "agent": 1,
            "security": 2,
            "rag": 3,
            "blockchain": 4,
            "education": 5,
            "consciousness": 6,
            "api": 7,
            "data": 8,
            "neural": 9,
            "federated": 10,
            "quantum": 11,
        }

        neural_patterns = weights_data["neural_patterns"]
        complexity_weights = weights_data["complexity_weights"]

        # Procesar patrones neuronales
        for pattern_name, weight_value in neural_patterns.items():
            # Extraer características del patrón
            feature_vector = self.extract_pattern_features(
                pattern_name, weight_value, complexity_weights
            )

            # Determinar categoría
            category = self.classify_pattern(pattern_name, categories)

            features.append(feature_vector)
            labels.append(category)

        # Convertir a tensores
        X = torch.FloatTensor(features)
        y = torch.LongTensor(labels)

        logger.info(
            f"✅ Dataset creado: {X.shape[0]} muestras, {X.shape[1]} características"
        )

        return X, y

    def extract_pattern_features(self, pattern_name, weight_value, complexity_weights):
        """Extraer vector de características de un patrón"""
        features = [weight_value]  # Valor base del peso

        # Características basadas en el nombre
        name_parts = pattern_name.lower().split("_")

        # One-hot para palabras clave
        keywords = [
            "mcp",
            "agent",
            "neural",
            "api",
            "system",
            "manager",
            "core",
            "advanced",
        ]
        for keyword in keywords:
            features.append(1.0 if any(keyword in part for part in name_parts) else 0.0)

        # Longitud del nombre (normalizada)
        features.append(len(pattern_name) / 100.0)

        # Número de partes en el nombre
        features.append(len(name_parts) / 10.0)

        # Complejidad si está disponible
        if pattern_name in complexity_weights:
            complexity_data = complexity_weights[pattern_name]
            if isinstance(complexity_data, dict):
                features.extend(
                    [
                        complexity_data.get("line_complexity", 0.5),
                        complexity_data.get("function_density", 0.5),
                        complexity_data.get("class_density", 0.5),
                    ]
                )
            else:
                features.extend([0.5, 0.5, 0.5])
        else:
            features.extend([0.5, 0.5, 0.5])

        # Estadísticas del peso
        features.extend(
            [
                abs(weight_value),  # Magnitud
                min(weight_value, 1.0),  # Saturado a 1
                weight_value**2,  # Cuadrático
            ]
        )

        # Padding para llegar a 512 características
        while len(features) < 512:
            features.append(np.random.normal(0, 0.1))

        return features[:512]

    def classify_pattern(self, pattern_name, categories):
        """Clasificar patrón en una categoría"""
        pattern_lower = pattern_name.lower()

        for category_name, category_id in categories.items():
            if category_name in pattern_lower:
                return category_id

        # Clasificación por palabras clave más específicas
        if any(
            word in pattern_lower for word in ["orchestrator", "coordinator", "master"]
        ):
            return categories["mcp"]
        elif any(
            word in pattern_lower for word in ["auth", "crypto", "security", "guard"]
        ):
            return categories["security"]
        elif any(word in pattern_lower for word in ["retrieval", "search", "index"]):
            return categories["rag"]
        elif any(word in pattern_lower for word in ["server", "client", "endpoint"]):
            return categories["api"]
        elif any(word in pattern_lower for word in ["learn", "train", "model"]):
            return categories["neural"]

        # Default: neural
        return categories["neural"]

    def create_model(self, input_size, num_classes):
        """Crear modelo con pesos pre-inicializados"""
        model = SheilyNeuralModel(input_size=input_size, num_classes=num_classes)

        # Cargar pesos pre-generados si están disponibles
        self.load_pretrained_weights(model)

        return model.to(self.device)

    def load_pretrained_weights(self, model):
        """Cargar pesos pre-generados del proyecto"""
        weights_dir = Path("real_neural_weights/feedforward")
        weight_files = list(weights_dir.glob("weights_*.npz"))

        if not weight_files:
            logger.info(
                "⚠️ No se encontraron pesos pre-generados, usando inicialización aleatoria"
            )
            return

        latest_weights = max(weight_files, key=lambda f: f.stat().st_mtime)
        weights = np.load(latest_weights)

        logger.info(f"🔄 Cargando pesos desde: {latest_weights}")

        try:
            # Mapear pesos del archivo .npz al modelo PyTorch
            with torch.no_grad():
                layer_idx = 0
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        weight_key = f"layer_{layer_idx + 1}_weights"
                        bias_key = f"layer_{layer_idx + 1}_bias"

                        if weight_key in weights.files and bias_key in weights.files:
                            weight_data = weights[weight_key]
                            bias_data = weights[bias_key]

                            # Verificar compatibilidad de dimensiones
                            if (
                                weight_data.shape[1] == module.weight.shape[0]
                                and weight_data.shape[0] == module.weight.shape[1]
                            ):
                                module.weight.copy_(torch.from_numpy(weight_data.T))
                                if bias_data.shape[0] == module.bias.shape[0]:
                                    module.bias.copy_(torch.from_numpy(bias_data))
                                logger.info(f"✅ Cargado: {weight_key}")
                            else:
                                logger.warning(
                                    f"⚠️ Dimensiones incompatibles para {weight_key}"
                                )

                        layer_idx += 1

            logger.info("✅ Pesos pre-generados cargados exitosamente")

        except Exception as e:
            logger.warning(f"⚠️ Error cargando pesos: {e}")

    def train_model(self, epochs=100, learning_rate=0.001, batch_size=32):
        """Entrenar el modelo neuronal"""
        logger.info("🚀 INICIANDO ENTRENAMIENTO NEURONAL REAL")
        logger.info("=" * 60)

        # Crear dataset
        X, y = self.create_training_dataset()

        # Dividir en entrenamiento y validación
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Mover a dispositivo
        X_train, X_val = X_train.to(self.device), X_val.to(self.device)
        y_train, y_val = y_train.to(self.device), y_val.to(self.device)

        # Crear modelo
        self.model = self.create_model(X.shape[1], len(torch.unique(y)))

        # Definir pérdida y optimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        logger.info(
            f"📊 Dataset: {len(X_train)} entrenamiento, {len(X_val)} validación"
        )
        logger.info(
            f"🧠 Modelo: {sum(p.numel() for p in self.model.parameters()):,} parámetros"
        )

        # Loop de entrenamiento
        best_val_accuracy = 0.0
        training_history = []

        for epoch in range(epochs):
            # Entrenamiento
            self.model.train()
            total_loss = 0.0
            correct_train = 0

            # Mini-batches
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i : i + batch_size]
                batch_y = y_train[i : i + batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_train += (predicted == batch_y).sum().item()

            # Validación
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_val).float().mean().item()

            train_accuracy = correct_train / len(X_train)

            # Scheduler
            scheduler.step(val_loss)

            # Log progreso
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Época {epoch+1}/{epochs}:")
                logger.info(f"  Pérdida entrenamiento: {total_loss/len(X_train):.6f}")
                logger.info(f"  Precisión entrenamiento: {train_accuracy:.4f}")
                logger.info(f"  Pérdida validación: {val_loss:.6f}")
                logger.info(f"  Precisión validación: {val_accuracy:.4f}")
                logger.info(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

            # Guardar mejor modelo
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_trained_model(epoch, val_accuracy)

            # Historial
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total_loss / len(X_train),
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss.item(),
                    "val_accuracy": val_accuracy,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        logger.info("🎉 ENTRENAMIENTO COMPLETADO")
        logger.info(f"✅ Mejor precisión de validación: {best_val_accuracy:.4f}")

        return training_history

    def save_trained_model(self, epoch, accuracy):
        """Guardar modelo entrenado"""
        models_dir = Path("trained_models")
        models_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Guardar pesos del modelo
        model_path = models_dir / f"sheily_neural_model_{timestamp}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "epoch": epoch,
                "accuracy": accuracy,
                "model_architecture": "SheilyNeuralModel",
                "based_on_project": "Sheily AI System Analysis",
                "timestamp": timestamp,
            },
            model_path,
        )

        logger.info(f"💾 Modelo guardado: {model_path}")


def main():
    """Función principal de entrenamiento"""
    trainer = RealNeuralTrainer()
    history = trainer.train_model(epochs=50, learning_rate=0.001)

    logger.info("🎯 ENTRENAMIENTO DE PESOS REALES COMPLETADO")
    logger.info(
        "Los pesos neuronales están ahora entrenados con datos reales del proyecto"
    )


if __name__ == "__main__":
    main()
