#!/usr/bin/env python3
"""
ENTRENAMIENTO REAL CON TRANSFORMERS
===================================
Sistema de entrenamiento verdadero usando Hugging Face Transformers,
datos reales del proyecto Sheily, y backpropagation real con GPU.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader

# Transformers y entrenamiento real
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RealTransformersTraining")


class SheilyRealTrainer:
    """Entrenador real con Transformers para el proyecto Sheily"""

    def __init__(self):
        # Configuración del dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔥 Dispositivo: {self.device}")

        # Verificar GPU
        if torch.cuda.is_available():
            logger.info(f"🚀 GPU detectada: {torch.cuda.get_device_name()}")
            logger.info(
                f"💾 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        else:
            logger.warning("⚠️ Usando CPU - el entrenamiento será más lento")

        # Configuración del modelo
        self.model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = None
        self.model = None

        # Directorios
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "extracted_project_data"
        self.models_dir = self.project_root / "trained_transformers"
        self.models_dir.mkdir(exist_ok=True)

        logger.info("🧠 Inicializando entrenamiento real con Transformers")

    def load_project_data(self) -> Dict[str, Any]:
        """Cargar datos reales del proyecto para entrenamiento"""
        logger.info("📂 Cargando datos reales del proyecto Sheily...")

        analysis_files = list(self.data_dir.glob("project_analysis_*.json"))
        if not analysis_files:
            raise FileNotFoundError(
                "❌ No se encontraron datos del proyecto. Ejecuta extract_project_files.py primero"
            )

        latest_file = max(analysis_files, key=lambda f: f.stat().st_mtime)

        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"✅ Datos cargados desde: {latest_file}")
        logger.info(f"📊 Archivos analizados: {data['totals']['files_processed']}")
        logger.info(f"🔧 Funciones extraídas: {data['totals']['functions_extracted']}")
        logger.info(f"📦 Clases extraídas: {data['totals']['classes_extracted']}")

        return data

    def create_training_dataset(self, project_data: Dict[str, Any]) -> Dataset:
        """Crear dataset real para entrenamiento de lenguaje"""
        logger.info("🔄 Creando dataset real de entrenamiento...")

        training_texts = []

        # 1. Extraer documentación y comentarios
        file_analysis = project_data.get("file_analysis", {})

        for file_path, analysis in file_analysis.items():
            if isinstance(analysis, dict):
                # Agregar docstrings de funciones
                functions = analysis.get("functions", [])
                for func in functions:
                    if isinstance(func, dict) and func.get("docstring"):
                        docstring = func["docstring"].strip()
                        if len(docstring) > 20:  # Filtrar docstrings muy cortos
                            training_texts.append(f"# {func['name']}\n{docstring}")

                # Agregar docstrings de clases
                classes = analysis.get("classes", [])
                for cls in classes:
                    if isinstance(cls, dict) and cls.get("docstring"):
                        docstring = cls["docstring"].strip()
                        if len(docstring) > 20:
                            training_texts.append(
                                f"class {cls['name']}:\n    \"\"\"{docstring}\"\"\""
                            )

        # 2. Crear ejemplos de código y documentación
        patterns = project_data["weights_dataset"]["neural_patterns"]

        for pattern_name, weight in patterns.items():
            # Generar texto descriptivo basado en patrones
            if "mcp" in pattern_name.lower():
                training_texts.append(
                    f"El sistema MCP {pattern_name} implementa protocolos de comunicación "
                    f"entre agentes con un peso de importancia de {weight:.3f}."
                )
            elif "agent" in pattern_name.lower():
                training_texts.append(
                    f"El agente {pattern_name} gestiona tareas específicas del sistema "
                    f"con una complejidad de {weight:.3f}."
                )
            elif "neural" in pattern_name.lower():
                training_texts.append(
                    f"El componente neural {pattern_name} procesa información con "
                    f"un factor de activación de {weight:.3f}."
                )

        # 3. Agregar descripciones de arquitectura
        architecture_descriptions = [
            "Sheily es un sistema de IA avanzado que integra múltiples agentes especializados.",
            "El sistema MCP (Model Context Protocol) coordina la comunicación entre componentes.",
            "Los agentes autónomos gestionan tareas específicas como seguridad, análisis y aprendizaje.",
            "El sistema de memoria distribuida permite el almacenamiento y recuperación eficiente de información.",
            "La arquitectura federada permite el entrenamiento distribuido de modelos de IA.",
            "Los componentes de seguridad implementan protocolos de cifrado y autenticación avanzados.",
            "El sistema de recompensas optimiza el comportamiento de los agentes mediante reinforcement learning.",
            "La integración blockchain proporciona trazabilidad y transparencia en las transacciones.",
            "El motor de procesamiento de lenguaje natural permite la comunicación fluida con usuarios.",
            "Los componentes de monitoreo proporcionan métricas en tiempo real del rendimiento del sistema.",
        ]

        training_texts.extend(architecture_descriptions)

        # 4. Filtrar y limpiar textos
        cleaned_texts = []
        for text in training_texts:
            if isinstance(text, str) and len(text.strip()) > 10:
                # Limpiar y normalizar
                cleaned_text = text.strip().replace("\t", "    ")
                if len(cleaned_text) < 1000:  # Evitar textos muy largos
                    cleaned_texts.append(cleaned_text)

        logger.info(
            f"✅ Dataset creado con {len(cleaned_texts)} ejemplos de entrenamiento"
        )

        # Crear dataset de Hugging Face
        dataset = Dataset.from_dict({"text": cleaned_texts})

        return dataset

    def load_model_and_tokenizer(self):
        """Cargar modelo y tokenizer reales de Hugging Face"""
        logger.info(f"🤖 Cargando modelo: {self.model_name}")

        try:
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Agregar token de padding si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(
                f"✅ Tokenizer cargado. Vocabulario: {len(self.tokenizer):,} tokens"
            )

            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else None,
            )

            # Obtener número de parámetros
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            logger.info(f"✅ Modelo cargado:")
            logger.info(f"   🔢 Parámetros totales: {total_params:,}")
            logger.info(f"   🎯 Parámetros entrenables: {trainable_params:,}")
            logger.info(
                f"   💾 Memoria del modelo: {total_params * 2 / 1e9:.1f} GB (fp16)"
            )

        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenizar dataset para entrenamiento"""
        logger.info("🔤 Tokenizando dataset...")

        def tokenize_function(examples):
            # Tokenizar con truncation y padding
            tokens = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,  # Reducir para ahorrar memoria
                return_tensors="pt",
            )

            # Para language modeling, labels = input_ids
            tokens["labels"] = tokens["input_ids"].clone()

            return tokens

        # Aplicar tokenización en lotes
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )

        logger.info(f"✅ Dataset tokenizado: {len(tokenized_dataset)} ejemplos")

        return tokenized_dataset

    def setup_training_args(self) -> TrainingArguments:
        """Configurar argumentos de entrenamiento real"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.models_dir / f"sheily_transformer_{timestamp}"

        # Configuración adaptada a recursos disponibles
        if torch.cuda.is_available():
            # Configuración para GPU
            batch_size = 4  # Reducido para evitar OOM
            gradient_accumulation_steps = 4
            fp16 = True
            dataloader_num_workers = 2
        else:
            # Configuración para CPU
            batch_size = 1
            gradient_accumulation_steps = 8
            fp16 = False
            dataloader_num_workers = 0

        training_args = TrainingArguments(
            # Directorio de salida
            output_dir=str(output_dir),
            # Configuración de entrenamiento
            num_train_epochs=3,  # Épocas reales
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # Optimización
            learning_rate=5e-5,  # Learning rate estándar para fine-tuning
            weight_decay=0.01,
            warmup_steps=100,
            # Precisión mixta
            fp16=fp16,
            # Logging y guardado
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            # Evaluación
            eval_strategy="steps",
            eval_steps=50,
            # Recursos
            dataloader_num_workers=dataloader_num_workers,
            remove_unused_columns=False,
            # Otras configuraciones
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Reproducibilidad
            seed=42,
            # Reportes
            report_to=[],  # Desactivar wandb/tensorboard por ahora
        )

        logger.info("⚙️ Configuración de entrenamiento:")
        logger.info(f"   📁 Directorio de salida: {output_dir}")
        logger.info(f"   🔄 Épocas: {training_args.num_train_epochs}")
        logger.info(
            f"   📦 Batch size: {batch_size} x {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}"
        )
        logger.info(f"   📈 Learning rate: {training_args.learning_rate}")
        logger.info(f"   🎯 FP16: {fp16}")

        return training_args

    def train_model(self):
        """Ejecutar entrenamiento real del modelo"""
        logger.info("🚀 INICIANDO ENTRENAMIENTO REAL CON TRANSFORMERS")
        logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # 1. Cargar datos del proyecto
            project_data = self.load_project_data()

            # 2. Crear dataset
            dataset = self.create_training_dataset(project_data)

            # 3. Cargar modelo y tokenizer
            self.load_model_and_tokenizer()

            # 4. Tokenizar dataset
            tokenized_dataset = self.tokenize_dataset(dataset)

            # 5. Dividir en train/eval
            train_size = int(0.9 * len(tokenized_dataset))
            train_dataset = tokenized_dataset.select(range(train_size))
            eval_dataset = tokenized_dataset.select(
                range(train_size, len(tokenized_dataset))
            )

            logger.info(f"📊 División del dataset:")
            logger.info(f"   🏋️ Entrenamiento: {len(train_dataset)} ejemplos")
            logger.info(f"   🎯 Evaluación: {len(eval_dataset)} ejemplos")

            # 6. Configurar argumentos de entrenamiento
            training_args = self.setup_training_args()

            # 7. Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # No es masked language modeling
                return_tensors="pt",
            )

            # 8. Crear trainer real
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )

            # 9. ¡ENTRENAMIENTO REAL!
            logger.info("🔥 Iniciando backpropagation real...")
            logger.info("⏰ Esto tomará tiempo real dependiendo de tu hardware...")

            train_result = trainer.train()

            # 10. Guardar modelo entrenado
            trainer.save_model()
            self.tokenizer.save_pretrained(training_args.output_dir)

            # 11. Generar reporte final
            end_time = datetime.now()
            training_time = end_time - start_time

            final_report = {
                "training_completed": True,
                "model_name": self.model_name,
                "training_time_seconds": training_time.total_seconds(),
                "training_time_formatted": str(training_time),
                "total_steps": train_result.global_step,
                "final_loss": train_result.training_loss,
                "device_used": str(self.device),
                "model_size_parameters": sum(
                    p.numel() for p in self.model.parameters()
                ),
                "dataset_size": len(train_dataset),
                "output_directory": str(training_args.output_dir),
                "based_on_project": "Sheily AI System - Real Code Analysis",
                "timestamp": datetime.now().isoformat(),
            }

            # Guardar reporte
            report_path = training_args.output_dir / "training_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)

            logger.info("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
            logger.info(f"⏱️ Tiempo total: {training_time}")
            logger.info(f"📉 Pérdida final: {train_result.training_loss:.6f}")
            logger.info(f"🔄 Pasos totales: {train_result.global_step}")
            logger.info(f"💾 Modelo guardado en: {training_args.output_dir}")
            logger.info(f"📋 Reporte: {report_path}")

            return final_report

        except Exception as e:
            logger.error(f"❌ Error durante entrenamiento: {e}")
            raise

    def test_trained_model(self, model_path: str):
        """Probar modelo entrenado con inferencia real"""
        logger.info("🧪 Probando modelo entrenado...")

        # Cargar modelo entrenado
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Prompt de prueba
        test_prompts = [
            "El sistema MCP",
            "Los agentes de Sheily",
            "La arquitectura neuronal",
            "# Función para",
        ]

        model.eval()

        for prompt in test_prompts:
            inputs = tokenizer.encode(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(
                f"Input: '{prompt}' → Output: '{generated_text[len(prompt):][:100]}...'"
            )


def main():
    """Función principal"""
    trainer = SheilyRealTrainer()

    try:
        report = trainer.train_model()

        print("\n" + "=" * 80)
        print("🎯 ENTRENAMIENTO REAL COMPLETADO")
        print("=" * 80)
        print("✅ Esto fue entrenamiento REAL con:")
        print("   - Modelo real de Hugging Face (DialoGPT)")
        print("   - Datos reales del proyecto Sheily")
        print("   - Backpropagation real con gradientes")
        print("   - Optimización real con Adam")
        print("   - Múltiples épocas de entrenamiento")
        print("   - Guardado real del modelo entrenado")
        print(f"📊 Parámetros entrenados: {report['model_size_parameters']:,}")
        print(f"⏱️ Tiempo real: {report['training_time_formatted']}")
        print(f"💾 Modelo guardado en: {report['output_directory']}")

    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        print("💡 Posibles soluciones:")
        print("   - Verificar que tengas suficiente memoria GPU/RAM")
        print("   - Reducir batch_size en setup_training_args()")
        print("   - Usar un modelo más pequeño (DistilGPT-2)")


if __name__ == "__main__":
    main()
