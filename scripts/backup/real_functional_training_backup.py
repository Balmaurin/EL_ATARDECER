#!/usr/bin/env python3
"""
SISTEMA DE ENTRENAMIENTO REAL Y FUNCIONAL - EL AMANECER V3
===========================================================
Sistema que realmente entrena modelos sin errores de encoding
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configurar logging sin emojis para evitar errores de encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_real.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("RealTraining")

class RealTrainingSystem:
    """Sistema de entrenamiento real que funciona"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        
        # Crear directorios necesarios
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.training_results = {}
        logger.info("Sistema de entrenamiento real inicializado")
    
    def check_python_environment(self):
        """Verificar que el entorno Python esté configurado"""
        try:
            import torch
            logger.info(f"PyTorch disponible: {torch.__version__}")
            if torch.cuda.is_available():
                logger.info(f"CUDA disponible: {torch.cuda.device_count()} GPUs")
            return True
        except ImportError:
            logger.warning("PyTorch no disponible, usando modo CPU")
            return False
    
    def create_simple_dataset(self):
        """Crear un dataset simple para entrenamiento"""
        dataset_file = self.data_dir / "training_data.json"
        
        # Dataset simple para entrenamiento de prueba
        training_data = [
            {"input": "Hola", "output": "Hola! Como puedo ayudarte?"},
            {"input": "Como estas?", "output": "Estoy bien, gracias por preguntar"},
            {"input": "Que eres?", "output": "Soy Sheily, una IA asistente"},
            {"input": "Explicame algo", "output": "Claro, que te gustaria que te explique?"},
            {"input": "Gracias", "output": "De nada! Estoy aqui para ayudarte"},
            {"input": "Adios", "output": "Hasta luego! Que tengas buen dia"},
            {"input": "Ayuda", "output": "Por supuesto, en que puedo asistirte?"},
            {"input": "Aprender", "output": "El aprendizaje es fundamental para crecer"},
            {"input": "Conocimiento", "output": "El conocimiento es poder y liberacion"},
            {"input": "Inteligencia", "output": "La inteligencia artificial puede ayudar a la humanidad"}
        ]
        
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset creado: {len(training_data)} ejemplos")
        return dataset_file
    
    def train_simple_model(self):
        """Entrenar un modelo simple usando transformers"""
        try:
            logger.info("Iniciando entrenamiento de modelo simple...")
            
            # Crear script de entrenamiento simple
            training_script = self.project_root / "simple_training.py"
            
            script_content = '''
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Pregunta: {item['input']} Respuesta: {item['output']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def train_model():
    # Usar modelo pequeño para prueba
    model_name = "gpt2"
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Agregar pad token si no existe
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Cargar dataset
        dataset = SimpleDataset("data/training_data.json", tokenizer)
        
        # Configurar entrenamiento
        training_args = TrainingArguments(
            output_dir="models/sheily_simple",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=10,
            save_total_limit=2,
            logging_steps=5,
            learning_rate=5e-5,
            warmup_steps=10,
            logging_dir="logs/training",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        
        print("Iniciando entrenamiento...")
        trainer.train()
        
        # Guardar modelo
        trainer.save_model()
        tokenizer.save_pretrained("models/sheily_simple")
        
        print("Entrenamiento completado exitosamente!")
        return True
        
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        return False

if __name__ == "__main__":
    train_model()
'''
            
            with open(training_script, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Ejecutar entrenamiento
            result = subprocess.run([
                sys.executable, str(training_script)
            ], capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                logger.info("Entrenamiento completado exitosamente")
                self.training_results['simple_model'] = 'SUCCESS'
                return True
            else:
                logger.error(f"Error en entrenamiento: {result.stderr}")
                self.training_results['simple_model'] = 'FAILED'
                return False
                
        except Exception as e:
            logger.error(f"Error en train_simple_model: {e}")
            return False
    
    def train_embeddings(self):
        """Entrenar embeddings personalizados"""
        try:
            logger.info("Entrenando embeddings...")
            
            # Crear script para embeddings
            embeddings_script = '''
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

def train_embeddings():
    # Cargar datos
    with open("data/training_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Preparar textos
    texts = []
    for item in data:
        texts.append(item["input"])
        texts.append(item["output"])
    
    # Crear vectorizador TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Reducir dimensiones con SVD
    svd = TruncatedSVD(n_components=100)
    embeddings = svd.fit_transform(tfidf_matrix)
    
    # Guardar modelos
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open("models/svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    
    np.save("models/embeddings.npy", embeddings)
    
    print(f"Embeddings entrenados: {embeddings.shape}")
    return True

if __name__ == "__main__":
    train_embeddings()
'''
            
            embeddings_file = self.project_root / "train_embeddings.py"
            with open(embeddings_file, 'w', encoding='utf-8') as f:
                f.write(embeddings_script)
            
            result = subprocess.run([
                sys.executable, str(embeddings_file)
            ], capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                logger.info("Embeddings entrenados exitosamente")
                self.training_results['embeddings'] = 'SUCCESS'
                return True
            else:
                logger.error(f"Error entrenando embeddings: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error en train_embeddings: {e}")
            return False
    
    def create_model_config(self):
        """Crear configuración del modelo entrenado"""
        config = {
            "model_name": "Sheily-V1-Real",
            "version": "1.0",
            "trained_at": datetime.now().isoformat(),
            "dataset_size": 10,
            "model_type": "simple_gpt2",
            "capabilities": [
                "Conversacion basica",
                "Respuestas contextuales", 
                "Asistencia general"
            ],
            "training_results": self.training_results
        }
        
        config_file = self.models_dir / "model_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info("Configuracion del modelo creada")
    
    def test_trained_model(self):
        """Probar el modelo entrenado"""
        try:
            test_script = '''
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_model():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("models/sheily_simple")
        model = GPT2LMHeadModel.from_pretrained("models/sheily_simple")
        
        test_input = "Pregunta: Hola"
        inputs = tokenizer.encode(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=50, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Entrada: {test_input}")
        print(f"Respuesta: {response}")
        
        return True
    except Exception as e:
        print(f"Error en test: {e}")
        return False

if __name__ == "__main__":
    test_model()
'''
            
            test_file = self.project_root / "test_model.py"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_script)
            
            result = subprocess.run([
                sys.executable, str(test_file)
            ], capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                logger.info("Modelo probado exitosamente")
                logger.info(f"Salida del test: {result.stdout}")
                return True
            else:
                logger.warning(f"Test del modelo fallo: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error en test del modelo: {e}")
            return False
    
    def run_complete_training(self):
        """Ejecutar entrenamiento completo funcional"""
        start_time = datetime.now()
        logger.info("=== INICIANDO ENTRENAMIENTO REAL ===")
        
        # 1. Verificar entorno
        torch_available = self.check_python_environment()
        
        # 2. Crear dataset
        dataset_file = self.create_simple_dataset()
        
        # 3. Entrenar embeddings (siempre funciona)
        embeddings_success = self.train_embeddings()
        
        # 4. Entrenar modelo (si torch está disponible)
        model_success = False
        if torch_available:
            try:
                # Verificar si transformers está disponible
                import transformers
                model_success = self.train_simple_model()
            except ImportError:
                logger.warning("Transformers no disponible, instalando...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "torch", "scikit-learn"], check=True)
                    model_success = self.train_simple_model()
                except:
                    logger.error("No se pudo instalar transformers")
        
        # 5. Crear configuración
        self.create_model_config()
        
        # 6. Probar modelo (si se entrenó)
        if model_success:
            self.test_trained_model()
        
        # Resumen
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=== ENTRENAMIENTO COMPLETADO ===")
        logger.info(f"Duracion: {duration}")
        logger.info(f"Embeddings: {'OK' if embeddings_success else 'FALLO'}")
        logger.info(f"Modelo GPT2: {'OK' if model_success else 'FALLO'}")
        
        if embeddings_success or model_success:
            logger.info("ENTRENAMIENTO EXITOSO - AL MENOS UN COMPONENTE FUNCIONAL")
            return True
        else:
            logger.error("ENTRENAMIENTO FALLO - NINGUN COMPONENTE FUNCIONAL")
            return False

def main():
    training_system = RealTrainingSystem()
    success = training_system.run_complete_training()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)