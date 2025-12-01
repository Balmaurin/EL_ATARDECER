"""
Fine-tuning REAL de embeddings BAAI/bge-m3 con datos de Hack-Memori
Sistema completo sin mocks ni fallbacks
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
    from torch.utils.data import DataLoader
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error("❌ sentence-transformers no disponible. Instalar: pip install sentence-transformers")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("❌ PyTorch no disponible. Instalar: pip install torch")


class EmbeddingFinetuner:
    """Fine-tuning REAL de embeddings BAAI/bge-m3"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers es requerido para fine-tuning de embeddings")
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch es requerido para fine-tuning de embeddings")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cargar modelo REAL
        logger.info(f"📥 Cargando modelo {model_name} en {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"✅ Modelo cargado: {self.model.get_sentence_embedding_dimension()} dimensiones")
        
        self.training_history = []
    
    def prepare_training_data(self, qa_data: List[Dict]) -> List[InputExample]:
        """Preparar datos de entrenamiento REAL desde Q&A de Hack-Memori"""
        training_examples = []
        
        for qa in qa_data:
            question = qa.get("question", "")
            response = qa.get("response", "")
            quality_score = qa.get("quality_score", 0.5)
            
            # Solo usar Q&A de alta calidad para entrenamiento
            if quality_score >= 0.7 and question and response:
                # Crear ejemplos positivos (pregunta-respuesta relacionadas)
                training_examples.append(
                    InputExample(texts=[question, response], label=1.0)
                )
                
                # Crear ejemplos negativos (pregunta-respuesta no relacionadas) si hay suficientes datos
                if len(qa_data) > 1:
                    # Usar otra respuesta aleatoria como negativo
                    import random
                    other_qa = random.choice([q for q in qa_data if q != qa])
                    if other_qa.get("response"):
                        training_examples.append(
                            InputExample(texts=[question, other_qa["response"]], label=0.0)
                        )
        
        logger.info(f"✅ Preparados {len(training_examples)} ejemplos de entrenamiento")
        return training_examples
    
    def finetune(
        self,
        training_data: List[InputExample],
        output_dir: str = "models/embeddings_finetuned",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100
    ) -> Dict[str, Any]:
        """Fine-tuning REAL del modelo de embeddings"""
        if not training_data:
            raise ValueError("No hay datos de entrenamiento")
        
        logger.info(f"🚀 Iniciando fine-tuning REAL de {self.model_name}")
        logger.info(f"   - Ejemplos: {len(training_data)}")
        logger.info(f"   - Epochs: {epochs}")
        logger.info(f"   - Batch size: {batch_size}")
        logger.info(f"   - Learning rate: {learning_rate}")
        logger.info(f"   - Device: {self.device}")
        
        # Crear DataLoader
        train_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size)
        
        # Loss function REAL (contrastive loss)
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Evaluación (usar subset de datos)
        evaluator = None
        if len(training_data) > 20:
            eval_examples = training_data[:min(20, len(training_data)//10)]
            evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
                eval_examples, name="eval"
            )
        
        # Entrenamiento REAL
        start_time = datetime.now()
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            evaluator=evaluator,
            evaluation_steps=100 if evaluator else None,
            output_path=output_dir,
            show_progress_bar=True
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Guardar modelo
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save(output_dir)
        
        # Guardar metadata
        metadata = {
            "model_name": self.model_name,
            "training_examples": len(training_data),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_time_seconds": training_time,
            "device": self.device,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(Path(output_dir) / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Fine-tuning completado en {training_time:.2f}s")
        logger.info(f"   - Modelo guardado en: {output_dir}")
        
        return {
            "success": True,
            "output_dir": output_dir,
            "training_time": training_time,
            "metadata": metadata
        }
    
    def update_embeddings(self, corpus_path: Path, output_path: Path):
        """Regenerar embeddings del corpus con modelo fine-tuneado"""
        logger.info(f"🔄 Regenerando embeddings del corpus...")
        
        # Cargar chunks
        chunks_dir = corpus_path / "chunks"
        if not chunks_dir.exists():
            raise FileNotFoundError(f"Directorio de chunks no encontrado: {chunks_dir}")
        
        texts = []
        for chunk_file in chunks_dir.glob("*.jsonl"):
            try:
                chunk_data = json.loads(chunk_file.read_text(encoding="utf-8"))
                texts.append(chunk_data.get("text", ""))
            except Exception as e:
                logger.warning(f"Error leyendo chunk {chunk_file}: {e}")
        
        if not texts:
            raise ValueError("No se encontraron textos para embedding")
        
        # Generar embeddings con modelo fine-tuneado
        logger.info(f"📤 Generando embeddings para {len(texts)} textos...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Guardar embeddings
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)
        
        logger.info(f"✅ Embeddings guardados en: {output_path}")
        logger.info(f"   - Shape: {embeddings.shape}")
        
        return embeddings

