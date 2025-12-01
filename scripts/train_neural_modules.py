"""
Script para entrenar módulos neurales con datos reales
======================================================

Entrena los módulos neurales usando los datasets construidos
desde Hack-Memori.
"""

import sys
import os
from pathlib import Path
import logging

# Añadir paths del proyecto
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_modules():
    """Entrena todos los módulos neurales."""
    logger.info("Starting neural modules training...")
    
    base_dir = project_root
    models_dir = base_dir / "data" / "consciousness" / "models"
    training_data_dir = base_dir / "data" / "consciousness" / "training_data"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar que existen datasets
    emotional_dataset = training_data_dir / "emotional_dataset.jsonl"
    decision_dataset = training_data_dir / "decision_dataset.jsonl"
    memory_dataset = training_data_dir / "memory_dataset.jsonl"
    
    if not emotional_dataset.exists():
        logger.error(f"Emotional dataset not found: {emotional_dataset}")
        logger.info("Please run collect_hack_memori_data.py first")
        return
    
    try:
        from packages.consciousness.src.conciencia.modulos.neural_modules.training.training_pipeline import TrainingPipeline
        from packages.consciousness.src.conciencia.modulos.neural_modules.ras_neural import RASNeuralSystem
        from packages.consciousness.src.conciencia.modulos.neural_modules.vmpfc_neural import VMPFCNeuralSystem
        
        device = "cpu"
        training_pipeline = TrainingPipeline(device=device)
        
        # Entrenar vmPFC
        logger.info("Training vmPFC Neural...")
        vmpfc_model = VMPFCNeuralSystem(device=device)
        vmpfc_metrics = training_pipeline.train_vmpfc(
            vmpfc_model.model,
            str(emotional_dataset),
            epochs=3,
            batch_size=4,
            lr=1e-4
        )
        logger.info(f"vmPFC training metrics: {vmpfc_metrics}")
        
        # Guardar modelo vmPFC
        vmpfc_model.save_model(str(models_dir / "vmpfc_model.pt"))
        logger.info(f"vmPFC model saved to: {models_dir / 'vmpfc_model.pt'}")
        
        # Entrenar RAS
        logger.info("Training RAS Neural...")
        ras_model = RASNeuralSystem(device=device)
        ras_metrics = training_pipeline.train_ras(
            ras_model.model,
            str(emotional_dataset),  # Reusar dataset emocional
            epochs=3,
            batch_size=4,
            lr=1e-4
        )
        logger.info(f"RAS training metrics: {ras_metrics}")
        
        # Guardar modelo RAS
        ras_model.save_model(str(models_dir / "ras_model.pt"))
        logger.info(f"RAS model saved to: {models_dir / 'ras_model.pt'}")
        
        logger.info("✅ Training complete!")
        logger.info(f"Models saved to: {models_dir}")
        
    except ImportError as e:
        logger.error(f"Failed to import training modules: {e}")
        logger.info("Make sure all dependencies are installed: pip install -r requirements_neural.txt")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)


if __name__ == "__main__":
    train_modules()

