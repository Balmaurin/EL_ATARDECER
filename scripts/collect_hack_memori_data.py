"""
Script para recopilar datos de Hack-Memori para entrenamiento inicial
========================================================================

Recolecta datos de sesiones de Hack-Memori y construye datasets
para entrenamiento de módulos neurales.
"""

import sys
import os
from pathlib import Path

# Añadir paths del proyecto
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from packages.consciousness.src.conciencia.modulos.neural_modules.training.dataset_builder import DatasetBuilder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Recopila datos de Hack-Memori y construye datasets."""
    logger.info("Starting Hack-Memori data collection...")
    
    # Buscar datos de Hack-Memori
    base_dir = project_root
    hack_memori_data_path = base_dir / "data" / "hack_memori"
    
    if not hack_memori_data_path.exists():
        logger.warning(f"Hack-Memori data directory not found: {hack_memori_data_path}")
        logger.info("Creating empty directory structure...")
        hack_memori_data_path.mkdir(parents=True, exist_ok=True)
        logger.info("Please add Hack-Memori session files to: {hack_memori_data_path}")
        return
    
    # Inicializar DatasetBuilder
    output_dir = base_dir / "data" / "consciousness" / "training_data"
    dataset_builder = DatasetBuilder(
        hack_memori_data_path=str(hack_memori_data_path),
        output_dir=str(output_dir)
    )
    
    # Recolectar datos
    logger.info("Collecting data from Hack-Memori sessions...")
    collected_data = dataset_builder.collect_from_hack_memori()
    
    if len(collected_data) == 0:
        logger.warning("No data collected. Make sure Hack-Memori session files exist.")
        logger.info(f"Expected location: {hack_memori_data_path}")
        return
    
    logger.info(f"Collected {len(collected_data)} entries")
    
    # Construir datasets
    logger.info("Building emotional dataset (vmPFC)...")
    emotional_dataset = dataset_builder.build_emotional_dataset(collected_data)
    logger.info(f"Emotional dataset saved to: {emotional_dataset}")
    
    logger.info("Building decision dataset (OFC)...")
    decision_dataset = dataset_builder.build_decision_dataset(collected_data)
    logger.info(f"Decision dataset saved to: {decision_dataset}")
    
    logger.info("Building memory dataset (Hippocampus)...")
    memory_dataset = dataset_builder.build_memory_dataset(collected_data)
    logger.info(f"Memory dataset saved to: {memory_dataset}")
    
    logger.info("✅ Data collection complete!")
    logger.info(f"Datasets ready for training in: {output_dir}")


if __name__ == "__main__":
    main()

