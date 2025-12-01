#!/usr/bin/env python3
"""
Script de Migración: Responses a Formato Optimizado
====================================================

Migra los archivos JSON de responses existentes al formato optimizado
que maximiza la eficacia en todos los sistemas de entrenamiento.

Uso:
    python scripts/migrate_responses_to_optimized_format.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.backend.hack_memori_service import HackMemoriService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_response_file(response_file: Path, service: HackMemoriService) -> bool:
    """
    Migrar un archivo de response al formato optimizado
    """
    try:
        # Cargar response existente
        with open(response_file, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
        
        # Verificar si ya está en formato optimizado
        if "quality_score" in old_data and "categories" in old_data:
            logger.debug(f"✅ {response_file.name} ya está en formato optimizado")
            return True
        
        # Extraer datos básicos
        prompt = old_data.get("prompt", "")
        response = old_data.get("response", "")
        accepted = old_data.get("accepted_for_training", False)
        pii_flag = old_data.get("pii_flag", False)
        tokens_used = old_data.get("tokens_used", 0)
        
        # Calcular campos nuevos usando métodos del servicio
        quality_score = service._calculate_optimized_quality_score(response, accepted, pii_flag, tokens_used)
        category_data = service._classify_categories(prompt, response)
        quality_metrics = service._calculate_quality_metrics(response, accepted, pii_flag, tokens_used)
        formatted_for_training = service._generate_training_formats(prompt, response, quality_score)
        linguistic_analysis = service._analyze_linguistics(response)
        
        # Crear nuevo formato manteniendo campos existentes
        new_data = {
            # Campos básicos (mantener todos los existentes)
            **old_data,
            
            # Agregar campos nuevos si no existen
            "quality_score": old_data.get("quality_score", quality_score),
            "quality_metrics": old_data.get("quality_metrics", quality_metrics),
            "categories": old_data.get("categories", category_data.get("categories", ["general"])),
            "primary_category": old_data.get("primary_category", category_data.get("primary_category", "general")),
            "subcategories": old_data.get("subcategories", category_data.get("subcategories", [])),
            "complexity_level": old_data.get("complexity_level", category_data.get("complexity_level", "intermediate")),
            "task_type": old_data.get("task_type", "instruction"),
            "formatted_for_training": old_data.get("formatted_for_training", formatted_for_training),
            "linguistic_analysis": old_data.get("linguistic_analysis", linguistic_analysis),
            
            # Metadata de entrenamiento
            "training_metadata": old_data.get("training_metadata", {
                "used_in_training": False,
                "training_systems": [],
                "training_timestamp": None,
                "training_improvement": None
            }),
            
            # Metadata de seguridad
            "security_metadata": old_data.get("security_metadata", {
                "pii_detected": pii_flag,
                "sensitive_content": False,
                "content_filter_score": 0.95 if not pii_flag else 0.5,
                "ethical_compliance": True
            }),
            
            # Generation metadata
            "generation_metadata": old_data.get("generation_metadata", {
                "response_time_ms": None,
                "retries": 0,
                "cache_hit": False
            })
        }
        
        # Guardar archivo migrado
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error migrando {response_file.name}: {e}")
        return False


def main():
    """
    Función principal de migración
    """
    logger.info("=" * 80)
    logger.info("🔄 MIGRACIÓN DE RESPONSES A FORMATO OPTIMIZADO")
    logger.info("=" * 80)
    
    service = HackMemoriService()
    responses_dir = service.responses_dir
    
    if not responses_dir.exists():
        logger.error(f"❌ Directorio no existe: {responses_dir}")
        return
    
    # Encontrar todos los archivos JSON
    response_files = list(responses_dir.glob("*.json"))
    logger.info(f"📊 Archivos encontrados: {len(response_files)}")
    
    if not response_files:
        logger.warning("⚠️ No hay archivos para migrar")
        return
    
    # Migrar cada archivo
    migrated = 0
    failed = 0
    already_optimized = 0
    
    for i, response_file in enumerate(response_files, 1):
        logger.info(f"📦 Procesando {i}/{len(response_files)}: {response_file.name}")
        
        # Verificar si ya está optimizado
        try:
            with open(response_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "quality_score" in data and "categories" in data:
                already_optimized += 1
                continue
        except:
            pass
        
        if migrate_response_file(response_file, service):
            migrated += 1
        else:
            failed += 1
    
    # Resumen
    logger.info("=" * 80)
    logger.info("📊 RESUMEN DE MIGRACIÓN")
    logger.info("=" * 80)
    logger.info(f"✅ Migrados: {migrated}")
    logger.info(f"⏭️  Ya optimizados: {already_optimized}")
    logger.info(f"❌ Fallidos: {failed}")
    logger.info(f"📊 Total: {len(response_files)}")
    logger.info("=" * 80)
    
    if migrated > 0:
        logger.info("✅ Migración completada exitosamente")
    else:
        logger.info("ℹ️  No se requirió migración (todos los archivos ya están optimizados)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Migración interrumpida por el usuario")
    except Exception as e:
        logger.error(f"❌ Error fatal en migración: {e}", exc_info=True)
        sys.exit(1)

