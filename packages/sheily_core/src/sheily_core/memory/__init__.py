import logging

logger = logging.getLogger(__name__)
"""
SISTEMA DE MEMORIA AVANZADO SHEILY - Módulo Principal

Este módulo contiene el sistema completo de memoria híbrida humano-IA:

COMPONENTES PRINCIPALES:
- Procesamiento automático de archivos de texto
- Sistema de atención avanzada neuronal
- Almacenamiento vectorial inteligente
- Recuperación contextual emocional
- Integración con chat y RAG

IMPORTS PRINCIPALES:
- SheilyMemoryV2: Motor principal de memoria híbrida
- SheilyFileProcessor: Procesador automático de archivos
- SheilyMemorySystem: Sistema completo integrado
- MemoryDatabase: Base de datos vectorial funcional

EJEMPLO DE USO:
    from sheily_core.memory import SheilyMemoryV2, SheilyMemorySystem

    # Crear sistema de memoria completo
    memory_system = SheilyMemorySystem()
    memory_system.initialize()

    # Procesar archivos automáticamente
    results = memory_system.process_pending_files()

    # Buscar conocimiento
    memories = memory_system.search_knowledge("explicar física cuántica")
"""

# ==============================================================================
# IMPORTS PRINCIPALES DEL MÓDULO memory
# ==============================================================================

__all__ = [
    "SheilyMemoryV2",
    "SheilyFileProcessor",
    "SheilyMemorySystem",
    "MemoryDatabase",
    "AdvancedAttentionV2",
    "OptimizedVectorStore",
]

# ==============================================================================
# CONFIGURACIÓN DEL MÓDULO
# ==============================================================================
__version__ = "2.0.0"
__author__ = "Sheily AI Team"
__description__ = "Sistema avanzado de memoria híbrida humano-IA con procesamiento automático de archivos"

# ==============================================================================
# IMPORTS CONDICIONALES PARA MEJOR COMPATIBILIDAD
# ==============================================================================

# REAL imports - fail clearly if modules not available
try:
    # Imports principales desde componentes internos
    from ..memory_system_complete import SheilyMemorySystem
    from ..process_files_for_memory import SheilyFileProcessor
    from .core.attention.advanced_attention_v2 import AdvancedAttentionV2
    from .core.database.memory_engine import MemoryDatabase, SheilyMemoryV2
    from .core.storage.optimized_vector_store_v2 import OptimizedVectorStore
    
    logger.info("✅ Memory system components loaded successfully")
    
except ImportError as e:
    # NO FALLBACKS - Fail clearly with helpful error message
    logger.error(
        f"❌ CRITICAL: Memory system components not available: {e}\n"
        f"   Required modules:\n"
        f"   - memory_system_complete (SheilyMemorySystem)\n"
        f"   - process_files_for_memory (SheilyFileProcessor)\n"
        f"   - core.attention.advanced_attention_v2 (AdvancedAttentionV2)\n"
        f"   - core.database.memory_engine (MemoryDatabase, SheilyMemoryV2)\n"
        f"   - core.storage.optimized_vector_store_v2 (OptimizedVectorStore)\n"
        f"   Ensure all memory system modules are properly installed and available."
    )
    
    # Raise exception instead of providing fake fallbacks
    raise ImportError(
        f"Memory system components not available. This is a required dependency. "
        f"Original error: {e}\n"
        f"Please ensure the memory system modules are properly installed."
    ) from e


# ==============================================================================
# FUNCIONES DE UTILIDAD DEL MÓDULO
# ==============================================================================


def create_memory_system(db_path: str = "sheily_core/memory/storage/memory.db"):
    """Crear sistema completo de memoria"""
    return SheilyMemoryV2(db_path)


def create_file_processor():
    """Crear procesador automático de archivos"""
    return SheilyFileProcessor()


def create_integrated_system():
    """Crear sistema de memoria completamente integrado"""
    return SheilyMemorySystem()


# ==============================================================================
# INICIALIZACIÓN DEL MÓDULO
# ==============================================================================


def initialize_memory_system():
    """Inicializar sistema completo de memoria"""
    logger.info(" Inicializando sistema de memoria Sheily...")
    try:
        # Inicializar componentes principales
        memory_engine = create_memory_system()
        file_processor = create_file_processor()
        integrated_system = create_integrated_system()

        logger.info("Sistema de memoria inicializado correctamente")
        return True

    except Exception as e:
        logger.error(f"Error inicializando sistema de memoria: {e}")
        return False


# ==============================================================================
# CONFIGURACIÓN AUTOMÁTICA
# ==============================================================================
# Inicializar sistema automáticamente si se ejecuta directamente
if __name__ == "__main__":
    initialize_memory_system()
