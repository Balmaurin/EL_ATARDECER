"""

SISTEMA DE CHAT AVANZADO SHEILY - Módulo Principal

Este módulo contiene el sistema completo de conversación inteligente:

COMPONENTES PRINCIPALES:
- Motor de chat funcional con detección de ramas
- Sistema de conversación unificado
- Chat ultra-rápido optimizado
- Adaptadores de memoria conversacional
- Integración con modelos GGUF

IMPORTS PRINCIPALES:
- ChatEngine: Motor principal de conversación funcional
- UnifiedChatSystem: Sistema unificado de chat
- FastChatV3: Chat ultra-rápido optimizado
- SheilyChatMemoryAdapter: Adaptador de memoria conversacional

EJEMPLO DE USO:
    from sheily_core.chat import ChatEngine, UnifiedChatSystem

    # Crear motor de chat avanzado
    chat_engine = ChatEngine()
    response = chat_engine.process_query("¿Qué es la física cuántica?")

    # Usar sistema unificado
    chat_system = UnifiedChatSystem()
    conversation = chat_system.start_conversation()
"""

import logging

# ==============================================================================
# IMPORTS PRINCIPALES DEL MÓDULO CHAT
# ==============================================================================

__all__ = [
    "UnifiedChatSystem",
    "FastChatV3",
]

# ==============================================================================
# CONFIGURACIÓN DEL MÓDULO
# ==============================================================================

__version__ = "2.0.0"
__author__ = "Sheily AI Team"
__description__ = "Sistema completo de conversación inteligente con detección automática de ramas académicas"

# ==============================================================================
# IMPORTS REALES - Sin comentarios, solo lo que está disponible
# ==============================================================================

logger = logging.getLogger(__name__)

# Imports reales - fail if not available
try:
    from .sheily_fast_chat_v3 import FastChatV3
    from .unified_chat_system import UnifiedChatSystem
    logger.info("✅ Chat system components loaded successfully")
except ImportError as e:
    logger.error(f"❌ CRITICAL: Chat system components not available: {e}")
    raise ImportError(
        f"Chat system components not available. Required modules:\n"
        f"  - sheily_fast_chat_v3 (FastChatV3)\n"
        f"  - unified_chat_system (UnifiedChatSystem)\n"
        f"Original error: {e}"
    ) from e

# Optional components - try to import but don't fail if not available
try:
    from .chat_engine import ChatContext, ChatEngine, ChatMessage, ChatResponse
    __all__.extend(["ChatEngine", "ChatContext", "ChatMessage", "ChatResponse"])
    logger.info("✅ ChatEngine components available")
except ImportError:
    logger.debug("ChatEngine components not available (optional)")

try:
    from .sheily_chat_memory_adapter import SheilyChatMemoryAdapter
    __all__.append("SheilyChatMemoryAdapter")
    logger.info("✅ Chat memory adapter available")
except ImportError:
    logger.debug("Chat memory adapter not available (optional)")


# ==============================================================================
# FUNCIONES DE UTILIDAD DEL MÓDULO
# ==============================================================================

def create_chat_engine(config_path: str = None):
    """Crear motor de chat con configuración opcional - REAL implementation"""
    try:
        from .chat_engine import ChatEngine
        return ChatEngine() if config_path is None else ChatEngine(config_path)
    except NameError:
        raise RuntimeError(
            "ChatEngine not available. Install chat_engine module or use UnifiedChatSystem instead."
        )


def create_unified_chat():
    """Crear sistema unificado de conversación"""
    return UnifiedChatSystem()


def create_fast_chat():
    """Crear chat ultra-rápido optimizado"""
    return FastChatV3()


# ==============================================================================
# INICIALIZACIÓN DEL MÓDULO
# ==============================================================================


def initialize_chat_system():
    """Inicializar sistema completo de chat"""
    logger.info("Inicializando sistema de chat Sheily...")
    try:
        # Inicializar componentes principales
        # chat_engine = create_chat_engine()
        unified_chat = create_unified_chat()
        fast_chat = create_fast_chat()

        logger.info("Sistema de chat inicializado correctamente")
        return True

    except Exception as e:
        logger.error(f"Error inicializando sistema de chat: {e}")
        return False


# ==============================================================================
# CONFIGURACIÓN AUTOMÁTICA
# ==============================================================================

# Inicializar sistema automáticamente si se ejecuta directamente
if __name__ == "__main__":
    initialize_chat_system()
