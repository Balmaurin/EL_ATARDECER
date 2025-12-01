"""
SHEILY_CORE - Núcleo Principal Del Sistema Sheily Ai Con Módulos Especializados

Este módulo forma parte del ecosistema Sheily AI y proporciona funcionalidades especializadas para:

FUNCIONALIDADES PRINCIPALES:
- Configuración central, APIs principales y enrutamiento del sistema
- Integración perfecta con otros módulos del sistema
- Configuración flexible y extensible
- Documentación técnica completa incluida

INTEGRACIÓN CON EL SISTEMA:
- Compatible con arquitectura modular de Sheily AI
- Sigue estándares de codificación profesionales
- Incluye tests y validación automática
- Soporte para múltiples entornos (desarrollo, producción)

USO TÍPICO:
    from sheily_core import SheilyApp
    # Ejemplo de uso del módulo sheily_core
"""

# ==============================================================================
# IMPORTS PRINCIPALES DEL MÓDULO SHEILY_CORE
# ==============================================================================

# Imports esenciales del módulo
__all__: list[str] = [
    # Agregar aquí las clases y funciones principales que se exportan
    # Ejemplo: "MainClass", "important_function", "CoreComponent"
]

# ==============================================================================
# CONFIGURACIÓN DEL MÓDULO
# ==============================================================================

# Versión del módulo
__version__ = "2.0.0"

# Información del módulo
__author__ = "Sheily AI Team"
__description__ = "Configuración central, APIs principales y enrutamiento del sistema"

# ==============================================================================
# IMPORTS CONDICIONALES PARA MEJOR COMPATIBILIDAD
# ==============================================================================

# REAL imports - export actual components that exist
import logging

logger = logging.getLogger(__name__)

# Import core components that actually exist
try:
    from .config import get_config, Config
    __all__.append("get_config")
    __all__.append("Config")
except ImportError as e:
    logger.warning(f"Config module not available: {e}")

try:
    from .logger import get_logger, ContextLogger
    __all__.extend(["get_logger", "ContextLogger"])
except ImportError as e:
    logger.warning(f"Logger module not available: {e}")

try:
    from .safety import get_security_monitor, SecurityMonitor
    __all__.extend(["get_security_monitor", "SecurityMonitor"])
except ImportError as e:
    logger.warning(f"Safety module not available: {e}")

# Import agent system if available
try:
    from .agents import (
        get_available_agents,
        get_system_status,
        get_system_info,
        create_agent_coordinator,
    )
    __all__.extend([
        "get_available_agents",
        "get_system_status", 
        "get_system_info",
        "create_agent_coordinator",
    ])
except ImportError as e:
    logger.debug(f"Agent system not available: {e}")

# ==============================================================================
# INICIALIZACIÓN DEL MÓDULO
# ==============================================================================

def get_main_component():
    """Obtener componente principal del módulo - REAL implementation"""
    # Return actual config as main component
    try:
        from .config import get_config
        return get_config()
    except ImportError:
        raise RuntimeError(
            "Main component (config) not available. "
            "Ensure sheily_core is properly installed."
        )


# ==============================================================================
