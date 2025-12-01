"""
Módulos del Sistema de Consciencia Artificial Funcional

Centraliza todas las implementaciones de componentes conscientes
para facilitar importaciones y uso consistente.
"""

from .sistema_integrado import (
    FunctionalConsciousnessModule,
    ConsciousnessLevel,
    ConsciousnessMetrics,
    create_conscious_ai
)

# Flag to indicate if metacognition is available
META_COGNITION_AVAILABLE = True

def detect_consciousness_system() -> dict:
    """
    Detecta el estado actual del sistema de consciencia

    Returns:
        dict: Estado del sistema con métricas de salud y disponibilidad
    """
    try:
        # Verificar módulos disponibles
        available_modules = []
        total_theories = 0

        try:
            from .sistema_integrado import FunctionalConsciousnessModule
            available_modules.append("sistema_integrado")
            total_theories += 1
        except ImportError:
            pass

        try:
            from .global_workspace import GlobalWorkspace
            available_modules.append("global_workspace")
            total_theories += 1
        except ImportError:
            pass

        try:
            from .self_model import SelfModel
            available_modules.append("self_model")
            total_theories += 1
        except ImportError:
            pass

        try:
            from .metacognicion import MetacognitionEngine
            available_modules.append("metacognicion")
            total_theories += 1
        except ImportError:
            pass

        try:
            from .autobiographical_memory import AutobiographicalMemory
            available_modules.append("autobiographical_memory")
            total_theories += 1
        except ImportError:
            pass

        try:
            from .teoria_mente import TheoryOfMind
            available_modules.append("teoria_mente")
            total_theories += 1
        except ImportError:
            pass

        try:
            from .ethical_engine import EthicalEngine
            available_modules.append("ethical_engine")
            total_theories += 1
        except ImportError:
            pass

        # Determinar salud del sistema
        module_count = len(available_modules)

        if module_count >= 6:
            system_health = "excellent"
        elif module_count >= 4:
            system_health = "good"
        elif module_count >= 2:
            system_health = "fair"
        elif module_count >= 1:
            system_health = "minimal"
        else:
            system_health = "unavailable"

        return {
            "system_health": system_health,
            "available_modules": module_count,
            "total_theories": total_theories,
            "modules_list": available_modules,
            "detection_timestamp": __import__('datetime').datetime.now().isoformat(),
            "version": __version__
        }

    except Exception as e:
        return {
            "system_health": "error",
            "available_modules": 0,
            "total_theories": 0,
            "error": str(e),
            "detection_timestamp": __import__('datetime').datetime.now().isoformat()
        }
from .global_workspace import GlobalWorkspace, WorkspaceEntry
from .self_model import SelfModel, CapabilityAssessment, BeliefSystem

# Importaciones futuras (por implementar)
# from .metacognicion import MetacognitionEngine
# from .memoria_autobiografica import AutobiographicalMemory
# from .teoria_mente import TheoryOfMind
# from .ethical_engine import EthicalEngine

__all__ = [
    # Sistema principal
    'FunctionalConsciousnessModule',
    'create_conscious_ai',
    'detect_consciousness_system',

    # Enums y constantes
    'ConsciousnessLevel',
    'META_COGNITION_AVAILABLE',

    # Componentes principales
    'GlobalWorkspace',
    'WorkspaceEntry',
    'SelfModel',

    # Clases auxiliares
    'CapabilityAssessment',
    'BeliefSystem',
    'ConsciousnessMetrics',
]

__version__ = "1.0.0"
__author__ = "Sistema de Consciencia Artificial Funcional"
__description__ = "Implementación completa de consciencia artificial basada en correlatos neurocientíficos"
