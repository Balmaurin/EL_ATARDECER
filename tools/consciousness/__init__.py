"""
Módulo de Consciencia - Tools
=============================
Herramientas para verificación y análisis del sistema de consciencia.
"""

__version__ = "1.0.0"
__author__ = "Sheily AI Team"

# Importar funciones principales si están disponibles
try:
    from .check_self_awareness import check_self_awareness
    __all__ = ["check_self_awareness"]
except ImportError:
    __all__ = []
