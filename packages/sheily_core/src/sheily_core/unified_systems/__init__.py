"""
Sistemas Unificados de Sheily
============================

Arquitectura unificada que integra todos los sistemas de IA, memoria,
seguridad y blockchain en una plataforma coherente.
"""

# from .unified_system_core import UnifiedSystemCore, get_unified_system  # Módulo no encontrado - comentado temporalmente

try:  # Optional: unified_master_system has many heavy dependencies
    from .unified_master_system import UnifiedMasterSystem
except ImportError as exc:  # pragma: no cover - exercised when deps missing

    def _unified_master_system_placeholder(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "UnifiedMasterSystem is unavailable because optional dependencies "
            f"could not be imported: {exc}"
        )

    UnifiedMasterSystem = _unified_master_system_placeholder  # type: ignore

__all__ = [
    "get_unified_system",
    "UnifiedSystemCore",
    "UnifiedMasterSystem",
]
