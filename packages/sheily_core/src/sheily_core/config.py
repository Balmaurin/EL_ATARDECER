"""
Config bridge for sheily_core
Connects to the unified settings from apps/backend
REAL IMPLEMENTATION - No fallbacks, fails if config cannot be loaded
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Detectar ruta base del proyecto de forma dinámica
def _find_project_root() -> Path:
    """Find project root by looking for markers like .git, pyproject.toml, etc."""
    current = Path(__file__).resolve()
    
    # Buscar hacia arriba desde el archivo actual
    for parent in current.parents:
        # Marcadores de raíz del proyecto
        markers = ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt', 'apps']
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    # Si no se encuentra, usar variable de entorno
    env_root = os.getenv("SHEILY_PROJECT_ROOT")
    if env_root:
        root_path = Path(env_root)
        if root_path.exists():
            return root_path
    
    # Último recurso: asumir estructura estándar
    # Buscar 'apps' o 'packages' en algún padre
    for parent in current.parents:
        if (parent / "apps").exists() or (parent / "packages").exists():
            return parent
    
    raise RuntimeError(
        "Could not determine project root. Set SHEILY_PROJECT_ROOT environment variable "
        "or ensure project structure is correct."
    )

# Obtener ruta del backend de forma dinámica
_project_root = _find_project_root()
_backend_config_path = _project_root / "apps" / "backend" / "src"

if not _backend_config_path.exists():
    # Intentar rutas alternativas
    alt_paths = [
        _project_root / "backend" / "src",
        _project_root / "apps" / "backend",
        Path(os.getenv("SHEILY_BACKEND_PATH", "")),
    ]
    
    for alt_path in alt_paths:
        if alt_path and alt_path.exists():
            _backend_config_path = alt_path
            break
    else:
        raise RuntimeError(
            f"Backend path not found. Expected: {_backend_config_path}. "
            f"Set SHEILY_BACKEND_PATH environment variable."
        )

if str(_backend_config_path) not in sys.path:
    sys.path.insert(0, str(_backend_config_path))

# Cargar configuración real - SIN FALLBACKS
_settings_file = _backend_config_path / "config" / "settings.py"

if not _settings_file.exists():
    raise FileNotFoundError(
        f"Settings file not found: {_settings_file}. "
        f"Ensure backend configuration is properly set up."
    )

try:
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("backend_settings", _settings_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {_settings_file}")
    
    backend_settings_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend_settings_module)
    
    if not hasattr(backend_settings_module, 'settings'):
        raise AttributeError(
            f"Module {_settings_file} does not have 'settings' attribute. "
            f"Ensure settings are properly defined."
        )
    
    unified_settings = backend_settings_module.settings
    
    # Usar el settings unificado como Config
    Config = type(unified_settings)
    
    def get_config():
        """Get global unified config instance - REAL IMPLEMENTATION"""
        if unified_settings is None:
            raise RuntimeError("Configuration not initialized. Check backend settings.")
        return unified_settings
        
    logger.info(f"✅ Configuration loaded successfully from {_settings_file}")
        
except Exception as e:
    logger.error(f"❌ CRITICAL: Failed to load configuration: {e}", exc_info=True)
    raise RuntimeError(
        f"Configuration loading failed. This is a critical error. "
        f"Details: {e}. "
        f"Ensure backend configuration is properly set up at {_settings_file}"
    ) from e
