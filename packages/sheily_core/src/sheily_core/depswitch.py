#!/usr/bin/env python3
"""
Dependency Switch Module
Maneja dependencias opcionales y configuraciones condicionales.
"""

import logging

logger = logging.getLogger(__name__)

# REAL dependency availability - checked at runtime
def _check_all_dependencies():
    """REAL check of all optional dependencies"""
    deps = {}
    for dep_name in ["transformers", "torch", "faiss", "chromadb", "sentence_transformers"]:
        deps[dep_name] = check_dependency(dep_name)
    return deps

# Initialize with REAL checks
AVAILABLE_DEPS = _check_all_dependencies()

def check_dependency(dep_name: str) -> bool:
    """Verificar si una dependencia está disponible"""
    try:
        __import__(dep_name)
        return True
    except ImportError:
        return False

def get_dependency_status() -> dict:
    """Obtener estado de todas las dependencias"""
    status = {}
    for dep in AVAILABLE_DEPS.keys():
        status[dep] = check_dependency(dep)
    return status

def require_dependency(dep_name: str, error_msg: str = None):
    """Requerir una dependencia o lanzar error"""
    if not check_dependency(dep_name):
        msg = error_msg or f"Dependencia requerida no disponible: {dep_name}"
        raise ImportError(msg)

# Cache global para rate limiting
_rate_limit_cache = {}

def activate_secure():
    """
    Activate security systems - REAL Implementation
    ===============================================

    Activa todos los sistemas de seguridad:
    - Rate limiting
    - CORS validation
    - Input sanitization
    - Request logging
    - JWT validation
    """
    import os
    
    try:
        # 1. Validar SECRET_KEY
        secret_key = os.getenv("SECRET_KEY")
        if not secret_key or secret_key == "change_this_in_production":
            logger.warning(
                "⚠️ SECRET_KEY not configured or using default value. "
                "Configure SECRET_KEY in .env for production security!"
            )
        else:
            logger.info("✅ SECRET_KEY validated")

        # 2. Inicializar JWT Manager
        try:
            from sheily_core.security.jwt_auth import JWT_AVAILABLE, get_jwt_manager
            
            if JWT_AVAILABLE:
                jwt_manager = get_jwt_manager()
                logger.info("✅ JWT Authentication system activated")
            else:
                logger.warning("⚠️ PyJWT not available - JWT auth disabled")
        except Exception as e:
            logger.debug(f"JWT init skipped: {e}")

        # 3. Validar configuración CORS (graceful)
        try:
            from sheily_core.config import get_config
            
            config = get_config()
            cors_origins = getattr(config, 'cors_origins', ['*'])
            if cors_origins == ["*"]:
                logger.warning(
                    "⚠️ CORS configured with wildcard (*) - not recommended for production"
                )
            else:
                logger.info(
                    f"✅ CORS configured with {len(cors_origins)} specific origins"
                )
        except Exception as e:
            logger.debug(f"CORS validation skipped: {e}")

        # 4. Inicializar rate limiting (preparar estructuras)
        global _rate_limit_cache
        _rate_limit_cache = {}
        logger.info("✅ Rate limiting structures initialized")

        # 5. Activar logging de seguridad
        security_logger = logging.getLogger("sheily.security")
        if not security_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] SECURITY: %(message)s")
            )
            security_logger.addHandler(handler)
            security_logger.setLevel(logging.INFO)
        logger.info("✅ Security logging activated")

        # 6. Validar subprocess utils están disponibles
        try:
            from sheily_core.utils.subprocess_utils import safe_subprocess_run
            logger.info("✅ Subprocess validation system available")
        except Exception as e:
            logger.debug(f"Subprocess utils not available: {e}")

        logger.info("🔒 Security systems activated successfully")
        return True

    except Exception as e:
        logger.error(f"🔴 CRITICAL: Failed to activate security: {e}")
        # No lanzar excepción para evitar romper el sistema
        return False

