"""
EL-AMANECER-V4 - PYTEST CONFIGURATION
======================================

Configuración global de pytest para todos los tests.
Arregla problemas de importación y configura fixtures compartidas.
"""

import sys
import os
from pathlib import Path
import pytest
from collections import defaultdict

# ====================================================================
# PATH CONFIGURATION
# ====================================================================

# Agregar el directorio raíz al PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "apps"))
sys.path.insert(0, str(ROOT_DIR / "packages"))
sys.path.insert(0, str(ROOT_DIR / "tools"))

# Configurar PYTHONPATH en el entorno
os.environ["PYTHONPATH"] = f"{ROOT_DIR};{ROOT_DIR / 'apps'};{ROOT_DIR / 'packages'};{ROOT_DIR / 'tools'}"

print(f"[OK] PYTHONPATH configurado: {ROOT_DIR}")


# ====================================================================
# PYTEST CONFIGURATION HOOKS
# ====================================================================

def pytest_configure(config):
    """Configuración inicial de pytest"""
    config.addinivalue_line(
        "markers", "enterprise: Enterprise-level tests"
    )
    config.addinivalue_line(
        "markers", "chaos: Chaos engineering tests"
    )
    config.addinivalue_line(
        "markers", "slow: Mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "security: Security and penetration tests"
    )
    config.addinivalue_line(
        "markers", "ui: UI/UX validation tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modificar items de test durante la colección"""
    for item in items:
        # Agregar marker 'enterprise' a todos los tests en tests/enterprise/
        if "enterprise" in str(item.fspath):
            item.add_marker(pytest.mark.enterprise)
        
        # Agregar markers específicos basados en el nombre del archivo
        if "chaos" in str(item.fspath):
            item.add_marker(pytest.mark.chaos)
            item.add_marker(pytest.mark.slow)
        
        if "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        
        if "ui" in str(item.fspath) or "frontend" in str(item.fspath):
            item.add_marker(pytest.mark.ui)
        
        if "performance" in str(item.fspath) or "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


# ====================================================================
# SHARED FIXTURES
# ====================================================================

@pytest.fixture(scope="session")
def test_config():
    """Configuración de test compartida"""
    return {
        "base_url": "http://localhost:8000",
        "frontend_url": "http://localhost:3000",
        "timeout": 30,
        "retry_attempts": 3
    }


@pytest.fixture(scope="function")
def temp_test_dir(tmp_path):
    """Directorio temporal para tests"""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(scope="function")
def mock_defaultdict():
    """Fixture para defaultdict (arregla importación faltante)"""
    return defaultdict


# ====================================================================
# ENTERPRISE TEST FIXTURES
# ====================================================================

@pytest.fixture(scope="session")
def enterprise_test_environment():
    """Configurar entorno de testing empresarial"""
    # Crear directorios necesarios
    results_dir = ROOT_DIR / "tests" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    (results_dir / "performance").mkdir(exist_ok=True)
    (results_dir / "security").mkdir(exist_ok=True)
    (results_dir / "chaos").mkdir(exist_ok=True)
    
    return {
        "results_dir": results_dir,
        "root_dir": ROOT_DIR
    }


print("[INIT] EL-AMANECER-V4 PYTEST CONFIGURATION LOADED")