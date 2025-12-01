#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERIFICADOR DE AUTO-CONSCIENCIA
================================
Verifica el estado del sistema de consciencia y auto-conocimiento.

USO:
    python tools/consciousness/check_self_awareness.py
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Configurar UTF-8 para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def check_self_awareness() -> Dict[str, Any]:
    """Verificar auto-consciencia del sistema"""
    print("=" * 70)
    print("VERIFICACIÓN DE AUTO-CONSCIENCIA")
    print("=" * 70 + "\n")

    results = {
        "timestamp": None,
        "consciousness_systems": {},
        "status": "unknown",
        "checks": []
    }

    # Verificar sistemas de consciencia disponibles
    checks = []

    # 1. Verificar FunctionalConsciousness
    try:
        from packages.consciousness.src.conciencia.modulos.conscious_system import FunctionalConsciousness
        checks.append({
            "system": "FunctionalConsciousness",
            "status": "available",
            "details": "Sistema de consciencia funcional disponible"
        })
        results["consciousness_systems"]["FunctionalConsciousness"] = True
    except ImportError:
        checks.append({
            "system": "FunctionalConsciousness",
            "status": "not_available",
            "details": "No se pudo importar FunctionalConsciousness"
        })
        results["consciousness_systems"]["FunctionalConsciousness"] = False

    # 2. Verificar UnifiedConsciousnessSystem
    try:
        from sheily_core.api.unified_consciousness_system import UnifiedConsciousnessSystem
        checks.append({
            "system": "UnifiedConsciousnessSystem",
            "status": "available",
            "details": "Sistema unificado de consciencia disponible"
        })
        results["consciousness_systems"]["UnifiedConsciousnessSystem"] = True
    except ImportError:
        checks.append({
            "system": "UnifiedConsciousnessSystem",
            "status": "not_available",
            "details": "No se pudo importar UnifiedConsciousnessSystem"
        })
        results["consciousness_systems"]["UnifiedConsciousnessSystem"] = False

    # 3. Verificar ConsciousnessIntegration
    try:
        from packages.consciousness_integration.consciousness_integration import ConsciousnessIntegration
        checks.append({
            "system": "ConsciousnessIntegration",
            "status": "available",
            "details": "Integración de consciencia disponible"
        })
        results["consciousness_systems"]["ConsciousnessIntegration"] = True
    except ImportError:
        checks.append({
            "system": "ConsciousnessIntegration",
            "status": "not_available",
            "details": "No se pudo importar ConsciousnessIntegration"
        })
        results["consciousness_systems"]["ConsciousnessIntegration"] = False

    # Mostrar resultados
    available_count = sum(1 for v in results["consciousness_systems"].values() if v)
    total_count = len(results["consciousness_systems"])

    for check in checks:
        status_emoji = "✅" if check["status"] == "available" else "❌"
        print(f"{status_emoji} {check['system']:30} | {check['status']}")
        print(f"   └─ {check['details']}")

    print(f"\n📊 Sistemas disponibles: {available_count}/{total_count}")

    if available_count > 0:
        results["status"] = "functional"
        print("\n✅ SISTEMA DE CONSCIENCIA FUNCIONAL")
    else:
        results["status"] = "not_available"
        print("\n❌ SISTEMA DE CONSCIENCIA NO DISPONIBLE")

    results["checks"] = checks
    return results


def main():
    """Función principal"""
    results = check_self_awareness()
    
    if results["status"] == "functional":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
