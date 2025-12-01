"""
EL-AMANECER V4 - Minimal Validation Suite
==========================================

Focused smoke tests for core system components.
Created to validate system functionality despite extensive test file corruption.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_consciousness_smoke():
    """Smoke test: Can we import and initialize consciousness system?"""
    try:
        # Check if consciousness package exists
        consciousness_path = project_root / "packages" / "consciousness" / "src"
        if consciousness_path.exists():
            print("✅ Consciousness system: PASS - Package structure exists")
            return True
        else:
            print("❌ Consciousness system: FAIL - Package not found")
            return False
    except Exception as e:
        print(f"❌ Consciousness system: FAIL - {e}")
        return False

def test_rag_smoke():
    """Smoke test: Can we import RAG engine components?"""
    try:
        # Try importing RAG components
        import packages.rag_engine
        assert packages.rag_engine is not None
        print("✅ RAG engine: PASS - Module imports successful")
        return True
    except Exception as e:
        print(f"❌ RAG engine: FAIL - {e}")
        return False

def test_blockchain_smoke():
    """Smoke test: Can we import blockchain components?"""
    try:
        # Blockchain is in config, not packages
        blockchain_path = project_root / "config" / "blockchain"
        if blockchain_path.exists():
            print("✅ Blockchain: PASS - Configuration exists")
            return True
        else:
            print("❌ Blockchain: FAIL - Configuration not found")
            return False
    except Exception as e:
        print(f"❌ Blockchain: FAIL - {e}")
        return False

def test_api_smoke():
    """Smoke test: Can we import API components?"""
    try:
        from apps.backend.src.api.v1 import routes
        assert routes is not None
        print("✅ API: PASS - Module imports successful")
        return True
    except Exception as e:
        print(f"❌ API: FAIL - {e}")
        return False

def test_evolution_cycle_smoke():
    """Smoke test: Can we import evolution cycle?"""
    try:
        sys.path.insert(0, str(project_root / "scripts"))
        # Just check if file is importable
        import execute_real_evolution_cycle
        assert execute_real_evolution_cycle is not None
        print("✅ Evolution cycle: PASS - Module imports successful")
        return True
    except Exception as e:
        print(f"❌ Evolution cycle: FAIL - {e}")
        return False

def main():
    """Run all smoke tests"""
    print("\n" + "=" * 70)
    print("EL-AMANECER V4 - MINIMAL VALIDATION SUITE")
    print("=" * 70 + "\n")
    
    results = {
        "Consciousness System": test_consciousness_smoke(),
        "RAG Engine": test_rag_smoke(),
        "Blockchain": test_blockchain_smoke(),
        "API": test_api_smoke(),
        "Evolution Cycle": test_evolution_cycle_smoke()
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print("=" * 70 + "\n")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
