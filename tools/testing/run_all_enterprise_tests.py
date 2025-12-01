#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EJECUTOR DE TESTS ENTERPRISE
============================
Ejecuta todos los tests enterprise del proyecto usando pytest.

USO:
    python tools/testing/run_all_enterprise_tests.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Configurar UTF-8 para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


class EnterpriseTestRunner:
    """Ejecutor de tests enterprise"""

    def __init__(self, project_root: Path = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()
        
        self.tests_dir = self.project_root / "tests" / "enterprise"
        self.results_dir = self.project_root / "tests" / "results" / "enterprise"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "tests_executed": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "duration_seconds": 0.0,
            "test_files": [],
            "errors": []
        }

    def find_enterprise_tests(self) -> List[Path]:
        """Encontrar todos los archivos de test enterprise"""
        if not self.tests_dir.exists():
            print(f"❌ Directorio de tests enterprise no encontrado: {self.tests_dir}")
            return []

        test_files = list(self.tests_dir.glob("test_*.py"))
        print(f"📁 Encontrados {len(test_files)} archivos de test enterprise")
        return test_files

    def run_pytest(self, test_files: List[Path] = None, verbose: bool = True) -> Dict[str, Any]:
        """Ejecutar pytest en los tests enterprise"""
        print("\n" + "=" * 70)
        print("EJECUTANDO TESTS ENTERPRISE")
        print("=" * 70 + "\n")

        if test_files is None:
            test_files = self.find_enterprise_tests()

        if not test_files:
            print("⚠️  No se encontraron archivos de test para ejecutar")
            return {"success": False, "error": "No test files found"}

        # Construir comando pytest
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir),
            "-v",  # Verbose
            "--tb=short",  # Traceback corto
            "--color=yes",  # Colores
        ]

        if verbose:
            cmd.append("-vv")  # Más verbose

        # Agregar reporte JSON si está disponible
        json_report = self.results_dir / f"pytest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            cmd.extend(["--json-report", "--json-report-file", str(json_report)])
        except:
            pass  # Si no está disponible el plugin, continuar sin él

        print(f"🚀 Ejecutando: {' '.join(cmd)}\n")

        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=False,  # Mostrar output en tiempo real
                text=True,
                timeout=3600  # 1 hora máximo
            )
            
            duration = time.time() - start_time
            self.results["duration_seconds"] = round(duration, 2)
            self.results["tests_executed"] = len(test_files)
            self.results["exit_code"] = result.returncode

            if result.returncode == 0:
                print(f"\n✅ Todos los tests pasaron en {duration:.2f} segundos")
                self.results["tests_passed"] = self.results["tests_executed"]
                self.results["success"] = True
            else:
                print(f"\n❌ Algunos tests fallaron (exit code: {result.returncode})")
                self.results["success"] = False

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "duration": duration
            }

        except subprocess.TimeoutExpired:
            print("\n⏱️  Timeout: Los tests tardaron más de 1 hora")
            self.results["errors"].append("Timeout después de 1 hora")
            return {"success": False, "error": "Timeout"}

        except Exception as e:
            print(f"\n❌ Error ejecutando tests: {e}")
            self.results["errors"].append(str(e))
            return {"success": False, "error": str(e)}

    def run_specific_tests(self, test_pattern: str = None):
        """Ejecutar tests específicos por patrón"""
        if test_pattern:
            cmd = [
                sys.executable, "-m", "pytest",
                str(self.tests_dir),
                "-k", test_pattern,
                "-v"
            ]
            print(f"🎯 Ejecutando tests que coinciden con: {test_pattern}\n")
            subprocess.run(cmd, cwd=str(self.project_root))

    def generate_summary(self):
        """Generar resumen de resultados"""
        print("\n" + "=" * 70)
        print("RESUMEN DE RESULTADOS")
        print("=" * 70 + "\n")

        print(f"📊 Tests ejecutados: {self.results['tests_executed']}")
        print(f"✅ Tests pasados: {self.results['tests_passed']}")
        print(f"❌ Tests fallidos: {self.results['tests_failed']}")
        print(f"⏭️  Tests omitidos: {self.results['tests_skipped']}")
        print(f"⏱️  Duración: {self.results['duration_seconds']} segundos")

        if self.results.get("success"):
            print("\n✅ TODOS LOS TESTS ENTERPRISE PASARON")
        else:
            print("\n❌ ALGUNOS TESTS ENTERPRISE FALLARON")

        # Guardar resultados
        results_file = self.results_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados guardados: {results_file}")

    def run_all(self):
        """Ejecutar todos los tests enterprise"""
        print("=" * 70)
        print("EJECUTOR DE TESTS ENTERPRISE")
        print("=" * 70)
        print(f"Proyecto: {self.project_root}")
        print(f"Tests: {self.tests_dir}")
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        test_files = self.find_enterprise_tests()
        
        if not test_files:
            print("\n⚠️  No se encontraron tests enterprise para ejecutar")
            return False

        result = self.run_pytest(test_files)
        self.generate_summary()

        return result.get("success", False)


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ejecutar tests enterprise")
    parser.add_argument(
        "-k", "--pattern",
        help="Ejecutar solo tests que coincidan con el patrón"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Modo verbose"
    )
    
    args = parser.parse_args()

    runner = EnterpriseTestRunner()
    
    if args.pattern:
        runner.run_specific_tests(args.pattern)
    else:
        success = runner.run_all()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
