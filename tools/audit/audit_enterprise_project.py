#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUDITORÍA ENTERPRISE DEL PROYECTO
==================================
Audita calidad, tests, seguridad y estándares enterprise del proyecto.

USO:
    python tools/audit/audit_enterprise_project.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Configurar UTF-8 para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


class EnterpriseProjectAuditor:
    """Auditor enterprise del proyecto"""

    def __init__(self, project_root: Path = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "categories": {
                "code_quality": {"status": "pending", "checks": [], "score": 0},
                "testing": {"status": "pending", "checks": [], "score": 0},
                "security": {"status": "pending", "checks": [], "score": 0},
                "documentation": {"status": "pending", "checks": [], "score": 0},
                "dependencies": {"status": "pending", "checks": [], "score": 0},
            },
            "total_score": 0,
            "max_score": 100,
            "critical_issues": [],
            "recommendations": []
        }

    def check_code_quality(self):
        """Verificar calidad del código"""
        print("\n" + "=" * 70)
        print("1. CALIDAD DE CÓDIGO")
        print("=" * 70 + "\n")

        checks = []
        score = 0
        max_score = 20

        # Verificar imports
        try:
            import ast
            python_files = list(self.project_root.rglob("*.py"))
            python_files = [f for f in python_files if "__pycache__" not in str(f)]
            
            total_files = len(python_files)
            valid_files = 0
            syntax_errors = []
            
            for py_file in python_files[:100]:  # Limitar para velocidad
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        ast.parse(f.read())
                    valid_files += 1
                except SyntaxError as e:
                    syntax_errors.append(f"{py_file.name}: {e}")
            
            if total_files > 0:
                valid_percentage = (valid_files / min(total_files, 100)) * 100
                checks.append({
                    "check": "Sintaxis Python válida",
                    "status": "passed" if valid_percentage >= 95 else "warning",
                    "details": f"{valid_files}/{min(total_files, 100)} archivos válidos ({valid_percentage:.1f}%)"
                })
                score += 10 if valid_percentage >= 95 else 5
                
                if syntax_errors:
                    self.results["critical_issues"].extend(syntax_errors[:5])
        except Exception as e:
            checks.append({
                "check": "Sintaxis Python",
                "status": "error",
                "details": f"Error: {e}"
            })

        # Verificar estructura de módulos
        tools_dir = self.project_root / "tools"
        if tools_dir.exists():
            checks.append({
                "check": "Estructura de herramientas",
                "status": "passed",
                "details": "Directorio tools/ presente"
            })
            score += 5
        else:
            checks.append({
                "check": "Estructura de herramientas",
                "status": "failed",
                "details": "Directorio tools/ no encontrado"
            })

        # Verificar tests
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.rglob("test_*.py"))
            checks.append({
                "check": "Archivos de test",
                "status": "passed",
                "details": f"{len(test_files)} archivos de test encontrados"
            })
            score += 5
        else:
            checks.append({
                "check": "Archivos de test",
                "status": "warning",
                "details": "Directorio tests/ no encontrado"
            })

        self.results["categories"]["code_quality"]["checks"] = checks
        self.results["categories"]["code_quality"]["score"] = score
        self.results["categories"]["code_quality"]["status"] = "passed" if score >= 15 else "warning"

        for check in checks:
            status_emoji = "✅" if check["status"] == "passed" else "⚠️" if check["status"] == "warning" else "❌"
            print(f"{status_emoji} {check['check']}: {check['details']}")

    def check_testing(self):
        """Verificar sistema de testing"""
        print("\n" + "=" * 70)
        print("2. TESTING")
        print("=" * 70 + "\n")

        checks = []
        score = 0
        max_score = 20

        # Verificar pytest
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                checks.append({
                    "check": "Pytest instalado",
                    "status": "passed",
                    "details": result.stdout.strip()
                })
                score += 5
            else:
                checks.append({
                    "check": "Pytest instalado",
                    "status": "failed",
                    "details": "Pytest no disponible"
                })
        except Exception:
            checks.append({
                "check": "Pytest instalado",
                "status": "failed",
                "details": "No se pudo verificar pytest"
            })

        # Verificar tests enterprise
        enterprise_tests = self.project_root / "tests" / "enterprise"
        if enterprise_tests.exists():
            test_files = list(enterprise_tests.glob("test_*.py"))
            checks.append({
                "check": "Tests enterprise",
                "status": "passed",
                "details": f"{len(test_files)} archivos de test enterprise"
            })
            score += 10
        else:
            checks.append({
                "check": "Tests enterprise",
                "status": "warning",
                "details": "Directorio tests/enterprise/ no encontrado"
            })

        # Verificar coverage
        try:
            result = subprocess.run(
                [sys.executable, "-m", "coverage", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                checks.append({
                    "check": "Coverage instalado",
                    "status": "passed",
                    "details": "Coverage disponible para análisis"
                })
                score += 5
        except Exception:
            checks.append({
                "check": "Coverage instalado",
                "status": "warning",
                "details": "Coverage no disponible"
            })

        self.results["categories"]["testing"]["checks"] = checks
        self.results["categories"]["testing"]["score"] = score
        self.results["categories"]["testing"]["status"] = "passed" if score >= 15 else "warning"

        for check in checks:
            status_emoji = "✅" if check["status"] == "passed" else "⚠️" if check["status"] == "warning" else "❌"
            print(f"{status_emoji} {check['check']}: {check['details']}")

    def check_security(self):
        """Verificar seguridad"""
        print("\n" + "=" * 70)
        print("3. SEGURIDAD")
        print("=" * 70 + "\n")

        checks = []
        score = 0
        max_score = 20

        # Verificar archivos de seguridad
        security_dir = self.project_root / "tools" / "security"
        if security_dir.exists():
            security_files = list(security_dir.glob("*.py"))
            checks.append({
                "check": "Módulos de seguridad",
                "status": "passed",
                "details": f"{len(security_files)} módulos de seguridad"
            })
            score += 10
        else:
            checks.append({
                "check": "Módulos de seguridad",
                "status": "warning",
                "details": "Directorio tools/security/ no encontrado"
            })

        # Verificar .gitignore
        gitignore = self.project_root / ".gitignore"
        if gitignore.exists():
            checks.append({
                "check": ".gitignore presente",
                "status": "passed",
                "details": "Archivo .gitignore encontrado"
            })
            score += 5
        else:
            checks.append({
                "check": ".gitignore presente",
                "status": "warning",
                "details": "Archivo .gitignore no encontrado"
            })

        # Verificar requirements.txt
        requirements = self.project_root / "requirements.txt"
        if requirements.exists():
            checks.append({
                "check": "requirements.txt presente",
                "status": "passed",
                "details": "Dependencias documentadas"
            })
            score += 5
        else:
            checks.append({
                "check": "requirements.txt presente",
                "status": "warning",
                "details": "Archivo requirements.txt no encontrado"
            })

        self.results["categories"]["security"]["checks"] = checks
        self.results["categories"]["security"]["score"] = score
        self.results["categories"]["security"]["status"] = "passed" if score >= 15 else "warning"

        for check in checks:
            status_emoji = "✅" if check["status"] == "passed" else "⚠️" if check["status"] == "warning" else "❌"
            print(f"{status_emoji} {check['check']}: {check['details']}")

    def check_documentation(self):
        """Verificar documentación"""
        print("\n" + "=" * 70)
        print("4. DOCUMENTACIÓN")
        print("=" * 70 + "\n")

        checks = []
        score = 0
        max_score = 20

        # Verificar README
        readme_files = list(self.project_root.glob("README*.md"))
        if readme_files:
            checks.append({
                "check": "README presente",
                "status": "passed",
                "details": f"{len(readme_files)} archivo(s) README"
            })
            score += 10
        else:
            checks.append({
                "check": "README presente",
                "status": "warning",
                "details": "No se encontraron archivos README"
            })

        # Verificar docs/
        docs_dir = self.project_root / "docs"
        if docs_dir.exists():
            doc_files = list(docs_dir.rglob("*.md"))
            checks.append({
                "check": "Documentación en docs/",
                "status": "passed",
                "details": f"{len(doc_files)} archivos de documentación"
            })
            score += 10
        else:
            checks.append({
                "check": "Documentación en docs/",
                "status": "warning",
                "details": "Directorio docs/ no encontrado"
            })

        self.results["categories"]["documentation"]["checks"] = checks
        self.results["categories"]["documentation"]["score"] = score
        self.results["categories"]["documentation"]["status"] = "passed" if score >= 15 else "warning"

        for check in checks:
            status_emoji = "✅" if check["status"] == "passed" else "⚠️" if check["status"] == "warning" else "❌"
            print(f"{status_emoji} {check['check']}: {check['details']}")

    def check_dependencies(self):
        """Verificar dependencias"""
        print("\n" + "=" * 70)
        print("5. DEPENDENCIAS")
        print("=" * 70 + "\n")

        checks = []
        score = 0
        max_score = 20

        # Verificar requirements.txt
        requirements = self.project_root / "requirements.txt"
        if requirements.exists():
            try:
                with open(requirements, "r", encoding="utf-8") as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                checks.append({
                    "check": "Dependencias documentadas",
                    "status": "passed",
                    "details": f"{len(deps)} dependencias en requirements.txt"
                })
                score += 10
            except Exception as e:
                checks.append({
                    "check": "Dependencias documentadas",
                    "status": "error",
                    "details": f"Error leyendo requirements.txt: {e}"
                })
        else:
            checks.append({
                "check": "Dependencias documentadas",
                "status": "warning",
                "details": "requirements.txt no encontrado"
            })

        # Verificar package.json si existe
        package_json = self.project_root / "package.json"
        if package_json.exists():
            checks.append({
                "check": "Dependencias Node.js",
                "status": "passed",
                "details": "package.json presente"
            })
            score += 10
        else:
            checks.append({
                "check": "Dependencias Node.js",
                "status": "info",
                "details": "package.json no encontrado (opcional)"
            })

        self.results["categories"]["dependencies"]["checks"] = checks
        self.results["categories"]["dependencies"]["score"] = score
        self.results["categories"]["dependencies"]["status"] = "passed" if score >= 15 else "warning"

        for check in checks:
            status_emoji = "✅" if check["status"] == "passed" else "⚠️" if check["status"] == "warning" else "ℹ️"
            print(f"{status_emoji} {check['check']}: {check['details']}")

    def generate_report(self):
        """Generar reporte final"""
        print("\n" + "=" * 70)
        print("REPORTE FINAL")
        print("=" * 70 + "\n")

        total_score = sum(cat["score"] for cat in self.results["categories"].values())
        self.results["total_score"] = total_score

        for category, data in self.results["categories"].items():
            status_emoji = "✅" if data["status"] == "passed" else "⚠️" if data["status"] == "warning" else "❌"
            category_name = category.replace("_", " ").title()
            print(f"{status_emoji} {category_name:20} | Score: {data['score']}/20 | {data['status']}")

        percentage = (total_score / self.results["max_score"]) * 100
        print(f"\n🏆 SCORE TOTAL: {total_score}/{self.results['max_score']} ({percentage:.1f}%)")

        if percentage >= 80:
            print("✅ PROYECTO EN BUEN ESTADO ENTERPRISE")
        elif percentage >= 60:
            print("⚠️ REQUIERE MEJORAS MODERADAS")
        else:
            print("❌ REQUIERE MEJORAS SIGNIFICATIVAS")

        if self.results["critical_issues"]:
            print(f"\n⚠️  Problemas críticos encontrados: {len(self.results['critical_issues'])}")
            for issue in self.results["critical_issues"][:5]:
                print(f"   - {issue}")

        # Guardar reporte
        report_file = self.project_root / "data" / "audit_reports" / f"enterprise_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Reporte guardado: {report_file}")

    def run_audit(self):
        """Ejecutar auditoría completa"""
        print("=" * 70)
        print("AUDITORÍA ENTERPRISE DEL PROYECTO")
        print("=" * 70)
        print(f"Proyecto: {self.project_root}")
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.check_code_quality()
        self.check_testing()
        self.check_security()
        self.check_documentation()
        self.check_dependencies()
        self.generate_report()

        return self.results


def main():
    """Función principal"""
    auditor = EnterpriseProjectAuditor()
    results = auditor.run_audit()
    
    # Exit code basado en score
    percentage = (results["total_score"] / results["max_score"]) * 100
    if percentage < 60:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
