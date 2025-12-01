#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUDITORÍA ESTRUCTURAL DEL PROYECTO
===================================
Audita la estructura del proyecto: archivos, directorios, tamaños, organización.

USO:
    python tools/audit/audit_project_structure.py
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configurar UTF-8 para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


class ProjectStructureAuditor:
    """Auditor de estructura del proyecto"""

    def __init__(self, project_root: Path = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "directories": {},
            "file_types": defaultdict(int),
            "total_files": 0,
            "total_size_mb": 0.0,
            "structure_issues": [],
            "statistics": {}
        }

    def audit_directories(self):
        """Auditar directorios principales"""
        print("\n" + "=" * 70)
        print("1. ESTRUCTURA DE DIRECTORIOS")
        print("=" * 70 + "\n")

        main_dirs = [
            "all-Branches",
            "packages",
            "tools",
            "scripts",
            "tests",
            "docs",
            "config",
            "data",
            "logs",
            "memory",
            "var",
        ]

        for dir_name in main_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                try:
                    files = list(dir_path.rglob("*"))
                    file_count = sum(1 for f in files if f.is_file())
                    dir_count = sum(1 for f in files if f.is_dir())
                    size_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
                    
                    self.results["directories"][dir_name] = {
                        "exists": True,
                        "files": file_count,
                        "directories": dir_count,
                        "size_mb": round(size_mb, 2)
                    }
                    
                    print(f"✅ {dir_name:20} | {file_count:>5} archivos | {dir_count:>4} dirs | {size_mb:>8.1f} MB")
                except Exception as e:
                    print(f"❌ {dir_name:20} | Error: {e}")
                    self.results["structure_issues"].append(f"Error accediendo {dir_name}: {e}")
            else:
                print(f"⚠️  {dir_name:20} | NO ENCONTRADO")
                self.results["directories"][dir_name] = {"exists": False}

    def audit_file_types(self):
        """Auditar tipos de archivos"""
        print("\n" + "=" * 70)
        print("2. TIPOS DE ARCHIVOS")
        print("=" * 70 + "\n")

        extensions = defaultdict(int)
        total_size = 0
        total_files = 0

        skip_dirs = {
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            "dist", "build", ".pytest_cache", ".mypy_cache", ".idea"
        }

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                # Saltar directorios especiales
                if any(skip in str(file_path) for skip in skip_dirs):
                    continue
                
                try:
                    ext = file_path.suffix.lower() or ".sin_extension"
                    extensions[ext] += 1
                    total_files += 1
                    total_size += file_path.stat().st_size
                except Exception:
                    continue

        # Ordenar por cantidad
        sorted_exts = sorted(extensions.items(), key=lambda x: x[1], reverse=True)

        for ext, count in sorted_exts[:20]:  # Top 20
            ext_size = sum(
                f.stat().st_size for f in self.project_root.rglob(f"*{ext}")
                if f.is_file() and not any(skip in str(f) for skip in skip_dirs)
            ) / (1024 * 1024)
            print(f"   {ext:15} | {count:>6} archivos | {ext_size:>8.1f} MB")
            self.results["file_types"][ext] = count

        self.results["total_files"] = total_files
        self.results["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        print(f"\n   TOTAL: {total_files} archivos | {self.results['total_size_mb']} MB")

    def audit_branches(self):
        """Auditar ramas de all-Branches si existe"""
        branches_dir = self.project_root / "all-Branches"
        if not branches_dir.exists():
            return

        print("\n" + "=" * 70)
        print("3. RAMAS (all-Branches)")
        print("=" * 70 + "\n")

        branches = []
        for item in branches_dir.iterdir():
            if item.is_dir():
                try:
                    files = list(item.rglob("*"))
                    file_count = sum(1 for f in files if f.is_file())
                    size_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
                    branches.append({
                        "name": item.name,
                        "files": file_count,
                        "size_mb": round(size_mb, 2)
                    })
                    print(f"   📁 {item.name:30} | {file_count:>5} archivos | {size_mb:>8.1f} MB")
                except Exception as e:
                    print(f"   ❌ {item.name:30} | Error: {e}")

        self.results["branches"] = branches
        print(f"\n   Total ramas: {len(branches)}")

    def generate_report(self):
        """Generar reporte final"""
        print("\n" + "=" * 70)
        print("4. REPORTE FINAL")
        print("=" * 70 + "\n")

        # Calcular estadísticas
        self.results["statistics"] = {
            "directories_found": sum(1 for d in self.results["directories"].values() if d.get("exists")),
            "total_directories_checked": len(self.results["directories"]),
            "file_types_count": len(self.results["file_types"]),
            "issues_count": len(self.results["structure_issues"])
        }

        print(f"✅ Directorios encontrados: {self.results['statistics']['directories_found']}/{self.results['statistics']['total_directories_checked']}")
        print(f"📊 Tipos de archivo únicos: {self.results['statistics']['file_types_count']}")
        print(f"📁 Total archivos: {self.results['total_files']}")
        print(f"💾 Tamaño total: {self.results['total_size_mb']} MB")
        
        if self.results["structure_issues"]:
            print(f"\n⚠️  Problemas encontrados: {len(self.results['structure_issues'])}")
            for issue in self.results["structure_issues"][:5]:
                print(f"   - {issue}")

        # Guardar reporte
        report_file = self.project_root / "data" / "audit_reports" / f"structure_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Reporte guardado: {report_file}")

    def run_audit(self):
        """Ejecutar auditoría completa"""
        print("=" * 70)
        print("AUDITORÍA ESTRUCTURAL DEL PROYECTO")
        print("=" * 70)
        print(f"Proyecto: {self.project_root}")
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.audit_directories()
        self.audit_file_types()
        self.audit_branches()
        self.generate_report()

        return self.results


def main():
    """Función principal"""
    auditor = ProjectStructureAuditor()
    results = auditor.run_audit()
    
    # Exit code basado en problemas encontrados
    if results["structure_issues"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
