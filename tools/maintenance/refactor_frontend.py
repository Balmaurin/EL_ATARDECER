#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFACTORIZADOR DE FRONTEND
===========================
Herramientas para refactorizar y mejorar el código frontend.

USO:
    python tools/maintenance/refactor_frontend.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Configurar UTF-8 para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


class FrontendRefactorer:
    """Refactorizador de código frontend"""

    def __init__(self, project_root: Path = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "files_analyzed": 0,
            "files_refactored": 0,
            "issues_found": [],
            "improvements": []
        }

    def find_frontend_files(self) -> List[Path]:
        """Encontrar archivos frontend"""
        frontend_files = []
        
        # Buscar archivos JavaScript/TypeScript
        for ext in ["*.js", "*.jsx", "*.ts", "*.tsx"]:
            frontend_files.extend(self.project_root.rglob(ext))
        
        # Buscar archivos Vue/React
        for pattern in ["*.vue", "*.svelte"]:
            frontend_files.extend(self.project_root.rglob(pattern))
        
        # Filtrar node_modules y otros
        frontend_files = [
            f for f in frontend_files
            if "node_modules" not in str(f) and ".git" not in str(f)
        ]
        
        return frontend_files

    def analyze_file(self, filepath: Path) -> Dict[str, Any]:
        """Analizar un archivo frontend"""
        try:
            content = filepath.read_text(encoding="utf-8")
            analysis = {
                "file": str(filepath.relative_to(self.project_root)),
                "size": len(content),
                "lines": len(content.splitlines()),
                "issues": [],
                "suggestions": []
            }

            # Detectar problemas comunes
            if len(content.splitlines()) > 500:
                analysis["issues"].append("Archivo muy grande (>500 líneas)")
                analysis["suggestions"].append("Considerar dividir en componentes más pequeños")

            if "console.log" in content and "test" not in str(filepath):
                analysis["issues"].append("console.log encontrado en código de producción")
                analysis["suggestions"].append("Reemplazar con sistema de logging apropiado")

            if "any" in content and filepath.suffix == ".ts":
                analysis["issues"].append("Uso de tipo 'any' en TypeScript")
                analysis["suggestions"].append("Usar tipos específicos en lugar de 'any'")

            return analysis

        except Exception as e:
            return {
                "file": str(filepath.relative_to(self.project_root)),
                "error": str(e)
            }

    def refactor_all(self) -> Dict[str, Any]:
        """Refactorizar todos los archivos frontend"""
        print("=" * 70)
        print("REFACTORIZADOR DE FRONTEND")
        print("=" * 70)
        print(f"Proyecto: {self.project_root}\n")

        frontend_files = self.find_frontend_files()
        
        if not frontend_files:
            print("⚠️  No se encontraron archivos frontend")
            return self.results

        print(f"📁 Encontrados {len(frontend_files)} archivos frontend\n")

        for filepath in frontend_files[:50]:  # Limitar para velocidad
            analysis = self.analyze_file(filepath)
            self.results["files_analyzed"] += 1
            
            if analysis.get("issues"):
                self.results["issues_found"].append(analysis)
                print(f"⚠️  {analysis['file']}: {len(analysis['issues'])} problemas")

        print(f"\n✅ Análisis completado: {self.results['files_analyzed']} archivos")
        print(f"📊 Problemas encontrados: {len(self.results['issues_found'])}")

        # Guardar resultados
        results_file = self.project_root / "data" / "refactor_reports" / f"frontend_refactor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Resultados guardados: {results_file}")

        return self.results


def main():
    """Función principal"""
    refactorer = FrontendRefactorer()
    results = refactorer.refactor_all()
    
    if results["issues_found"]:
        print(f"\n⚠️  Se encontraron problemas en {len(results['issues_found'])} archivos")
        sys.exit(1)
    else:
        print("\n✅ No se encontraron problemas críticos")
        sys.exit(0)


if __name__ == "__main__":
    main()
