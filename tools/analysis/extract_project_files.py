#!/usr/bin/env python3
"""
Extract Project Files - Extracción Real de Archivos del Proyecto
==================================================================

Extrae y analiza todos los archivos del proyecto para generar
datos estructurados para entrenamiento y análisis.
"""

import ast
import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProjectFileExtractor:
    """Extractor real de archivos del proyecto con análisis completo"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # Calcular desde este archivo
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.output_dir = self.project_root / "data" / "extracted_project_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Estadísticas
        self.stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "errors": []
        }
        
        logger.info(f"📁 Project root: {self.project_root}")
        logger.info(f"📂 Output directory: {self.output_dir}")

    def extract_all(self, extensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extrae todos los archivos del proyecto.
        
        Args:
            extensions: Lista de extensiones a procesar. Si None, procesa todas.
        """
        if extensions is None:
            extensions = [".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml", ".md"]
        
        logger.info(f"🚀 Starting extraction of project files...")
        logger.info(f"   Extensions: {', '.join(extensions)}")
        
        extracted_data = {
            "project_root": str(self.project_root),
            "extraction_timestamp": datetime.now().isoformat(),
            "extensions": extensions,
            "files": {},
            "statistics": {},
            "weights_dataset": {
                "neural_patterns": {},
                "complexity_weights": {}
            }
        }
        
        # Procesar archivos
        for ext in extensions:
            files = list(self.project_root.rglob(f"*{ext}"))
            logger.info(f"   Found {len(files)} {ext} files")
            
            for file_path in files:
                # Saltar directorios especiales
                if any(skip in str(file_path) for skip in [
                    "__pycache__", "node_modules", ".git", ".venv", "venv",
                    "dist", "build", ".pytest_cache", ".mypy_cache"
                ]):
                    continue
                
                try:
                    file_data = self._extract_file(file_path, ext)
                    if file_data:
                        relative_path = str(file_path.relative_to(self.project_root))
                        extracted_data["files"][relative_path] = file_data
                        self.stats["files_processed"] += 1
                except Exception as e:
                    self.stats["errors"].append({
                        "file": str(file_path),
                        "error": str(e)
                    })
                    self.stats["files_skipped"] += 1
                    logger.warning(f"⚠️ Error processing {file_path}: {e}")
        
        # Calcular estadísticas
        extracted_data["statistics"] = self._calculate_statistics(extracted_data["files"])
        
        # Generar weights dataset
        extracted_data["weights_dataset"] = self._generate_weights_dataset(extracted_data["files"])
        
        # Guardar resultado
        output_file = self.output_dir / f"project_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Extraction completed: {self.stats['files_processed']} files")
        logger.info(f"💾 Saved to: {output_file}")
        
        return extracted_data

    def _extract_file(self, file_path: Path, extension: str) -> Optional[Dict[str, Any]]:
        """Extraer datos de un archivo individual"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return None
        
        file_data = {
            "path": str(file_path.relative_to(self.project_root)),
            "extension": extension,
            "size_bytes": file_path.stat().st_size,
            "lines": content.count("\n") + 1,
            "characters": len(content),
            "hash": hashlib.md5(content.encode()).hexdigest(),
            "complexity_metrics": {}
        }
        
        # Análisis específico por tipo
        if extension == ".py":
            py_analysis = self._analyze_python_file(content, file_path)
            file_data.update(py_analysis)
        elif extension in [".ts", ".tsx", ".js", ".jsx"]:
            js_analysis = self._analyze_javascript_file(content)
            file_data.update(js_analysis)
        elif extension == ".json":
            json_analysis = self._analyze_json_file(content)
            file_data.update(json_analysis)
        
        # Métricas de complejidad
        file_data["complexity_metrics"] = self._calculate_complexity_metrics(file_data, content)
        
        self.stats["total_lines"] += file_data["lines"]
        
        return file_data

    def _analyze_python_file(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Análisis específico de archivos Python"""
        analysis = {
            "functions": [],
            "classes": [],
            "imports": [],
            "total_functions": 0,
            "total_classes": 0
        }
        
        try:
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "line": node.lineno,
                        "args_count": len(node.args.args),
                        "decorators": [d.id if isinstance(d, ast.Name) else "unknown" 
                                     for d in node.decorator_list],
                        "complexity": self._estimate_function_complexity(node)
                    }
                    analysis["functions"].append(func_info)
                    analysis["total_functions"] += 1
                    self.stats["total_functions"] += 1
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "line": node.lineno,
                        "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                        "bases": [self._get_name(base) for base in node.bases]
                    }
                    analysis["classes"].append(class_info)
                    analysis["total_classes"] += 1
                    self.stats["total_classes"] += 1
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis["imports"].append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            analysis["imports"].append(f"{module}.{alias.name}")
        
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            analysis["syntax_error"] = str(e)
        
        return analysis

    def _analyze_javascript_file(self, content: str) -> Dict[str, Any]:
        """Análisis básico de archivos JavaScript/TypeScript"""
        analysis = {
            "functions": [],
            "classes": [],
            "imports": []
        }
        
        # Análisis simple basado en regex (no hay parser AST fácil para JS/TS)
        import re
        
        # Detectar funciones
        function_pattern = r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>|function))"
        functions = re.findall(function_pattern, content)
        analysis["functions"] = [f[0] or f[1] for f in functions if f[0] or f[1]]
        analysis["total_functions"] = len(analysis["functions"])
        
        # Detectar clases
        class_pattern = r"class\s+(\w+)"
        classes = re.findall(class_pattern, content)
        analysis["classes"] = classes
        analysis["total_classes"] = len(classes)
        
        # Detectar imports
        import_pattern = r"(?:import|from)\s+['\"]([^'\"]+)['\"]"
        imports = re.findall(import_pattern, content)
        analysis["imports"] = imports
        
        return analysis

    def _analyze_json_file(self, content: str) -> Dict[str, Any]:
        """Análisis de archivos JSON"""
        try:
            data = json.loads(content)
            return {
                "json_valid": True,
                "json_type": type(data).__name__,
                "json_size": len(str(data))
            }
        except json.JSONDecodeError:
            return {
                "json_valid": False,
                "json_error": "Invalid JSON"
            }

    def _estimate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Estimar complejidad ciclomática de una función"""
        complexity = 1  # Base
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity

    def _get_name(self, node) -> str:
        """Obtener nombre de un nodo AST"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"

    def _calculate_complexity_metrics(
        self, file_data: Dict[str, Any], content: str
    ) -> Dict[str, Any]:
        """Calcular métricas de complejidad del archivo"""
        lines = file_data.get("lines", 0)
        functions = file_data.get("total_functions", 0)
        classes = file_data.get("total_classes", 0)
        
        return {
            "line_complexity": lines / max(functions, 1) if functions > 0 else 0,
            "function_density": functions / max(lines / 100, 1),
            "class_density": classes / max(lines / 200, 1),
            "avg_function_complexity": sum(
                f.get("complexity", 1) for f in file_data.get("functions", [])
            ) / max(functions, 1) if functions > 0 else 0
        }

    def _calculate_statistics(self, files: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular estadísticas agregadas"""
        total_files = len(files)
        total_lines = sum(f.get("lines", 0) for f in files.values())
        total_functions = sum(f.get("total_functions", 0) for f in files.values())
        total_classes = sum(f.get("total_classes", 0) for f in files.values())
        
        # Por extensión
        by_extension = defaultdict(int)
        for f in files.values():
            by_extension[f.get("extension", "unknown")] += 1
        
        return {
            "total_files": total_files,
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "files_by_extension": dict(by_extension),
            "avg_lines_per_file": total_lines / max(total_files, 1),
            "avg_functions_per_file": total_functions / max(total_files, 1)
        }

    def _generate_weights_dataset(self, files: Dict[str, Any]) -> Dict[str, Any]:
        """Generar dataset de pesos neuronales desde archivos extraídos"""
        neural_patterns = {}
        complexity_weights = {}
        
        for file_path, file_data in files.items():
            # Patrones neuronales basados en estructura
            functions = file_data.get("functions", [])
            classes = file_data.get("classes", [])
            
            # Crear patrón por archivo
            pattern_key = f"{file_path}_pattern"
            neural_patterns[pattern_key] = {
                "function_count": len(functions),
                "class_count": len(classes),
                "complexity_score": file_data.get("complexity_metrics", {}).get(
                    "avg_function_complexity", 0
                ),
                "structure_density": len(functions) + len(classes) * 2
            }
            
            # Pesos de complejidad
            complexity_weights[file_path] = file_data.get("complexity_metrics", {})
        
        return {
            "neural_patterns": neural_patterns,
            "complexity_weights": complexity_weights
        }


def main():
    """Función principal"""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    extractor = ProjectFileExtractor()
    result = extractor.extract_all()
    
    print(f"\n✅ Extraction completed!")
    print(f"   Files processed: {extractor.stats['files_processed']}")
    print(f"   Files skipped: {extractor.stats['files_skipped']}")
    print(f"   Total lines: {extractor.stats['total_lines']}")
    print(f"   Total functions: {extractor.stats['total_functions']}")
    print(f"   Total classes: {extractor.stats['total_classes']}")
    
    if extractor.stats['errors']:
        print(f"\n⚠️ Errors: {len(extractor.stats['errors'])}")
        for error in extractor.stats['errors'][:5]:
            print(f"   - {error['file']}: {error['error']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
