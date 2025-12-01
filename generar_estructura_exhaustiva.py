#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera la estructura COMPLETA y EXHAUSTIVA del proyecto
Incluye TODOS los archivos y carpetas sin excepción (excepto temporales)
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Carpetas y archivos a excluir (SOLO temporales)
EXCLUDE_DIRS = {
    'node_modules', 'venv', '__pycache__', '.git', '.next', 'dist', 'build',
    '.pytest_cache', '.mypy_cache', '.ruff_cache', 'mlruns', '.vscode', '.idea',
    'env', '.env', '.egg-info', 'Solution1', 'Videos', '.turbo'
}

EXCLUDE_EXTENSIONS = {'.pyc', '.pyo', '.pth', '.incomplete'}

def should_exclude(path: Path) -> bool:
    """Verifica si un path debe ser excluido (SOLO temporales)"""
    # Excluir por nombre de carpeta
    if any(excluded in path.parts for excluded in EXCLUDE_DIRS):
        return True
    
    # Excluir archivos con extensiones específicas
    if path.suffix in EXCLUDE_EXTENSIONS:
        return True
    
    # Excluir archivos ocultos que empiezan con . (excepto .env si es necesario)
    if path.name.startswith('.') and path.name not in ['.env', '.gitignore', '.gitkeep']:
        # Pero incluir archivos de configuración importantes
        if path.name in ['.env', '.gitignore', '.gitkeep', '.dockerignore']:
            return False
        return True
    
    return False

def get_file_size(path: Path) -> str:
    """Obtiene el tamaño del archivo en formato legible"""
    try:
        size = path.stat().st_size
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size/1024:.1f}KB"
        else:
            return f"{size/(1024*1024):.1f}MB"
    except:
        return "?"

def generate_tree_exhaustive(root: Path, prefix: str = "", is_last: bool = True, max_depth: int = None, current_depth: int = 0):
    """Genera el árbol de directorios de forma EXHAUSTIVA"""
    if max_depth and current_depth > max_depth:
        return
    
    if should_exclude(root):
        return
    
    # Obtener TODOS los elementos
    try:
        items = sorted([item for item in root.iterdir() if not should_exclude(item)])
    except PermissionError:
        return
    
    # Separar directorios y archivos
    dirs = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    # Mostrar el directorio actual
    connector = "+-- " if is_last else "|-- "
    print(f"{prefix}{connector}{root.name}/")
    
    # Actualizar prefijo
    extension = "    " if is_last else "|   "
    new_prefix = prefix + extension
    
    # Procesar TODOS los directorios primero
    all_items = dirs + files
    for i, item in enumerate(all_items):
        is_last_item = (i == len(all_items) - 1)
        
        if item.is_dir():
            generate_tree_exhaustive(item, new_prefix, is_last_item, max_depth, current_depth + 1)
        else:
            # Mostrar archivo con tamaño
            connector = "+-- " if is_last_item else "|-- "
            size = get_file_size(item)
            print(f"{new_prefix}{connector}{item.name} ({size})")

def count_files_by_type(root: Path, stats: defaultdict):
    """Cuenta archivos por tipo"""
    if should_exclude(root):
        return
    
    try:
        for item in root.iterdir():
            if should_exclude(item):
                continue
            
            if item.is_file():
                ext = item.suffix or '(sin extension)'
                stats[ext] += 1
            elif item.is_dir():
                count_files_by_type(item, stats)
    except PermissionError:
        pass

def main():
    """Función principal"""
    root_path = Path.cwd()
    
    print("=" * 100)
    print(f"ESTRUCTURA COMPLETA Y EXHAUSTIVA DEL PROYECTO: {root_path.name}")
    print("=" * 100)
    print()
    print("NOTA: Esta estructura incluye TODOS los archivos y carpetas excepto temporales")
    print("(node_modules, venv, __pycache__, .git, .next, dist, build, etc.)")
    print()
    print("=" * 100)
    print()
    
    # Generar árbol desde la raíz
    print(f"{root_path.name}/")
    
    # Procesar cada elemento en la raíz
    try:
        items = sorted([item for item in root_path.iterdir() if not should_exclude(item)])
    except PermissionError as e:
        print(f"Error de permisos: {e}")
        return
    
    # Separar directorios y archivos
    dirs = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    all_items = dirs + files
    
    for i, item in enumerate(all_items):
        is_last = (i == len(all_items) - 1)
        
        if item.is_dir():
            generate_tree_exhaustive(item, "", is_last)
        else:
            connector = "+-- " if is_last else "|-- "
            size = get_file_size(item)
            print(f"{connector}{item.name} ({size})")
    
    print()
    print("=" * 100)
    
    # Estadísticas
    print()
    print("ESTADÍSTICAS DE ARCHIVOS POR TIPO:")
    print("-" * 100)
    stats = defaultdict(int)
    count_files_by_type(root_path, stats)
    
    for ext, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext:20} : {count:5} archivos")
    
    print()
    print("=" * 100)
    print("FIN DE LA ESTRUCTURA COMPLETA")
    print("=" * 100)

if __name__ == "__main__":
    main()

