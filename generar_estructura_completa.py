#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera la estructura completa del proyecto excluyendo carpetas temporales
"""
import os
import sys
from pathlib import Path

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Carpetas y archivos a excluir
EXCLUDE_DIRS = {
    'node_modules', 'venv', '__pycache__', '.git', '.next', 'dist', 'build',
    '.pytest_cache', '.mypy_cache', '.ruff_cache', 'mlruns', '.vscode', '.idea',
    'env', '.env', '*.pyc', '*.pyo', '.egg-info', 'Solution1', 'Videos'
}

EXCLUDE_EXTENSIONS = {'.pyc', '.pyo', '.pth', '.faiss', '.incomplete'}

def should_exclude(path: Path) -> bool:
    """Verifica si un path debe ser excluido"""
    # Excluir por nombre de carpeta
    if any(excluded in path.parts for excluded in EXCLUDE_DIRS):
        return True
    
    # Excluir archivos con extensiones específicas
    if path.suffix in EXCLUDE_EXTENSIONS:
        return True
    
    # Excluir archivos ocultos (que empiezan con .)
    if path.name.startswith('.') and path.name != '.env':
        return True
    
    return False

def generate_tree(root: Path, prefix: str = "", is_last: bool = True, max_depth: int = None, current_depth: int = 0):
    """Genera el árbol de directorios recursivamente"""
    if max_depth and current_depth > max_depth:
        return
    
    if should_exclude(root):
        return
    
    # Obtener todos los elementos
    try:
        items = sorted([item for item in root.iterdir() if not should_exclude(item)])
    except PermissionError:
        return
    
    # Filtrar solo directorios primero
    dirs = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    # Mostrar el directorio actual (usar caracteres ASCII para compatibilidad)
    connector = "+-- " if is_last else "|-- "
    print(f"{prefix}{connector}{root.name}/")
    
    # Actualizar prefijo
    extension = "    " if is_last else "|   "
    new_prefix = prefix + extension
    
    # Procesar directorios
    all_items = dirs + files
    for i, item in enumerate(all_items):
        is_last_item = (i == len(all_items) - 1)
        
        if item.is_dir():
            generate_tree(item, new_prefix, is_last_item, max_depth, current_depth + 1)
        else:
            connector = "+-- " if is_last_item else "|-- "
            print(f"{new_prefix}{connector}{item.name}")

def main():
    """Función principal"""
    root_path = Path.cwd()
    
    print("=" * 80)
    print(f"ESTRUCTURA COMPLETA DEL PROYECTO: {root_path.name}")
    print("=" * 80)
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
            generate_tree(item, "", is_last)
        else:
            connector = "+-- " if is_last else "|-- "
            print(f"{connector}{item.name}")
    
    print()
    print("=" * 80)
    print("FIN DE LA ESTRUCTURA")
    print("=" * 80)

if __name__ == "__main__":
    main()

