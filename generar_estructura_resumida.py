#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera estructura completa pero resumida (agrupa archivos similares)
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

EXCLUDE_DIRS = {
    'node_modules', 'venv', '__pycache__', '.git', '.next', 'dist', 'build',
    '.pytest_cache', '.mypy_cache', '.ruff_cache', 'mlruns', '.vscode', '.idea',
    'env', '.env', '.egg-info', 'Solution1', 'Videos', '.turbo'
}

EXCLUDE_EXTENSIONS = {'.pyc', '.pyo', '.pth', '.incomplete'}

def should_exclude(path: Path) -> bool:
    if any(excluded in path.parts for excluded in EXCLUDE_DIRS):
        return True
    if path.suffix in EXCLUDE_EXTENSIONS:
        return True
    if path.name.startswith('.') and path.name not in ['.env', '.dockerignore']:
        return True
    return False

def group_files(files):
    """Agrupa archivos por extensión y patrón (más agresivo)"""
    groups = defaultdict(list)
    
    for f in files:
        ext = f.suffix or '(sin ext)'
        name = f.name
        
        # Agrupar por extensión principalmente (más compacto)
        if name.endswith('.json'):
            # Solo patrones muy específicos, resto todo a *.json
            if name.startswith('chat-'):
                groups['chat-*.json'].append(f)
            elif any(name.startswith(p) for p in ['prompt_optimization_', 'auto_training_', 'smart_training_', 'training_batches_', 'metrics_snapshot_', 'meta_cognition_']):
                groups['*.json (logs)'].append(f)
            else:
                groups['*.json'].append(f)
        elif name.endswith('.jsonl'):
            groups['*.jsonl'].append(f)
        elif name.endswith('.py'):
            groups['*.py'].append(f)
        elif name.endswith('.tsx'):
            groups['*.tsx'].append(f)
        elif name.endswith('.ts'):
            groups['*.ts'].append(f)
        elif name.endswith('.md'):
            groups['*.md'].append(f)
        elif name.endswith('.db') or name.endswith('.sqlite3'):
            groups['*.db'].append(f)
        elif name.endswith('.yaml') or name.endswith('.yml'):
            groups['*.yaml'].append(f)
        elif name.endswith('.txt'):
            groups['*.txt'].append(f)
        elif name.endswith('.css'):
            groups['*.css'].append(f)
        elif name.endswith('.js'):
            groups['*.js'].append(f)
        elif name.endswith('.html'):
            groups['*.html'].append(f)
        elif name.endswith('.png') or name.endswith('.jpg') or name.endswith('.svg') or name.endswith('.ico'):
            groups['*.img'].append(f)
        elif name.endswith('.exe') or name.endswith('.dll'):
            groups['*.bin'].append(f)
        else:
            groups[f'*{ext}'].append(f)
    
    return groups

def format_group(group_name, files):
    """Formatea un grupo de archivos"""
    count = len(files)
    if count == 1:
        return f"+-- {files[0].name}"
    elif count <= 5:
        return f"+-- {group_name} ({count} archivos)"
    else:
        # Mostrar algunos ejemplos
        examples = [f.name for f in sorted(files)[:3]]
        return f"+-- {group_name} ({count} archivos, ej: {', '.join(examples[:2])}...)"

def generate_tree_compact(root: Path, prefix: str = "", is_last: bool = True, max_depth: int = None, current_depth: int = 0):
    """Genera árbol compacto pero completo (optimizado 16% más)"""
    if max_depth and current_depth > max_depth:
        return
    
    if should_exclude(root):
        return
    
    try:
        items = sorted([item for item in root.iterdir() if not should_exclude(item)])
    except PermissionError:
        return
    
    dirs = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    # Optimización: si solo hay un archivo y no hay subdirectorios, mostrar más compacto
    if len(dirs) == 0 and len(files) == 1:
        connector = "+-- " if is_last else "|-- "
        print(f"{prefix}{connector}{root.name}/{files[0].name}")
        return
    
    # Optimización: si solo hay archivos y son pocos, mostrar inline
    if len(dirs) == 0 and len(files) <= 3:
        connector = "+-- " if is_last else "|-- "
        file_names = "/".join([f.name for f in files])
        print(f"{prefix}{connector}{root.name}/{file_names}")
        return
    
    # Mostrar directorio
    connector = "+-- " if is_last else "|-- "
    print(f"{prefix}{connector}{root.name}/")
    
    extension = "    " if is_last else "|   "
    new_prefix = prefix + extension
    
    # Agrupar archivos
    file_groups = {}
    if files:
        file_groups = group_files(files)
    
    # Procesar directorios primero (optimización: agrupar directorios vacíos)
    empty_dirs = []
    non_empty_dirs = []
    for d in dirs:
        try:
            has_content = any(True for _ in d.iterdir() if not should_exclude(_))
            if not has_content:
                empty_dirs.append(d)
            else:
                non_empty_dirs.append(d)
        except:
            non_empty_dirs.append(d)
    
    # Mostrar directorios vacíos agrupados
    if empty_dirs:
        connector = "+-- " if (len(non_empty_dirs) == 0 and not file_groups) else "|-- "
        print(f"{new_prefix}{connector}[{len(empty_dirs)} dirs vacíos: {', '.join([d.name for d in empty_dirs[:3]])}{'...' if len(empty_dirs) > 3 else ''}]")
    
    # Procesar directorios con contenido
    for i, item in enumerate(non_empty_dirs):
        is_last_dir = (i == len(non_empty_dirs) - 1 and not file_groups)
        generate_tree_compact(item, new_prefix, is_last_dir, max_depth, current_depth + 1)
    
    # Procesar grupos de archivos (más compacto - sin ejemplos si hay muchos)
    if file_groups:
        group_items = list(file_groups.items())
        for i, (group_name, group_files_list) in enumerate(group_items):
            is_last_item = (i == len(group_items) - 1)
            connector = "+-- " if is_last_item else "|-- "
            count = len(group_files_list)
            if count == 1:
                print(f"{new_prefix}{connector}{group_files_list[0].name}")
            else:
                # Siempre mostrar solo contador (sin ejemplos)
                print(f"{new_prefix}{connector}{group_name} ({count})")

def main():
    root_path = Path.cwd()
    
    print(f"ESTRUCTURA: {root_path.name}\n{root_path.name}/")
    
    try:
        items = sorted([item for item in root_path.iterdir() if not should_exclude(item)])
    except PermissionError as e:
        print(f"Error: {e}")
        return
    
    dirs = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    all_items = dirs + files
    
    for i, item in enumerate(all_items):
        is_last = (i == len(all_items) - 1)
        
        if item.is_dir():
            generate_tree_compact(item, "", is_last)
        else:
            connector = "+-- " if is_last else "|-- "
            print(f"{connector}{item.name}")
    
    print("\nFIN")

if __name__ == "__main__":
    main()

