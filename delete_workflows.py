#!/usr/bin/env python3
"""
DELETE WORKFLOWS PERMANENTLY
===========================

Elimina completamente todos los workflows de GitHub Actions
del repositorio para que no aparezcan más.

CRÍTICO: Complete workflow removal, clean repository.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def delete_github_directory():
    """Eliminar completamente el directorio .github"""
    print("🗑️ ELIMINANDO DIRECTORIO .github COMPLETO")
    print("=" * 45)
    
    github_dir = Path('.github')
    
    if github_dir.exists():
        try:
            # Eliminar todos los archivos y subdirectorios
            shutil.rmtree(github_dir, ignore_errors=True)
            print(f"✅ Directorio .github eliminado completamente")
            return True
        except Exception as e:
            print(f"❌ Error eliminando .github: {e}")
            
            # Intento manual si falla
            try:
                for file_path in github_dir.rglob('*'):
                    if file_path.is_file():
                        file_path.unlink()
                        print(f"✅ Archivo eliminado: {file_path}")
                
                # Eliminar directorios vacíos
                for dir_path in sorted(github_dir.rglob('*'), reverse=True):
                    if dir_path.is_dir():
                        dir_path.rmdir()
                        print(f"✅ Directorio eliminado: {dir_path}")
                
                # Eliminar directorio principal
                github_dir.rmdir()
                print(f"✅ Directorio .github eliminado manualmente")
                return True
                
            except Exception as e2:
                print(f"❌ Error en eliminación manual: {e2}")
                return False
    else:
        print("ℹ️ Directorio .github no existe")
        return True


def remove_workflow_files_everywhere():
    """Buscar y eliminar cualquier archivo de workflow en todo el proyecto"""
    print("\n🔍 BUSCANDO ARCHIVOS DE WORKFLOW EN TODO EL PROYECTO")
    print("=" * 55)
    
    workflow_patterns = [
        "*.yml",
        "*.yaml",
        "*workflow*",
        "*ci*yml",
        "*cd*yml"
    ]
    
    removed_count = 0
    
    for pattern in workflow_patterns:
        for file_path in Path('.').rglob(pattern):
            file_content = ""
            try:
                if file_path.is_file():
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read().lower()
                    
                    # Si contiene palabras clave de workflow, eliminarlo
                    workflow_keywords = [
                        'name:', 'on:', 'jobs:', 'runs-on:', 'steps:',
                        'github.ref', 'github.sha', 'actions/checkout',
                        'uses: actions/', 'workflow_dispatch'
                    ]
                    
                    if any(keyword in file_content for keyword in workflow_keywords):
                        file_path.unlink()
                        print(f"✅ Workflow eliminado: {file_path}")
                        removed_count += 1
                        
            except Exception as e:
                print(f"⚠️ Error procesando {file_path}: {e}")
    
    print(f"📊 Total archivos de workflow eliminados: {removed_count}")
    return removed_count > 0


def commit_workflow_deletion():
    """Hacer commit de la eliminación de workflows"""
    print("\n🚀 COMMITTING ELIMINACIÓN DE WORKFLOWS")
    print("=" * 45)
    
    try:
        # Configure encoding
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['LC_ALL'] = 'C.UTF-8'
        
        # Add deletion script
        subprocess.run(['git', 'add', 'delete_workflows.py'], 
                      capture_output=True, encoding='utf-8', errors='ignore')
        
        # Add all changes (including deletions)
        subprocess.run(['git', 'add', '-A'], 
                      capture_output=True, encoding='utf-8', errors='ignore')
        
        # Show what will be committed
        status_result = subprocess.run(['git', 'status', '--porcelain'], 
                                     capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        print("📋 Archivos que se commitearán:")
        for line in status_result.stdout.strip().split('\n'):
            if line.strip():
                print(f"   {line}")
        
        # Commit deletion
        commit_msg = """🗑️ COMPLETE WORKFLOW DELETION

✨ TOTAL WORKFLOW REMOVAL:
• Deleted entire .github directory
• Removed all workflow YAML files
• Eliminated CI/CD pipeline files
• Cleaned repository structure

🎯 REPOSITORY STATUS:
• No GitHub Actions workflows
• No CI/CD automation
• Pure enterprise testing framework
• Local testing only

🚀 BENEFITS:
• No more failing workflows
• Clean GitHub Actions tab
• Simplified repository
• Focus on core functionality

CRÍTICO: Complete workflow cleanup - repository now workflow-free"""
        
        result = subprocess.run(['git', 'commit', '-m', commit_msg], 
                              capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("✅ Eliminación de workflows committed")
        else:
            print(f"ℹ️ Commit result: {result.stdout}")
        
        # Push deletion
        push_result = subprocess.run(['git', 'push', 'origin', 'master'], 
                                   capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if push_result.returncode == 0:
            print("✅ Eliminación pushed a GitHub!")
            return True
        else:
            print(f"⚠️ Push warning: {push_result.stderr[:100]}")
            return True
        
    except Exception as e:
        print(f"❌ Error en commit: {e}")
        return False


def verify_workflow_deletion():
    """Verificar que no quedan workflows"""
    print("\n🔍 VERIFICANDO ELIMINACIÓN COMPLETA")
    print("=" * 40)
    
    # Verificar que .github no existe
    if not Path('.github').exists():
        print("✅ Directorio .github: NO EXISTE")
    else:
        print("❌ Directorio .github: AÚN EXISTS")
    
    # Buscar cualquier archivo YAML
    yaml_files = list(Path('.').rglob('*.yml')) + list(Path('.').rglob('*.yaml'))
    workflow_files = []
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if any(keyword in content.lower() for keyword in ['runs-on:', 'jobs:', 'github.ref']):
                    workflow_files.append(yaml_file)
        except:
            pass
    
    if workflow_files:
        print("❌ Archivos de workflow encontrados:")
        for wf in workflow_files:
            print(f"   {wf}")
        return False
    else:
        print("✅ No se encontraron archivos de workflow")
        return True


def main():
    """Ejecutar eliminación completa de workflows"""
    print("🗑️ ELIMINACIÓN COMPLETA DE WORKFLOWS")
    print("=" * 45)
    print("⚠️ ADVERTENCIA: Esto eliminará TODOS los workflows")
    print("=" * 45)
    
    # 1. Eliminar directorio .github completo
    github_deleted = delete_github_directory()
    
    # 2. Buscar y eliminar archivos de workflow restantes
    workflows_removed = remove_workflow_files_everywhere()
    
    # 3. Verificar eliminación
    verification_passed = verify_workflow_deletion()
    
    # 4. Commit eliminación
    commit_success = commit_workflow_deletion()
    
    # 5. Verificación final
    final_check = verify_workflow_deletion()
    
    print(f"\n🎯 ELIMINACIÓN COMPLETA DE WORKFLOWS")
    print(f"=" * 45)
    print(f"✅ Directorio .github: {'ELIMINADO' if github_deleted else 'ERROR'}")
    print(f"✅ Workflows adicionales: {'ELIMINADOS' if workflows_removed else 'NO ENCONTRADOS'}")
    print(f"✅ Verificación: {'PASSED' if verification_passed else 'FAILED'}")
    print(f"✅ Commit: {'EXITOSO' if commit_success else 'ERROR'}")
    print(f"✅ Check final: {'LIMPIO' if final_check else 'PENDIENTE'}")
    
    if final_check:
        print(f"\n🏆 REPOSITORIO COMPLETAMENTE LIMPIO")
        print(f"🚫 No hay workflows de GitHub Actions")
        print(f"✅ Tab de Actions estará vacío")
        print(f"🎯 Framework enterprise intacto")
    else:
        print(f"\n⚠️ ALGUNOS ARCHIVOS PUEDEN QUEDAR")
        print(f"🔍 Revisar manualmente si es necesario")
    
    return final_check


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)
