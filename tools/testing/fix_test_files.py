#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REPARADOR DE ARCHIVOS DE TEST
==============================
Corrige problemas comunes en archivos de test:
- Convierte returns en asserts
- Elimina setup_module problemáticos
- Corrige imports faltantes

USO:
    python tools/testing/fix_test_files.py
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Configurar UTF-8 para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


class TestFileFixer:
    """Reparador de archivos de test"""

    def __init__(self, project_root: Path = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()
        
        self.tests_dir = self.project_root / "tests"
        self.fixed_count = 0
        self.errors = []

    def fix_test_file(self, filepath: Path) -> bool:
        """Corregir un archivo de test"""
        try:
            content = filepath.read_text(encoding="utf-8")
            original = content
            changes = []

            # 1. Eliminar setup_module problemáticos
            if "def setup_module" in content:
                content = re.sub(
                    r'def setup_module\([^)]*\):.*?(?=\ndef|\Z)',
                    '',
                    content,
                    flags=re.DOTALL
                )
                changes.append("Eliminado setup_module")

            # 2. Convertir returns booleanos en asserts dentro de funciones test_
            def replace_test_returns(match):
                func_content = match.group(0)
                original_func = func_content
                
                # Reemplazar return True con assert True
                func_content = re.sub(
                    r'return True\s*$',
                    'assert True',
                    func_content,
                    flags=re.MULTILINE
                )
                
                # Reemplazar return False con assert False
                func_content = re.sub(
                    r'return False\s*$',
                    'assert False, "Test failed"',
                    func_content,
                    flags=re.MULTILINE
                )
                
                # Eliminar try-except que solo retornan False
                func_content = re.sub(
                    r'try:\s*\n(.*?)except.*?:\s*\n\s*return False',
                    r'\1',
                    func_content,
                    flags=re.DOTALL
                )
                
                if func_content != original_func:
                    changes.append("Convertidos returns en asserts")
                
                return func_content

            content = re.sub(
                r'def test_[^(]+\([^)]*\):.*?(?=\ndef|\Z)',
                replace_test_returns,
                content,
                flags=re.DOTALL
            )

            # 3. Agregar imports comunes si faltan
            if "import pytest" not in content and "test_" in content:
                # Intentar agregar al inicio después de otros imports
                import_section = re.search(r'^(.*?)(def |class )', content, re.DOTALL | re.MULTILINE)
                if import_section:
                    imports = import_section.group(1)
                    if "import pytest" not in imports:
                        # Agregar después del último import
                        last_import = list(re.finditer(r'^(import |from .*? import)', imports, re.MULTILINE))
                        if last_import:
                            last_pos = last_import[-1].end()
                            imports = imports[:last_pos] + "\nimport pytest" + imports[last_pos:]
                            content = imports + content[import_section.end(1):]
                            changes.append("Agregado import pytest")

            # 4. Corregir asserts mal formados
            content = re.sub(
                r'assert\s+True\s*$',
                'assert True',
                content,
                flags=re.MULTILINE
            )

            # Guardar si hubo cambios
            if content != original:
                filepath.write_text(content, encoding="utf-8")
                print(f"✅ Corregido: {filepath.relative_to(self.project_root)}")
                if changes:
                    print(f"   Cambios: {', '.join(changes)}")
                return True
            else:
                print(f"ℹ️  Sin cambios: {filepath.relative_to(self.project_root)}")
                return False

        except Exception as e:
            error_msg = f"Error corrigiendo {filepath.name}: {e}"
            print(f"❌ {error_msg}")
            self.errors.append(error_msg)
            return False

    def find_test_files(self) -> List[Path]:
        """Encontrar todos los archivos de test"""
        test_files = []
        
        if self.tests_dir.exists():
            test_files.extend(self.tests_dir.rglob("test_*.py"))
        
        # También buscar en tools/testing si tiene tests
        tools_tests = self.project_root / "tools" / "testing"
        if tools_tests.exists():
            test_files.extend(tools_tests.rglob("test_*.py"))

        return test_files

    def fix_all(self) -> Tuple[int, int]:
        """Corregir todos los archivos de test"""
        print("=" * 70)
        print("REPARADOR DE ARCHIVOS DE TEST")
        print("=" * 70)
        print(f"Proyecto: {self.project_root}")
        print(f"Buscando tests en: {self.tests_dir}\n")

        test_files = self.find_test_files()
        
        if not test_files:
            print("⚠️  No se encontraron archivos de test")
            return 0, 0

        print(f"📁 Encontrados {len(test_files)} archivos de test\n")

        fixed = 0
        for test_file in test_files:
            if self.fix_test_file(test_file):
                fixed += 1

        print(f"\n✅ Corregidos {fixed}/{len(test_files)} archivos")

        if self.errors:
            print(f"\n⚠️  Errores encontrados: {len(self.errors)}")
            for error in self.errors[:5]:
                print(f"   - {error}")

        return fixed, len(test_files)


def main():
    """Función principal"""
    fixer = TestFileFixer()
    fixed, total = fixer.fix_all()
    
    if fixed > 0:
        print(f"\n✅ Reparación completada: {fixed} archivos corregidos")
        sys.exit(0)
    else:
        print("\nℹ️  No se requirieron correcciones")
        sys.exit(0)


if __name__ == "__main__":
    main()
