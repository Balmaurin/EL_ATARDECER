#!/usr/bin/env python3
"""
ENTERPRISE TEST FILE FIXER
==========================

Automatically fixes common test file issues:
- Converts return statements to proper assertions
- Removes problematic setup_method functions
- Standardizes test structure for enterprise compliance
- Creates backups before modifications

CRÍTICO: Test automation, enterprise quality assurance.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Any, Tuple
import shutil
from datetime import datetime


class EnterpriseTestFixer:
    """Enterprise test file fixer and validator"""
    
    def __init__(self, project_root: str = "."):
        """Initialize test fixer with enterprise configuration
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.fixes_applied = 0
        self.files_processed = 0
        self.backup_dir = self.project_root / "test_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Enterprise patterns to fix
        self.fix_patterns = {
            'return_to_assert': True,
            'setup_cleanup': True,
            'docstring_addition': True,
            'assertion_standardization': True,
            'import_optimization': True
        }
    
    def fix_all_test_files(self) -> Dict[str, Any]:
        """Fix all test files in the project with enterprise standards
        
        Returns:
            Dictionary containing fix statistics and results
        """
        print("🔧 ENTERPRISE TEST FILE FIXER")
        print("=" * 45)
        print("🎯 Applying enterprise quality standards...")
        
        test_files = list(self.project_root.rglob("test_*.py"))
        test_files.extend(self.project_root.rglob("*_test.py"))
        
        results = []
        
        for test_file in test_files:
            if self._should_fix_file(test_file):
                try:
                    fixes = self._fix_single_test_file(test_file)
                    if fixes > 0:
                        print(f"✅ Fixed {test_file.name}: {fixes} issues resolved")
                        results.append({
                            'file': str(test_file),
                            'fixes_applied': fixes,
                            'status': 'success'
                        })
                    else:
                        print(f"ℹ️ {test_file.name}: Already compliant")
                        results.append({
                            'file': str(test_file),
                            'fixes_applied': 0,
                            'status': 'compliant'
                        })
                    self.files_processed += 1
                except Exception as e:
                    print(f"❌ Could not fix {test_file.name}: {e}")
                    results.append({
                        'file': str(test_file),
                        'fixes_applied': 0,
                        'status': 'error',
                        'error': str(e)
                    })
        
        summary = {
            'files_processed': self.files_processed,
            'total_fixes_applied': self.fixes_applied,
            'backup_location': str(self.backup_dir),
            'file_results': results,
            'success_rate': (len([r for r in results if r['status'] in ['success', 'compliant']]) / 
                           max(len(results), 1)) * 100
        }
        
        print(f"\n📊 ENTERPRISE TEST FIXING COMPLETE:")
        print(f"   Files processed: {summary['files_processed']}")
        print(f"   Total fixes applied: {summary['total_fixes_applied']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Backups saved to: {summary['backup_location']}")
        
        return summary
    
    def _should_fix_file(self, file_path: Path) -> bool:
        """Determine if file should be processed for enterprise fixes
        
        Args:
            file_path: Path to potential test file
            
        Returns:
            True if file should be fixed
        """
        return (
            file_path.suffix == '.py' and
            ('test' in file_path.name) and
            not str(file_path).startswith('.') and
            file_path.stat().st_size > 0
        )
    
    def _fix_single_test_file(self, file_path: Path) -> int:
        """Apply enterprise fixes to a single test file
        
        Args:
            file_path: Path to test file to fix
            
        Returns:
            Number of fixes applied
        """
        # Create timestamped backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"{file_path.name}_{timestamp}.bak"
        shutil.copy2(file_path, backup_path)
        
        # Read original content
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        content = original_content
        total_fixes = 0
        
        # Apply enterprise fixes
        if self.fix_patterns['return_to_assert']:
            content, fixes = self._convert_returns_to_assertions(content)
            total_fixes += fixes
        
        if self.fix_patterns['setup_cleanup']:
            content, fixes = self._fix_setup_methods(content)
            total_fixes += fixes
        
        if self.fix_patterns['docstring_addition']:
            content, fixes = self._add_enterprise_docstrings(content)
            total_fixes += fixes
        
        if self.fix_patterns['assertion_standardization']:
            content, fixes = self._standardize_assertions(content)
            total_fixes += fixes
        
        if self.fix_patterns['import_optimization']:
            content, fixes = self._optimize_imports(content)
            total_fixes += fixes
        
        # Write fixed content if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        self.fixes_applied += total_fixes
        return total_fixes
    
    def _convert_returns_to_assertions(self, content: str) -> Tuple[str, int]:
        """Convert return statements to proper assertions
        
        Args:
            content: File content to process
            
        Returns:
            Tuple of (fixed_content, number_of_fixes)
        """
        fixes = 0
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            # Convert return True/False patterns
            if stripped == 'return True':
                lines[i] = ' ' * indent + 'assert True  # Enterprise: Test passed'
                fixes += 1
            elif stripped == 'return False':
                lines[i] = ' ' * indent + 'assert False, "Enterprise: Test validation failed"'
                fixes += 1
            elif re.match(r'return\s+(success|result|passed)\s*$', stripped):
                var_name = re.search(r'return\s+(\w+)', stripped).group(1)
                lines[i] = ' ' * indent + f'assert {var_name}, "Enterprise: {var_name} validation failed"'
                fixes += 1
        
        return '\n'.join(lines), fixes
    
    def _fix_setup_methods(self, content: str) -> Tuple[str, int]:
        """Fix problematic setup_method functions
        
        Args:
            content: File content to process
            
        Returns:
            Tuple of (fixed_content, number_of_fixes)
        """
        fixes = 0
        
        # Remove empty setup_method functions
        empty_setup_pattern = r'def setup_method\(self.*?\):\s*\n(\s*"""[^"]*""")?\s*\n\s*pass\s*\n'
        content, count = re.subn(empty_setup_pattern, '', content, flags=re.DOTALL)
        fixes += count
        
        # Fix setup_method parameter signature
        if re.search(r'def setup_method\(self\):', content):
            content = re.sub(r'def setup_method\(self\):', 'def setup_method(self, method):', content)
            fixes += 1
        
        # Add proper enterprise setup template where needed
        if 'def setup_method(' in content and 'self.start_time' not in content:
            setup_replacement = '''def setup_method(self, method):
        """Enterprise test setup with metrics tracking
        
        Args:
            method: Test method being executed
        """
        self.start_time = time.time()
        self.test_metrics = {
            'assertions_checked': 0,
            'enterprise_validations': 0
        }'''
            content = re.sub(
                r'def setup_method\(self, method\):\s*\n(\s*"""[^"]*""")?\s*\n',
                setup_replacement + '\n\n',
                content
            )
            fixes += 1
        
        return content, fixes
    
    def _add_enterprise_docstrings(self, content: str) -> Tuple[str, int]:
        """Add enterprise-grade docstrings to test functions
        
        Args:
            content: File content to process
            
        Returns:
            Tuple of (fixed_content, number_of_fixes)
        """
        fixes = 0
        lines = content.split('\n')
        new_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            new_lines.append(line)
            
            # Check for test function without docstring
            if re.match(r'\s*def test_\w+\(', line):
                function_name = re.search(r'def (test_\w+)\(', line).group(1)
                
                # Look ahead to check if docstring exists
                next_non_empty = i + 1
                while (next_non_empty < len(lines) and 
                       not lines[next_non_empty].strip()):
                    next_non_empty += 1
                
                if (next_non_empty < len(lines) and 
                    not lines[next_non_empty].strip().startswith('"""')):
                    
                    # Add enterprise docstring
                    indent = len(line) - len(line.lstrip()) + 4
                    test_description = function_name.replace('_', ' ').replace('test ', '').title()
                    
                    enterprise_docstring = f'"""Enterprise test case: {test_description}\n' + \
                                         f'{" " * indent}\n' + \
                                         f'{" " * indent}Validates enterprise-grade functionality with comprehensive assertions.\n' + \
                                         f'{" " * indent}CRÍTICO: Production readiness validation.\n' + \
                                         f'{" " * indent}"""'
                    
                    new_lines.append(' ' * indent + enterprise_docstring)
                    fixes += 1
            
            i += 1
        
        return '\n'.join(new_lines), fixes
    
    def _standardize_assertions(self, content: str) -> Tuple[str, int]:
        """Standardize assertion patterns for enterprise compliance
        
        Args:
            content: File content to process
            
        Returns:
            Tuple of (fixed_content, number_of_fixes)
        """
        fixes = 0
        
        # Convert unittest assertions to pytest format
        conversions = [
            (r'self\.assertEqual\(([^,]+),\s*([^)]+)\)', r'assert \1 == \2'),
            (r'self\.assertTrue\(([^)]+)\)', r'assert \1'),
            (r'self\.assertFalse\(([^)]+)\)', r'assert not \1'),
            (r'self\.assertIsNone\(([^)]+)\)', r'assert \1 is None'),
            (r'self\.assertIsNotNone\(([^)]+)\)', r'assert \1 is not None'),
            (r'self\.assertIn\(([^,]+),\s*([^)]+)\)', r'assert \1 in \2'),
            (r'self\.assertNotIn\(([^,]+),\s*([^)]+)\)', r'assert \1 not in \2'),
        ]
        
        for pattern, replacement in conversions:
            content, count = re.subn(pattern, replacement, content)
            fixes += count
        
        return content, fixes
    
    def _optimize_imports(self, content: str) -> Tuple[str, int]:
        """Optimize imports for enterprise test files
        
        Args:
            content: File content to process
            
        Returns:
            Tuple of (fixed_content, number_of_fixes)
        """
        fixes = 0
        lines = content.split('\n')
        
        # Ensure essential enterprise imports are present
        has_pytest = any('import pytest' in line for line in lines)
        has_time = any('import time' in line for line in lines)
        has_typing = any('from typing import' in line for line in lines)
        
        if not has_pytest and 'def test_' in content:
            # Find appropriate location to insert pytest import
            import_section_end = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_section_end = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            lines.insert(import_section_end, 'import pytest')
            fixes += 1
        
        if not has_time and ('time.time()' in content or 'setup_method' in content):
            lines.insert(0, 'import time')
            fixes += 1
        
        return '\n'.join(lines), fixes


def remove_deprecated_documentation_tools():
    """Remove deprecated documentation generation scripts"""
    print("\n🗑️ REMOVING DEPRECATED DOCUMENTATION TOOLS")
    print("=" * 50)
    
    deprecated_files = [
        'auto_doc_generator.py',
        'generate_inventory_docs.py',
        'living_docs_generator.py',
        'doc_automation.py',
        'legacy_docs.py'
    ]
    
    removed_count = 0
    
    for file_name in deprecated_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✅ Removed: {file_name}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {file_name}: {e}")
        else:
            print(f"ℹ️ Not found: {file_name}")
    
    print(f"\n📊 Removed {removed_count} deprecated documentation files")


def create_vscode_settings_template():
    """Create enterprise VSCode settings template for Python testing"""
    print("\n⚙️ CREATING VSCODE SETTINGS TEMPLATE")
    print("=" * 45)
    
    vscode_dir = Path('.vscode')
    vscode_dir.mkdir(exist_ok=True)
    
    # Enterprise VSCode settings
    settings_config = {
        "python.testing.pytestEnabled": True,
        "python.testing.unittestEnabled": False,
        "python.testing.pytestArgs": [
            "tests/",
            "--verbose",
            "--tb=short",
            "--disable-warnings"
        ],
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": True,
        "python.formatting.provider": "black",
        "python.defaultInterpreterPath": "python",
        "files.associations": {
            "*.py": "python"
        },
        "python.testing.autoTestDiscoverOnSaveEnabled": True,
        "editor.rulers": [88, 100],
        "editor.formatOnSave": True,
        "python.sortImports.args": ["--profile", "black"],
        "python.analysis.typeCheckingMode": "basic",
        "python.analysis.autoImportCompletions": True,
        "files.exclude": {
            "**/__pycache__": True,
            "**/*.pyc": True,
            "**/test_backups": True,
            "**/audit_results": True
        },
        "search.exclude": {
            "**/test_backups": True,
            "**/audit_results": True,
            "**/__pycache__": True
        }
    }
    
    # Enterprise launch configurations
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Current File",
                "type": "python",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal"
            },
            {
                "name": "Enterprise: Run All Tests",
                "type": "python",
                "request": "launch",
                "module": "pytest",
                "args": ["tests/", "-v", "--tb=short"],
                "console": "integratedTerminal"
            },
            {
                "name": "Enterprise: Test Suite Orchestrator",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/run_all_enterprise_tests.py",
                "console": "integratedTerminal"
            },
            {
                "name": "Enterprise: Project Audit",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/audit_enterprise_project.py",
                "console": "integratedTerminal"
            },
            {
                "name": "Enterprise: Fix Test Files",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/fix_test_files.py",
                "console": "integratedTerminal"
            }
        ]
    }
    
    # Write settings.json
    settings_file = vscode_dir / 'settings.json'
    with open(settings_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(settings_config, f, indent=4)
    print(f"✅ Created: {settings_file}")
    
    # Write launch.json
    launch_file = vscode_dir / 'launch.json'
    with open(launch_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(launch_config, f, indent=4)
    print(f"✅ Created: {launch_file}")
    
    # Create tasks.json for build automation
    tasks_config = {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "Enterprise: Run Test Fixer",
                "type": "shell",
                "command": "python",
                "args": ["fix_test_files.py"],
                "group": "build",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "focus": False,
                    "panel": "shared"
                }
            },
            {
                "label": "Enterprise: Execute All Tests",
                "type": "shell",
                "command": "python",
                "args": ["run_all_enterprise_tests.py"],
                "group": "test",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "focus": False,
                    "panel": "shared"
                }
            },
            {
                "label": "Enterprise: Project Audit",
                "type": "shell",
                "command": "python",
                "args": ["audit_enterprise_project.py"],
                "group": "build",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "focus": False,
                    "panel": "shared"
                }
            }
        ]
    }
    
    tasks_file = vscode_dir / 'tasks.json'
    with open(tasks_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(tasks_config, f, indent=4)
    print(f"✅ Created: {tasks_file}")
    
    print(f"\n📁 VSCode configuration created in: {vscode_dir}")


def main():
    """Execute enterprise test maintenance and setup"""
    print("🚀 ENTERPRISE TEST MAINTENANCE & SETUP")
    print("=" * 50)
    
    # Step 1: Remove deprecated documentation tools
    remove_deprecated_documentation_tools()
    
    # Step 2: Fix all test files
    test_fixer = EnterpriseTestFixer()
    fix_results = test_fixer.fix_all_test_files()
    
    # Step 3: Create VSCode settings template
    create_vscode_settings_template()
    
    # Summary report
    print(f"\n🎯 ENTERPRISE MAINTENANCE COMPLETE")
    print(f"=" * 45)
    print(f"✅ Test files processed: {fix_results['files_processed']}")
    print(f"✅ Total fixes applied: {fix_results['total_fixes_applied']}")
    print(f"✅ Success rate: {fix_results['success_rate']:.1f}%")
    print(f"✅ VSCode configuration created")
    print(f"\n📂 Backups available in: {fix_results['backup_location']}")
    print(f"📂 VSCode settings in: .vscode/")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Enterprise maintenance interrupted")
        exit(1)
    except Exception as e:
        print(f"\n💥 Enterprise maintenance failed: {e}")
        exit(1)
