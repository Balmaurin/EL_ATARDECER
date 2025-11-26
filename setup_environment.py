#!/usr/bin/env python3
"""
ENTERPRISE ENVIRONMENT SETUP
============================

Automated setup script for enterprise testing environment.
Installs dependencies, configures paths, validates setup.

CRÍTICO: Environment configuration, dependency management.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_dependencies():
    """Install required dependencies for enterprise testing"""
    print("📦 INSTALLING ENTERPRISE DEPENDENCIES")
    print("=" * 40)
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements if file exists
        requirements_file = Path("requirements.txt")
        if requirements_file.exists():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        else:
            # Install essential packages directly
            essential_packages = [
                "pytest>=7.0.0",
                "numpy>=1.21.0", 
                "psutil>=5.9.0"
            ]
            for package in essential_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def validate_environment():
    """Validate the enterprise testing environment"""
    print("\n🔍 VALIDATING ENTERPRISE ENVIRONMENT")
    print("=" * 40)
    
    validation_results = {}
    
    # Check Python version
    python_version = sys.version_info
    validation_results['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    
    if python_version >= (3, 8):
        print(f"✅ Python version: {validation_results['python_version']}")
    else:
        print(f"❌ Python version too old: {validation_results['python_version']} (requires 3.8+)")
        return False
    
    # Check essential imports
    essential_modules = ['pytest', 'numpy', 'time', 'json', 'pathlib']
    
    for module in essential_modules:
        try:
            __import__(module)
            print(f"✅ Module available: {module}")
            validation_results[f'module_{module}'] = True
        except ImportError:
            print(f"❌ Module missing: {module}")
            validation_results[f'module_{module}'] = False
    
    # Check test files exist
    test_files = [
        'tests/enterprise/test_api_enterprise_suites.py',
        'tests/enterprise/test_blockchain_enterprise.py',
        'tests/enterprise/test_rag_system_enterprise.py'
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"✅ Test suite found: {test_file}")
            validation_results[f'test_{Path(test_file).stem}'] = True
        else:
            print(f"❌ Test suite missing: {test_file}")
            validation_results[f'test_{Path(test_file).stem}'] = False
    
    return all(validation_results.values())


def setup_project_structure():
    """Setup required project directories"""
    print("\n📁 SETTING UP PROJECT STRUCTURE")
    print("=" * 35)
    
    required_dirs = [
        'tests/enterprise',
        'tests/results/enterprise',
        'audit_results',
        'test_backups',
        '.vscode'
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory ready: {dir_path}")
    
    return True


def main():
    """Execute complete environment setup"""
    print("🚀 ENTERPRISE ENVIRONMENT SETUP")
    print("=" * 35)
    
    # Setup project structure
    setup_success = setup_project_structure()
    if not setup_success:
        print("❌ Failed to setup project structure")
        return False
    
    # Install dependencies
    install_success = install_dependencies()
    if not install_success:
        print("❌ Failed to install dependencies")
        return False
    
    # Validate environment
    validation_success = validate_environment()
    if not validation_success:
        print("❌ Environment validation failed")
        return False
    
    print("\n🎯 ENTERPRISE ENVIRONMENT READY!")
    print("=" * 35)
    print("✅ Project structure created")
    print("✅ Dependencies installed")
    print("✅ Environment validated")
    print("\nNext steps:")
    print("  1. Run: python run_all_enterprise_tests.py")
    print("  2. Run: python audit_enterprise_project.py")
    print("  3. Run: python fix_test_files.py")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Setup failed: {e}")
        sys.exit(1)
