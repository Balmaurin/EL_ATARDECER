"""
Dapr Installation and Health Verification Script
Verifies that Dapr is properly installed and all components are healthy
"""

import subprocess
import sys
import json
from pathlib import Path

def run_command(cmd: list[str]) -> tuple[bool, str]:
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def check_dapr_cli():
    """Check if Dapr CLI is installed"""
    print("🔍 Checking Dapr CLI installation...")
    success, output = run_command(["dapr", "--version"])
    
    if success:
        print(f"✅ Dapr CLI installed: {output.strip()}")
        return True
    else:
        print("❌ Dapr CLI not found. Run: scripts/install_dapr.ps1")
        return False

def check_dapr_components():
    """Check if Dapr components are running"""
    print("\n🔍 Checking Dapr components...")
    success, output = run_command(["dapr", "components", "-k"])
    
    if success:
        print("✅ Dapr components:")
        print(output)
        return True
    else:
        print("⚠️  Could not list Dapr components")
        print(output)
        return False

def check_redis():
    """Check if Redis (state store) is accessible"""
    print("\n🔍 Checking Redis state store...")
    success, output = run_command(["docker", "ps", "--filter", "name=dapr_redis", "--format", "{{.Names}}"])
    
    if success and "dapr_redis" in output:
        print("✅ Redis is running (Dapr state store)")
        return True
    else:
        print("⚠️  Redis not running. State store unavailable.")
        return False

def check_zipkin():
    """Check if Zipkin (tracing) is accessible"""
    print("\n🔍 Checking Zipkin tracing...")
    success, output = run_command(["docker", "ps", "--filter", "name=dapr_zipkin", "--format", "{{.Names}}"])
    
    if success and "dapr_zipkin" in output:
        print("✅ Zipkin is running (Dapr tracing)")
        return True
    else:
        print("ℹ️  Zipkin not running (optional)")
        return False

def verify_component_configs():
    """Verify that component configuration files exist"""
    print("\n🔍 Checking component configuration files...")
    
    components_dir = Path("dapr/components")
    required_files = ["pubsub.yaml", "statestore.yaml", "secrets.yaml"]
    
    all_exist = True
    for file in required_files:
        file_path = components_dir / file
        if file_path.exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def main():
    """Main verification function"""
    print("=" * 60)
    print("🚀 Dapr Installation & Health Verification")
    print("=" * 60)
    
    checks = [
        ("Dapr CLI", check_dapr_cli),
        ("Component Configs", verify_component_configs),
        ("Dapr Components", check_dapr_components),
        ("Redis State Store", check_redis),
        ("Zipkin Tracing", check_zipkin),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Verification Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}")
    
    print(f"\n🎯 Score: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✨ All checks passed! Dapr is ready for consciousness integration.")
        return 0
    elif passed >= 3:
        print("\n⚠️  Some checks failed, but core functionality should work.")
        return 0
    else:
        print("\n❌ Critical checks failed. Please fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
