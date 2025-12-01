#!/usr/bin/env python3
"""
🚀 EL-AMANECER Frontend Launcher
Inicia el frontend Next.js en el puerto 3000
Lee configuración desde .env
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from {env_path}")

# Obtener puerto del frontend
port = os.getenv("NEXT_PUBLIC_PORT", "3000")
api_url = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000")

if __name__ == "__main__":
    print("="*80)
    print("🌐 EL-AMANECER Frontend - Next.js")
    print("="*80)
    print(f"📡 Server: http://localhost:{port}")
    print(f"🔗 Backend API: {api_url}")
    print("="*80)
    
    try:
        # Cambiar al directorio de apps
        apps_dir = Path(__file__).parent / "apps"
        if not apps_dir.exists():
            print(f"❌ ERROR: No se encontró el directorio {apps_dir}")
            sys.exit(1)
        
        os.chdir(apps_dir)
        print(f"📂 Working directory: {apps_dir}")
        
        # Verificar que package.json existe
        package_json = apps_dir / "package.json"
        if not package_json.exists():
            print(f"❌ ERROR: No se encontró {package_json}")
            sys.exit(1)
        
        # Iniciar Next.js con turbo
        print("\n🚀 Starting Next.js with Turbopack...")
        process = subprocess.run(
            ["npx", "next", "dev", "--turbo", "--port", port],
            cwd=apps_dir,
            shell=True
        )
        
        sys.exit(process.returncode)
        
    except KeyboardInterrupt:
        print("\n🛑 Frontend detenido por usuario")
    except Exception as e:
        print(f"❌ ERROR al iniciar frontend: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
