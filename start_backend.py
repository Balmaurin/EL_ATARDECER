#!/usr/bin/env python3
"""
🚀 EL-AMANECER Backend Launcher
Inicia el backend GraphQL Federation Gateway en el puerto 8000
Lee configuración desde .env y settings.py
"""

import os
import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from {env_path}")

# Agregar paths necesarios
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Importar settings
try:
    from apps.backend.src.core.config.settings import settings
    host = settings.server_host
    port = settings.server_port
    print(f"✅ Using settings: {host}:{port}")
except Exception as e:
    print(f"⚠️ Could not load settings: {e}")
    # Fallback a variables de entorno o valores por defecto
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    print(f"✅ Using environment/defaults: {host}:{port}")

if __name__ == "__main__":
    print("="*80)
    print("🚀 EL-AMANECER Backend - GraphQL Federation Gateway")
    print("="*80)
    print(f"📡 Server: http://localhost:{port}")
    print(f"📊 GraphQL Playground: http://localhost:{port}/graphql")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print("="*80)
    
    try:
        # Verificar que el módulo existe
        federation_module = project_root / "apps" / "backend" / "federation_server.py"
        if not federation_module.exists():
            print(f"❌ ERROR: No se encontró {federation_module}")
            sys.exit(1)
        
        # Iniciar servidor con uvicorn
        uvicorn.run(
            "apps.backend.federation_server:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Backend detenido por usuario")
    except Exception as e:
        print(f"❌ ERROR al iniciar backend: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
