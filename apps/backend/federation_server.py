#!/usr/bin/env python3
"""
[TARGET] GraphQL Federation Gateway - API Consolidation
Consolida 100+ endpoints REST → 4 queries GraphQL federadas + TODO System

Extrae todas las rutas de EL-AMANECER en schema único tipado:
- Consciousness: Procesamiento neural y emocional
- Chat: Conversaciones + WebSocket state + LLM Integration
- Agent: Orchestration + MCP coordination
- Enterprise: Users, analytics, marketplace
- TODO System: Gestión unificada de tareas (único sistema)

Con factores de consolidación:
- Auth+Users → 7 endpoints → 1 query
- Chat+Conversations → 8 endpoints → 1 query + subscriptions + LLM
- Consciousness → 7 endpoints → 1 query + streaming
- Sistema → 15 endpoints → 1 query admin
- TODO → Sistema unificado completo

Total: 35+ archivos de rutas → 8 queries GraphQL + 10 mutations principales
"""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime
from typing import List, Optional, AsyncGenerator, Dict, Any
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL
from strawberry.types import Info
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import Schema and Service
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow absolute imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from apps.backend.graphql_schema import Query, Mutation, Subscription
from apps.backend.todo_service import TodoService
from apps.backend.hack_memori_service import HackMemoriService
from apps.backend.neural_consciousness_api import router as neural_consciousness_router

# Configuración básica - sin imports problemáticos
logger = logging.getLogger(__name__)
logger.info("[OK] Federation Gateway inicializando con sistemas reales")

class FederationGateway:
    """GraphQL Federation Gateway para consolidation de APIs"""

    def __init__(self):
        self.app: Optional[FastAPI] = None
        self.graphql_app = None
        self.consciousness_system = None
        self.neural_system = None
        self.todo_service = TodoService()
        self.hack_memori_service = HackMemoriService()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Lifecycle management para servicios críticos"""
        # Startup
        logger.info("[START] Initializing Federation Gateway...")

        # Initialize consciousness system with auto-detection
        try:
            # Import reorganized consciousness system
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
            consciousness_path = os.path.join(root_dir, "packages", "consciousness", "src")
            if consciousness_path not in sys.path:
                sys.path.append(consciousness_path)

            # Use reorganized detection system with robust error handling
            try:
                from conciencia.modulos import detect_consciousness_system, META_COGNITION_AVAILABLE
                consciousness_available = True
            except ImportError as e:
                logger.warning(f"[WARN] Consciousness detection module not available: {e}")
                consciousness_available = False
                META_COGNITION_AVAILABLE = False

            if consciousness_available:
                # Auto-detect consciousness system status
                try:
                    consciousness_status = detect_consciousness_system()
                    logger.info(f"[DETECT] Consciousness system: {consciousness_status['system_health']} ({consciousness_status['available_modules']}/{consciousness_status['total_theories']} theories)")
                except Exception as e:
                    logger.warning(f"[WARN] Consciousness detection failed: {e}")
                    consciousness_status = {"system_health": "unknown", "available_modules": 0, "total_theories": 0}

                # Initialize if available
                if META_COGNITION_AVAILABLE:
                    try:
                        from conciencia.meta_cognition_system import MetaCognitionSystem
                        consciousness_dir = os.path.join(root_dir, "data", "consciousness")
                        self.consciousness_system = MetaCognitionSystem(
                            consciousness_dir=consciousness_dir,
                            emergence_threshold=0.85
                        )
                        logger.info("[OK] Consciousness system initialized with auto-detection")
                    except ImportError as e:
                        logger.warning(f"[WARN] Meta cognition system not available: {e}")
                        # Try to use basic IIT engine as fallback
                        try:
                            from conciencia.modulos import IITEngine
                            self.consciousness_system = IITEngine()
                            logger.info("[OK] Using IIT Engine as fallback consciousness system")
                        except ImportError:
                            logger.error("[ERROR] No consciousness system available")
                            self.consciousness_system = None
                else:
                    logger.warning("[WARN] Consciousness system modules not fully available")
                    self.consciousness_system = None
            else:
                logger.warning("[WARN] Consciousness system completely unavailable - proceeding without it")
                self.consciousness_system = None

        except Exception as e:
            logger.error(f"[ERROR] Consciousness system auto-detection failed: {e}")
            self.consciousness_system = None

        # Initialize Neural Consciousness System
        try:
            from packages.consciousness.src.conciencia.modulos.neural_modules.neural_consciousness_system import NeuralConsciousnessSystem
            from apps.backend.src.core.config.settings import settings
            from apps.backend.neural_consciousness_api import set_neural_system
            
            # Configurar modelo local GGUF si está disponible
            llm_model_path = None
            if hasattr(settings, 'llm') and hasattr(settings.llm, 'model_path'):
                model_path = Path(settings.llm.model_path)
                if model_path.exists():
                    llm_model_path = str(model_path)
                    logger.info(f"Using local GGUF model: {llm_model_path}")
            
            neural_config = {
                "llm_service_url": settings.llm_service_url,
                "llm_model_id": settings.llm_model_id,
                "llm_model_path": llm_model_path,  # Ruta al modelo local GGUF
                "llm_n_ctx": getattr(settings.llm, 'n_ctx', 4096) if hasattr(settings, 'llm') else 4096,
                "llm_n_threads": getattr(settings.llm, 'n_threads', 4) if hasattr(settings, 'llm') else 4,
                "llm_chat_format": getattr(settings.llm, 'chat_format', 'chatml') if hasattr(settings, 'llm') else 'chatml',
                "training_interval": 100,
                "brain_state_file": str(project_root / "data" / "consciousness" / "brain_state.json")
            }
            
            self.neural_system = NeuralConsciousnessSystem(config=neural_config, device="cpu")
            set_neural_system(self.neural_system)  # Hacer disponible para API
            logger.info("[OK] Neural Consciousness System initialized")
        except Exception as e:
            logger.warning(f"[WARN] Neural Consciousness System initialization failed: {e}")
            self.neural_system = None

        # Initialize HACK-MEMORI Automatic Service
        hack_memori_task = None
        try:
            import os
            auto_mode = os.getenv('HACK_MEMORI_AUTO_MODE', 'true').lower() == 'true'
            
            if auto_mode:
                logger.info("🤖 Starting HACK-MEMORI Auto-Generation Service...")
                from apps.backend.hack_memori_service import run_automatic_service
                
                # Start as background task
                hack_memori_task = asyncio.create_task(run_automatic_service())
                logger.info("✅ HACK-MEMORI Auto-Generation Service started")
            else:
                logger.info("ℹ️ HACK-MEMORI Auto-Generation disabled (set HACK_MEMORI_AUTO_MODE=true to enable)")
        except Exception as e:
            logger.warning(f"[WARN] HACK-MEMORI Auto-Generation initialization failed: {e}")

        # Keep existing services running
        yield

        # Shutdown
        logger.info("[STOP] Federation Gateway shutting down...")
        
        # Stop HACK-MEMORI service
        if hack_memori_task and not hack_memori_task.done():
            logger.info("🛑 Stopping HACK-MEMORI Auto-Generation Service...")
            hack_memori_task.cancel()
            try:
                await hack_memori_task
            except asyncio.CancelledError:
                logger.info("✅ HACK-MEMORI Auto-Generation Service stopped")
        
        # Shutdown Neural System
        if self.neural_system:
            try:
                self.neural_system.shutdown()
                logger.info("[OK] Neural Consciousness System shutdown complete")
            except Exception as e:
                logger.error(f"[ERROR] Error shutting down Neural System: {e}")
        
        # Shutdown Neural System
        if self.neural_system:
            try:
                self.neural_system.shutdown()
                logger.info("[OK] Neural Consciousness System shutdown complete")
            except Exception as e:
                logger.error(f"[ERROR] Error shutting down Neural System: {e}")

    def create_app(self) -> FastAPI:
        """Crear FastAPI app con GraphQL gateway"""

        @asynccontextmanager
        async def app_lifespan(app: FastAPI):
            async with self.lifespan(app):
                yield

        app = FastAPI(
            title="EL-AMANECER V4 - GraphQL Federation Gateway",
            description="API unificada que consolida 100+ endpoints REST en 4 queries federadas",
            version="4.0.0",
            lifespan=app_lifespan
        )

        # CORS para frontend
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Health check endpoint (monitoring only)
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "gateway": "operational"}

        # ═══════════════════════════════════════════════════════════════════
        # [TARGET] GRAPHQL-ONLY ARCHITECTURE - ZERO REST ENDPOINTS
        # ═══════════════════════════════════════════════════════════════════
        logger.info("[TARGET] GraphQL Federation Gateway - Pure GraphQL-native architecture initialized")

        async def get_context():
            return {
                "gateway": self,
                "todo_service": self.todo_service,
                "hack_memori_service": self.hack_memori_service,
                "neural_system": self.neural_system
            }

        # GraphQL Federation Gateway
        schema = strawberry.Schema(
            query=Query,
            mutation=Mutation,
            subscription=Subscription
        )

        self.graphql_app = GraphQLRouter(
            schema=schema,
            path="/graphql",
            context_getter=get_context,
            subscription_protocols=[GRAPHQL_WS_PROTOCOL, GRAPHQL_TRANSPORT_WS_PROTOCOL]
        )

        app.include_router(self.graphql_app, prefix="")
        
        # Include Neural Consciousness API router
        app.include_router(neural_consciousness_router)
        
        self.app = app

        logger.info("[OK] GraphQL Federation Gateway creado exitosamente")
        logger.info("[OK] Neural Consciousness API integrated")
        return app

    async def _fetch_from_system_direct_imports(self) -> dict:
        """Fetch data using direct EL-AMANECER system imports (no HTTP)"""
        try:
            # Import main system components directly - native to EL-AMANECER
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.abspath(os.path.join(current_dir, "../../"))

            # Add packages to sys.path for native imports
            consciousness_path = os.path.join(root_dir, "packages", "consciousness", "src")
            if consciousness_path not in sys.path:
                sys.path.append(consciousness_path)

            sheily_core_path = os.path.join(root_dir, "packages", "sheily_core", "src")
            if sheily_core_path not in sys.path:
                sys.path.append(sheily_core_path)

            # Try importing real consciousness system
            try:
                from conciencia.meta_cognition_system import MetaCognitionSystem
                consciousness_dir = os.path.join(root_dir, "data", "consciousness")
                meta_system = MetaCognitionSystem(
                    consciousness_dir=consciousness_dir,
                    emergence_threshold=0.85
                )
                real_consciousness = True
            except ImportError:
                meta_system = None
                real_consciousness = False

            # Import dashboard function directly
            try:
                dashboard_path = os.path.join(root_dir, "apps", "backend", "src", "api", "v1", "routes")
                if dashboard_path not in sys.path:
                    sys.path.append(dashboard_path)
                from dashboard import get_consciousness_dashboard
                dashboard_available = True
            except ImportError:
                dashboard_available = False

            # Aggregate real data from systems
            data_aggregated = {}

            # Consciousness data
            if meta_system and real_consciousness:
                try:
                    state = meta_system.current_cognitive_state
                    # Get real phi value from system
                    phi_value = getattr(state, 'phi_value', getattr(state, 'meta_awareness', 0.0))
                    # Get real cognitive state
                    cognitive_state = getattr(state, 'cognitive_state', 'neutral')
                    # Calculate real cognitive load
                    working_memory = getattr(state, 'working_memory', [])
                    cognitive_load = min(1.0, len(working_memory) / 10.0) if working_memory else 0.0
                    # Get real memory counts from persistence if available
                    total_memories = getattr(meta_system, 'total_memories', 0)
                    learning_experiences = getattr(meta_system, 'learning_experiences', 0)
                    
                    data_aggregated["consciousness"] = {
                        "score": phi_value,
                        "emotion": cognitive_state,
                        "load": cognitive_load,
                        "last_thought": getattr(state, 'current_thought', ''),
                        "phi_value": phi_value,
                        "arousal": getattr(state, 'arousal', 0.5),
                        "complexity": getattr(state, 'complexity', 0.85),
                        "active_circuits": len(working_memory) if working_memory else 0,
                        "cognitive_load": cognitive_load,
                        "awareness_level": getattr(state, 'meta_awareness', phi_value),
                        "total_memories": total_memories,
                        "learning_experiences": learning_experiences
                    }
                except Exception as e:
                    logger.error(f"Error getting consciousness data: {e}")
                    raise

            # Dashboard data if available
            if dashboard_available:
                try:
                    dashboard_data = await get_consciousness_dashboard()
                    if real_consciousness:
                        data_aggregated.update(dashboard_data)
                    else:
                        data_aggregated["consciousness"] = dashboard_data.get("consciousness", {})
                except Exception:
                    pass

            # System metrics from native sources
            data_aggregated["system_status"] = {
                "health": "healthy",
                "direct_imports_active": True,
                "consciousness_system_integrated": real_consciousness,
                "last_native_access": datetime.now().isoformat(),
                "architecture_type": "integrated_direct_imports"
            }

            return data_aggregated

        except Exception as e:
            logger.error(f"Native system imports failed: {e}")
            # Fail explicitly instead of returning mock data
            raise RuntimeError(f"Failed to fetch system data: {e}") from e

def create_federation_gateway() -> FastAPI:
    """Factory function para crear el gateway consolidado"""
    gateway = FederationGateway()
    return gateway.create_app()

# Create app instance for uvicorn
app = create_federation_gateway()

# Para ejecución standalone
if __name__ == "__main__":
    print("[TARGET] EL-AMANECER GraphQL Federation Gateway")
    print("Consolidación: 100+ endpoints → 4 queries federadas")
    print("[CHART] Schema: Consciousness + Chat + Agent + Enterprise")
    from apps.backend.src.core.config.settings import settings
    base_url = f"http://localhost:{settings.server_port}"
    print(f"[START] {base_url}/graphql")
    print(f"📚 Docs: {base_url}/graphql")

    from apps.backend.src.core.config.settings import settings
    
    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port,
        log_level="info"
    )
