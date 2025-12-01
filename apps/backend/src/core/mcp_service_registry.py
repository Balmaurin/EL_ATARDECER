"""
MCP Service Registry
Registro centralizado de todos los servicios backend para el MCP Orchestrator
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Estado de un servicio"""
    AVAILABLE = "available"
    INITIALIZING = "initializing"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ServiceCategory(Enum):
    """Categoría de servicio"""
    CORE = "core"
    API = "api"
    SERVICE = "service"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class ServiceEndpoint:
    """Endpoint de un servicio"""
    path: str
    method: str
    description: str
    requires_auth: bool = True


@dataclass
class RegisteredService:
    """Servicio registrado en el MCP"""
    id: str
    name: str
    category: ServiceCategory
    module_path: str
    status: ServiceStatus
    version: str = "1.0.0"
    description: str = ""
    endpoints: List[ServiceEndpoint] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = None
    health_score: float = 1.0


class MCPServiceRegistry:
    """
    Registro centralizado de servicios para MCP Orchestrator
    Mantiene inventario de todos los módulos backend disponibles
    """
    
    def __init__(self):
        self.services: Dict[str, RegisteredService] = {}
        self._initialize_core_services()
        
    def _initialize_core_services(self):
        """Registrar los 9 módulos core verificados"""
        
        core_services = [
            # CORE INFRASTRUCTURE
            RegisteredService(
                id="core.config",
                name="Configuration Service",
                category=ServiceCategory.CORE,
                module_path="apps.backend.src.core.config.settings",
                status=ServiceStatus.AVAILABLE,
                description="Sistema de configuración centralizado",
                capabilities=["settings_management", "environment_config"],
                metadata={"location": "core/config/"}
            ),
            RegisteredService(
                id="core.auth",
                name="Authentication Service",
                category=ServiceCategory.CORE,
                module_path="apps.backend.src.core.auth.service",
                status=ServiceStatus.AVAILABLE,
                description="Servicio de autenticación JWT y gestión de usuarios",
                capabilities=["jwt_auth", "password_hashing", "token_management"],
                dependencies={"core.config", "core.database"},
                metadata={"location": "core/auth/"}
            ),
            RegisteredService(
                id="core.security",
                name="Security Manager",
                category=ServiceCategory.CORE,
                module_path="apps.backend.src.core.security.manager",
                status=ServiceStatus.AVAILABLE,
                description="Gestor de seguridad enterprise con rate limiting y sanitización",
                capabilities=["rate_limiting", "input_sanitization", "csrf_protection"],
                dependencies={"core.config"},
                metadata={"location": "core/security/"}
            ),
            RegisteredService(
                id="core.database",
                name="Database Session Manager",
                category=ServiceCategory.CORE,
                module_path="apps.backend.src.core.database.session",
                status=ServiceStatus.AVAILABLE,
                description="Gestión de sesiones de base de datos PostgreSQL con pooling",
                capabilities=["connection_pooling", "session_management", "rls_support"],
                dependencies={"core.config"},
                metadata={"location": "core/database/"}
            ),
            
            # SERVICES
            RegisteredService(
                id="service.ai",
                name="AI Service",
                category=ServiceCategory.SERVICE,
                module_path="apps.backend.src.services.ai.ai_service",
                status=ServiceStatus.AVAILABLE,
                description="Servicio principal de AI con LLM y razonamiento",
                capabilities=["llm_inference", "reasoning_engine", "agent_orchestration"],
                dependencies={"core.config", "service.chat"},
                metadata={"location": "services/ai/"}
            ),
            
            # API ROUTES
            RegisteredService(
                id="api.auth",
                name="Auth API Routes",
                category=ServiceCategory.API,
                module_path="apps.backend.src.api.v1.routes.auth",
                status=ServiceStatus.AVAILABLE,
                description="Endpoints de autenticación (login, registro, tokens)",
                endpoints=[
                    ServiceEndpoint("/api/auth/login", "POST", "User login"),
                    ServiceEndpoint("/api/auth/register", "POST", "User registration"),
                    ServiceEndpoint("/api/auth/refresh", "POST", "Refresh JWT token"),
                    ServiceEndpoint("/api/auth/profile", "GET", "Get user profile"),
                ],
                dependencies={"core.auth", "core.database"},
                capabilities=["user_authentication", "token_refresh"],
                metadata={"location": "api/v1/routes/"}
            ),
            RegisteredService(
                id="api.chat",
                name="Chat API Routes",
                category=ServiceCategory.API,
                module_path="apps.backend.src.api.v1.routes.chat",
                status=ServiceStatus.AVAILABLE,
                description="Endpoints de chat con LLM local y streaming",
                endpoints=[
                    ServiceEndpoint("/api/chat/message", "POST", "Send chat message"),
                    ServiceEndpoint("/api/chat/stream", "POST", "Stream chat response"),
                    ServiceEndpoint("/api/chat/history", "GET", "Get chat history"),
                ],
                dependencies={"service.ai", "core.database"},
                capabilities=["llm_chat", "streaming_responses", "conversation_management"],
                metadata={"location": "api/v1/routes/"}
            ),
            RegisteredService(
                id="api.users",
                name="Users API Routes",
                category=ServiceCategory.API,
                module_path="apps.backend.src.api.v1.routes.users",
                status=ServiceStatus.AVAILABLE,
                description="Endpoints de gestión de usuarios y perfiles",
                endpoints=[
                    ServiceEndpoint("/api/users/profile", "GET", "Get user profile"),
                    ServiceEndpoint("/api/users/update", "PUT", "Update user profile"),
                    ServiceEndpoint("/api/users/tokens", "GET", "Get token balance"),
                    ServiceEndpoint("/api/users/avatar", "POST", "Upload avatar"),
                ],
                dependencies={"core.auth", "core.database"},
                capabilities=["profile_management", "token_management", "avatar_upload"],
                metadata={"location": "api/v1/routes/"}
            ),
            RegisteredService(
                id="api.datasets",
                name="Datasets API Routes",
                category=ServiceCategory.API,
                module_path="apps.backend.src.api.v1.routes.datasets",
                status=ServiceStatus.AVAILABLE,
                description="Endpoints de gestión de datasets y ejercicios",
                endpoints=[
                    ServiceEndpoint("/api/datasets/list", "GET", "List all datasets"),
                    ServiceEndpoint("/api/datasets/create", "POST", "Create new dataset"),
                    ServiceEndpoint("/api/datasets/train", "POST", "Train on dataset"),
                ],
                dependencies={"core.auth", "core.database"},
                capabilities=["dataset_management", "training", "exercises"],
                metadata={"location": "api/v1/routes/"}
            ),
        ]
        
        for service in core_services:
            self.services[service.id] = service
            logger.info(f"🔌 Registered service: {service.name} ({service.id})")
    
    def register_service(self, service: RegisteredService) -> bool:
        """Registrar un nuevo servicio"""
        try:
            self.services[service.id] = service
            logger.info(f"[OK] Service registered: {service.name}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to register service {service.name}: {e}")
            return False
    
    def get_service(self, service_id: str) -> Optional[RegisteredService]:
        """Obtener servicio por ID"""
        return self.services.get(service_id)
    
    def get_services_by_category(self, category: ServiceCategory) -> List[RegisteredService]:
        """Obtener servicios por categoría"""
        return [s for s in self.services.values() if s.category == category]
    
    def get_all_services(self) -> Dict[str, RegisteredService]:
        """Obtener todos los servicios"""
        return self.services.copy()
    
    def get_service_status(self, service_id: str) -> Optional[ServiceStatus]:
        """Obtener estado de un servicio"""
        service = self.get_service(service_id)
        return service.status if service else None
    
    def update_service_status(self, service_id: str, status: ServiceStatus) -> bool:
        """Actualizar estado de un servicio"""
        if service_id in self.services:
            self.services[service_id].status = status
            return True
        return False
    
    def health_check_service(self, service_id: str) -> bool:
        """Verificar salud de un servicio"""
        service = self.get_service(service_id)
        if not service:
            return False
        
        # Actualizar timestamp de health check
        service.last_health_check = datetime.now()
        
        # Por ahora, asumimos que los servicios están healthy si están registrados
        service.health_score = 1.0
        service.status = ServiceStatus.AVAILABLE
        return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del registro"""
        return {
            "total_services": len(self.services),
            "by_category": {
                cat.value: len(self.get_services_by_category(cat))
                for cat in ServiceCategory
            },
            "by_status": {
                status.value: len([s for s in self.services.values() if s.status == status])
                for status in ServiceStatus
            },
            "total_endpoints": sum(len(s.endpoints) for s in self.services.values()),
            "total_capabilities": sum(len(s.capabilities) for s in self.services.values()),
        }
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Obtener grafo de dependencias"""
        return {
            service_id: list(service.dependencies)
            for service_id, service in self.services.items()
        }
    
    def export_service_manifest(self) -> Dict[str, Any]:
        """Exportar manifiesto completo de servicios para MCP"""
        return {
            "registry_version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "services": {
                service_id: {
                    "name": service.name,
                    "category": service.category.value,
                    "module_path": service.module_path,
                    "status": service.status.value,
                    "version": service.version,
                    "description": service.description,
                    "capabilities": service.capabilities,
                    "dependencies": list(service.dependencies),
                    "endpoints": [
                        {
                            "path": ep.path,
                            "method": ep.method,
                            "description": ep.description,
                            "requires_auth": ep.requires_auth
                        }
                        for ep in service.endpoints
                    ],
                    "health_score": service.health_score,
                    "last_health_check": service.last_health_check.isoformat() if service.last_health_check else None
                }
                for service_id, service in self.services.items()
            },
            "stats": self.get_registry_stats(),
            "dependency_graph": self.get_dependency_graph()
        }


# Instancia global del registro
service_registry = MCPServiceRegistry()

# Export
__all__ = [
    "MCPServiceRegistry",
    "service_registry",
    "ServiceStatus",
    "ServiceCategory",
    "RegisteredService",
    "ServiceEndpoint"
]
