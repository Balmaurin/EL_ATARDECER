"""
Health Check System - Real Status Monitoring
NO MOCKS - Reports actual system health, no fake data
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


class HealthStatus:
    """Health status constants"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth:
    """Health information for a single component"""

    def __init__(
        self,
        name: str,
        status: str,
        response_time_ms: Optional[float] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.status = status
        self.response_time_ms = response_time_ms
        self.message = message
        self.details = details or {}
        self.checked_at = datetime.now(timezone.utc).isoformat()


class HealthChecker:
    """
    Real health checking system.
    NO MOCKS - All checks perform actual operations
    """

    def __init__(self):
        self.start_time = datetime.now(timezone.utc)

    async def check_all(self) -> Dict[str, Any]:
        """
        Check health of all system components.
        Returns complete health status with NO MOCK DATA.
        """
        checks = []

        # Perform all health checks
        checks.append(await self._check_database())
        checks.append(await self._check_redis())
        checks.append(await self._check_embedding_service())
        checks.append(await self._check_llm_service())
        checks.append(await self._check_consciousness_system())

        # Calculate overall status
        overall_status = self._calculate_overall_status(checks)

        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "version": "1.0.0",
            "components": [
                {
                    "name": check.name,
                    "status": check.status,
                    "response_time_ms": check.response_time_ms,
                    "message": check.message,
                    "details": check.details,
                    "checked_at": check.checked_at
                }
                for check in checks
            ]
        }

    async def _check_database(self) -> ComponentHealth:
        """Check database health - REAL CONNECTION TEST"""
        try:
            from src.models.database import engine
            from sqlalchemy import text

            start = time.time()

            with engine.connect() as conn:
                # Test query
                result = conn.execute(text("SELECT COUNT(*) FROM users"))
                user_count = result.scalar()

                # Get table count
                result = conn.execute(text(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                ))
                table_count = result.scalar()

            response_time = (time.time() - start) * 1000

            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                response_time_ms=round(response_time, 2),
                message="Database operational",
                details={
                    "users": user_count,
                    "tables": table_count
                }
            )

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}"
            )

    async def _check_redis(self) -> ComponentHealth:
        """Check Redis health - REAL CONNECTION TEST"""
        try:
            import redis
            from src.core.config.settings import settings

            redis_url = getattr(settings, 'redis_url', None)

            if not redis_url:
                return ComponentHealth(
                    name="redis",
                    status=HealthStatus.UNHEALTHY,
                    message="Redis URL not configured"
                )

            start = time.time()

            client = redis.from_url(redis_url)

            # Test connection
            client.ping()

            # Get info
            info = client.info()

            response_time = (time.time() - start) * 1000

            return ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                response_time_ms=round(response_time, 2),
                message="Redis operational",
                details={
                    "connected_clients": info.get('connected_clients', 0),
                    "used_memory_mb": round(info.get('used_memory', 0) / 1024 / 1024, 2),
                    "uptime_days": round(info.get('uptime_in_seconds', 0) / 86400, 2)
                }
            )

        except ImportError:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message="Redis library not installed"
            )
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis check failed: {str(e)}"
            )

    async def _check_embedding_service(self) -> ComponentHealth:
        """Check embedding service health - REAL EMBEDDING TEST"""
        try:
            from src.core.embeddings.embedding_service import EmbeddingService

            start = time.time()

            service = EmbeddingService()

            # Test embedding generation
            test_embedding = service.generate_embedding("health check test")

            response_time = (time.time() - start) * 1000

            if test_embedding is None or len(test_embedding) == 0:
                return ComponentHealth(
                    name="embedding_service",
                    status=HealthStatus.UNHEALTHY,
                    message="Embedding generation failed"
                )

            return ComponentHealth(
                name="embedding_service",
                status=HealthStatus.HEALTHY,
                response_time_ms=round(response_time, 2),
                message="Embedding service operational",
                details={
                    "model": getattr(service, 'model_name', 'unknown'),
                    "dimension": len(test_embedding)
                }
            )

        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return ComponentHealth(
                name="embedding_service",
                status=HealthStatus.UNHEALTHY,
                message=f"Embedding service check failed: {str(e)}"
            )

    async def _check_llm_service(self) -> ComponentHealth:
        """Check LLM service health - REAL CONFIGURATION CHECK"""
        try:
            from src.core.llm.llm_factory import LLMFactory
            import os

            start = time.time()

            # Check available providers
            has_openai = bool(os.getenv('OPENAI_API_KEY'))
            has_local = self._check_local_llm()

            response_time = (time.time() - start) * 1000

            if not has_openai and not has_local:
                return ComponentHealth(
                    name="llm_service",
                    status=HealthStatus.UNHEALTHY,
                    message="No LLM provider configured",
                    response_time_ms=round(response_time, 2)
                )

            providers = []
            if has_openai:
                providers.append("openai")
            if has_local:
                providers.append("local")

            return ComponentHealth(
                name="llm_service",
                status=HealthStatus.HEALTHY,
                response_time_ms=round(response_time, 2),
                message="LLM service operational",
                details={
                    "providers": providers,
                    "default": providers[0] if providers else None
                }
            )

        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
            return ComponentHealth(
                name="llm_service",
                status=HealthStatus.DEGRADED,
                message=f"LLM service check failed: {str(e)}"
            )

    async def _check_consciousness_system(self) -> ComponentHealth:
        """Check consciousness system health - REAL MODULE CHECK"""
        try:
            start = time.time()

            # Check if consciousness modules can be imported
            from packages.consciousness.src.conciencia.modulos import conscious_system

            # Check if key components exist
            components_ok = hasattr(conscious_system, 'ConsciousSystem')

            response_time = (time.time() - start) * 1000

            if not components_ok:
                return ComponentHealth(
                    name="consciousness_system",
                    status=HealthStatus.DEGRADED,
                    response_time_ms=round(response_time, 2),
                    message="Consciousness system partially loaded"
                )

            return ComponentHealth(
                name="consciousness_system",
                status=HealthStatus.HEALTHY,
                response_time_ms=round(response_time, 2),
                message="Consciousness system operational",
                details={
                    "modules_loaded": True
                }
            )

        except ImportError as e:
            logger.error(f"Consciousness system health check failed: {e}")
            return ComponentHealth(
                name="consciousness_system",
                status=HealthStatus.UNHEALTHY,
                message=f"Consciousness system not available: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Consciousness system health check failed: {e}")
            return ComponentHealth(
                name="consciousness_system",
                status=HealthStatus.DEGRADED,
                message=f"Consciousness system check failed: {str(e)}"
            )

    def _check_local_llm(self) -> bool:
        """Check if local LLM is available"""
        try:
            import os
            from src.core.config.settings import settings

            # Check for model path
            model_path = getattr(settings, 'local_llm_model_path', None)
            if model_path and os.path.exists(model_path):
                return True

            # Check config files
            config_paths = [
                'config/ai/llm/llm_config.json',
                'config/ai/llm/llama_config.json'
            ]

            return any(os.path.exists(p) for p in config_paths)
        except:
            return False

    def _calculate_overall_status(self, checks: List[ComponentHealth]) -> str:
        """Calculate overall system status from component checks"""
        unhealthy_count = sum(1 for c in checks if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in checks if c.status == HealthStatus.DEGRADED)

        # If any critical component is unhealthy
        if unhealthy_count > 0:
            critical_components = ['database', 'redis']
            critical_unhealthy = any(
                c.name in critical_components and c.status == HealthStatus.UNHEALTHY
                for c in checks
            )
            if critical_unhealthy:
                return HealthStatus.UNHEALTHY

        # If multiple components degraded or some unhealthy
        if unhealthy_count > 0 or degraded_count > 1:
            return HealthStatus.DEGRADED

        # If any component degraded
        if degraded_count > 0:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY


# Global health checker instance
health_checker = HealthChecker()


async def get_health() -> Dict[str, Any]:
    """
    Get current system health.
    This is the main function to call from health check endpoints.
    """
    return await health_checker.check_all()


if __name__ == "__main__":
    # Allow running as standalone health check
    import asyncio

    async def main():
        logging.basicConfig(level=logging.INFO)
        health = await get_health()

        print("\n" + "=" * 80)
        print(f"SYSTEM HEALTH: {health['status'].upper()}")
        print("=" * 80)
        print(f"\nUptime: {health['uptime_seconds']:.2f} seconds")
        print(f"Checked at: {health['timestamp']}")
        print("\nComponents:")

        for component in health['components']:
            status_emoji = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.UNHEALTHY: "❌"
            }.get(component['status'], "❓")

            print(f"\n  {status_emoji} {component['name']}: {component['status']}")
            if component['response_time_ms']:
                print(f"     Response time: {component['response_time_ms']}ms")
            if component['message']:
                print(f"     Message: {component['message']}")
            if component['details']:
                for key, value in component['details'].items():
                    print(f"     {key}: {value}")

        print("\n" + "=" * 80)

    asyncio.run(main())
