"""
Middleware de Rate Limiting para FastAPI
Implementa control de tasa de requests usando Redis
"""

import logging
import time
from typing import Any, Callable, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from redis import Redis

from apps.backend.src.core.config.settings import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Clase para manejar rate limiting con Redis"""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    def is_allowed(self, key: str, limit: int, window: int) -> Dict[str, Any]:
        """
        Verifica si una request está dentro del límite

        Args:
            key: Identificador único (ej: "user:123", "ip:192.168.1.1")
            limit: Número máximo de requests en la ventana
            window: Ventana de tiempo en segundos

        Returns:
            Dict con información del rate limit
        """
        current_time = int(time.time())
        window_start = current_time - window

        # Usar Redis sorted set para sliding window
        # Remover requests fuera de la ventana
        self.redis.zremrangebyscore(key, 0, window_start)

        # Contar requests en la ventana actual
        request_count = self.redis.zcard(key)

        # Calcular tiempo hasta reset
        oldest_request = self.redis.zrange(key, 0, 0, withscores=True)
        if oldest_request:
            reset_time = int(oldest_request[0][1]) + window
        else:
            reset_time = current_time + window

        # Verificar si está dentro del límite
        if request_count >= limit:
            return {
                "allowed": False,
                "current": request_count,
                "limit": limit,
                "remaining": 0,
                "reset_time": reset_time,
                "retry_after": max(1, reset_time - current_time),
            }

        # Agregar request actual
        self.redis.zadd(key, {str(current_time): current_time})

        # Establecer expiración para limpiar automáticamente
        self.redis.expire(key, window * 2)

        return {
            "allowed": True,
            "current": request_count + 1,
            "limit": limit,
            "remaining": limit - (request_count + 1),
            "reset_time": reset_time,
            "retry_after": 0,
        }


class RateLimitMiddleware:
    """Middleware de FastAPI para rate limiting"""

    def __init__(self, redis_client: Redis):
        self.rate_limiter = RateLimiter(redis_client)
        self.exempt_paths = {"/health", "/docs", "/redoc", "/openapi.json"}

    async def __call__(self, request: Request, call_next: Callable):
        """Middleware principal"""

        # Eximir ciertos paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Obtener identificadores para rate limiting
        identifiers = self._get_identifiers(request)

        # Aplicar rate limiting por IP
        ip_limit = self.rate_limiter.is_allowed(
            f"ip:{identifiers['ip']}",
            limit=settings.security.rate_limit_requests,
            window=settings.security.rate_limit_window,
        )

        if not ip_limit["allowed"]:
            logger.warning(f"Rate limit exceeded for IP: {identifiers['ip']}")
            return self._rate_limit_response(ip_limit)

        # Aplicar rate limiting por usuario si está autenticado
        if identifiers.get("user_id"):
            user_limit = self.rate_limiter.is_allowed(
                f"user:{identifiers['user_id']}",
                limit=settings.security.rate_limit_requests
                * 2,  # Límite más alto para usuarios
                window=settings.security.rate_limit_window,
            )

            if not user_limit["allowed"]:
                logger.warning(
                    f"Rate limit exceeded for user: {identifiers['user_id']}"
                )
                return self._rate_limit_response(user_limit)

        # Continuar con la request
        response = await call_next(request)

        # Agregar headers de rate limit a la respuesta
        response.headers["X-RateLimit-Limit"] = str(ip_limit["limit"])
        response.headers["X-RateLimit-Remaining"] = str(ip_limit["remaining"])
        response.headers["X-RateLimit-Reset"] = str(ip_limit["reset_time"])

        return response

    def _get_identifiers(self, request: Request) -> Dict[str, str]:
        """Extraer identificadores de la request"""
        identifiers = {}

        # IP del cliente
        identifiers["ip"] = self._get_client_ip(request)

        # User ID si está autenticado (extraer del token JWT)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                # Aquí iría la lógica para decodificar el JWT y extraer user_id
                # Por ahora, solo extraemos un identificador básico
                identifiers["user_id"] = self._extract_user_from_token(auth_header[7:])
            except Exception:
                pass

        return identifiers

    def _get_client_ip(self, request: Request) -> str:
        """Obtener IP real del cliente considerando proxies"""
        # Verificar headers de proxy en orden de prioridad
        for header in ["X-Forwarded-For", "X-Real-IP", "CF-Connecting-IP"]:
            if header in request.headers:
                # Tomar la primera IP si hay múltiples
                ip = request.headers[header].split(",")[0].strip()
                if ip:
                    return ip

        # Fallback a la IP directa
        return request.client.host if request.client else "unknown"

    def _extract_user_from_token(self, token: str) -> str:
        """Extraer user_id del token JWT (simplificado)"""
        # En una implementación real, validaríamos el token completo
        # Por ahora, retornamos un identificador basado en el token
        try:
            # Aquí iría la lógica real de decodificación JWT
            return f"token_{hash(token) % 10000}"  # Simplificado
        except Exception:
            return "unknown"

    def _rate_limit_response(self, limit_info: Dict[str, Any]) -> JSONResponse:
        """Generar respuesta de rate limit excedido"""
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Try again in {limit_info['retry_after']} seconds.",  # noqa: E501
                "retry_after": limit_info["retry_after"],
                "limit": limit_info["limit"],
                "reset_time": limit_info["reset_time"],
            },
            headers={
                "Retry-After": str(limit_info["retry_after"]),
                "X-RateLimit-Limit": str(limit_info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(limit_info["reset_time"]),
            },
        )


# Instancia global del middleware
def create_rate_limit_middleware():
    """Crear instancia del middleware de rate limiting"""
    try:
        redis_client = Redis(
            host=settings.database.redis_host,
            port=settings.database.redis_port,
            password=settings.database.redis_password or None,
            db=settings.database.redis_db,
            decode_responses=True,
        )
        return RateLimitMiddleware(redis_client)
    except Exception as e:
        logger.error(f"Error creando rate limit middleware: {e}")
        # Fallback: middleware que no hace rate limiting

        class NoOpMiddleware:
            async def __call__(self, request, call_next):
                return await call_next(request)

        return NoOpMiddleware()
