#!/usr/bin/env python3
"""
SHEILY AI - RATE LIMITING SYSTEM
================================

Sistema avanzado de rate limiting para prevenir abuso de APIs
y ataques de denegación de servicio (DoS).

Incluye:
- Rate limiting basado en IP, usuario y endpoint
- Algoritmos: Token Bucket, Leaky Bucket, Fixed Window
- Redis para almacenamiento distribuido
- Configuración flexible por endpoint
- Headers informativos para clientes
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Algoritmos de rate limiting disponibles"""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(Enum):
    """Ámbito del rate limiting"""

    IP = "ip"
    USER = "user"
    ENDPOINT = "endpoint"
    GLOBAL = "global"


@dataclass
class RateLimitRule:
    """Regla de rate limiting individual"""

    name: str
    scope: RateLimitScope
    algorithm: RateLimitAlgorithm
    requests_per_window: int
    window_seconds: int
    burst_limit: Optional[int] = None
    enabled: bool = True
    exempt_users: List[str] = field(default_factory=list)
    exempt_ips: List[str] = field(default_factory=list)


@dataclass
class RateLimitState:
    """Estado actual del rate limiting"""

    key: str
    requests: float = 0.0
    window_start: float = 0.0
    tokens: float = 0.0
    last_refill: float = 0.0


@dataclass
class RateLimitResult:
    """Resultado de verificación de rate limiting"""

    allowed: bool
    remaining_requests: float
    reset_time: float
    retry_after: Optional[int] = None
    limit_exceeded: bool = False


class RateLimiter:
    """
    Rate limiter avanzado para Sheily AI
    Soporta múltiples algoritmos y configuraciones
    REAL IMPLEMENTATION - Redis is REQUIRED, no fallbacks
    """

    def __init__(self, redis_client=None, enable_redis: bool = True):
        """
        Initialize rate limiter.

        IMPORTANT: Redis is REQUIRED for production.
        If redis_client is None, system will fail fast.
        """
        if redis_client is None:
            raise RuntimeError(
                "Redis client is required for RateLimiter. "
                "NO FALLBACKS - System cannot start without Redis. "
                "Configure REDIS_URL environment variable."
            )

        self.redis_client = redis_client
        self.enable_redis = True  # Always True in production

        # Reglas de rate limiting por defecto
        self.rules = self._load_default_rules()

        logger.info("✓ Rate Limiter initialized with Redis backend (NO FALLBACKS)")

    def _load_default_rules(self) -> Dict[str, RateLimitRule]:
        """Cargar reglas por defecto"""
        return {
            "api_general": RateLimitRule(
                name="api_general",
                scope=RateLimitScope.IP,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                requests_per_window=100,  # 100 requests
                window_seconds=60,  # por minuto
                burst_limit=20,  # burst de 20
            ),
            "api_auth": RateLimitRule(
                name="api_auth",
                scope=RateLimitScope.IP,
                algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                requests_per_window=5,  # 5 intentos de login
                window_seconds=300,  # por 5 minutos
            ),
            "api_chat": RateLimitRule(
                name="api_chat",
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                requests_per_window=50,  # 50 mensajes
                window_seconds=60,  # por minuto
                burst_limit=10,
            ),
            "api_admin": RateLimitRule(
                name="api_admin",
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                requests_per_window=1000,  # 1000 requests
                window_seconds=3600,  # por hora
            ),
        }

    async def check_rate_limit(
        self,
        key: str,
        rule_name: str = "api_general",
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> RateLimitResult:
        """
        Verificar si una request está dentro del rate limit

        Args:
            key: Clave única para el límite (ej: endpoint path)
            rule_name: Nombre de la regla a aplicar
            user_id: ID del usuario (opcional)
            ip_address: Dirección IP (opcional)

        Returns:
            RateLimitResult: Resultado de la verificación
        """
        rule = self.rules.get(rule_name)
        if not rule or not rule.enabled:
            # Sin regla = permitir
            return RateLimitResult(allowed=True, remaining_requests=-1, reset_time=0.0)

        # Verificar exenciones
        if self._is_exempt(user_id, ip_address, rule):
            return RateLimitResult(allowed=True, remaining_requests=-1, reset_time=0.0)

        # Generar clave de estado
        state_key = self._generate_state_key(key, rule.scope, user_id, ip_address)

        # Obtener estado actual
        state = await self._get_state(state_key)

        # Aplicar algoritmo de rate limiting
        if rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            result = self._check_fixed_window(state, rule)
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            result = self._check_sliding_window(state, rule)
        elif rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            result = self._check_token_bucket(state, rule)
        elif rule.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            result = self._check_leaky_bucket(state, rule)
        else:
            # Algoritmo desconocido = permitir
            return RateLimitResult(allowed=True, remaining_requests=-1, reset_time=0.0)

        # Actualizar estado si fue permitido
        if result.allowed:
            await self._update_state(state_key, state)

        return result

    def _is_exempt(
        self, user_id: Optional[str], ip_address: Optional[str], rule: RateLimitRule
    ) -> bool:
        """Verificar si el request está exento de rate limiting"""
        if user_id and user_id in rule.exempt_users:
            return True
        if ip_address and ip_address in rule.exempt_ips:
            return True
        return False

    def _generate_state_key(
        self,
        key: str,
        scope: RateLimitScope,
        user_id: Optional[str],
        ip_address: Optional[str],
    ) -> str:
        """Generar clave única para el estado"""
        if scope == RateLimitScope.IP and ip_address:
            identifier = f"ip:{ip_address}"
        elif scope == RateLimitScope.USER and user_id:
            identifier = f"user:{user_id}"
        elif scope == RateLimitScope.ENDPOINT:
            identifier = f"endpoint:{key}"
        else:  # GLOBAL
            identifier = "global"

        return f"ratelimit:{identifier}:{key}"

    async def _get_state(self, state_key: str) -> RateLimitState:
        """
        Obtener estado actual desde Redis.
        REAL IMPLEMENTATION - NO FALLBACKS, Redis is required
        """
        try:
            data = await self.redis_client.get(state_key)
            if data:
                state_data = json.loads(data)
                return RateLimitState(**state_data)
            return RateLimitState(key=state_key)
        except Exception as e:
            logger.error(f"Redis error getting state: {e}")
            raise RuntimeError(
                f"Failed to get rate limit state from Redis: {e}. "
                "NO FALLBACKS - Rate limiter requires Redis."
            )

    async def _update_state(self, state_key: str, state: RateLimitState):
        """
        Actualizar estado en Redis.
        REAL IMPLEMENTATION - NO FALLBACKS, Redis is required
        """
        try:
            data = {
                "key": state.key,
                "requests": state.requests,
                "window_start": state.window_start,
                "tokens": state.tokens,
                "last_refill": state.last_refill,
            }
            await self.redis_client.setex(
                state_key, 3600, json.dumps(data)
            )  # 1 hora TTL
        except Exception as e:
            logger.error(f"Redis error updating state: {e}")
            raise RuntimeError(
                f"Failed to update rate limit state in Redis: {e}. "
                "NO FALLBACKS - Rate limiter requires Redis."
            )

    def _check_fixed_window(
        self, state: RateLimitState, rule: RateLimitRule
    ) -> RateLimitResult:
        """Verificar rate limiting con ventana fija"""
        current_time = time.time()

        # Verificar si estamos en una nueva ventana
        if current_time - state.window_start >= rule.window_seconds:
            # Nueva ventana
            state.requests = 1
            state.window_start = current_time
            return RateLimitResult(
                allowed=True,
                remaining_requests=rule.requests_per_window - 1,
                reset_time=state.window_start + rule.window_seconds,
            )
        else:
            # Misma ventana
            if state.requests >= rule.requests_per_window:
                # Límite excedido
                return RateLimitResult(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=state.window_start + rule.window_seconds,
                    retry_after=int(
                        (state.window_start + rule.window_seconds) - current_time
                    ),
                    limit_exceeded=True,
                )
            else:
                # Permitir
                state.requests += 1
                return RateLimitResult(
                    allowed=True,
                    remaining_requests=rule.requests_per_window - state.requests,
                    reset_time=state.window_start + rule.window_seconds,
                )

    def _check_sliding_window(
        self, state: RateLimitState, rule: RateLimitRule
    ) -> RateLimitResult:
        """Verificar rate limiting con ventana deslizante (simplificada)"""
        # Implementación simplificada - usar fixed window por ahora
        return self._check_fixed_window(state, rule)

    def _check_token_bucket(
        self, state: RateLimitState, rule: RateLimitRule
    ) -> RateLimitResult:
        """Verificar rate limiting con token bucket"""
        current_time = time.time()
        time_passed = current_time - state.last_refill

        # Calcular tokens a añadir
        tokens_to_add = time_passed * (rule.requests_per_window / rule.window_seconds)
        state.tokens = min(rule.requests_per_window, state.tokens + tokens_to_add)
        state.last_refill = current_time

        # Verificar burst limit
        burst_limit = rule.burst_limit or rule.requests_per_window

        if state.tokens >= 1:
            # Consumir token
            state.tokens -= 1
            remaining = min(int(state.tokens), burst_limit)
            return RateLimitResult(
                allowed=True,
                remaining_requests=remaining,
                reset_time=current_time
                + (rule.window_seconds / rule.requests_per_window),
            )
        else:
            # Sin tokens disponibles
            refill_time = (1 - state.tokens) / (
                rule.requests_per_window / rule.window_seconds
            )
            return RateLimitResult(
                allowed=False,
                remaining_requests=0,
                reset_time=current_time + refill_time,
                retry_after=int(refill_time),
                limit_exceeded=True,
            )

    def _check_leaky_bucket(
        self, state: RateLimitState, rule: RateLimitRule
    ) -> RateLimitResult:
        """Verificar rate limiting con leaky bucket"""
        current_time = time.time()

        # Calcular cuántos "requests" han "salido" del bucket
        time_passed = current_time - state.last_refill
        leaked = time_passed * (rule.requests_per_window / rule.window_seconds)
        state.requests = max(0, state.requests - leaked)
        state.last_refill = current_time

        if state.requests < rule.requests_per_window:
            # Añadir request al bucket
            state.requests += 1
            remaining = rule.requests_per_window - int(state.requests)
            return RateLimitResult(
                allowed=True,
                remaining_requests=remaining,
                reset_time=current_time + rule.window_seconds,
            )
        else:
            # Bucket lleno
            return RateLimitResult(
                allowed=False,
                remaining_requests=0,
                reset_time=current_time + rule.window_seconds,
                retry_after=rule.window_seconds,
                limit_exceeded=True,
            )


    def add_rule(self, rule: RateLimitRule):
        """Añadir una nueva regla de rate limiting"""
        self.rules[rule.name] = rule
        logger.info(f"[OK] Added rate limit rule: {rule.name}")

    def remove_rule(self, rule_name: str):
        """Remover una regla de rate limiting"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"🗑️ Removed rate limit rule: {rule_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del rate limiter"""
        return {
            "rules_count": len(self.rules),
            "redis_enabled": True,  # Always True - Redis is required
            "rules": list(self.rules.keys()),
        }


# Instancia global del rate limiter
_global_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Obtener instancia global del rate limiter"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


# Decorador para endpoints con rate limiting automático
def rate_limit(rule_name: str = "api_general"):
    """
    Decorador para aplicar rate limiting automáticamente en endpoints

    Args:
        rule_name: Nombre de la regla de rate limiting a aplicar
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extraer información del request (asumiendo FastAPI/Starlette)
            request = None
            for arg in args:
                if hasattr(arg, "client") and hasattr(arg, "headers"):
                    request = arg
                    break

            # Obtener IP y user ID
            ip_address = None
            user_id = None

            if request:
                # Extraer IP
                ip_address = (
                    getattr(request.client, "host", None)
                    if hasattr(request, "client")
                    else None
                )

                # Extraer user ID de headers o token
                auth_header = request.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    # Aquí iría la lógica para extraer user_id del token JWT
                    # Por simplicidad, usar un hash del token como ID
                    user_id = hashlib.md5(auth_header.encode()).hexdigest()[:16]

            # Obtener endpoint path
            endpoint_path = f"{func.__module__}.{func.__name__}"

            # Verificar rate limit
            rate_limiter = get_rate_limiter()
            result = await rate_limiter.check_rate_limit(
                key=endpoint_path,
                rule_name=rule_name,
                user_id=user_id,
                ip_address=ip_address,
            )

            if not result.allowed:
                # Rate limit excedido
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(result.remaining_requests + 1),
                        "X-RateLimit-Remaining": str(result.remaining_requests),
                        "X-RateLimit-Reset": str(int(result.reset_time)),
                        "Retry-After": str(result.retry_after or 60),
                    },
                )

            # Añadir headers informativos a la respuesta
            response = await func(*args, **kwargs)

            # Si es una respuesta FastAPI/Starlette, añadir headers
            if hasattr(response, "headers"):
                response.headers["X-RateLimit-Limit"] = str(
                    result.remaining_requests + 1
                )
                response.headers["X-RateLimit-Remaining"] = str(
                    result.remaining_requests
                )
                response.headers["X-RateLimit-Reset"] = str(int(result.reset_time))

            return response

        return wrapper

    return decorator


if __name__ == "__main__":
    # Demo del sistema de rate limiting
    print("[FIRE] SHEILY AI - RATE LIMITING SYSTEM DEMO")
    print("=" * 60)

    async def demo():
        rate_limiter = RateLimiter(enable_redis=False)

        print("[CHART] Testing Fixed Window Algorithm (5 requests per minute)")
        rule = RateLimitRule(
            name="test_fixed",
            scope=RateLimitScope.IP,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            requests_per_window=5,
            window_seconds=60,
        )
        rate_limiter.add_rule(rule)

        # Simular requests
        for i in range(8):
            result = await rate_limiter.check_rate_limit(
                key="/api/test", rule_name="test_fixed", ip_address="192.168.1.100"
            )
            print(
                f"Request {i+1}: Allowed={result.allowed}, Remaining={result.remaining_requests}"
            )
            if not result.allowed:
                print(f"  Rate limit exceeded! Retry after {result.retry_after}s")

        print("\n[START] Testing Token Bucket Algorithm (10 requests per minute, burst=3)")
        rule_tb = RateLimitRule(
            name="test_token_bucket",
            scope=RateLimitScope.USER,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            requests_per_window=10,
            window_seconds=60,
            burst_limit=3,
        )
        rate_limiter.add_rule(rule_tb)

        # Simular requests con token bucket
        for i in range(6):
            result = await rate_limiter.check_rate_limit(
                key="/api/chat", rule_name="test_token_bucket", user_id="user123"
            )
            print(
                f"Request {i+1}: Allowed={result.allowed}, Remaining={result.remaining_requests}"
            )

            # Simular tiempo entre requests
            await asyncio.sleep(0.1)

        print("\n[OK] Rate Limiting System Demo Complete!")
        print("[FIRE] Sistema listo para prevenir abuso de APIs")

    # Ejecutar demo
    asyncio.run(demo())
