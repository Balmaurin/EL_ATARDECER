#!/usr/bin/env python3
"""
Métricas Prometheus para el sistema de indexación vectorial.
Exporta métricas críticas para observabilidad del sistema RAG.

Métricas exportadas:
- queries_processed: Total de queries procesadas
- avg_query_time_ms: Tiempo promedio de query en milisegundos
- index_build_time_s: Tiempo de construcción de índices en segundos
- cache_hits: Hits del cache de embeddings
- cache_misses: Misses del cache de embeddings
- n_vectors: Número de vectores en el índice
- embedding_batch_size: Tamaño de batch para embeddings
- last_index_time: Timestamp de última indexación
"""

import os
import time
from datetime import datetime
from typing import Dict, Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Dummy classes para cuando Prometheus no está disponible
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self

    class Summary:
        def __init__(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass

    def generate_latest():
        return b""

    CONTENT_TYPE_LATEST = "text/plain"

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# MÉTRICAS PROMETHEUS
# =============================================================================

# Contadores
QUERIES_PROCESSED = Counter(
    "rag_queries_processed_total",
    "Total number of queries processed",
    ["index_name", "search_type"],
)

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Total number of cache hits",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Total number of cache misses",
    ["cache_type"],
)

INDEX_BUILDS = Counter(
    "rag_index_builds_total",
    "Total number of index builds",
    ["index_type", "success"],
)

# Gauges (valores actuales)
N_VECTORS = Gauge(
    "rag_n_vectors",
    "Current number of vectors in the index",
    ["index_name", "index_type"],
)

EMBEDDING_BATCH_SIZE = Gauge(
    "rag_embedding_batch_size",
    "Current embedding batch size",
)

LAST_INDEX_TIME = Gauge(
    "rag_last_index_time_seconds",
    "Timestamp of last index build",
    ["index_name"],
)

# Histogramas (distribuciones)
QUERY_TIME_MS = Histogram(
    "rag_query_time_ms",
    "Query processing time in milliseconds",
    ["index_name", "search_type"],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)

INDEX_BUILD_TIME_S = Histogram(
    "rag_index_build_time_seconds",
    "Index build time in seconds",
    ["index_type"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)

EMBEDDING_TIME_MS = Histogram(
    "rag_embedding_time_ms",
    "Embedding generation time in milliseconds",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)

# Summary (para percentiles)
CACHE_HIT_RATIO = Summary(
    "rag_cache_hit_ratio",
    "Cache hit ratio",
    ["cache_type"],
)

# =============================================================================
# HELPER CLASS PARA MANEJO DE MÉTRICAS
# =============================================================================


class IndexingMetrics:
    """Helper class para gestionar métricas del sistema de indexación"""

    def __init__(self, index_name: str = "default"):
        self.index_name = index_name
        self._query_times = []
        self._build_start_time: Optional[float] = None

    def record_query(
        self, query_time_ms: float, search_type: str = "hybrid"
    ) -> None:
        """Registrar una query procesada"""
        QUERIES_PROCESSED.labels(
            index_name=self.index_name, search_type=search_type
        ).inc()
        QUERY_TIME_MS.labels(
            index_name=self.index_name, search_type=search_type
        ).observe(query_time_ms)

    def record_cache_hit(self, cache_type: str = "embedding") -> None:
        """Registrar un cache hit"""
        CACHE_HITS.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str = "embedding") -> None:
        """Registrar un cache miss"""
        CACHE_MISSES.labels(cache_type=cache_type).inc()

    def update_cache_ratio(self, cache_type: str = "embedding") -> None:
        """Actualizar ratio de cache hits/misses"""
        try:
            hits = self._get_counter_value(CACHE_HITS, cache_type)
            misses = self._get_counter_value(CACHE_MISSES, cache_type)
            total = hits + misses
            if total > 0:
                ratio = hits / total
                CACHE_HIT_RATIO.labels(cache_type=cache_type).observe(ratio)
        except Exception as e:
            logger.warning(f"Error updating cache ratio: {e}")

    def set_n_vectors(self, count: int, index_type: str = "faiss") -> None:
        """Actualizar número de vectores en el índice"""
        N_VECTORS.labels(
            index_name=self.index_name, index_type=index_type
        ).set(count)

    def set_embedding_batch_size(self, batch_size: int) -> None:
        """Actualizar tamaño de batch de embeddings"""
        EMBEDDING_BATCH_SIZE.set(batch_size)

    def set_last_index_time(self) -> None:
        """Actualizar timestamp de última indexación"""
        LAST_INDEX_TIME.labels(index_name=self.index_name).set(
            time.time()
        )

    def start_index_build(self) -> None:
        """Iniciar tracking de construcción de índice"""
        self._build_start_time = time.time()

    def record_index_build(
        self, index_type: str, success: bool = True
    ) -> None:
        """Registrar construcción de índice completada"""
        INDEX_BUILDS.labels(
            index_type=index_type, success="true" if success else "false"
        ).inc()

        if self._build_start_time is not None:
            build_time = time.time() - self._build_start_time
            INDEX_BUILD_TIME_S.labels(index_type=index_type).observe(
                build_time
            )
            self._build_start_time = None

        if success:
            self.set_last_index_time()

    def record_embedding_time(self, time_ms: float) -> None:
        """Registrar tiempo de generación de embedding"""
        EMBEDDING_TIME_MS.observe(time_ms)

    def _get_counter_value(self, counter: Counter, label_value: str) -> float:
        """Obtener valor de un counter con label específico"""
        # Prometheus client no expone fácilmente valores de counters con labels
        # Esta es una implementación simplificada
        # En producción, deberías usar prometheus_client.REGISTRY
        return 0.0

    def get_metrics_summary(self) -> Dict[str, any]:
        """Obtener resumen de métricas actuales"""
        return {
            "index_name": self.index_name,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "metrics": {
                "queries": "Tracked via QUERIES_PROCESSED",
                "cache": "Tracked via CACHE_HITS/CACHE_MISSES",
                "vectors": "Tracked via N_VECTORS",
                "index_builds": "Tracked via INDEX_BUILDS",
            },
        }


# =============================================================================
# FUNCIÓN PARA EXPORTAR MÉTRICAS
# =============================================================================


def export_metrics() -> bytes:
    """Exportar métricas en formato Prometheus"""
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not available\n"
    return generate_latest()


def get_metrics_content_type() -> str:
    """Obtener content type para endpoint de métricas"""
    return CONTENT_TYPE_LATEST


# =============================================================================
# INTEGRACIÓN CON HYBRIDVECTORSTORE
# =============================================================================


def create_metrics_for_index(index_name: str = "default") -> IndexingMetrics:
    """Crear instancia de métricas para un índice específico"""
    return IndexingMetrics(index_name=index_name)


# Verificar si Prometheus está habilitado
PROMETHEUS_ENABLED = os.getenv("RAG_PROMETHEUS_ENABLED", "true").lower() == "true"

if __name__ == "__main__":
    # Demo de uso
    metrics = create_metrics_for_index("demo_index")

    # Simular algunas operaciones
    metrics.set_n_vectors(1000, "faiss")
    metrics.set_embedding_batch_size(32)
    metrics.record_query(25.5, "hybrid")
    metrics.record_cache_hit("embedding")
    metrics.record_cache_miss("embedding")

    metrics.start_index_build()
    time.sleep(0.1)  # Simular construcción
    metrics.record_index_build("faiss", success=True)

    print("✅ Métricas registradas correctamente")
    print(f"📊 Prometheus disponible: {PROMETHEUS_AVAILABLE}")
    print(f"📈 Resumen: {metrics.get_metrics_summary()}")

