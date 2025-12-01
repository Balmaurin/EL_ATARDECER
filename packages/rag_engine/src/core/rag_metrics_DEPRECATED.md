# ⚠️ DEPRECATED: rag_metrics.py

**Fecha de deprecación:** 2025-01-XX  
**Estado:** DEPRECATED - Migrar a `indexing_metrics.py`

## Razón de Deprecación

Este módulo está siendo deprecado en favor de `indexing_metrics.py` que proporciona:
- Métricas Prometheus integradas
- Mejor estructura y organización
- Soporte para observabilidad enterprise

## Migración

### Antes (rag_metrics.py):
```python
from packages.rag_engine.src.core.rag_metrics import RAGMetrics

metrics = RAGMetrics()
precision = metrics.calculate_precision(retrieved_docs, relevant_docs)
```

### Después (indexing_metrics.py):
```python
from packages.rag_engine.src.core.indexing_metrics import (
    record_query_processed,
    get_metrics_summary
)

record_query_processed(index_name="rag", search_type="hybrid")
summary = get_metrics_summary()
```

## Funcionalidades Mantenidas

Algunas clases específicas de RAG se mantienen temporalmente:
- `HybridSearchEngine`: Motor de búsqueda híbrida
- `RAGEvaluator`: Evaluador específico de RAG

Estas se migrarán a módulos específicos en el futuro.

## Timeline

- **Fase 1 (Actual):** Deprecación anunciada, imports con warnings
- **Fase 2 (Próxima):** Migración completa de funcionalidades
- **Fase 3 (Futuro):** Eliminación del módulo

## Referencias

- Ver `indexing_metrics.py` para el nuevo sistema de métricas
- Ver `DOCUMENTACION_COMPLETA_RAG_ENGINE.md` para detalles completos

