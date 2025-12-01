#!/usr/bin/env python3
"""
Testbench de 200 queries para evaluación de baseline recall@5
=============================================================

Este script genera y mantiene un conjunto de 200 queries de prueba
para evaluar el rendimiento del sistema RAG y establecer un baseline
de recall@5 que se puede usar en CI/CD.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class TestQuery:
    """Query de prueba con su ground truth"""

    query_id: str
    question: str
    expected_chunks: List[str]  # Textos o IDs de chunks esperados
    category: Optional[str] = None
    difficulty: Optional[str] = "medium"  # easy, medium, hard


class TestbenchGenerator:
    """Generador y gestor del testbench de queries"""

    def __init__(self, output_path: Path = Path("corpus/_registry/testbench_queries.jsonl")):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate_default_queries(self) -> List[TestQuery]:
        """Generar conjunto de 200 queries de prueba por defecto"""

        queries = []

        # Categoría: Información General (50 queries)
        general_queries = [
            ("¿Qué es la inteligencia artificial?", ["IA", "artificial intelligence", "machine learning"]),
            ("¿Cómo funciona el machine learning?", ["machine learning", "algoritmos", "entrenamiento"]),
            ("¿Qué es un modelo de lenguaje?", ["modelo", "lenguaje", "NLP", "transformers"]),
            ("¿Cuál es la diferencia entre IA y ML?", ["IA", "ML", "machine learning", "diferencias"]),
            ("¿Qué es deep learning?", ["deep learning", "redes neuronales", "aprendizaje profundo"]),
            ("¿Cómo se entrena un modelo de IA?", ["entrenamiento", "datos", "gradiente", "backpropagation"]),
            ("¿Qué son los embeddings?", ["embeddings", "vectores", "representación"]),
            ("¿Qué es RAG?", ["RAG", "retrieval", "augmented generation"]),
            ("¿Cómo funciona la búsqueda semántica?", ["búsqueda semántica", "vectores", "similitud"]),
            ("¿Qué es un vector database?", ["vector database", "búsqueda vectorial", "FAISS"]),
            ("¿Qué es ChromaDB?", ["ChromaDB", "base de datos", "vectores"]),
            ("¿Qué es FAISS?", ["FAISS", "Facebook", "búsqueda", "vectores"]),
            ("¿Qué es HNSW?", ["HNSW", "gráfico", "búsqueda aproximada"]),
            ("¿Qué es BM25?", ["BM25", "búsqueda léxica", "ranking"]),
            ("¿Cómo funciona la indexación de documentos?", ["indexación", "documentos", "chunks"]),
            ("¿Qué es chunking?", ["chunking", "fragmentación", "segmentación"]),
            ("¿Qué es reranking?", ["reranking", "reordenamiento", "ranking"]),
            ("¿Qué es la deduplicación?", ["deduplicación", "duplicados", "hash"]),
            ("¿Qué es la normalización de texto?", ["normalización", "texto", "preprocessing"]),
            ("¿Qué es el preprocessing?", ["preprocessing", "limpieza", "normalización"]),
        ]

        for i, (question, expected) in enumerate(general_queries):
            queries.append(
                TestQuery(
                    query_id=f"general_{i+1:03d}",
                    question=question,
                    expected_chunks=expected,
                    category="general",
                    difficulty="easy" if i < 10 else "medium"
                )
            )

        # Categoría: Técnica Avanzada (50 queries)
        technical_queries = [
            ("¿Cómo optimizar un índice FAISS?", ["optimización", "FAISS", "parámetros", "nprobe"]),
            ("¿Qué es product quantization?", ["quantization", "compresión", "PQ"]),
            ("¿Cómo funciona HNSW?", ["HNSW", "gráfico", "navegación", "eficiencia"]),
            ("¿Qué es el recall en búsqueda?", ["recall", "precisión", "métricas"]),
            ("¿Cómo medir la calidad de un índice?", ["métricas", "calidad", "evaluación"]),
            ("¿Qué es el drift de embeddings?", ["drift", "embeddings", "degradación"]),
            ("¿Cómo hacer incremental indexing?", ["incremental", "indexación", "actualización"]),
            ("¿Qué es un snapshot?", ["snapshot", "versión", "backup"]),
            ("¿Cómo hacer backup de índices?", ["backup", "restauración", "índices"]),
            ("¿Qué es versionado de índices?", ["versionado", "migración", "esquema"]),
            ("¿Cómo implementar deduplicación?", ["deduplicación", "hash", "Jaccard"]),
            ("¿Qué es fuzzy matching?", ["fuzzy", "matching", "similitud"]),
            ("¿Cómo optimizar embeddings?", ["embeddings", "optimización", "batch"]),
            ("¿Qué es batch processing?", ["batch", "procesamiento", "eficiencia"]),
            ("¿Cómo implementar cache?", ["cache", "embeddings", "SQLite"]),
            ("¿Qué es TTL en cache?", ["TTL", "cache", "expiración"]),
            ("¿Cómo hacer compactación?", ["compactación", "VACUUM", "SQLite"]),
            ("¿Qué es observabilidad?", ["observabilidad", "métricas", "monitoring"]),
            ("¿Cómo implementar métricas Prometheus?", ["Prometheus", "métricas", "exportación"]),
            ("¿Qué es OpenTelemetry?", ["OpenTelemetry", "tracing", "observabilidad"]),
        ]

        for i, (question, expected) in enumerate(technical_queries):
            queries.append(
                TestQuery(
                    query_id=f"technical_{i+1:03d}",
                    question=question,
                    expected_chunks=expected,
                    category="technical",
                    difficulty="medium" if i < 10 else "hard"
                )
            )

        # Categoría: Uso y Aplicación (50 queries)
        usage_queries = [
            ("¿Cómo indexar documentos PDF?", ["PDF", "indexación", "parsing"]),
            ("¿Cómo buscar documentos?", ["búsqueda", "query", "retrieval"]),
            ("¿Cómo usar ChromaDB?", ["ChromaDB", "uso", "API"]),
            ("¿Cómo integrar RAG en una aplicación?", ["RAG", "integración", "API"]),
            ("¿Qué es un pipeline RAG?", ["pipeline", "RAG", "flujo"]),
            ("¿Cómo hacer búsqueda híbrida?", ["híbrida", "vector", "BM25"]),
            ("¿Cómo implementar reranking?", ["reranking", "modelo", "implementación"]),
            ("¿Qué es CRAG?", ["CRAG", "corrective", "gating"]),
            ("¿Cómo hacer búsqueda multi-modal?", ["multi-modal", "imágenes", "texto"]),
            ("¿Qué es query expansion?", ["query expansion", "reescritura", "sinónimos"]),
            ("¿Cómo hacer query rewriting?", ["query rewriting", "transformación", "optimización"]),
            ("¿Qué es HyDE?", ["HyDE", "hypothetical", "document"]),
            ("¿Cómo implementar caching de queries?", ["cache", "queries", "optimización"]),
            ("¿Qué es rate limiting?", ["rate limiting", "throttling", "API"]),
            ("¿Cómo hacer logging?", ["logging", "logs", "monitoring"]),
            ("¿Qué es error handling?", ["error handling", "excepciones", "resiliencia"]),
            ("¿Cómo implementar retries?", ["retries", "backoff", "resiliencia"]),
            ("¿Qué es circuit breaker?", ["circuit breaker", "fallback", "resiliencia"]),
            ("¿Cómo hacer testing?", ["testing", "tests", "validación"]),
            ("¿Qué es CI/CD?", ["CI/CD", "integración continua", "deployment"]),
        ]

        for i, (question, expected) in enumerate(usage_queries):
            queries.append(
                TestQuery(
                    query_id=f"usage_{i+1:03d}",
                    question=question,
                    expected_chunks=expected,
                    category="usage",
                    difficulty="easy"
                )
            )

        # Categoría: Seguridad y Compliance (30 queries)
        security_queries = [
            ("¿Cómo asegurar datos en índices?", ["seguridad", "encriptación", "datos"]),
            ("¿Qué es RBAC?", ["RBAC", "permisos", "roles"]),
            ("¿Cómo implementar autenticación?", ["autenticación", "JWT", "OAuth"]),
            ("¿Qué es audit logging?", ["audit", "logging", "compliance"]),
            ("¿Cómo proteger datos sensibles?", ["datos sensibles", "privacidad", "encriptación"]),
            ("¿Qué es encriptación en reposo?", ["encriptación", "reposo", "seguridad"]),
            ("¿Cómo hacer encriptación en tránsito?", ["encriptación", "tránsito", "TLS"]),
            ("¿Qué es SQL injection?", ["SQL injection", "seguridad", "vulnerabilidad"]),
            ("¿Cómo prevenir ataques?", ["seguridad", "ataques", "prevención"]),
            ("¿Qué es data governance?", ["governance", "datos", "compliance"]),
            ("¿Cómo implementar retención de datos?", ["retención", "datos", "archivado"]),
            ("¿Qué es GDPR compliance?", ["GDPR", "compliance", "privacidad"]),
            ("¿Cómo hacer backup seguro?", ["backup", "seguridad", "encriptación"]),
            ("¿Qué es disaster recovery?", ["disaster recovery", "backup", "recuperación"]),
            ("¿Cómo implementar acceso controlado?", ["acceso", "control", "permisos"]),
        ]

        for i, (question, expected) in enumerate(security_queries):
            queries.append(
                TestQuery(
                    query_id=f"security_{i+1:03d}",
                    question=question,
                    expected_chunks=expected,
                    category="security",
                    difficulty="medium"
                )
            )

        # Categoría: Rendimiento y Escalabilidad (20 queries)
        performance_queries = [
            ("¿Cómo escalar índices?", ["escalado", "distribuido", "sharding"]),
            ("¿Qué es sharding?", ["sharding", "particionamiento", "escalado"]),
            ("¿Cómo optimizar latencia?", ["latencia", "optimización", "rendimiento"]),
            ("¿Qué es throughput?", ["throughput", "rendimiento", "QPS"]),
            ("¿Cómo hacer load balancing?", ["load balancing", "distribución", "carga"]),
            ("¿Qué es caching distribuido?", ["cache distribuido", "Redis", "memoria"]),
            ("¿Cómo optimizar memoria?", ["memoria", "optimización", "compresión"]),
            ("¿Qué es GPU acceleration?", ["GPU", "aceleración", "CUDA"]),
            ("¿Cómo hacer profiling?", ["profiling", "rendimiento", "optimización"]),
            ("¿Qué es benchmarking?", ["benchmarking", "rendimiento", "métricas"]),
            ("¿Cómo optimizar queries?", ["queries", "optimización", "índices"]),
            ("¿Qué es query optimization?", ["optimización", "queries", "rendimiento"]),
            ("¿Cómo reducir costos?", ["costos", "optimización", "recursos"]),
            ("¿Qué es resource monitoring?", ["monitoring", "recursos", "CPU", "memoria"]),
            ("¿Cómo hacer auto-scaling?", ["auto-scaling", "escalado", "carga"]),
            ("¿Qué es horizontal scaling?", ["escalado horizontal", "nodos", "distribuido"]),
            ("¿Cómo optimizar embeddings batch?", ["batch", "embeddings", "optimización"]),
            ("¿Qué es parallel processing?", ["paralelización", "threading", "multiprocessing"]),
            ("¿Cómo hacer async processing?", ["async", "asíncrono", "rendimiento"]),
            ("¿Qué es connection pooling?", ["connection pooling", "bases de datos", "eficiencia"]),
        ]

        for i, (question, expected) in enumerate(performance_queries):
            queries.append(
                TestQuery(
                    query_id=f"performance_{i+1:03d}",
                    question=question,
                    expected_chunks=expected,
                    category="performance",
                    difficulty="hard"
                )
            )

        return queries

    def save_queries(self, queries: List[TestQuery]) -> None:
        """Guardar queries en formato JSONL"""
        with open(self.output_path, "w", encoding="utf-8") as f:
            for query in queries:
                f.write(json.dumps(asdict(query), ensure_ascii=False) + "\n")
        print(f"✅ {len(queries)} queries guardadas en {self.output_path}")

    def load_queries(self) -> List[TestQuery]:
        """Cargar queries desde archivo JSONL"""
        queries = []
        if self.output_path.exists():
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        queries.append(TestQuery(**data))
        return queries

    def generate_and_save(self) -> List[TestQuery]:
        """Generar y guardar queries por defecto"""
        queries = self.generate_default_queries()
        self.save_queries(queries)
        return queries


def main():
    """Generar testbench de queries"""
    generator = TestbenchGenerator()
    queries = generator.generate_and_save()
    
    print(f"\n📊 Estadísticas del testbench:")
    print(f"  Total queries: {len(queries)}")
    
    categories = {}
    difficulties = {}
    for q in queries:
        categories[q.category] = categories.get(q.category, 0) + 1
        difficulties[q.difficulty] = difficulties.get(q.difficulty, 0) + 1
    
    print(f"\n📂 Por categoría:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print(f"\n🎯 Por dificultad:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}")
    
    print(f"\n✅ Testbench generado exitosamente en: {generator.output_path}")


if __name__ == "__main__":
    main()

