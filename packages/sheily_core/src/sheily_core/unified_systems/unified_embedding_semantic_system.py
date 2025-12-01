#!/usr/bin/env python3
"""
Unified Embedding Semantic System - Sistema Unificado de Embeddings y Búsqueda Semántica

Este módulo implementa un sistema avanzado de embeddings semánticos
para procesamiento inteligente de texto y búsqueda semántica.

Autor: Unified Systems Team
Fecha: 2025-11-12
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Importar sentence-transformers para embeddings
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ImportError:
    SentenceTransformer = None
    cos_sim = None

from .unified_system_core import SystemConfig

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuración del sistema de embeddings"""

    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    cache_enabled: bool = True
    performance_tracking: bool = True
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    similarity_threshold: float = 0.7
    cache_size: int = 10000
    device: str = "cpu"  # 'cpu', 'cuda', 'auto'


@dataclass
class EmbeddingResult:
    """Resultado de generación de embedding"""

    text: str
    embedding: np.ndarray
    model_used: str
    processing_time: float
    dimensions: int
    normalized: bool
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "text": self.text,
            "embedding": (
                self.embedding.tolist()
                if isinstance(self.embedding, np.ndarray)
                else self.embedding
            ),
            "model_used": self.model_used,
            "processing_time": self.processing_time,
            "dimensions": self.dimensions,
            "normalized": self.normalized,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SemanticSearchResult:
    """Resultado de búsqueda semántica"""

    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float
    similarity_threshold: float
    search_type: str  # 'cosine', 'dot_product', 'euclidean'


class UnifiedEmbeddingSemanticSystem:
    """
    Sistema unificado de embeddings y búsqueda semántica
    para procesamiento inteligente de texto.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Inicializar sistema de embeddings"""
        self.config = config or EmbeddingConfig()

        # Modelo de embeddings
        self.model: Optional[SentenceTransformer] = None
        self.model_loaded = False

        # Caché de embeddings
        self.embedding_cache: Dict[str, EmbeddingResult] = {}
        self.text_cache: Dict[str, str] = {}

        # Estadísticas de rendimiento
        self.stats = {
            "total_embeddings_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_processing_time": 0.0,
            "total_searches": 0,
        }

        logger.info("🔍 Unified Embedding Semantic System inicializado")

    async def initialize(self) -> bool:
        """Inicializar el modelo de embeddings"""
        try:
            if SentenceTransformer is None:
                logger.error("❌ sentence-transformers no está instalado")
                return False

            logger.info(f"📥 Cargando modelo de embeddings: {self.config.model_name}")

            # Cargar modelo (esto puede tomar tiempo)
            self.model = SentenceTransformer(
                self.config.model_name, device=self.config.device
            )

            # Configurar modelo
            if hasattr(self.model, "max_seq_length"):
                self.model.max_seq_length = self.config.max_seq_length

            self.model_loaded = True
            logger.info("✅ Modelo de embeddings cargado correctamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error inicializando modelo de embeddings: {e}")
            self.model_loaded = False
            return False

    def generate_embedding(self, text: str, domain: str = "general") -> EmbeddingResult:
        """
        Generar embedding para un texto

        Args:
            text: Texto a procesar
            domain: Dominio del texto (para optimizaciones futuras)

        Returns:
            Resultado del embedding
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Modelo de embeddings no inicializado")

        start_time = datetime.now()

        try:
            # Verificar caché
            cache_key = self._get_cache_key(text, domain)
            if self.config.cache_enabled and cache_key in self.embedding_cache:
                cached_result = self.embedding_cache[cache_key]
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit para texto: {text[:50]}...")
                return cached_result

            # Generar embedding
            self.stats["cache_misses"] += 1

            # Procesar texto
            processed_text = self._preprocess_text(text)

            # Generar embedding usando sentence-transformers
            embedding = self.model.encode(
                processed_text,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
            )

            # Crear resultado
            processing_time = (datetime.now() - start_time).total_seconds()
            result = EmbeddingResult(
                text=text,
                embedding=embedding,
                model_used=self.config.model_name,
                processing_time=processing_time,
                dimensions=len(embedding),
                normalized=self.config.normalize_embeddings,
            )

            # Actualizar estadísticas
            self.stats["total_embeddings_generated"] += 1
            self._update_avg_processing_time(processing_time)

            # Almacenar en caché
            if self.config.cache_enabled:
                self._cache_embedding(cache_key, result)

            logger.debug(
                f"Embedding generado para texto: {text[:50]}... ({processing_time:.3f}s)"
            )
            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error generando embedding: {e}")
            raise RuntimeError(f"Error generando embedding: {e}")

    def _preprocess_text(self, text: str) -> str:
        """Preprocesar texto antes de generar embedding"""
        # Limpieza básica
        processed = text.strip()

        # Limitar longitud si es necesario
        if (
            len(processed) > self.config.max_seq_length * 4
        ):  # Aproximadamente 4 chars por token
            processed = processed[: self.config.max_seq_length * 4]
            processed = processed.rsplit(" ", 1)[0]  # Cortar en límite de palabra

        return processed

    def _get_cache_key(self, text: str, domain: str) -> str:
        """Generar clave de caché"""
        import hashlib

        content = f"{text}:{domain}".encode("utf-8")
        return hashlib.md5(content).hexdigest()

    def _cache_embedding(self, key: str, result: EmbeddingResult):
        """Almacenar embedding en caché"""
        if len(self.embedding_cache) >= self.config.cache_size:
            # Eliminar entrada más antigua
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]

        self.embedding_cache[key] = result
        self.text_cache[key] = result.text

    def _update_avg_processing_time(self, new_time: float):
        """Actualizar tiempo promedio de procesamiento"""
        total_embeddings = self.stats["total_embeddings_generated"]
        current_avg = self.stats["avg_processing_time"]

        # Calcular nuevo promedio
        self.stats["avg_processing_time"] = (
            current_avg * (total_embeddings - 1) + new_time
        ) / total_embeddings

    def semantic_search(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> SemanticSearchResult:
        """
        Realizar búsqueda semántica

        Args:
            query: Consulta de búsqueda
            corpus: Lista de textos a buscar
            top_k: Número máximo de resultados
            threshold: Umbral de similitud mínimo

        Returns:
            Resultados de búsqueda semántica
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Modelo de embeddings no inicializado")

        start_time = datetime.now()
        threshold = threshold or self.config.similarity_threshold

        try:
            # Generar embedding de la consulta
            query_embedding = self.generate_embedding(query, domain="search")

            # Generar embeddings del corpus
            corpus_embeddings = []
            for text in corpus:
                embedding = self.generate_embedding(text, domain="corpus")
                corpus_embeddings.append(embedding.embedding)

            # Convertir a numpy array
            corpus_embeddings = np.array(corpus_embeddings)

            # Calcular similitudes
            similarities = cos_sim(
                query_embedding.embedding.reshape(1, -1), corpus_embeddings
            )[0]

            # Crear resultados
            results = []
            for idx, similarity in enumerate(similarities):
                if similarity >= threshold:
                    results.append(
                        {
                            "text": corpus[idx],
                            "similarity": float(similarity),
                            "rank": len(results) + 1,
                        }
                    )

            # Ordenar por similitud descendente
            results.sort(key=lambda x: x["similarity"], reverse=True)

            # Limitar resultados
            results = results[:top_k]

            processing_time = (datetime.now() - start_time).total_seconds()

            # Actualizar estadísticas
            self.stats["total_searches"] += 1

            return SemanticSearchResult(
                query=query,
                results=results,
                total_results=len(results),
                processing_time=processing_time,
                similarity_threshold=threshold,
                search_type="cosine",
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error en búsqueda semántica: {e}")
            raise RuntimeError(f"Error en búsqueda semántica: {e}")

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calcular similitud semántica entre dos textos

        Args:
            text1: Primer texto
            text2: Segundo texto

        Returns:
            Similitud entre 0 y 1
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Modelo de embeddings no inicializado")

        try:
            # Generar embeddings
            emb1 = self.generate_embedding(text1)
            emb2 = self.generate_embedding(text2)

            # Calcular similitud coseno
            similarity = cos_sim(
                emb1.embedding.reshape(1, -1), emb2.embedding.reshape(1, -1)
            )[0][0]

            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculando similitud: {e}")
            return 0.0

    def calculate_semantic_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embedding vectors using numpy.
        Useful when sentence_transformers is not available.
        """
        if emb1.size == 0 or emb2.size == 0:
            return 0.0
        
        # Ensure 1D arrays
        v1 = emb1.flatten()
        v2 = emb2.flatten()
        
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def batch_generate_embeddings(
        self, texts: List[str], domain: str = "batch"
    ) -> List[EmbeddingResult]:
        """
        Generar embeddings para múltiples textos (optimizado)

        Args:
            texts: Lista de textos
            domain: Dominio de los textos

        Returns:
            Lista de resultados de embeddings
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Modelo de embeddings no inicializado")

        start_time = datetime.now()

        try:
            # Preprocesar textos
            processed_texts = [self._preprocess_text(text) for text in texts]

            # Generar embeddings en batch
            embeddings = self.model.encode(
                processed_texts,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
                batch_size=32,  # Optimizado para rendimiento
            )

            # Crear resultados
            results = []
            processing_time = (datetime.now() - start_time).total_seconds()
            avg_time_per_text = processing_time / len(texts)

            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                result = EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model_used=self.config.model_name,
                    processing_time=avg_time_per_text,
                    dimensions=len(embedding),
                    normalized=self.config.normalize_embeddings,
                )
                results.append(result)

                # Actualizar estadísticas
                self.stats["total_embeddings_generated"] += 1
                self._update_avg_processing_time(avg_time_per_text)

            logger.info(
                f"Batch embeddings generados: {len(texts)} textos en {processing_time:.3f}s"
            )
            return results

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error en batch embedding generation: {e}")
            raise RuntimeError(f"Error en batch embedding generation: {e}")

    def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.config.model_name,
            "device": self.config.device,
            "cache_enabled": self.config.cache_enabled,
            "cache_size": len(self.embedding_cache),
            "max_cache_size": self.config.cache_size,
            "performance": self.stats,
            "config": {
                "max_seq_length": self.config.max_seq_length,
                "normalize_embeddings": self.config.normalize_embeddings,
                "similarity_threshold": self.config.similarity_threshold,
            },
        }

    def clear_cache(self):
        """Limpiar caché de embeddings"""
        self.embedding_cache.clear()
        self.text_cache.clear()
        logger.info("🧹 Caché de embeddings limpiado")

    async def cleanup(self):
        """Limpiar recursos"""
        self.clear_cache()
        if self.model is not None:
            # Liberar memoria del modelo si es posible
            del self.model
            self.model = None
            self.model_loaded = False
        logger.info("🧹 Unified Embedding Semantic System limpiado")


# Funciones de compatibilidad
async def generate_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Función de compatibilidad para generar embeddings"""
    system = UnifiedEmbeddingSemanticSystem()
    await system.initialize()

    results = system.batch_generate_embeddings(texts)
    return [result.embedding for result in results]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Función de compatibilidad para calcular similitud semántica entre textos"""
    try:
        # Nota: Esta función síncrona tiene limitaciones con la inicialización async
        # En producción se debería usar la versión async del sistema

        # Crear sistema con configuración básica
        config = EmbeddingConfig(device="cpu", cache_enabled=False)
        system = UnifiedEmbeddingSemanticSystem(config)

        # Intentar inicializar (esto puede fallar en entornos sin dependencias)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Intentar inicialización síncrona forzada
            success = loop.run_until_complete(system.initialize())
            if not success:
                logger.warning("No se pudo inicializar modelo de embeddings, retornando similitud básica")
                return _basic_text_similarity(text1, text2)

            # Calcular similitud usando el sistema inicializado
            return system.calculate_similarity(text1, text2)

        finally:
            loop.close()

    except Exception as e:
        logger.warning(f"Error en cálculo de similitud semántica: {e}, usando similitud básica")
        # Fallback a similitud básica basada en texto
        return _basic_text_similarity(text1, text2)


def _basic_text_similarity(text1: str, text2: str) -> float:
    """Calcular similitud básica basada en overlapping de palabras"""
    if not text1 or not text2:
        return 0.0

    # Normalizar textos
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    # Calcular Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    if union == 0:
        return 0.0

    return intersection / union


if __name__ == "__main__":
    # Demo del sistema de embeddings
    async def demo():
        system = UnifiedEmbeddingSemanticSystem()

        # Inicializar
        print("🔍 Unified Embedding Semantic System Demo")
        print("=" * 50)

        success = await system.initialize()
        if not success:
            print("❌ Error inicializando sistema")
            return

        # Generar embedding de ejemplo
        sample_text = "La inteligencia artificial está transformando el mundo"
        print(f"📝 Generando embedding para: '{sample_text}'")

        try:
            result = system.generate_embedding(sample_text)
            print("✅ Embedding generado:")
            print(f"   Dimensiones: {result.dimensions}")
            print(f"   Tiempo: {result.processing_time:.3f}s")
            print(f"   Normalizado: {result.normalized}")

            # Búsqueda semántica de ejemplo
            corpus = [
                "La IA está cambiando la sociedad",
                "El clima está cambiando rápidamente",
                "Los coches eléctricos son el futuro",
                "La inteligencia artificial transforma industrias",
            ]

            print(f"\n🔍 Búsqueda semántica en corpus de {len(corpus)} textos")
            search_result = system.semantic_search(sample_text, corpus, top_k=3)

            print("📊 Resultados:")
            for i, res in enumerate(search_result.results, 1):
                print(f"   {i}. '{res['text']}' (similitud: {res['similarity']:.3f})")

            # Estadísticas del sistema
            stats = system.get_system_stats()
            print("\n📈 Estadísticas del sistema:")
            print(f"   Modelo: {stats['model_name']}")
            print(f"   Cache: {stats['cache_size']}/{stats['max_cache_size']}")
            print(
                f"   Total embeddings: {stats['performance']['total_embeddings_generated']}"
            )

        except Exception as e:
            print(f"❌ Error en demo: {e}")

        # Limpiar
        await system.cleanup()

    asyncio.run(demo())
