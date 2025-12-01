#!/usr/bin/env python3
"""
Advanced Vector Indexing System - ChromaDB + Faiss Integration
==============================================================

Sistema de indexación vectorial híbrido para RAG enterprise:
- ChromaDB para persistencia y búsquedas vectoriales escalables
- Faiss para búsquedas de alta velocidad en memoria
- Hybrid search combinando BM25 + vector search
- Métricas avanzadas de rendimiento y calidad
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

# Importar métricas Prometheus
try:
    from core.indexing_metrics import (
        IndexingMetrics,
        create_metrics_for_index,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    IndexingMetrics = None
    create_metrics_for_index = None


class HybridVectorStore:
    """Sistema híbrido de indexación vectorial ChromaDB + Faiss"""

    def __init__(
        self,
        collection_name: str = "sheily_rag",
        persist_directory: str = "./rag_advanced/data",
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Modelos de embeddings
        self.embedding_model = None
        self.embedding_dimension = 384  # MiniLM default

        # ChromaDB setup
        self.chroma_client = None
        self.chroma_collection = None

        # Faiss setup
        self.faiss_index = None
        self.faiss_id_mapping = {}
        self.faiss_texts = []
        self.faiss_metadata = []

        # BM25 para hybrid search
        self.bm25_index = None
        self.bm25_texts = []

        # Métricas de rendimiento (legacy)
        self.metrics = {
            "queries_processed": 0,
            "avg_query_time": 0,
            "cache_hits": 0,
            "hybrid_search_ratio": 0.7,
        }

        # Métricas Prometheus
        if METRICS_AVAILABLE:
            self.prometheus_metrics = create_metrics_for_index(collection_name)
        else:
            self.prometheus_metrics = None

        # Initialization state
        self.initialized = False

    async def initialize(self):
        """Inicializar todos los componentes del sistema de manera asíncrona"""
        if self.initialized:
            return

        print("🚀 Inicializando sistema híbrido de indexación vectorial...")
        loop = asyncio.get_running_loop()

        # Run heavy initialization in executor
        await loop.run_in_executor(None, self._initialize_components_sync)
        self.initialized = True
        print("✅ Sistema híbrido inicializado")

    def _initialize_components_sync(self):
        """Inicialización síncrona de componentes (para correr en thread)"""
        # Modelo de embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                self.embedding_model = SentenceTransformer(model_name)
                self.embedding_dimension = (
                    self.embedding_model.get_sentence_embedding_dimension()
                )
                print(
                    f"✅ Modelo de embeddings cargado: {model_name} (dim: {self.embedding_dimension})"
                )
            except Exception as e:
                print(f"⚠️ Error cargando modelo de embeddings: {e}")
        else:
            print("❌ Sentence Transformers no disponible")

        # ChromaDB initialization
        if CHROMADB_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.persist_directory / "chromadb")
                )

                # Crear colección si no existe
                try:
                    self.chroma_collection = (
                        self.chroma_client.get_or_create_collection(
                            name=self.collection_name, metadata={"hnsw:space": "cosine"}
                        )
                    )
                    print(f"✅ ChromaDB colección inicializada: {self.collection_name}")
                except Exception as e:
                    print(f"⚠️ Error creando colección ChromaDB: {e}")
            except Exception as e:
                print(f"⚠️ Error inicializando ChromaDB: {e}")
        else:
            print("❌ ChromaDB no disponible")

        # Faiss initialization
        if FAISS_AVAILABLE:
            try:
                # Index HNSW para búsquedas eficientes
                self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dimension, 32)
                print("✅ Faiss index HNSW inicializado")
            except Exception as e:
                print(f"⚠️ Error inicializando Faiss: {e}")
        else:
            print("❌ Faiss no disponible")

    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Agregar documentos al índice híbrido"""
        if not self.initialized:
            await self.initialize()

        if not documents:
            return False

        print(f"📄 Procesando {len(documents)} documentos para indexación...")

        try:
            # Extraer textos y metadata
            texts = []
            metadatas = []
            ids = []

            for i, doc in enumerate(documents):
                text = doc.get("content", doc.get("text", ""))
                if text:
                    texts.append(text)
                    metadatas.append(doc.get("metadata", {}))
                    ids.append(doc.get("id", f"doc_{i}"))

            if not texts:
                return False

            # Generar embeddings (async)
            embeddings = await self._generate_embeddings(texts)

            if embeddings is None:
                return False

            # Registrar inicio de construcción de índice
            if self.prometheus_metrics:
                self.prometheus_metrics.start_index_build()

            # Run DB operations in executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, 
                lambda: self._add_to_stores_sync(texts, metadatas, ids, embeddings)
            )

            # Registrar finalización de construcción de índice
            if self.prometheus_metrics:
                self.prometheus_metrics.record_index_build("hybrid", success=True)
                # Actualizar número de vectores
                if self.faiss_index is not None:
                    self.prometheus_metrics.set_n_vectors(
                        self.faiss_index.ntotal, "faiss"
                    )

            return True

        except Exception as e:
            print(f"❌ Error procesando documentos: {e}")
            return False

    def _add_to_stores_sync(self, texts, metadatas, ids, embeddings):
        """Operaciones síncronas de base de datos"""
        # 1. Agregar a ChromaDB
        if self.chroma_collection is not None:
            try:
                self.chroma_collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
                print(f"✅ Documentos agregados a ChromaDB: {len(texts)}")
            except Exception as e:
                print(f"⚠️ Error agregando a ChromaDB: {e}")

        # 2. Agregar a Faiss
        if self.faiss_index is not None:
            try:
                # Agregar embeddings a Faiss
                self.faiss_index.add(embeddings.astype(np.float32))

                # Mantener mapping y textos para Faiss
                for i, (text, metadata, doc_id) in enumerate(
                    zip(texts, metadatas, ids)
                ):
                    self.faiss_id_mapping[len(self.faiss_texts)] = doc_id
                    self.faiss_texts.append(text)
                    self.faiss_metadata.append(metadata)

                print(f"✅ Documentos agregados a Faiss: {len(texts)}")
            except Exception as e:
                print(f"⚠️ Error agregando a Faiss: {e}")

        # 3. Actualizar BM25 index
        if BM25_AVAILABLE:
            try:
                self.bm25_texts.extend(texts)
                # Tokenizar para BM25
                tokenized_texts = [text.lower().split() for text in self.bm25_texts]
                self.bm25_index = BM25Okapi(tokenized_texts)
                print(
                    f"✅ BM25 index actualizado: {len(self.bm25_texts)} documentos"
                )
            except Exception as e:
                print(f"⚠️ Error actualizando BM25: {e}")

    async def hybrid_search(
        self, query: str, limit: int = 10, score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Búsqueda híbrida combinando vector search + BM25"""
        if not self.initialized:
            await self.initialize()

        start_time = datetime.now()

        print(f"🔍 Ejecutando búsqueda híbrida para: '{query[:50]}...'")

        try:
            # Generar embedding para la query
            query_embedding = await self._generate_embeddings([query])
            if query_embedding is None:
                return []

            query_embedding = query_embedding[0]

            # Run search in executor
            loop = asyncio.get_running_loop()
            combined_results = await loop.run_in_executor(
                None,
                lambda: self._search_sync(query, query_embedding, limit, score_threshold)
            )

            # Actualizar métricas
            query_time_delta = datetime.now() - start_time
            query_time_ms = query_time_delta.total_seconds() * 1000
            self._update_metrics(query_time_delta, len(combined_results))
            
            # Registrar en Prometheus
            if self.prometheus_metrics:
                self.prometheus_metrics.record_query(query_time_ms, "hybrid")

            print(
                f"✅ Búsqueda híbrida completada - Resultados: {len(combined_results)}"
            )
            return combined_results[:limit]

        except Exception as e:
            print(f"❌ Error en búsqueda híbrida: {e}")
            return []

    def _search_sync(self, query, query_embedding, limit, score_threshold):
        """Búsqueda síncrona"""
        vector_results = []
        bm25_results = []

        # 1. Vector search con Faiss
        if (
            FAISS_AVAILABLE
            and self.faiss_index is not None
            and len(self.faiss_texts) > 0
        ):
            try:
                # Búsqueda en Faiss
                distances, indices = self.faiss_index.search(
                    query_embedding.reshape(1, -1).astype(np.float32),
                    min(
                        limit * 2, len(self.faiss_texts)
                    ),  # Buscar más para luego filtrar
                )

                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(self.faiss_texts):
                        similarity = (
                            1 - distances[0][i]
                        )  # Convertir distancia a similitud
                        if similarity >= score_threshold:
                            vector_results.append(
                                {
                                    "text": self.faiss_texts[idx],
                                    "metadata": (
                                        self.faiss_metadata[idx]
                                        if idx < len(self.faiss_metadata)
                                        else {}
                                    ),
                                    "similarity": float(similarity),
                                    "source": "faiss",
                                    "doc_id": self.faiss_id_mapping.get(
                                        idx, f"faiss_{idx}"
                                    ),
                                }
                            )
            except Exception as e:
                print(f"⚠️ Error en búsqueda Faiss: {e}")

        # 2. BM25 search
        if BM25_AVAILABLE and self.bm25_index is not None:
            try:
                query_tokens = query.lower().split()
                bm25_scores = self.bm25_index.get_scores(query_tokens)

                # Obtener top resultados BM25
                top_bm25_indices = np.argsort(bm25_scores)[::-1][: limit * 2]

                for idx in top_bm25_indices:
                    score = bm25_scores[idx]
                    if score > 0.1:  # Umbral mínimo
                        bm25_results.append(
                            {
                                "text": self.bm25_texts[idx],
                                "bm25_score": float(score),
                                "source": "bm25",
                                "doc_id": f"bm25_{idx}",
                            }
                        )
            except Exception as e:
                print(f"⚠️ Error en búsqueda BM25: {e}")

        # 3. Combinar resultados
        return self._combine_results_sync(vector_results, bm25_results, query, limit)

    def _combine_results_sync(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        query: str,
        limit: int,
    ) -> List[Dict]:
        """Combinar resultados de vector search y BM25 de manera inteligente"""
        # (This logic was previously async but didn't await anything, so it can be sync)
        combined_scores = {}

        # Peso para cada tipo de resultado
        vector_weight = 0.7
        bm25_weight = 0.3

        # Procesar resultados vectoriales
        for result in vector_results:
            doc_id = result["doc_id"]
            combined_scores[doc_id] = {
                "vector_score": result["similarity"] * vector_weight,
                "text": result["text"],
                "metadata": result.get("metadata", {}),
                "source": "vector",
            }

        # Procesar resultados BM25
        for result in bm25_results:
            doc_id = result["doc_id"]
            bm25_score = result["bm25_score"] * bm25_weight

            if doc_id in combined_scores:
                combined_scores[doc_id]["bm25_score"] = bm25_score
                combined_scores[doc_id]["combined_score"] = (
                    combined_scores[doc_id]["vector_score"] + bm25_score
                )
            else:
                combined_scores[doc_id] = {
                    "bm25_score": bm25_score,
                    "vector_score": 0,
                    "text": result["text"],
                    "combined_score": bm25_score,
                    "source": "bm25",
                }

        # Ordenar por score combinado y devolver top resultados
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1].get("combined_score", 0),
            reverse=True,
        )

        # Formatear resultados finales
        final_results = []
        for doc_id, scores in sorted_results[:limit]:
            result = {
                "doc_id": doc_id,
                "text": scores["text"],
                "combined_score": scores.get("combined_score", 0),
                "vector_score": scores.get("vector_score", 0),
                "bm25_score": scores.get("bm25_score", 0),
                "source": scores.get("source", "combined"),
                "metadata": scores.get("metadata", {}),
            }
            final_results.append(result)

        return final_results

    # Kept for compatibility but redirects to sync version
    async def _combine_results(self, *args, **kwargs):
        return self._combine_results_sync(*args, **kwargs)

    async def _generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generar embeddings para textos dados (async)"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("❌ Modelo de embeddings no disponible")
            return None

        if not texts:
            return None

        try:
            # Run inference in executor
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None, 
                lambda: self.embedding_model.encode(texts, convert_to_numpy=True)
            )
            return embeddings
        except Exception as e:
            print(f"❌ Error generando embeddings: {e}")
            return None

    def _update_metrics(self, query_time, num_results):
        """Actualizar métricas de rendimiento"""
        self.metrics["queries_processed"] += 1

        # Calcular tiempo promedio
        current_avg = self.metrics["avg_query_time"]
        total_queries = self.metrics["queries_processed"]

        self.metrics["avg_query_time"] = (
            (current_avg * (total_queries - 1)) + query_time.total_seconds()
        ) / total_queries

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        stats = {
            "collections": {},
            "performance": self.metrics.copy(),
            "capabilities": {
                "chromadb": CHROMADB_AVAILABLE,
                "faiss": FAISS_AVAILABLE,
                "bm25": BM25_AVAILABLE,
                "embeddings": SENTENCE_TRANSFORMERS_AVAILABLE,
            },
        }

        # Estadísticas de ChromaDB
        if self.chroma_collection is not None:
            try:
                count = self.chroma_collection.count()
                stats["collections"]["chromadb"] = count
            except:
                stats["collections"]["chromadb"] = 0

        # Estadísticas de Faiss
        if self.faiss_index is not None:
            faiss_count = self.faiss_index.ntotal
            stats["collections"]["faiss"] = faiss_count
            # Actualizar métricas Prometheus
            if self.prometheus_metrics:
                self.prometheus_metrics.set_n_vectors(faiss_count, "faiss")
        else:
            stats["collections"]["faiss"] = 0

        # Estadísticas de BM25
        if self.bm25_texts is not None:
            stats["collections"]["bm25"] = len(self.bm25_texts)
        else:
            stats["collections"]["bm25"] = 0

        return stats

    async def save_index(self, filepath: str = None):
        """Guardar índices en disco"""
        if filepath is None:
            filepath = self.persist_directory / "vector_index_backup.json"

        try:
            backup_data = {
                "faiss_id_mapping": self.faiss_id_mapping,
                "faiss_texts": self.faiss_texts,
                "faiss_metadata": self.faiss_metadata,
                "bm25_texts": self.bm25_texts,
                "timestamp": datetime.now().isoformat(),
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)

            # Guardar índice Faiss si está disponible
            if FAISS_AVAILABLE and self.faiss_index is not None:
                faiss_index_file = filepath.with_suffix(".faiss")
                faiss.write_index(self.faiss_index, str(faiss_index_file))

            print(f"✅ Índices guardados: {filepath}")
            return True

        except Exception as e:
            print(f"❌ Error guardando índices: {e}")
            return False

    async def load_index(self, filepath: str = None):
        """Cargar índices desde disco"""
        if filepath is None:
            filepath = self.persist_directory / "vector_index_backup.json"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                backup_data = json.load(f)

            self.faiss_id_mapping = backup_data.get("faiss_id_mapping", {})
            self.faiss_texts = backup_data.get("faiss_texts", [])
            self.faiss_metadata = backup_data.get("faiss_metadata", [])
            self.bm25_texts = backup_data.get("bm25_texts", [])

            # Inicializar BM25 si hay datos
            if BM25_AVAILABLE and self.bm25_texts:
                tokenized_texts = [text.lower().split() for text in self.bm25_texts]
                self.bm25_index = BM25Okapi(tokenized_texts)

            # Cargar índice Faiss si existe
            if FAISS_AVAILABLE:
                faiss_index_file = filepath.with_suffix(".faiss")
                if faiss_index_file.exists():
                    self.faiss_index = faiss.read_index(str(faiss_index_file))

            print(f"✅ Índices cargados: {filepath}")
            return True

        except Exception as e:
            print(f"❌ Error cargando índices: {e}")
            return False


class VectorIndexingAPI:
    """API para operaciones de indexación vectorial"""

    def __init__(self):
        self.vector_store = None
        self.indexes = {}

    async def initialize(self, collection_name: str = "sheily_rag"):
        """Inicializar el sistema de indexación"""
        if self.vector_store is None:
            self.vector_store = HybridVectorStore(collection_name)
            await self.vector_store.initialize()
            self.indexes[collection_name] = self.vector_store
            print("🎯 API de Vector Indexing inicializada")
        return True

    async def create_index(self, index_name: str, config: Dict[str, Any] = None):
        """Crear un nuevo índice"""
        if config is None:
            config = {}

        try:
            index = HybridVectorStore(
                collection_name=f"{index_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                persist_directory=config.get(
                    "persist_directory", "./rag_advanced/data"
                ),
            )

            self.indexes[index_name] = index
            print(f"✅ Índice creado: {index_name}")
            return {"status": "success", "index_id": index_name}
        except Exception as e:
            print(f"❌ Error creando índice {index_name}: {e}")
            return {"status": "error", "message": str(e)}

    async def add_documents(self, index_name: str, documents: List[Dict[str, Any]]):
        """Agregar documentos a un índice específico"""
        if index_name not in self.indexes:
            return {"status": "error", "message": f"Índice no encontrado: {index_name}"}

        try:
            success = await self.indexes[index_name].add_documents(documents)
            if success:
                return {"status": "success", "documents_added": len(documents)}
            else:
                return {"status": "error", "message": "Error agregando documentos"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def search(self, index_name: str, query: str, limit: int = 10):
        """Buscar en un índice específico"""
        if index_name not in self.indexes:
            return {"status": "error", "message": f"Índice no encontrado: {index_name}"}

        try:
            results = await self.indexes[index_name].hybrid_search(query, limit)
            return {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_stats(self, index_name: str = None):
        """Obtener estadísticas"""
        if index_name:
            if index_name not in self.indexes:
                return {
                    "status": "error",
                    "message": f"Índice no encontrado: {index_name}",
                }

            stats = self.indexes[index_name].get_stats()
            return {"status": "success", "stats": stats}

        # Estadísticas generales
        all_stats = {}
        for name, index in self.indexes.items():
            all_stats[name] = index.get_stats()

        return {"status": "success", "all_stats": all_stats}

    async def save_indexes(self):
        """Guardar todos los índices"""
        results = {}
        for name, index in self.indexes.items():
            success = await index.save_index()
            results[name] = "saved" if success else "error"

        return {"status": "success", "results": results}

    async def load_indexes(self):
        """Cargar todos los índices"""
        results = {}
        for name, index in self.indexes.items():
            success = await index.load_index()
            results[name] = "loaded" if success else "error"

        return {"status": "success", "results": results}


if __name__ == "__main__":
    # Demo del sistema
    async def demo():
        print("🚀 Demo del sistema de indexación vectorial híbrido")
        print("=" * 60)

        # Inicializar API
        api = VectorIndexingAPI()
        await api.initialize()

        # Crear índice de ejemplo
        await api.create_index("demo_index")

        # Documentos de ejemplo
        sample_docs = [
            {
                "id": "doc1",
                "content": "La inteligencia artificial está transformando el mundo empresarial.",
                "metadata": {"category": "AI", "source": "demo"},
            },
            {
                "id": "doc2",
                "content": "El aprendizaje automático permite sistemas predictivos avanzados.",
                "metadata": {"category": "ML", "source": "demo"},
            },
            {
                "id": "doc3",
                "content": "La seguridad es fundamental en sistemas de IA modernos.",
                "metadata": {"category": "Security", "source": "demo"},
            },
        ]

        # Agregar documentos
        result = await api.add_documents("demo_index", sample_docs)
        print(f"📄 Documentos agregados: {result}")

        # Buscar
        search_result = await api.search("demo_index", "¿Qué es la IA?", limit=5)
        print(f"🔍 Resultados de búsqueda: {len(search_result.get('results', []))}")

        # Estadísticas
        stats = await api.get_stats("demo_index")
        print(f"📊 Estadísticas: {stats}")

        print("✅ Demo completada exitosamente!")

    asyncio.run(demo())
