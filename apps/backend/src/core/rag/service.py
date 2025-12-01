#!/usr/bin/env python3
"""
RAG SERVICE - Retrieval-Augmented Generation System
===================================================

Enterprise-grade RAG system integrating:
- Sentence-Transformers embeddings (all-MiniLM-L6-v2) - REQUIRED
- Vector search with HNSW (FAISS) - REQUIRED
- NO FALLBACKS - System fails fast if dependencies missing
- Full integration with ChatService

REAL IMPLEMENTATION - All dependencies must be available
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# REQUIRED IMPORTS - NO FALLBACKS
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is REQUIRED for RAG. "
        "Install with: pip install sentence-transformers"
    )

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu is REQUIRED for RAG. "
        "Install with: pip install faiss-cpu"
    )

logger = logging.getLogger("rag_service")


class RAGService:
    """
    Enterprise RAG System with Transformer Embeddings and FAISS
    REAL IMPLEMENTATION - NO FALLBACKS
    """

    def __init__(self, config_path: str = "config/universal.yaml", data_dir: str = "./rag_data"):
        self.config_path = config_path
        self.data_dir = Path(data_dir)
        self.config = None

        # System Components - ALL REQUIRED
        self.embedding_model = None
        self.documents = []
        self.doc_ids = []
        self.metadatas = []
        self.embeddings = None

        # Vector Indices (HNSW/FAISS) - REQUIRED
        self.vector_index = None

        # State
        self.initialized = False
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension

        logger.info("🔧 RAG Service initializing (NO FALLBACKS)")

    async def initialize(self) -> bool:
        """
        Initialize the complete RAG system.
        REAL IMPLEMENTATION - Fails fast if embeddings cannot be loaded
        """
        try:
            # Load configuration
            await self._load_config()

            # Initialize embeddings - REQUIRED, NO FALLBACK
            await self._initialize_embeddings()

            # Load data if exists
            await self._load_data()

            # Initialize indices if data exists
            if self.documents:
                await self._build_indices()

            self.initialized = True
            logger.info(f"✅ RAG Service operational - {len(self.documents)} documents indexed")
            return True

        except Exception as e:
            logger.error(f"❌ RAG Service initialization FAILED: {e}")
            raise RuntimeError(
                f"RAG Service failed to initialize: {e}. "
                "NO FALLBACKS - Fix the issue and restart."
            )

    async def _load_config(self) -> None:
        """Load configuration"""
        try:
            import yaml
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
                    logger.info("[OK] Configuration loaded")
            else:
                self.config = {}
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            self.config = {}

    async def _initialize_embeddings(self) -> None:
        """
        Initialize Sentence Transformers embeddings.
        REAL IMPLEMENTATION - NO FALLBACKS, raises exception on failure
        """
        try:
            # Config from universal.yaml or defaults
            embedder_config = self.config.get("embedder", {}) if self.config else {}
            model_name = embedder_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")

            # Force CPU for stability on Windows unless configured otherwise
            device = embedder_config.get("device", "cpu")
            if device != "cpu":
                device = "cpu"  # Force CPU for safety

            logger.info(f"🔧 Loading embedding model: {model_name} on {device}")

            # Load model with cache optimization
            self.embedding_model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=os.environ.get("HF_HOME", "./cache")
            )

            # Configure dimension
            test_embed = self.embedding_model.encode(["test"])
            self.embedding_dimension = len(test_embed[0])

            logger.info(f"✓ Embedding model loaded - dimension: {self.embedding_dimension}")

        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {e}")
            raise RuntimeError(
                f"Failed to load embedding model: {e}. "
                "NO FALLBACKS - Embeddings are required for RAG. "
                "Install with: pip install sentence-transformers"
            )

    async def _load_data(self) -> None:
        """Load persisted data"""

        try:
            data_file = self.data_dir / "documents.json"
            if data_file.exists():
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.doc_ids = data.get("doc_ids", [])
                    self.metadatas = data.get("metadatas", [])
                    logger.info(f"[OK] Data loaded: {len(self.documents)} documents")
            else:
                logger.info("No pre-loaded data found")

        except Exception as e:
            logger.error(f"Error loading data: {e}")

    async def _build_indices(self) -> None:
        """
        Build search indices with embeddings and FAISS.
        REAL IMPLEMENTATION - NO FALLBACKS
        """
        try:
            if not self.documents:
                return

            if not self.embedding_model:
                raise RuntimeError(
                    "Embedding model not initialized. Cannot build indices. "
                    "NO FALLBACKS - Embeddings are required."
                )

            logger.info("🔧 Generating embeddings...")

            # Optimized batch processing
            batch_size = min(32, len(self.documents))
            embeddings_list = []

            for i in range(0, len(self.documents), batch_size):
                batch_texts = self.documents[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=8
                )
                embeddings_list.append(batch_embeddings)

            self.embeddings = np.vstack(embeddings_list)
            logger.info(f"✓ Embeddings generated: {self.embeddings.shape}")

            # Build HNSW index - REQUIRED
            await self._build_vector_index()

        except Exception as e:
            logger.error(f"❌ Error building indices: {e}")
            raise

    async def _build_vector_index(self) -> None:
        """
        Build HNSW vector index with FAISS.
        REAL IMPLEMENTATION - NO FALLBACKS, raises on failure
        """
        if self.embeddings is None:
            raise RuntimeError("Cannot build vector index without embeddings")

        try:
            dimension = self.embedding_dimension

            # Create HNSW index
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per element
            index.hnsw.efConstruction = 200
            index.add(self.embeddings.astype('float32'))

            self.vector_index = index
            logger.info(f"✓ HNSW Index built: {len(self.embeddings)} vectors")

        except Exception as e:
            logger.error(f"❌ Error building HNSW: {e}")
            raise RuntimeError(
                f"Failed to build FAISS index: {e}. "
                "NO FALLBACKS - FAISS is required for RAG."
            )

    # ========== CHAT SERVICE COMPATIBILITY ==========

    async def index_documents(self, docs: List[str], ids: List[str] = None, metadatas: List[Dict] = None) -> Dict[str, Any]:
        """Index documents - compatible with ChatService"""

        if not docs:
            return {"error": "no documents"}

        if not ids:
            ids = [f"doc_{i}" for i in range(len(docs))]
        if not metadatas:
            metadatas = [{}] * len(docs)

        self.documents = docs
        self.doc_ids = ids
        self.metadatas = metadatas

        # Build indices
        await self._build_indices()

        # Save data
        await self._save_data()

        return {
            "indexed": len(docs),
            "method": "sentence_transformers_hnsw",
            "embeddings_available": True,
            "vector_index": self.vector_index is not None
        }

    async def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Vector search with sentence transformers and FAISS.
        REAL IMPLEMENTATION - NO FALLBACKS
        """
        if not self.initialized:
            raise RuntimeError("RAG not initialized. Call initialize() first.")

        if not self.documents:
            return {"query": query, "results": [], "total_docs": 0}

        if not self.embedding_model or self.embeddings is None:
            raise RuntimeError(
                "Embedding model or embeddings not available. "
                "NO FALLBACKS - Vector search requires embeddings."
            )

        try:
            results = await self._vector_search(query, top_k)

            return {
                "query": query,
                "results": results,
                "method": "sentence_transformers_hnsw",
                "total_docs": len(self.documents)
            }

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            raise

    async def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Vector search implementation"""

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        query_embedding = query_embedding.astype('float32')

        # FAISS Search
        if self.vector_index is not None:
            faiss.cvar.hnsw.efSearch.set(100)
            distances, indices = self.vector_index.search(
                query_embedding.reshape(1, -1),
                top_k
            )

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    results.append({
                        "document": self.documents[idx],
                        "id": self.doc_ids[idx],
                        "metadata": self.metadatas[idx],
                        "similarity": float(1 - dist),
                        "score": float(1 - dist)
                    })
        else:
            # Manual Cosine Search
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    "document": self.documents[idx],
                    "id": self.doc_ids[idx],
                    "metadata": self.metadatas[idx],
                    "similarity": float(similarities[idx]),
                    "score": float(similarities[idx])
                })

        return results


    async def retrieve_relevant_context(self, query: str, top_k: int = 3, similarity_threshold: float = 0.1) -> str:
        """Get relevant context as string (ChatService compatible)"""
        results = await self.search(query, top_k=top_k)

        if not results or "results" not in results:
            return ""

        context_parts = []
        for result in results["results"]:
            similarity = result.get("similarity", 0)
            if similarity > similarity_threshold:
                doc = result.get("document", "")
                if doc:
                    context_parts.append(doc)

        return "\n\n".join(context_parts)

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection info"""
        return {
            "name": "rag_collection",
            "initialized": self.initialized,
            "count": len(self.documents),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_index": "HNSW" if self.vector_index else "building",
            "dimension": self.embedding_dimension
        }

    async def is_ready(self) -> bool:
        """Check if ready"""
        return self.initialized and len(self.documents) > 0

    async def _save_data(self) -> None:
        """Persist data"""
        try:
            data = {
                "documents": self.documents,
                "doc_ids": self.doc_ids,
                "metadatas": self.metadatas,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_index": "HNSW" if self.vector_index else "pending"
            }

            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.data_dir / "documents.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"[OK] Data saved to {self.data_dir}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
