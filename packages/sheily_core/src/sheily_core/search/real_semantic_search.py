"""
Real Semantic Search Engine - NO SIMULATIONS
Uses Sentence Transformers + FAISS for real vector search
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RealSemanticSearch:
    """
    Real semantic search using Sentence Transformers and FAISS
    NO MOCKS - Actual vector embeddings and similarity search
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic search
        
        Args:
            model_name: Sentence transformer model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index = None
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.dimension = 384  # Default for MiniLM
        
        logger.info(f"🔍 Real Semantic Search initialized with {model_name}")
    
    def _load_model(self):
        """Load sentence transformer model"""
        if self.model is None:
            try:
                logger.info(f"📥 Loading model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"✅ Model loaded (dimension: {self.dimension})")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                raise
    
    def _create_index(self):
        """Create FAISS index"""
        try:
            import faiss
            
            # Use L2 distance for similarity
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"✅ FAISS index created (dimension: {self.dimension})")
            
        except ImportError:
            logger.error("❌ FAISS not installed. Run: pip install faiss-cpu")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add documents to the search index
        
        Args:
            documents: List of text documents
            metadata: Optional metadata for each document
        """
        try:
            self._load_model()
            
            if self.index is None:
                self._create_index()
            
            logger.info(f"📝 Adding {len(documents)} documents...")
            
            # Generate embeddings
            embeddings = self.model.encode(
                documents,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store documents and metadata
            self.documents.extend(documents)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{} for _ in documents])
            
            logger.info(f"✅ Added {len(documents)} documents (total: {len(self.documents)})")
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            raise
    
    def search(
        self,
        query: str,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Optional similarity threshold
            
        Returns:
            List of results with document, score, and metadata
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Add documents first.")
            
            if self.index is None or len(self.documents) == 0:
                logger.warning("⚠️ No documents in index")
                return []
            
            logger.info(f"🔍 Searching for: '{query[:50]}...'")
            
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search in FAISS
            distances, indices = self.index.search(
                query_embedding.astype('float32'),
                min(k, len(self.documents))
            )
            
            # Format results
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    # Convert L2 distance to similarity score (0-1)
                    similarity = 1 / (1 + dist)
                    
                    if threshold is None or similarity >= threshold:
                        results.append({
                            "document": self.documents[idx],
                            "score": float(similarity),
                            "distance": float(dist),
                            "metadata": self.metadata[idx],
                            "index": int(idx)
                        })
            
            logger.info(f"✅ Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def save_index(self, path: str):
        """Save index to disk"""
        try:
            import faiss
            
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(path_obj / "index.faiss"))
            
            # Save documents and metadata
            with open(path_obj / "documents.pkl", "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "model_name": self.model_name,
                    "dimension": self.dimension
                }, f)
            
            logger.info(f"💾 Index saved to: {path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save index: {e}")
    
    def load_index(self, path: str):
        """Load index from disk"""
        try:
            import faiss
            
            path_obj = Path(path)
            
            # Load FAISS index
            self.index = faiss.read_index(str(path_obj / "index.faiss"))
            
            # Load documents and metadata
            with open(path_obj / "documents.pkl", "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]
                self.model_name = data["model_name"]
                self.dimension = data["dimension"]
            
            # Load model
            self._load_model()
            
            logger.info(f"📂 Index loaded from: {path}")
            logger.info(f"   Documents: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            raise


# Singleton
_real_semantic_search: Optional[RealSemanticSearch] = None


def get_real_semantic_search(model_name: str = "all-MiniLM-L6-v2") -> RealSemanticSearch:
    """Get singleton instance"""
    global _real_semantic_search
    
    if _real_semantic_search is None:
        _real_semantic_search = RealSemanticSearch(model_name)
    
    return _real_semantic_search


# Demo
if __name__ == "__main__":
    print("🔍 Real Semantic Search Demo")
    print("=" * 50)
    
    # Initialize
    search = get_real_semantic_search()
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "Neural networks are inspired by biological brains",
        "Deep learning uses multiple layers of neural networks",
        "Natural language processing helps computers understand text"
    ]
    
    # Add documents
    search.add_documents(documents)
    
    # Search
    results = search.search("What is AI?", k=3)
    
    print("\n📊 Search Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   {result['document'][:60]}...")
