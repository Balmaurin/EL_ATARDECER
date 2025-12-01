#!/usr/bin/env python3
"""
Simple RAG System Adapter for EL-AMANECERV3
Wrapper simplificado del UltraRAGSystem que funciona sin dependencias complejas
"""

import time
import logging
import numpy as np
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    """
    Sistema RAG simplificado para integración con AutonomousSystemController
    Proporciona funcionalidad básica de retrieval y generación
    """
    
    def __init__(self):
        self.documents = {}
        self.document_embeddings = {}
        self.is_initialized = False
        logger.info("📚 Simple RAG System initialized")
    
    def initialize_system(self) -> bool:
        """Inicializar sistema"""
        self.is_initialized = True
        return True
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> bool:
        """Añadir documentos al sistema"""
        try:
            for idx, doc in enumerate(documents):
                doc_id = f"doc_{idx}"
                self.documents[doc_id] = {
                    'content': doc,
                    'metadata': metadata[idx] if metadata and idx < len(metadata) else {}
                }
                # Embedding simple (hash-based)
                self.document_embeddings[doc_id] = self._simple_embedding(doc)
            
            logger.info(f"✅ {len(documents)} documents added to RAG")
            return True
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            return False
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Genera embedding simple basado en hash del texto"""
        # Usar hash para generar embedding reproducible
        text_hash = hash(text.lower())
        np.random.seed(text_hash % (2**32))
        return np.random.randn(384).astype(np.float32)  # Embedding de 384 dims
    
    def process_query(self, query: str, use_advanced_processing: bool = True) -> Dict[str, Any]:
        """Procesa una query y retorna respuesta basada en documentos"""
        try:
            start_time = time.time()
            
            if not self.is_initialized:
                return {"error": "System not initialized"}
            
            # Obtener embedding de la query
            query_embedding = self._simple_embedding(query)
            
            # Buscar documentos relevantes (similitud coseno)
            relevant_docs = []
            for doc_id, embedding in self.document_embeddings.items():
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                relevant_docs.append({
                    'doc_id': doc_id,
                    'similarity': float(similarity),
                    'content': self.documents[doc_id]['content']
                })
            
            # Ordenar por similitud
            relevant_docs.sort(key=lambda x: x['similarity'], reverse=True)
            top_docs = relevant_docs[:3]
            
            # Generar respuesta
            if top_docs:
                # Combinar contenido de documentos relevantes
                context = " ".join([doc['content'] for doc in top_docs])
                response = f"Based on the available knowledge: {context[:200]}... This relates to your query about '{query}'."
            else:
                response = f"I don't have enough information to answer '{query}'."
            
            processing_time = time.time() - start_time
            
            return {
                'query': query,
                'response': response,
                'documents_retrieved': len(top_docs),
                'processing_time': processing_time,
                'top_similarities': [doc['similarity'] for doc in top_docs]
            }
            
        except Exception as e:
            return {'error': str(e), 'query': query}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Estado del sistema RAG"""
        return {
            'initialized': self.is_initialized,
            'documents_count': len(self.documents),
            'embeddings_count': len(self.document_embeddings)
        }


# Cargar conocimiento base al inicializar
def initialize_rag_with_base_knowledge():
    """Inicializa RAG con conocimiento base del sistema"""
    rag = SimpleRAGSystem()
    rag.initialize_system()
    
    # Conocimiento base sobre el sistema
    base_knowledge = [
        """
        EL-AMANECERV3 is a complete AI consciousness system that integrates:
        - Global Workspace Theory for conscious processing
        - Digital Nervous System for emotions and neurotransmitters
        - Metacognition for self-aware thinking
        - Ethical Engine for moral decision-making
        - Digital DNA for unique personality
        """,
        """
        System optimization strategies:
        - When CPU load is high (>70%), reduce background tasks
        - Consolidate memories during low-load periods
        - Use ethical validation before critical actions
        - Learn from experiences and store in episodic memory
        """,
        """
        The conscious loop integrates pre-conscious inputs from:
        - Hardware sensors (CPU, RAM)
        - Emotional state (Digital Nervous System)
        - Memory associations (Vector Memory)
        - Subjective experience (Qualia Simulator)
        These compete in the Global Workspace to become conscious content.
        """
    ]
    
    rag.add_documents(base_knowledge)
    return rag
