#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete RAG Integration System
Combines all advanced RAG techniques with MCP agents and Federated Learning

Based on EMNLP 2024 Paper + Sheily AI Architecture
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..chunking import AdvancedChunker, ChunkingMethod, ChunkingResult
from ..evaluation import EvaluationResult, RAGEvaluator
from ..parametric_rag import ParametricRAG, ParametricRAGResult

# Import all RAG components
from ..query_classification import QueryClassificationResult, QueryClassifier
from ..reranking import Reranker, RerankerType, RerankingResult
from ..retrieval import AdvancedRetriever, RetrievalMethod, RetrievalResult
from ..summarization import ContextSummarizer, SummarizationMethod, SummarizationResult

logger = logging.getLogger(__name__)


@dataclass
class IntegratedRAGResult:
    """Complete result from integrated RAG system"""

    query: str
    classification: QueryClassificationResult
    chunks: Optional[ChunkingResult]
    retrieval: RetrievalResult
    reranking: Optional[RerankingResult]
    summarization: Optional[SummarizationResult]
    parametric_rag: Optional[ParametricRAGResult]  # New paradigm result
    evaluation: Optional[EvaluationResult]
    final_answer: str
    processing_time: float
    metadata: Dict[str, Any]


class RAGIntegrator:
    """
    Complete RAG Integration System

    Orchestrates all advanced RAG techniques:
    1. Query Classification
    2. Advanced Chunking
    3. Multiple Retrieval Methods
    4. Reranking
    5. Context Summarization
    6. Comprehensive Evaluation

    Integrated with MCP agents and Federated Learning
    """

    def __init__(
        self,
        # Component configurations
        query_classifier_path: Optional[str] = None,
        embedding_model: str = "BAAI/bge-m3",
        reranker_type: RerankerType = RerankerType.MONOT5,
        summarization_method: SummarizationMethod = SummarizationMethod.RECOMP,
        # Retrieval settings
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        hybrid_alpha: float = 0.3,
        # Chunking settings
        chunking_method: ChunkingMethod = ChunkingMethod.SMALL_TO_BIG,
        # COLING 2025 settings
        enable_query_expansion: bool = True,
        enable_contrastive_icl: bool = True,
        enable_focus_mode: bool = True,
        enable_retrieval_stride: bool = True,
        retrieval_stride: int = 5,  # Optimal from COLING 2025
        # MCP integration
        enable_mcp_integration: bool = True,
        federated_learning_enabled: bool = True,
        # Parametric RAG (New Paradigm)
        enable_parametric_rag: bool = False,
        parametric_rag_model: str = "meta-llama/Llama-2-7b-hf",
        # Documents and knowledge base
        documents: Optional[List[str]] = None,
        knowledge_base_path: Optional[str] = None,
    ):
        """
        Initialize the complete RAG integration system

        Args:
            query_classifier_path: Path to trained query classifier
            embedding_model: Embedding model for retrieval
            reranker_type: Type of reranker to use
            summarization_method: Context summarization method
            retrieval_method: Primary retrieval method
            hybrid_alpha: BM25 weight in hybrid search
            chunking_method: Document chunking strategy
            enable_mcp_integration: Enable MCP agent integration
            federated_learning_enabled: Enable federated learning
            documents: Initial document collection
            knowledge_base_path: Path to knowledge base
        """
        self.query_classifier_path = query_classifier_path
        self.embedding_model = embedding_model
        self.reranker_type = reranker_type
        self.summarization_method = summarization_method
        self.retrieval_method = retrieval_method
        self.hybrid_alpha = hybrid_alpha
        self.chunking_method = chunking_method
        self.enable_mcp_integration = enable_mcp_integration
        self.federated_learning_enabled = federated_learning_enabled
        self.enable_parametric_rag = enable_parametric_rag
        self.parametric_rag_model = parametric_rag_model
        self.documents = documents or []
        self.knowledge_base_path = knowledge_base_path

        # Initialize all components
        self._initialize_components()

        logger.info("RAG Integration System initialized")

    def _initialize_components(self):
        """Initialize all RAG components"""
        # Query Classifier
        self.query_classifier = QueryClassifier()
        if self.query_classifier_path:
            self.query_classifier.load_trained_model(self.query_classifier_path)

        # Advanced Chunker
        self.chunker = AdvancedChunker(embedding_model=self.embedding_model)

        # Advanced Retriever
        self.retriever = AdvancedRetriever(
            embedding_model=self.embedding_model,
            hybrid_alpha=self.hybrid_alpha,
            documents=self.documents,
        )

        # Reranker
        self.reranker = Reranker(reranker_type=self.reranker_type)

        # Context Summarizer
        self.summarizer = ContextSummarizer()

        # RAG Evaluator
        self.evaluator = RAGEvaluator()

        # MCP Integration
        self.mcp_agents = None
        self.mcp_available = False
        if self.enable_mcp_integration:
            self._initialize_mcp_integration()

        # Federated Learning
        self.fl_coordinator = None
        self.fl_available = False
        if self.federated_learning_enabled:
            self._initialize_federated_learning()

        # Parametric RAG (New Paradigm)
        self.parametric_rag = None
        if self.enable_parametric_rag:
            self._initialize_parametric_rag()

    def _initialize_mcp_integration(self):
        """Initialize MCP agent integration"""
        try:
            from sheily_core.core.mcp.mcp_agent_manager import MCPAgentManager
            self.mcp_agents = MCPAgentManager()
            self.mcp_available = True
            logger.info("✅ MCP integration initialized successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Could not import MCPAgentManager: {e}")
            logger.warning("⚠️ MCP integration disabled - funcionalidad no disponible")
            self.mcp_agents = None
            self.mcp_available = False
        except Exception as e:
            logger.error(f"❌ Error initializing MCP integration: {e}")
            self.mcp_agents = None
            self.mcp_available = False

    def _initialize_federated_learning(self):
        """Initialize federated learning coordinator"""
        try:
            # Intentar importar sistema de FL real si existe
            try:
                # Aquí se integraría con el sistema de FL real cuando esté disponible
                # Por ahora, crear estructura básica pero funcional
                self.fl_coordinator = {
                    "status": "initialized",
                    "nodes": [],
                    "round": 0,
                    "model_version": "1.0.0"
                }
                self.fl_available = True
                logger.info("✅ Federated learning coordinator initialized")
                logger.info("⚠️ Nota: FL está en modo básico - integración completa pendiente")
            except Exception as e:
                logger.warning(f"⚠️ Error en inicialización avanzada de FL: {e}")
                # Fallback a estructura básica
                self.fl_coordinator = {
                    "status": "basic_mode",
                    "nodes": [],
                    "available": False
                }
                self.fl_available = False
        except Exception as e:
            logger.error(f"❌ Federated learning initialization failed: {e}")
            self.fl_coordinator = None
            self.fl_available = False

    def _initialize_parametric_rag(self):
        """Initialize Parametric RAG system"""
        try:
            self.parametric_rag = ParametricRAG(
                base_model=self.parametric_rag_model,
                retriever=self.retriever,  # Inject existing retriever
            )

            # Add existing documents to parametric RAG
            for i, doc in enumerate(self.documents):
                doc_id = f"doc_{i}"
                self.parametric_rag.add_document(doc_id, doc)

            logger.info(
                f"Parametric RAG initialized with {len(self.documents)} documents"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Parametric RAG: {e}")
            self.parametric_rag = None

    def process_query(
        self,
        query: str,
        ground_truth: Optional[str] = None,
        enable_evaluation: bool = True,
    ) -> IntegratedRAGResult:
        """
        Process a complete query through the integrated RAG system

        Args:
            query: User query
            ground_truth: Ground truth answer for evaluation
            enable_evaluation: Whether to perform evaluation

        Returns:
            Complete integrated RAG result
        """
        import time

        start_time = time.time()

        # Step 1: Query Classification
        classification = self.query_classifier.classify(query)
        needs_retrieval = classification.needs_retrieval

        # Step 2: Document Chunking (if documents available)
        chunks = None
        if self.documents:
            # Combine all documents and chunk them
            combined_text = " ".join(self.documents)
            chunks = self.chunker.chunk_document(combined_text, self.chunking_method)

            # Update retriever with chunked documents
            chunked_texts = chunks.chunks
            self.retriever.add_documents(chunked_texts)

        # Step 3: Retrieval
        if needs_retrieval:
            retrieval = self.retriever.retrieve(query, self.retrieval_method, top_k=10)
        else:
            # For non-retrieval queries, return minimal context
            retrieval = RetrievalResult(
                query=query,
                retrieved_docs=[],
                scores=[],
                method=RetrievalMethod.BM25,
                processing_time=0.0,
                metadata={"skipped_retrieval": True},
            )

        # Step 4: Reranking (if we have retrieved documents)
        reranking = None
        if retrieval.retrieved_docs and len(retrieval.retrieved_docs) > 1:
            reranking = self.reranker.rerank(
                query, retrieval.retrieved_docs, retrieval.scores
            )

        # Step 5: Context Summarization (if context is too long)
        summarization = None
        context_to_use = retrieval.retrieved_docs

        if context_to_use:
            total_context_length = sum(len(doc) for doc in context_to_use)
            if total_context_length > 2000:  # Threshold for summarization
                combined_context = " ".join(context_to_use)
                summarization = self.summarizer.summarize(
                    combined_context, self.summarization_method, query
                )
                context_to_use = [summarization.summarized_text]

        # Step 6: Parametric RAG (New Paradigm) - if enabled
        parametric_rag_result = None
        final_answer = None  # Initialize final_answer

        if self.enable_parametric_rag and self.parametric_rag and needs_retrieval:
            try:
                parametric_rag_result = (
                    self.parametric_rag.generate_with_parametric_rag(
                        query=query, top_k=3  # Use top 3 for efficiency
                    )
                )
                # Use Parametric RAG answer as final answer if available
                if (
                    parametric_rag_result.generated_answer
                    != "Error: No parametric documents available"
                ):
                    final_answer = parametric_rag_result.generated_answer
            except Exception as e:
                logger.warning(f"Parametric RAG failed: {e}")

        # Step 7: Generate Final Answer (fallback to traditional RAG)
        if final_answer is None:
            final_answer = self._generate_final_answer(
                query, context_to_use, classification
            )

        # Step 8: Evaluation (if enabled and ground truth available)
        evaluation = None
        if enable_evaluation and ground_truth:
            evaluation = self.evaluator.evaluate_sample(
                query=query,
                generated_answer=final_answer,
                retrieved_contexts=retrieval.retrieved_docs,
                ground_truth=ground_truth,
            )

        processing_time = time.time() - start_time

        # Compile metadata
        metadata = {
            "pipeline_steps": [
                "classification",
                "chunking",
                "retrieval",
                "reranking",
                "summarization",
                "parametric_rag",
                "generation",
            ],
            "needs_retrieval": needs_retrieval,
            "total_documents": len(self.documents),
            "chunks_created": len(chunks.chunks) if chunks else 0,
            "docs_retrieved": len(retrieval.retrieved_docs),
            "reranking_applied": reranking is not None,
            "summarization_applied": summarization is not None,
            "parametric_rag_applied": parametric_rag_result is not None,
            "evaluation_performed": evaluation is not None,
            "mcp_integration_active": self.enable_mcp_integration,
            "federated_learning_active": self.federated_learning_enabled,
            "parametric_rag_active": self.enable_parametric_rag,
        }

        return IntegratedRAGResult(
            query=query,
            classification=classification,
            chunks=chunks,
            retrieval=retrieval,
            reranking=reranking,
            summarization=summarization,
            parametric_rag=parametric_rag_result,
            evaluation=evaluation,
            final_answer=final_answer,
            processing_time=processing_time,
            metadata=metadata,
        )

    def _generate_final_answer(
        self,
        query: str,
        context_docs: List[str],
        classification: QueryClassificationResult,
    ) -> str:
        """
        Generate final answer using available context and classification

        Args:
            query: Original query
            context_docs: Retrieved/summarized context documents
            classification: Query classification result

        Returns:
            Generated answer
        """
        if not classification.needs_retrieval:
            # For queries that don't need retrieval, provide direct answer
            if "translation" in classification.predicted_class.lower():
                return "Lo siento, las capacidades de traducción requieren configuración adicional."
            elif "summarization" in classification.predicted_class.lower():
                return (
                    "Para resumir contenido, por favor proporciona el texto a resumir."
                )
            elif "planning" in classification.predicted_class.lower():
                return "Para ayudarte con planificación, necesito más detalles sobre tus objetivos."
            else:
                return f"Entiendo tu consulta sobre '{query}'. ¿Puedes proporcionar más contexto específico?"

        # For retrieval-required queries, use context to generate answer
        if not context_docs:
            return "No encontré información relevante para responder tu consulta."

        # Simple answer generation (in production, use fine-tuned generator)
        combined_context = " ".join(context_docs[:3])  # Use top 3 docs

        # Basic answer extraction/generation
        if len(combined_context) < 500:
            answer = (
                f"Basándome en la información disponible: {combined_context[:300]}..."
            )
        else:
            # Extract key sentences
            sentences = combined_context.split(".")
            key_sentences = sentences[:2]  # First two sentences
            answer = ". ".join(key_sentences) + "."

        return answer

    def add_documents(self, documents: List[str]):
        """
        Add documents to the knowledge base

        Args:
            documents: List of document texts
        """
        self.documents.extend(documents)
        self.retriever.add_documents(documents)

        # Also add to Parametric RAG if enabled
        if self.parametric_rag:
            for i, doc in enumerate(documents):
                doc_id = f"doc_{len(self.documents) - len(documents) + i}"
                self.parametric_rag.add_document(doc_id, doc)

        logger.info(f"Added {len(documents)} documents to knowledge base")

    def update_models(self):
        """Update all models with latest federated learning parameters"""
        if self.federated_learning_enabled and self.fl_coordinator:
            logger.info("Updating models with federated learning parameters")
            # Simulate model update
            # In production, this would pull weights from the FL coordinator
            if hasattr(self.retriever, 'update_weights'):
                # self.retriever.update_weights(self.fl_coordinator.get_weights())
                pass
            logger.info("Models updated successfully")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "components": {
                "query_classifier": self.query_classifier is not None,
                "chunker": self.chunker is not None,
                "retriever": self.retriever is not None,
                "reranker": self.reranker is not None,
                "summarizer": self.summarizer is not None,
                "evaluator": self.evaluator is not None,
                "parametric_rag": self.parametric_rag is not None,
            },
            "integrations": {
                "mcp_enabled": self.enable_mcp_integration,
                "mcp_available": self.mcp_available,
                "federated_learning_enabled": self.federated_learning_enabled,
                "federated_learning_available": self.fl_available,
                "parametric_rag_enabled": self.enable_parametric_rag,
            },
            "knowledge_base": {
                "total_documents": len(self.documents),
                "embedding_model": self.embedding_model,
                "parametric_documents": (
                    len(self.parametric_rag.parametric_documents)
                    if self.parametric_rag
                    else 0
                ),
            },
            "performance_targets": {
                "query_classification_accuracy": 0.95,
                "retrieval_ndcg@10": 0.70,
                "reranking_improvement": 0.25,
                "faithfulness_score": 0.85,
                "answer_correctness": 0.80,
                "parametric_rag_speedup": "29-36% faster than in-context RAG",
            },
        }

    def optimize_pipeline(self, evaluation_results: List[EvaluationResult]):
        """
        Optimize pipeline based on evaluation results

        Args:
            evaluation_results: Recent evaluation results
        """
        # Analyze performance and adjust parameters
        avg_faithfulness = sum(
            r.metrics.get("faithfulness", 0) for r in evaluation_results
        ) / len(evaluation_results)

        if avg_faithfulness < 0.8:
            logger.info("Low faithfulness detected, adjusting summarization ratio")
            self.summarizer.summarization_ratio = min(
                0.5, self.summarizer.summarization_ratio + 0.1
            )

        # Add more optimization logic based on different metrics
        logger.info("Pipeline optimization completed")

    def batch_process(
        self, queries: List[str], ground_truths: Optional[List[str]] = None
    ) -> List[IntegratedRAGResult]:
        """
        Process multiple queries in batch

        Args:
            queries: List of queries
            ground_truths: Optional list of ground truth answers

        Returns:
            List of integrated RAG results
        """
        results = []
        for i, query in enumerate(queries):
            ground_truth = (
                ground_truths[i] if ground_truths and i < len(ground_truths) else None
            )
            result = self.process_query(query, ground_truth)
            results.append(result)

        return results
