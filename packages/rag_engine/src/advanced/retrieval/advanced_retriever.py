#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Retrieval Methods for RAG
Based on EMNLP 2024 Paper Section A.3

Implements multiple retrieval strategies:
- HyDE (Hypothetical Document Embedding)
- Query Rewriting with Zephyr-7b-alpha
- Query Decomposition with GPT-3.5-turbo
- Hybrid Search with optimal α weighting
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RetrievalMethod(Enum):
    """Available retrieval methods"""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYDE = "hyde"
    QUERY_REWRITING = "query_rewriting"
    QUERY_DECOMPOSITION = "query_decomposition"


@dataclass
class RetrievalResult:
    """Result of retrieval operation"""

    query: str
    retrieved_docs: List[str]
    scores: List[float]
    method: RetrievalMethod
    processing_time: float
    metadata: Dict[str, Any]
    rewritten_queries: Optional[List[str]] = None
    decomposed_queries: Optional[List[str]] = None
    hypothetical_docs: Optional[List[str]] = None


class AdvancedRetriever:
    """
    Advanced retriever implementing multiple strategies from EMNLP 2024 and COLING 2025 papers

    Based on paper configurations:
    - Zephyr-7b-alpha for query rewriting
    - GPT-3.5-turbo for query decomposition
    - LLM-Embedder for dense retrieval
    - HyDE with default temperature 0.7
    - Hybrid search with α=0.3 (optimal from paper)
    - T5-based Query Expansion (COLING 2025)
    - Contrastive ICL support
    - Focus Mode (sentence-level retrieval)
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/LLM-Embedder",
        rewriting_model: str = "HuggingFaceH4/zephyr-7b-alpha",
        decomposition_model: str = "gpt-3.5-turbo",
        query_expansion_model: str = "google/flan-t5-small",  # COLING 2025
        bm25_index_path: Optional[str] = None,
        dense_index_path: Optional[str] = None,
        documents: Optional[List[str]] = None,
        hybrid_alpha: float = 0.3,  # Optimal from paper Table 9
        enable_query_expansion: bool = True,
        enable_contrastive_icl: bool = True,
        enable_focus_mode: bool = True,
    ):
        """
        Initialize the advanced retriever

        Args:
            embedding_model: Model for dense embeddings
            rewriting_model: Model for query rewriting
            decomposition_model: Model for query decomposition
            bm25_index_path: Path to BM25 index
            dense_index_path: Path to dense index
            documents: Document collection
            hybrid_alpha: Weight for BM25 in hybrid search (0.3 optimal)
        """
        self.embedding_model_name = embedding_model
        self.rewriting_model_name = rewriting_model
        self.decomposition_model_name = decomposition_model
        self.query_expansion_model_name = query_expansion_model
        self.bm25_index_path = bm25_index_path
        self.dense_index_path = dense_index_path
        self.documents = documents or []
        self.hybrid_alpha = hybrid_alpha
        self.enable_query_expansion = enable_query_expansion
        self.enable_contrastive_icl = enable_contrastive_icl
        self.enable_focus_mode = enable_focus_mode

        # COLING 2025 additions
        self.query_expansion_pipeline = None
        self.contrastive_examples = None

        # Initialize components
        self.embedding_model = None
        self.rewriting_pipeline = None
        self.bm25_index = None
        self.dense_index = None
        self.doc_embeddings = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all retrieval components"""
        # Initialize embedding model
        if TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")

        # Initialize rewriting pipeline
        if TRANSFORMERS_AVAILABLE:
            try:
                self.rewriting_pipeline = pipeline(
                    "text-generation",
                    model=self.rewriting_model_name,
                    device_map="auto",
                    torch_dtype="auto",
                )
                logger.info(f"Loaded rewriting model: {self.rewriting_model_name}")
            except Exception as e:
                logger.warning(f"Could not load rewriting model: {e}")

        # Initialize indexes if paths provided
        self._load_indexes()

        # Create embeddings for documents if available
        if self.documents and self.embedding_model:
            self._create_document_embeddings()

    def _load_indexes(self):
        """Load BM25 and dense indexes"""
        # BM25 index loading would go here
        # Dense index loading would go here
        pass

    def _create_document_embeddings(self):
        """Create embeddings for the document collection"""
        if self.embedding_model and self.documents:
            logger.info(f"Creating embeddings for {len(self.documents)} documents")
            self.doc_embeddings = self.embedding_model.encode(
                self.documents, show_progress_bar=True
            )
            logger.info("Document embeddings created")

    def bm25_retrieval(
        self, query: str, top_k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        BM25 retrieval

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (documents, scores)
        """
        if not self.documents:
            return [], []

        # Simple BM25 implementation (would use Pyserini in production)
        scores = []
        for doc in self.documents:
            # Calculate BM25 score (simplified)
            score = self._calculate_bm25_score(query, doc)
            scores.append(score)

        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_docs = [self.documents[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]

        return top_docs, top_scores

    def _calculate_bm25_score(self, query: str, document: str) -> float:
        """Calculate BM25 score for query-document pair"""
        # Simplified BM25 calculation
        query_terms = query.lower().split()
        doc_terms = document.lower().split()

        score = 0.0
        doc_len = len(doc_terms)
        avg_doc_len = np.mean([len(doc.split()) for doc in self.documents])

        for term in query_terms:
            if term in doc_terms:
                tf = doc_terms.count(term)
                idf = np.log(
                    len(self.documents)
                    / sum(1 for doc in self.documents if term in doc.lower().split())
                )
                score += (
                    idf
                    * (tf * (1.5 + 1))
                    / (tf + 1.5 * (1 - 0.75 + 0.75 * doc_len / avg_doc_len))
                )

        return score

    def dense_retrieval(
        self, query: str, top_k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        Dense retrieval using embeddings

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (documents, scores)
        """
        if (
            not self.embedding_model
            or self.doc_embeddings is None
            or not self.documents
        ):
            return [], []

        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]

        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.doc_embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_docs = [self.documents[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]

        return top_docs, top_scores

    def hybrid_retrieval(
        self, query: str, top_k: int = 10, alpha: Optional[float] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Hybrid retrieval combining BM25 and dense

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            alpha: BM25 weight (if None, uses self.hybrid_alpha)

        Returns:
            Tuple of (documents, scores)
        """
        if alpha is None:
            alpha = self.hybrid_alpha

        # Get BM25 results
        bm25_docs, bm25_scores = self.bm25_retrieval(
            query, top_k * 2
        )  # Get more candidates

        # Get dense results
        dense_docs, dense_scores = self.dense_retrieval(query, top_k * 2)

        # Normalize scores
        if bm25_scores:
            bm25_scores = np.array(bm25_scores)
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (
                np.max(bm25_scores) - np.min(bm25_scores) + 1e-8
            )

        if dense_scores:
            dense_scores = np.array(dense_scores)
            dense_scores = (dense_scores - np.min(dense_scores)) / (
                np.max(dense_scores) - np.min(dense_scores) + 1e-8
            )

        # Combine scores: alpha * BM25 + (1-alpha) * Dense
        combined_scores = {}
        for doc, score in zip(bm25_docs, bm25_scores):
            combined_scores[doc] = alpha * score

        for doc, score in zip(dense_docs, dense_scores):
            if doc in combined_scores:
                combined_scores[doc] += (1 - alpha) * score
            else:
                combined_scores[doc] = (1 - alpha) * score

        # Sort by combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in sorted_docs[:top_k]]
        top_scores = [score for _, score in sorted_docs[:top_k]]

        return top_docs, top_scores

    def query_rewriting(self, query: str) -> List[str]:
        """
        Rewrite query using Zephyr-7b-alpha (from paper)

        Args:
            query: Original query

        Returns:
            List of rewritten queries
        """
        if not self.rewriting_pipeline:
            return [query]  # Return original if no model

        prompt = f"""Rewrite the following search query to be more effective for information retrieval.
Make it clearer and more specific while preserving the original intent.

Original query: {query}

Rewritten query:"""

        try:
            outputs = self.rewriting_pipeline(
                prompt,
                max_new_tokens=100,
                num_return_sequences=3,  # Generate multiple rewrites
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.rewriting_pipeline.tokenizer.eos_token_id,
            )

            rewritten_queries = []
            for output in outputs:
                rewritten = output["generated_text"].replace(prompt, "").strip()
                # Clean up the response
                rewritten = rewritten.split("\n")[0].strip()
                if rewritten and len(rewritten) > 10:  # Filter out too short rewrites
                    rewritten_queries.append(rewritten)

            return rewritten_queries if rewritten_queries else [query]

        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return [query]

    def query_decomposition(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries using GPT-3.5-turbo (from paper)

        Args:
            query: Complex query

        Returns:
            List of sub-queries
        """
        # For now, implement a rule-based decomposition
        # In production, this would use GPT-3.5-turbo API

        sub_queries = []

        # Simple decomposition rules
        if " and " in query.lower():
            parts = query.lower().split(" and ")
            sub_queries.extend([part.strip() for part in parts])
        elif " or " in query.lower():
            parts = query.lower().split(" or ")
            sub_queries.extend([part.strip() for part in parts])
        elif "?" in query:
            # Break down questions
            sub_queries.append(query)
            # Add related sub-questions
            if "what" in query.lower() and "how" in query.lower():
                sub_queries.append(query.replace("what", "how"))
        else:
            sub_queries.append(query)

        return sub_queries

    def hyde_retrieval(
        self, query: str, top_k: int = 10
    ) -> Tuple[List[str], List[float], List[str]]:
        """
        HyDE (Hypothetical Document Embedding) retrieval

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (documents, scores, hypothetical_docs)
        """
        if not self.embedding_model:
            return [], [], []

        # Generate hypothetical document (simplified - would use GPT-3.5-turbo-instruct)
        hypothetical_doc = self._generate_hypothetical_document(query)

        # Embed hypothetical document
        hypo_embedding = self.embedding_model.encode([hypothetical_doc])[0]

        # Find similar documents using hypothetical embedding
        similarities = cosine_similarity([hypo_embedding], self.doc_embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_docs = [self.documents[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]

        return top_docs, top_scores, [hypothetical_doc]

    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate hypothetical document for HyDE
        
        Intenta usar LLM real si está disponible, sino usa templates mejorados.
        """
        # Intentar usar LLM real para generar documento hipotético
        try:
            # Opción 1: OpenAI (si está disponible)
            try:
                import openai
                if hasattr(openai, 'OpenAI'):
                    client = openai.OpenAI()
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that generates hypothetical documents that would answer a query."},
                            {"role": "user", "content": f"Generate a hypothetical document that would answer this query: {query}"}
                        ],
                        max_tokens=200,
                        temperature=0.7
                    )
                    return response.choices[0].message.content.strip()
            except (ImportError, Exception) as e:
                logger.debug(f"OpenAI no disponible para HyDE: {e}")
            
            # Opción 2: Usar modelo local si está disponible
            try:
                from transformers import pipeline
                if not hasattr(self, '_hyde_generator'):
                    self._hyde_generator = pipeline(
                        "text-generation",
                        model="gpt2",  # Modelo base, puede mejorarse
                        max_length=150,
                        do_sample=True,
                        temperature=0.7
                    )
                
                prompt = f"Document that answers '{query}':"
                result = self._hyde_generator(prompt, max_length=150, num_return_sequences=1)
                generated_text = result[0]['generated_text']
                # Extraer solo la parte generada (después del prompt)
                if prompt in generated_text:
                    return generated_text.split(prompt, 1)[1].strip()
                return generated_text.strip()
            except (ImportError, Exception) as e:
                logger.debug(f"Transformers pipeline no disponible para HyDE: {e}")
            
        except Exception as e:
            logger.warning(f"Error usando LLM para HyDE, usando templates mejorados: {e}")
        
        # Fallback: Templates mejorados basados en tipo de query
        query_lower = query.lower()
        
        # Detectar tipo de query para template más apropiado
        if any(word in query_lower for word in ["qué es", "what is", "define", "definición"]):
            template = f"This is a comprehensive definition of {query}. It explains the core concepts, key characteristics, provides clear examples, and discusses practical applications. The document covers both theoretical foundations and real-world usage."
        elif any(word in query_lower for word in ["cómo", "how", "pasos", "steps"]):
            template = f"This document provides a step-by-step guide to {query}. It includes detailed instructions, best practices, common pitfalls to avoid, and practical examples. The guide is structured for both beginners and advanced users."
        elif any(word in query_lower for word in ["por qué", "why", "causa", "reason"]):
            template = f"This document explains the reasons and causes behind {query}. It analyzes underlying factors, provides evidence-based explanations, discusses different perspectives, and explores implications."
        else:
            # Template general mejorado
            template = f"This document comprehensively addresses {query}. It covers fundamental concepts, provides detailed explanations with examples, discusses current best practices, explores advanced topics, and includes practical applications. The content is structured to be both informative and actionable."
        
        return template

    def retrieve(
        self,
        query: str,
        method: RetrievalMethod = RetrievalMethod.HYBRID,
        top_k: int = 10,
        **kwargs,
    ) -> RetrievalResult:
        """
        Main retrieval method supporting all strategies

        Args:
            query: Search query
            method: Retrieval method to use
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters

        Returns:
            RetrievalResult with documents and metadata
        """
        import time

        start_time = time.time()

        docs = []
        scores = []
        rewritten_queries = None
        decomposed_queries = None
        hypothetical_docs = None

        if method == RetrievalMethod.BM25:
            docs, scores = self.bm25_retrieval(query, top_k)

        elif method == RetrievalMethod.DENSE:
            docs, scores = self.dense_retrieval(query, top_k)

        elif method == RetrievalMethod.HYBRID:
            alpha = kwargs.get("alpha", self.hybrid_alpha)
            docs, scores = self.hybrid_retrieval(query, top_k, alpha)

        elif method == RetrievalMethod.QUERY_REWRITING:
            rewritten = self.query_rewriting(query)
            rewritten_queries = rewritten

            # Retrieve using each rewritten query and combine results
            all_docs = []
            all_scores = []
            for rw_query in rewritten:
                rw_docs, rw_scores = self.hybrid_retrieval(
                    rw_query, top_k // len(rewritten) + 1
                )
                all_docs.extend(rw_docs)
                all_scores.extend(rw_scores)

            # Remove duplicates and sort by score
            doc_score_pairs = list(set(zip(all_docs, all_scores)))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            docs = [doc for doc, _ in doc_score_pairs[:top_k]]
            scores = [score for _, score in doc_score_pairs[:top_k]]

        elif method == RetrievalMethod.QUERY_DECOMPOSITION:
            decomposed = self.query_decomposition(query)
            decomposed_queries = decomposed

            # Retrieve using each sub-query and combine results
            all_docs = []
            all_scores = []
            for sub_query in decomposed:
                sub_docs, sub_scores = self.hybrid_retrieval(
                    sub_query, top_k // len(decomposed) + 1
                )
                all_docs.extend(sub_docs)
                all_scores.extend(sub_scores)

            # Remove duplicates and sort by score
            doc_score_pairs = list(set(zip(all_docs, all_scores)))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            docs = [doc for doc, _ in doc_score_pairs[:top_k]]
            scores = [score for _, score in doc_score_pairs[:top_k]]

        elif method == RetrievalMethod.HYDE:
            docs, scores, hypothetical_docs = self.hyde_retrieval(query, top_k)

        processing_time = time.time() - start_time

        metadata = {
            "method": method.value,
            "top_k_requested": top_k,
            "docs_returned": len(docs),
            "hybrid_alpha": (
                self.hybrid_alpha if method == RetrievalMethod.HYBRID else None
            ),
            **kwargs,
        }

        return RetrievalResult(
            query=query,
            retrieved_docs=docs,
            scores=scores,
            method=method,
            processing_time=processing_time,
            metadata=metadata,
            rewritten_queries=rewritten_queries,
            decomposed_queries=decomposed_queries,
            hypothetical_docs=hypothetical_docs,
        )

    def add_documents(self, documents: List[str]):
        """
        Add documents to the collection

        Args:
            documents: List of document texts
        """
        self.documents.extend(documents)
        if self.embedding_model:
            self._create_document_embeddings()

    # COLING 2025 Methods
    def query_expansion_t5(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        Query Expansion using T5 model (COLING 2025)

        Args:
            query: Original query
            num_expansions: Number of expanded queries to generate

        Returns:
            List of expanded queries
        """
        if not self.query_expansion_pipeline:
            # Initialize T5 pipeline if not already done
            if TRANSFORMERS_AVAILABLE:
                try:
                    from transformers import pipeline

                    self.query_expansion_pipeline = pipeline(
                        "text2text-generation",
                        model=self.query_expansion_model_name,
                        device="cpu",  # Use CPU for compatibility
                    )
                    logger.info(
                        f"Loaded T5 query expansion model: {self.query_expansion_model_name}"
                    )
                except Exception as e:
                    logger.warning(f"Could not load T5 query expansion model: {e}")
                    return [query]

        if not self.query_expansion_pipeline:
            return [query]

        # Create prompt for query expansion
        prompt = f"Expand this search query with relevant keywords: {query}"

        try:
            outputs = self.query_expansion_pipeline(
                prompt,
                max_length=100,
                num_return_sequences=num_expansions,
                do_sample=True,
                temperature=0.7,
            )

            expanded_queries = []
            for output in outputs:
                expanded = output["generated_text"].strip()
                # Clean up the response
                if expanded and len(expanded) > len(
                    query
                ):  # Only keep meaningful expansions
                    expanded_queries.append(expanded)

            return expanded_queries if expanded_queries else [query]

        except Exception as e:
            logger.warning(f"T5 query expansion failed: {e}")
            return [query]

    def retrieval_stride(
        self, query: str, stride: int = 5, max_steps: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieval Stride: Dynamic context update during generation (COLING 2025)

        Args:
            query: Original query
            stride: Update frequency (every N steps)
            max_steps: Maximum generation steps

        Returns:
            Tuple of (documents, scores) - best documents found
        """
        # For now, implement as enhanced retrieval with multiple rounds
        # In full implementation, this would integrate with generation process

        all_docs = []
        all_scores = []

        # Initial retrieval
        docs, scores = self.hybrid_retrieval(query, top_k=10)
        all_docs.extend(docs)
        all_scores.extend(scores)

        # Simulate multiple retrieval rounds (simplified)
        for step in range(1, min(max_steps // stride, 3)):  # Limit to 3 rounds for demo
            # Generate pseudo "generated text" based on current context
            context_preview = " ".join(docs[:3])[:200]  # First 200 chars of top 3 docs
            augmented_query = f"{query} {context_preview}"

            # Retrieve with augmented query
            new_docs, new_scores = self.hybrid_retrieval(augmented_query, top_k=5)
            all_docs.extend(new_docs)
            all_scores.extend(new_scores)

        # Remove duplicates and return top results
        doc_score_pairs = list(set(zip(all_docs, all_scores)))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        top_docs = [doc for doc, _ in doc_score_pairs[:10]]
        top_scores = [score for _, score in doc_score_pairs[:10]]

        return top_docs, top_scores

    def contrastive_icl_retrieval(
        self, query: str, top_k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        Contrastive In-Context Learning retrieval (COLING 2025)

        Uses correct and incorrect examples as knowledge base for better discrimination.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (documents, scores)
        """
        if not self.contrastive_examples:
            # Initialize with some example Q&A pairs (simplified)
            self.contrastive_examples = [
                {
                    "question": "What is the capital of France?",
                    "correct_answer": "Paris",
                    "incorrect_answer": "London",
                },
                {
                    "question": "What is 2 + 2?",
                    "correct_answer": "4",
                    "incorrect_answer": "5",
                },
                {
                    "question": "What is the largest planet?",
                    "correct_answer": "Jupiter",
                    "incorrect_answer": "Mars",
                },
            ]

        # For now, fall back to regular retrieval
        # In full implementation, this would use contrastive examples
        return self.hybrid_retrieval(query, top_k)

    def focus_mode_retrieval(
        self, query: str, doc_count: int = 2, sentence_count: int = 1
    ) -> Tuple[List[str], List[float]]:
        """
        Focus Mode: Sentence-level retrieval (COLING 2025)

        Splits retrieved documents into sentences and ranks them by relevance.

        Args:
            query: Search query
            doc_count: Number of documents to retrieve initially
            sentence_count: Number of sentences per document to keep

        Returns:
            Tuple of (sentences, scores)
        """
        try:
            import nltk

            nltk.download("punkt", quiet=True)
            from nltk.tokenize import sent_tokenize
        except ImportError:
            # Fallback without NLTK
            def sent_tokenize(text):
                return text.split(".")

        # First, retrieve documents
        docs, doc_scores = self.hybrid_retrieval(query, top_k=doc_count)

        focused_sentences = []
        focused_scores = []

        for doc, doc_score in zip(docs, doc_scores):
            # Split document into sentences
            sentences = sent_tokenize(doc)

            # Score each sentence
            sentence_scores = []
            for sentence in sentences:
                if len(sentence.strip()) > 10:  # Filter out very short sentences
                    # Simple relevance scoring (could be improved with embeddings)
                    relevance_score = self._calculate_sentence_relevance(
                        sentence, query
                    )
                    sentence_scores.append((sentence.strip(), relevance_score))

            # Keep top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = sentence_scores[:sentence_count]

            focused_sentences.extend([sent for sent, _ in top_sentences])
            focused_scores.extend(
                [score * doc_score for _, score in top_sentences]
            )  # Combine scores

        return focused_sentences, focused_scores

    def _calculate_sentence_relevance(self, sentence: str, query: str) -> float:
        """
        Calculate relevance score for a sentence given a query

        Args:
            sentence: Sentence to score
            query: Query for relevance

        Returns:
            Relevance score (0-1)
        """
        query_terms = set(query.lower().split())
        sentence_terms = set(sentence.lower().split())

        # Jaccard similarity
        intersection = len(query_terms.intersection(sentence_terms))
        union = len(query_terms.union(sentence_terms))

        if union == 0:
            return 0.0

        return intersection / union

    def get_coling2025_optimal_configs(self) -> Dict[str, Any]:
        """
        Get optimal configurations from COLING 2025 experiments

        Based on comprehensive evaluation of 74 experiment runs across:
        - TruthfulQA (commonsense knowledge)
        - MMLU (specialized knowledge)
        - Metrics: ROUGE-1/2/L F1, Embedding Cosine Similarity, MAUVE, FActScore

        Returns:
            Dictionary with optimal settings for each technique
        """
        return {
            "query_expansion": {
                "model": "google/flan-t5-small",
                "num_expansions": 3,
                "filter_sizes_tested": [9, 15, 21],  # ExpendS, ExpendM, ExpendL
                "improvement": "+2-3% ROUGE scores on TruthfulQA",
                "best_config": "ExpendL (21 articles filter)",
                "limitation": "Marginal gains on datasets with good baseline retrieval",
            },
            "retrieval_stride": {
                "optimal_stride": 5,  # Best performance from experiments
                "stride_range_tested": [1, 2, 5],  # Stride1, Stride2, Baseline(5)
                "improvement": "Better context coherence vs frequent updates",
                "performance_decline": "Stride1 hurts ROUGE/ECS/MAUVE scores",
                "finding": "Larger strides preserve context stability",
            },
            "contrastive_icl": {
                "examples_needed": "1Doc+ (correct + incorrect)",
                "configurations_tested": ["ICL1Doc", "ICL2Doc", "ICL1Doc+", "ICL2Doc+"],
                "improvement": "+3.93% ROUGE-L on TruthfulQA, +2.99% MAUVE on MMLU",
                "best_config": "ICL1Doc+",
                "factuality_boost": "+7.00 FActScore on TruthfulQA",
                "key_insight": "Contrastive examples help differentiate correct vs incorrect information",
            },
            "focus_mode": {
                "sentence_extraction": "Split documents into sentences, rank by relevance",
                "configurations_tested": [
                    "2Doc1S",
                    "20Doc20S",
                    "40Doc40S",
                    "80Doc80S",
                    "120Doc120S",
                ],
                "optimal_config_truthfulqa": "80Doc80S (+1.65% ROUGE-L)",
                "optimal_config_mmlu": "120Doc120S (+0.81% Embedding Cosine Similarity)",
                "improvement": "+1.65% ROUGE-L overall",
                "application": "Effective for text summarization and simplification",
            },
            "knowledge_base": {
                "source": "Wikipedia Vital Articles",
                "levels_used": ["Level 3 (999 articles)", "Level 4 (10,011 articles)"],
                "multilingual_support": "French and German articles tested",
                "finding": "Quality > Quantity - larger KB provides marginal gains",
                "multilingual_limitation": "Performance decline due to synthesis challenges",
            },
            "experimental_setup": {
                "total_runs": 74,
                "datasets": ["TruthfulQA (817 samples)", "MMLU (1,824 samples)"],
                "llm_models": [
                    "Mistral-7B-Instruct-v0.2",
                    "Mixtral-8x7B-Instruct-v0.1",
                ],
                "chunk_size": 64,
                "embedding_model": "all-MiniLM-L6-v2",
                "indexing": "FAISS for efficient similarity search",
            },
            "overall_best": {
                "technique": "Contrastive ICL (ICL1Doc+)",
                "improvement_over_baseline": "+3.93% ROUGE-L, +2.99% MAUVE",
                "factuality_improvement": "+7.00 FActScore on TruthfulQA",
                "second_best": "Focus Mode (80Doc80S/120Doc120S)",
                "key_findings": [
                    "Prompt design remains crucial even in RAG",
                    "Knowledge base quality > size",
                    "Contrastive learning significantly boosts factuality",
                    "Focus Mode enhances precision through sentence-level selection",
                ],
            },
        }

    def get_optimal_hybrid_alpha(self) -> Dict[str, float]:
        """
        Get optimal alpha values from paper experiments

        From paper Table 9: Hybrid Search with Different α
        - α=0.3: Best performance on TREC DL19/20
        """
        return {
            "optimal_alpha": 0.3,  # From paper experiments
            "alpha_range_tested": [0.1, 0.3, 0.5, 0.7, 0.9],
            "best_performing_alpha": 0.3,
            "performance_gain_over_bm25": "+47% nDCG@10, +51% R@1k",
        }
