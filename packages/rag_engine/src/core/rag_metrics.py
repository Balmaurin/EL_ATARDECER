#!/usr/bin/env python3
"""
Enhanced RAG Metrics - BM25 + Hybrid Search System
==================================================

⚠️ DEPRECATED: Este módulo está siendo deprecado.
Migrar a `indexing_metrics.py` para métricas Prometheus integradas.

Sistema avanzado de métricas para RAG (Retrieval Augmented Generation):
- Métricas de precisión, recall y F1-score para evaluaciones RAG
- Sistema híbrido BM25 + Vector Search con re-ranking
- Evaluación de calidad de respuesta y relevancia
- Métricas de diversidad y cobertura de contexto
- Análisis de latencia y rendimiento

NOTA: Las clases HybridSearchEngine y RAGEvaluator se mantienen temporalmente
para compatibilidad, pero se recomienda migrar a los nuevos módulos.
"""

import warnings

warnings.warn(
    "rag_metrics.py está deprecated. Migrar a indexing_metrics.py",
    DeprecationWarning,
    stacklevel=2
)

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

# Métricas estándar para evaluación RAG
RAG_METRICS = {
    "precision": "Proporción de documentos relevantes recuperados",
    "recall": "Proporción de documentos relevantes que se recuperaron",
    "f1_score": "Media armónica de precisión y recall",
    "map_score": "Mean Average Precision",
    "ndcg_score": "Normalized Discounted Cumulative Gain",
    "context_relevance": "Relevancia del contexto proporcionado",
    "answer_quality": "Calidad de la respuesta generada",
    "answer_correctness": "Exactitud factual de la respuesta",
    "answer_relevance": "Relevancia de la respuesta a la pregunta",
    "diversity_score": "Diversidad de los documentos recuperados",
    "latency_ms": "Tiempo de respuesta en milisegundos",
    "throughput": "Consultas por segundo",
}


class RAGMetrics:
    """Sistema de métricas avanzadas para evaluación RAG"""

    def __init__(self):
        self.metrics_history = []
        self.baseline_metrics = {}
        self.performance_thresholds = {
            "precision": 0.8,
            "recall": 0.75,
            "f1_score": 0.77,
            "context_relevance": 0.85,
            "answer_quality": 0.8,
            "latency_ms": 2000,  # 2 segundos máximo
            "throughput": 5,  # 5 consultas/segundo mínimo
        }

    def calculate_precision(
        self, retrieved_docs: List[str], relevant_docs: List[str]
    ) -> float:
        """Calcular precisión: TP / (TP + FP)"""
        if not retrieved_docs:
            return 0.0

        true_positives = len(set(retrieved_docs) & set(relevant_docs))
        return true_positives / len(retrieved_docs)

    def calculate_recall(
        self, retrieved_docs: List[str], relevant_docs: List[str]
    ) -> float:
        """Calcular recall: TP / (TP + FN)"""
        if not relevant_docs:
            return 0.0

        true_positives = len(set(retrieved_docs) & set(relevant_docs))
        return true_positives / len(relevant_docs)

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calcular F1-score: 2 * (precision * recall) / (precision + recall)"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_map_score(
        self, retrieved_docs: List[str], relevant_docs: List[str]
    ) -> float:
        """Calcular Mean Average Precision (MAP)"""
        if not retrieved_docs or not relevant_docs:
            return 0.0

        relevant_set = set(relevant_docs)
        precision_sum = 0.0
        relevant_found = 0

        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_set:
                relevant_found += 1
                precision_sum += relevant_found / i

        return precision_sum / len(relevant_docs)

    def calculate_ndcg_score(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        relevance_scores: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calcular Normalized Discounted Cumulative Gain (NDCG)"""
        if not retrieved_docs:
            return 0.0

        # Asignar scores de relevancia (por defecto 1.0 para relevantes, 0.0 para no relevantes)
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant_docs}

        # Calcular DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs, 1):
            relevance = relevance_scores.get(doc, 0.0)
            dcg += relevance / np.log2(i + 1)

        # Calcular IDCG (ideal DCG)
        ideal_relevance_scores = sorted(relevance_scores.values(), reverse=True)
        idcg = sum(
            score / np.log2(i + 1) for i, score in enumerate(ideal_relevance_scores, 1)
        )

        return dcg / idcg if idcg > 0 else 0.0

    def calculate_context_relevance(self, query: str, context: str) -> float:
        """Calcular relevancia del contexto proporcionado a la query"""
        if not context:
            return 0.0

        # Tokenizar
        query_tokens = set(query.lower().split())
        context_tokens = set(context.lower().split())

        # Calcular similitud Jaccard
        intersection = len(query_tokens & context_tokens)
        union = len(query_tokens | context_tokens)

        if union == 0:
            return 0.0

        # Bonus por términos relevantes encontrados
        keyword_bonus = min(intersection / len(query_tokens), 1.0) * 0.3

        return min((intersection / union) + keyword_bonus, 1.0)

    def calculate_answer_quality(
        self, answer: str, context: str, query: str
    ) -> Dict[str, float]:
        """Calcular calidad de la respuesta considerando múltiples factores"""
        quality_scores = {
            "completeness": self._calculate_answer_completeness(answer, context),
            "conciseness": self._calculate_answer_conciseness(answer),
            "relevance": self._calculate_answer_relevance(answer, query),
            "factual_accuracy": self._calculate_factual_accuracy(answer, context),
        }

        # Score compuesto
        quality_scores["overall_quality"] = np.mean(list(quality_scores.values()))
        return quality_scores

    def _calculate_answer_completeness(self, answer: str, context: str) -> float:
        """Calcular completitud de la respuesta"""
        if not answer or not context:
            return 0.0

        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())

        coverage = (
            len(answer_words & context_words) / len(answer_words) if answer_words else 0
        )
        return min(coverage, 1.0)

    def _calculate_answer_conciseness(self, answer: str) -> float:
        """Calcular concisión de la respuesta (penalizar respuestas demasiado largas)"""
        word_count = len(answer.split())
        if word_count <= 50:
            return 1.0
        elif word_count <= 100:
            return 0.8
        elif word_count <= 200:
            return 0.6
        else:
            return max(0.2, 300 / word_count)

    def _calculate_answer_relevance(self, answer: str, query: str) -> float:
        """Calcular relevancia de la respuesta a la pregunta"""
        query_tokens = set(query.lower().split())
        answer_tokens = set(answer.lower().split())

        intersection = len(query_tokens & answer_tokens)
        return min(intersection / len(query_tokens), 1.0) if query_tokens else 0.0

    def _calculate_factual_accuracy(self, answer: str, context: str) -> float:
        """Calcular precisión factual (simplificado)"""
        # Versión básica: verificar que la información de la respuesta esté en el contexto
        context_lower = context.lower()
        answer_sentences = answer.split(".")

        facts_supported = 0
        facts_unverified = 0

        for sentence in answer_sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Saltar oraciones cortas
                continue

            # Verificar si la información está soportada por el contexto
            if sentence.lower() in context_lower:
                facts_supported += 1
            else:
                facts_unverified += 1

        total_facts = facts_supported + facts_unverified
        return facts_supported / total_facts if total_facts > 0 else 0.5

    def calculate_diversity_score(self, documents: List[str]) -> float:
        """Calcular diversidad de documentos (similitud semántica entre ellos)"""
        if len(documents) <= 1:
            return 1.0

        # Calcular similitud promedio entre pares
        similarities = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                sim = self._calculate_text_similarity(documents[i], documents[j])
                similarities.append(sim)

        # Mayor diversidad = menor similitud
        avg_similarity = np.mean(similarities)
        return max(0.0, 1.0 - avg_similarity)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud simple entre textos usando Jaccard"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def evaluate_rag_system(
        self,
        query: str,
        retrieved_docs: List[str],
        ground_truth: List[str],
        generated_answer: str,
        context: str,
        latency_ms: float,
    ) -> Dict[str, Any]:
        """Evaluación completa de un sistema RAG"""

        # Métricas de recuperación
        precision = self.calculate_precision(retrieved_docs, ground_truth)
        recall = self.calculate_recall(retrieved_docs, ground_truth)
        f1_score = self.calculate_f1_score(precision, recall)
        map_score = self.calculate_map_score(retrieved_docs, ground_truth)
        ndcg_score = self.calculate_ndcg_score(retrieved_docs, ground_truth)

        # Métricas de comprensión de contexto
        context_relevance = self.calculate_context_relevance(query, context)

        # Métricas de calidad de respuesta
        answer_metrics = self.calculate_answer_quality(generated_answer, context, query)

        # Métricas adicionales
        diversity = self.calculate_diversity_score(retrieved_docs)

        # Métricas de rendimiento
        throughput = 1000 / latency_ms if latency_ms > 0 else 0

        # Calcular thresholds antes de crear el diccionario
        metrics_dict = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "context_relevance": context_relevance,
            "latency_ms": latency_ms,
            "throughput": throughput,
        }
        meets_thresholds = self._check_performance_thresholds(metrics_dict)
        
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            # Retrieval metrics
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "map_score": map_score,
            "ndcg_score": ndcg_score,
            # Context understanding
            "context_relevance": context_relevance,
            # Answer quality
            "answer_quality": answer_metrics,
            # System metrics
            "diversity_score": diversity,
            "latency_ms": latency_ms,
            "throughput": throughput,
            # Metadata
            "retrieved_docs_count": len(retrieved_docs),
            "ground_truth_count": len(ground_truth),
            "context_length": len(context.split()),
            "answer_length": len(generated_answer.split()),
            # Performance flags
            "meets_thresholds": meets_thresholds,
        }

        # Agregar a historial
        self.metrics_history.append(evaluation)

        return evaluation

    def _check_performance_thresholds(self, evaluation: Dict[str, Any]) -> bool:
        """Verificar si la evaluación cumple con los umbrales de rendimiento"""
        checks = {
            "precision": evaluation.get("precision", 0)
            >= self.performance_thresholds["precision"],
            "recall": evaluation.get("recall", 0)
            >= self.performance_thresholds["recall"],
            "f1_score": evaluation.get("f1_score", 0)
            >= self.performance_thresholds["f1_score"],
            "context_relevance": evaluation.get("context_relevance", 0)
            >= self.performance_thresholds["context_relevance"],
            "answer_quality": evaluation.get("answer_quality", {}).get(
                "overall_quality", 0
            )
            >= self.performance_thresholds["answer_quality"],
            "latency": evaluation.get("latency_ms", float("inf"))
            <= self.performance_thresholds["latency_ms"],
            "throughput": evaluation.get("throughput", 0)
            >= self.performance_thresholds["throughput"],
        }

        return all(checks.values())

    def get_performance_summary(self, last_n: int = 50) -> Dict[str, Any]:
        """Obtener resumen de rendimiento del sistema RAG"""
        recent_metrics = self.metrics_history[-last_n:] if self.metrics_history else []

        if not recent_metrics:
            return {"status": "no_data"}

        # Calcular métricas promedio
        metric_keys = [
            "precision",
            "recall",
            "f1_score",
            "map_score",
            "ndcg_score",
            "context_relevance",
            "diversity_score",
            "latency_ms",
            "throughput",
        ]

        summary = {
            "total_evaluations": len(recent_metrics),
            "average_metrics": {},
            "performance_trends": {},
            "threshold_compliance": {},
            "recommendations": [],
        }

        # Calcular promedios
        for metric in metric_keys:
            values = [
                m.get(metric, 0)
                for m in recent_metrics
                if isinstance(m.get(metric), (int, float))
            ]
            if values:
                summary["average_metrics"][metric] = np.mean(values)

        # Calcular tendencias (última vs promedio)
        if len(recent_metrics) >= 2:
            latest = recent_metrics[-1]
            for metric in metric_keys:
                if (
                    isinstance(latest.get(metric), (int, float))
                    and metric in summary["average_metrics"]
                ):
                    latest_value = latest[metric]
                    avg_value = summary["average_metrics"][metric]
                    summary["performance_trends"][metric] = {
                        "latest": latest_value,
                        "average": avg_value,
                        "trend": (
                            "improving" if latest_value > avg_value else "declining"
                        ),
                    }

        # Verificar cumplimiento de thresholds
        for threshold_metric in [
            "precision",
            "recall",
            "f1_score",
            "context_relevance",
            "latency_ms",
            "throughput",
        ]:
            if threshold_metric in summary["average_metrics"]:
                threshold_value = self.performance_thresholds.get(threshold_metric, 0)
                actual_value = summary["average_metrics"][threshold_metric]

                if threshold_metric in ["latency_ms"]:
                    compliant = actual_value <= threshold_value
                else:
                    compliant = actual_value >= threshold_value

                summary["threshold_compliance"][threshold_metric] = {
                    "value": actual_value,
                    "threshold": threshold_value,
                    "compliant": compliant,
                }

        # Generar recomendaciones
        summary["recommendations"] = self._generate_rag_recommendations(summary)

        return summary

    def _generate_rag_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generar recomendaciones basadas en el análisis de métricas"""
        recommendations = []

        avg_metrics = summary.get("average_metrics", {})

        # Recomendaciones de precisión
        precision = avg_metrics.get("precision", 0)
        recall = avg_metrics.get("recall", 0)

        if precision < 0.7:
            recommendations.append(
                "Mejorar precisión de recuperación - considerar re-ranking o query expansion"
            )
        if recall < 0.6:
            recommendations.append(
                "Mejorar recall - aumentar el número de documentos recuperados"
            )

        # Recomendaciones de latencia
        latency = avg_metrics.get("latency_ms", 0)
        if latency > 3000:
            recommendations.append(
                "Reducir latencia - optimizar indexación vectorial o implementar caching"
            )

        # Recomendaciones de calidad de respuesta
        context_relevance = avg_metrics.get("context_relevance", 0)
        if context_relevance < 0.8:
            recommendations.append(
                "Mejorar selección de contexto - implementar técnicas de re-ranking avanzadas"
            )

        # Recomendaciones de diversidad
        diversity = avg_metrics.get("diversity_score", 0)
        if diversity < 0.3:
            recommendations.append(
                "Aumentar diversidad - evitar recuperación de documentos muy similares"
            )

        # Recomendaciones por no cumplimiento de thresholds
        threshold_compliance = summary.get("threshold_compliance", {})
        non_compliant = [
            k for k, v in threshold_compliance.items() if not v.get("compliant", True)
        ]

        if non_compliant:
            recommendations.append(
                f"Mejorar métricas por debajo del threshold: {', '.join(non_compliant)}"
            )

        # Si no hay problemas específicos, recomendar mejoras generales
        if not recommendations:
            recommendations.append(
                "Mandar sistema RAG funcionando bien - considerar optimizaciones menores"
            )

        return recommendations


class HybridSearchEngine:
    """Motor de búsqueda híbrido BM25 + Vector Search con re-ranking"""

    def __init__(self, vector_index=None):
        self.vector_index = vector_index
        self.bm25_index = None
        self.bm25_texts = []

        # Parámetros de búsqueda híbrida
        self.vector_weight = 0.7
        self.bm25_weight = 0.3
        self.rerank_limit = 100
        self.final_limit = 10

    async def hybrid_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Ejecutar búsqueda híbrida con re-ranking inteligente"""
        start_time = time.time()

        # Fase 1: Recuperación inicial con ambas estrategias
        vector_results = await self._vector_search(query, self.rerank_limit)
        bm25_results = self._bm25_search(query, self.rerank_limit)

        # Fase 2: Combinación inicial de resultados
        combined_results = self._merge_search_results(vector_results, bm25_results)

        # Fase 3: Re-ranking inteligente
        reranked_results = self._rerank_results(
            query, combined_results, self.final_limit
        )

        # Agregar metadata de rendimiento
        search_time = (time.time() - start_time) * 1000  # ms

        for result in reranked_results:
            result["search_time_ms"] = search_time
            result["search_strategy"] = "hybrid_bm25_vector"

        return reranked_results[:limit]

    async def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Búsqueda vectorial usando el índice vectorial"""
        if not self.vector_index:
            return []

        try:
            results = await self.vector_index.hybrid_search(query, limit)
            # Formatear para compatibilidad
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "text": result.get("text", ""),
                        "metadata": result.get("metadata", {}),
                        "vector_score": result.get("vector_score", 0),
                        "combined_score": result.get("combined_score", 0),
                        "doc_id": result.get("doc_id", ""),
                        "source": "vector",
                    }
                )
            return formatted_results
        except Exception as e:
            print(f"Error en búsqueda vectorial: {e}")
            return []

    def _bm25_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Búsqueda BM25"""
        if not self.bm25_index or not self.bm25_texts:
            return []

        try:
            query_tokens = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(query_tokens)

            # Obtener top resultados
            top_indices = np.argsort(bm25_scores)[::-1][:limit]

            results = []
            for idx in top_indices:
                score = bm25_scores[idx]
                if score > 0:
                    results.append(
                        {
                            "text": self.bm25_texts[idx],
                            "bm25_score": float(score),
                            "doc_id": f"bm25_{idx}",
                            "source": "bm25",
                        }
                    )

            return results
        except Exception as e:
            print(f"Error en búsqueda BM25: {e}")
            return []

    def _merge_search_results(
        self, vector_results: List[Dict], bm25_results: List[Dict]
    ) -> List[Dict]:
        """Combinar resultados de ambas estrategias de búsqueda"""
        merged_scores = {}

        # Procesar resultados vectoriales
        for result in vector_results:
            doc_id = result.get("doc_id", "")
            merged_scores[doc_id] = {
                "vector_score": result.get("vector_score", 0),
                "bm25_score": 0,
                "text": result.get("text", ""),
                "metadata": result.get("metadata", {}),
                "sources": ["vector"],
            }

        # Procesar resultados BM25
        for result in bm25_results:
            doc_id = result.get("doc_id", "")
            bm25_score = result.get("bm25_score", 0)

            if doc_id in merged_scores:
                merged_scores[doc_id]["bm25_score"] = bm25_score
                merged_scores[doc_id]["sources"].append("bm25")
            else:
                merged_scores[doc_id] = {
                    "vector_score": 0,
                    "bm25_score": bm25_score,
                    "text": result.get("text", ""),
                    "metadata": {},
                    "sources": ["bm25"],
                }

        # Calcular scores combinados
        for doc_id, scores in merged_scores.items():
            vector_score = scores["vector_score"] * self.vector_weight
            bm25_score = scores["bm25_score"] * self.bm25_weight
            scores["combined_score"] = vector_score + bm25_score
            scores["doc_id"] = doc_id

        return list(merged_scores.values())

    def _rerank_results(
        self, query: str, results: List[Dict], limit: int
    ) -> List[Dict]:
        """Re-ranking inteligente basado en múltiples factores"""
        if not results:
            return []

        scored_results = []

        for result in results:
            text = result.get("text", "").lower()
            query_lower = query.lower()

            # Factor 1: Coincidencia exacta de términos clave
            exact_match_bonus = 0
            query_words = set(query_lower.split())
            text_words = set(text.split())
            exact_matches = len(query_words & text_words)
            exact_match_bonus = min(exact_matches * 0.1, 0.5)

            # Factor 2: Distancia semántica (aproximada)
            semantic_score = result.get("combined_score", 0)

            # Factor 3: Diversidad (penalizar resultados muy similares)
            diversity_penalty = 0

            # Factor 4: Frescura del contenido (bonus por términos recientes)
            recency_bonus = 0
            current_year = datetime.now().year
            if str(current_year) in text or str(current_year - 1) in text:
                recency_bonus = 0.05

            # Score final
            final_score = (
                semantic_score + exact_match_bonus + recency_bonus - diversity_penalty
            )
            result["final_score"] = final_score
            result["rerank_factors"] = {
                "semantic_score": semantic_score,
                "exact_match_bonus": exact_match_bonus,
                "recency_bonus": recency_bonus,
                "diversity_penalty": diversity_penalty,
            }

            scored_results.append(result)

        # Ordenar por score final y devolver top resultados
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_results[:limit]

    def update_bm25_index(self, documents: List[str]):
        """Actualizar índice BM25 con nuevos documentos"""
        if not BM25_AVAILABLE:
            print("BM25 no disponible - búsqueda híbrida limitada")
            return

        try:
            self.bm25_texts.extend(documents)
            tokenized_docs = [doc.lower().split() for doc in self.bm25_texts]
            self.bm25_index = BM25Okapi(tokenized_docs)
            print(f"✅ BM25 index actualizado con {len(documents)} documentos nuevos")
        except Exception as e:
            print(f"Error actualizando BM25 index: {e}")


class RAGEvaluator:
    """Evaluador automatizado de sistemas RAG"""

    def __init__(self):
        self.metrics = RAGMetrics()
        self.test_cases = []
        self.baseline_performance = {}

    def add_test_case(
        self,
        query: str,
        ground_truth_docs: List[str],
        expected_answer_pattern: str = "",
    ):
        """Agregar caso de prueba para evaluación"""
        test_case = {
            "query": query,
            "ground_truth_docs": ground_truth_docs,
            "expected_answer_pattern": expected_answer_pattern,
            "timestamp": datetime.now().isoformat(),
        }

        self.test_cases.append(test_case)

    def run_evaluation_suite(self, rag_system, max_queries: int = 50) -> Dict[str, Any]:
        """Ejecutar suite completa de evaluación"""
        print(
            f"🧪 Ejecutando evaluación RAG con {min(len(self.test_cases), max_queries)} casos de prueba"
        )

        evaluation_results = []
        start_time = time.time()

        test_cases_to_run = self.test_cases[:max_queries]

        for i, test_case in enumerate(test_cases_to_run):
            print(
                f"  [{i+1}/{len(test_cases_to_run)}] Evaluando: {test_case['query'][:50]}..."
            )

            try:
                # Ejecutar búsqueda
                search_start = time.time()
                search_results = rag_system.hybrid_search(
                    test_case["query"], limit=5  # Recuperar top 5 documentos
                )
                search_time = (time.time() - search_start) * 1000

                retrieved_docs = [r.get("text", "") for r in search_results]

                # Generar respuesta simulada basada en documentos recuperados
                context = " ".join(
                    retrieved_docs[:3]
                )  # Usar top 3 documentos como contexto
                generated_answer = self._generate_real_answer(
                    test_case["query"], context
                )

                # Evaluar con métricas RAG
                evaluation = self.metrics.evaluate_rag_system(
                    query=test_case["query"],
                    retrieved_docs=retrieved_docs,
                    ground_truth=test_case["ground_truth_docs"],
                    generated_answer=generated_answer,
                    context=context,
                    latency_ms=search_time,
                )

                evaluation_results.append(evaluation)

            except Exception as e:
                print(f"    ❌ Error evaluando caso: {e}")
                evaluation_results.append(
                    {
                        "query": test_case["query"],
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        evaluation_time = time.time() - start_time

        # Generar reporte de evaluación
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_test_cases": len(test_cases_to_run),
            "successful_evaluations": len(
                [r for r in evaluation_results if "error" not in r]
            ),
            "average_metrics": {},
            "performance_summary": self.metrics.get_performance_summary(),
            "recommendations": [],
            "evaluation_time_seconds": evaluation_time,
        }

        # Calcular métricas promedio
        if evaluation_results:
            successful_results = [r for r in evaluation_results if "error" not in r]
            if successful_results:
                metric_keys = [
                    "precision",
                    "recall",
                    "f1_score",
                    "context_relevance",
                    "latency_ms",
                ]
                for metric in metric_keys:
                    values = [r.get(metric, 0) for r in successful_results]
                    report["average_metrics"][metric] = np.mean(values) if values else 0

        # Agregar recomendaciones basadas en resultados
        summary = report["performance_summary"]
        if "recommendations" in summary:
            report["recommendations"] = summary["recommendations"]

        return report

    def _generate_real_answer(self, query: str, context: str) -> str:
        """Generar respuesta real usando LLM con el contexto recuperado"""
        if not context:
            return "No se pudo encontrar información relevante para responder esta pregunta."

        try:
            # Intentar usar el LLM real
            import sys
            from pathlib import Path
            
            # Agregar path para importar RealLLMInference
            root = Path(__file__).resolve().parents[5]
            sys.path.insert(0, str(root / "packages" / "sheily_core" / "src"))
            
            if get_real_llm_inference is None:
                raise ImportError("RealLLMInference no disponible")
            
            llm = get_real_llm_inference()
            
            # Crear prompt con contexto
            prompt = f"""Basándote en el siguiente contexto, responde la pregunta de manera clara y precisa.

Contexto:
{context[:2000]}  # Limitar contexto a 2000 caracteres

Pregunta: {query}

Respuesta:"""
            
            results = llm.generate(
                prompt,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9
            )
            
            if results and len(results) > 0:
                answer = results[0].strip()
                # Limpiar respuesta si contiene el prompt
                if "Respuesta:" in answer:
                    answer = answer.split("Respuesta:")[-1].strip()
                return answer if answer else "No pude generar una respuesta clara."
            
            return "No pude generar una respuesta clara."
            
        except Exception as e:
            logger.warning(f"Error usando LLM real, usando método básico: {e}")
            # Fallback básico solo si el LLM falla completamente
            sentences = context.split(".")
            relevant_sentences = []
            query_keywords = set(query.lower().split())
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in query_keywords):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                answer = ". ".join(relevant_sentences[:3])
                return answer + "." if not answer.endswith(".") else answer
            else:
                return "Basado en la información disponible, esta pregunta requiere más contexto específico."


# ================================
# DEMO Y TESTING FUNCTIONS
# ================================


async def demo_rag_metrics():
    """Demo del sistema de métricas RAG"""
    print("🎯 DEMO: Enhanced RAG Metrics System")
    print("=" * 50)

    metrics = RAGMetrics()

    # Caso de prueba simulado
    query = "¿Cómo funciona la inteligencia artificial?"
    retrieved_docs = [
        "La inteligencia artificial es una rama de la informática que busca crear máquinas capaces de simular la inteligencia humana.",
        "El aprendizaje automático es una técnica fundamental de IA que permite a los sistemas aprender de datos.",
        "La IA moderna utiliza redes neuronales profundas para procesar información compleja.",
        "Los algoritmos de IA requieren grandes cantidades de datos para entrenar modelos efectivos.",
    ]

    ground_truth_docs = [
        "La inteligencia artificial es una rama de la informática que busca crear máquinas capaces de simular la inteligencia humana.",
        "El aprendizaje automático es una técnica fundamental de IA que permite a los sistemas aprender de datos.",
    ]

    context = " ".join(retrieved_docs[:2])
    generated_answer = "La inteligencia artificial es una rama de la informática que busca crear máquinas capaces de simular la inteligencia humana, utilizando técnicas como el aprendizaje automático."

    # Evaluar
    evaluation = metrics.evaluate_rag_system(
        query=query,
        retrieved_docs=retrieved_docs,
        ground_truth=ground_truth_docs,
        generated_answer=generated_answer,
        context=context,
        latency_ms=1500,
    )

    print("📊 Resultados de Evaluación RAG:")
    print(f"  • Precision: {evaluation['precision']:.3f}")
    print(f"  • Recall: {evaluation['recall']:.3f}")
    print(f"  • F1-Score: {evaluation['f1_score']:.3f}")
    print(f"  • Context Relevance: {evaluation['context_relevance']:.3f}")
    print(f"  • Latency: {evaluation['latency_ms']:.0f}ms")
    print(f"  • Meets Thresholds: {evaluation['meets_thresholds']}")

    return evaluation


if __name__ == "__main__":
    # Ejecutar demo
    import asyncio

    asyncio.run(demo_rag_metrics())
