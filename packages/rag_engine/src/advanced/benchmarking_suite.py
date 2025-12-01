#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Benchmarking Suite for RAG Systems
Based on ANN-Benchmarks and RAG evaluation frameworks

Implements comprehensive benchmarking:
- Performance benchmarks (latency, throughput)
- Accuracy benchmarks (recall, precision)
- Scalability benchmarks (dataset size, concurrency)
- Comparative analysis across configurations
"""

import concurrent.futures
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .advanced_evaluation import ComprehensiveRAGEvaluator
from .advanced_indexing import AdvancedVectorIndex, MultiIndexManager


@dataclass
class BenchmarkResult:
    """Result of a benchmark run"""

    benchmark_name: str
    configuration: Dict[str, Any]
    metrics: Dict[str, float]
    raw_data: Dict[str, Any]
    timestamp: str
    duration: float


@dataclass
class PerformanceMetrics:
    """Performance benchmark metrics"""

    avg_latency_ms: float
    throughput_qps: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float


@dataclass
class AccuracyMetrics:
    """Accuracy benchmark metrics"""

    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    precision_at_1: float
    precision_at_10: float
    precision_at_100: float
    ndcg_at_10: float
    ndcg_at_100: float
    map_score: float


@dataclass
class ScalabilityMetrics:
    """Scalability benchmark metrics"""

    dataset_sizes: List[int]
    latencies_by_size: List[float]
    throughputs_by_size: List[float]
    memory_by_size: List[float]
    index_build_times: List[float]


class PerformanceBenchmarker:
    """
    Performance benchmarking for RAG systems

    Measures latency, throughput, and resource usage
    """

    def __init__(self, index_manager: MultiIndexManager):
        self.index_manager = index_manager
        self.evaluator = ComprehensiveRAGEvaluator()

    def benchmark_latency(
        self,
        queries: List[np.ndarray],
        index_name: str,
        k: int = 10,
        num_runs: int = 100,
    ) -> PerformanceMetrics:
        """
        Benchmark query latency

        Args:
            queries: List of query vectors
            index_name: Name of index to benchmark
            k: Number of neighbors to retrieve
            num_runs: Number of benchmark runs

        Returns:
            Performance metrics
        """
        latencies = []

        # Warm-up
        for _ in range(10):
            self.index_manager.search_index(index_name, queries[0], k)

        # Benchmark runs
        for _ in range(num_runs):
            start_time = time.time()
            for query in queries[:10]:  # Use subset for speed
                self.index_manager.search_index(index_name, query, k)
            end_time = time.time()

            latency = (end_time - start_time) / len(queries[:10]) * 1000  # ms
            latencies.append(latency)

        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        p50 = np.percentile(latencies_sorted, 50)
        p95 = np.percentile(latencies_sorted, 95)
        p99 = np.percentile(latencies_sorted, 99)

        # Throughput calculation
        avg_latency = sum(latencies) / len(latencies)
        throughput = 1000 / avg_latency  # queries per second

        return PerformanceMetrics(
            avg_latency_ms=avg_latency,
            throughput_qps=throughput,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            memory_usage_mb=self._get_memory_usage(),
            cpu_usage_percent=self._get_cpu_usage(),
        )

    def benchmark_concurrent_throughput(
        self,
        queries: List[np.ndarray],
        index_name: str,
        k: int = 10,
        num_threads: int = 4,
        duration_seconds: int = 10,
    ) -> float:
        """
        Benchmark concurrent throughput

        Args:
            queries: Query vectors
            index_name: Index name
            k: Neighbors to retrieve
            num_threads: Number of concurrent threads
            duration_seconds: Benchmark duration

        Returns:
            Queries per second
        """
        query_count = 0
        start_time = time.time()

        def worker():
            nonlocal query_count
            local_count = 0
            while time.time() - start_time < duration_seconds:
                for query in queries:
                    self.index_manager.search_index(index_name, query, k)
                    local_count += 1
            query_count += local_count

        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            concurrent.futures.wait(futures)

        total_time = time.time() - start_time
        return query_count / total_time

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil

            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0


class AccuracyBenchmarker:
    """
    Accuracy benchmarking for RAG systems

    Measures retrieval quality metrics
    """

    def __init__(self, index_manager: MultiIndexManager):
        self.index_manager = index_manager

    def benchmark_accuracy(
        self,
        query_vectors: List[np.ndarray],
        ground_truth_indices: List[List[int]],
        index_name: str,
        k_values: List[int] = [1, 10, 100],
    ) -> AccuracyMetrics:
        """
        Benchmark retrieval accuracy

        Args:
            query_vectors: Query vectors
            ground_truth_indices: Ground truth relevant indices for each query
            index_name: Index to benchmark
            k_values: k values for evaluation

        Returns:
            Accuracy metrics
        """
        all_recalls = {k: [] for k in k_values}
        all_precisions = {k: [] for k in k_values}
        all_ndcgs = {k: [] for k in k_values}

        for query, gt_indices in zip(query_vectors, ground_truth_indices):
            result = self.index_manager.search_index(index_name, query, max(k_values))

            for k in k_values:
                retrieved = set(result.indices[0][:k])
                relevant = set(gt_indices)

                # Recall
                recall = (
                    len(retrieved.intersection(relevant)) / len(relevant)
                    if relevant
                    else 0.0
                )
                all_recalls[k].append(recall)

                # Precision
                precision = len(retrieved.intersection(relevant)) / k if k > 0 else 0.0
                all_precisions[k].append(precision)

                # NDCG
                ndcg = self._calculate_ndcg(retrieved, relevant, k)
                all_ndcgs[k].append(ndcg)

        # Average metrics
        recall_at_1 = (
            sum(all_recalls[1]) / len(all_recalls[1]) if all_recalls[1] else 0.0
        )
        recall_at_10 = (
            sum(all_recalls[10]) / len(all_recalls[10]) if all_recalls[10] else 0.0
        )
        recall_at_100 = (
            sum(all_recalls[100]) / len(all_recalls[100]) if all_recalls[100] else 0.0
        )

        precision_at_1 = (
            sum(all_precisions[1]) / len(all_precisions[1])
            if all_precisions[1]
            else 0.0
        )
        precision_at_10 = (
            sum(all_precisions[10]) / len(all_precisions[10])
            if all_precisions[10]
            else 0.0
        )
        precision_at_100 = (
            sum(all_precisions[100]) / len(all_precisions[100])
            if all_precisions[100]
            else 0.0
        )

        ndcg_at_10 = sum(all_ndcgs[10]) / len(all_ndcgs[10]) if all_ndcgs[10] else 0.0
        ndcg_at_100 = (
            sum(all_ndcgs[100]) / len(all_ndcgs[100]) if all_ndcgs[100] else 0.0
        )

        # MAP score
        map_score = self._calculate_map(all_precisions[10])

        return AccuracyMetrics(
            recall_at_1=recall_at_1,
            recall_at_10=recall_at_10,
            recall_at_100=recall_at_100,
            precision_at_1=precision_at_1,
            precision_at_10=precision_at_10,
            precision_at_100=precision_at_100,
            ndcg_at_10=ndcg_at_10,
            ndcg_at_100=ndcg_at_100,
            map_score=map_score,
        )

    def _calculate_ndcg(self, retrieved: set, relevant: set, k: int) -> float:
        """Calculate NDCG@k"""
        if not relevant:
            return 0.0

        dcg = 0.0
        idcg = 0.0

        # Calculate DCG
        for i, item in enumerate(list(retrieved)[:k]):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)

        # Calculate IDCG (ideal DCG)
        for i in range(min(k, len(relevant))):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_map(self, precisions: List[float]) -> float:
        """Calculate Mean Average Precision"""
        return sum(precisions) / len(precisions) if precisions else 0.0


class ScalabilityBenchmarker:
    """
    Scalability benchmarking for RAG systems

    Tests performance across different dataset sizes
    """

    def __init__(self, index_manager: MultiIndexManager):
        self.index_manager = index_manager
        self.performance_benchmarker = PerformanceBenchmarker(index_manager)

    def benchmark_scalability(
        self,
        base_vectors: np.ndarray,
        query_vectors: List[np.ndarray],
        index_type: str,
        dataset_sizes: List[int] = [1000, 10000, 50000, 100000],
    ) -> ScalabilityMetrics:
        """
        Benchmark scalability across dataset sizes

        Args:
            base_vectors: Base dataset vectors
            query_vectors: Query vectors for testing
            index_type: Type of index to use
            dataset_sizes: Dataset sizes to test

        Returns:
            Scalability metrics
        """
        latencies = []
        throughputs = []
        memories = []
        build_times = []

        for size in dataset_sizes:
            print(f"Benchmarking dataset size: {size}")

            # Subset dataset
            subset_vectors = base_vectors[:size]

            # Create and build index
            index_name = f"scalability_{size}"
            self.index_manager.create_index(index_name, index_type)

            start_time = time.time()
            self.index_manager.add_to_index(index_name, subset_vectors)
            build_time = time.time() - start_time
            build_times.append(build_time)

            # Benchmark performance
            perf_metrics = self.performance_benchmarker.benchmark_latency(
                query_vectors, index_name, k=10, num_runs=50
            )

            latencies.append(perf_metrics.avg_latency_ms)
            throughputs.append(perf_metrics.throughput_qps)
            memories.append(perf_metrics.memory_usage_mb)

        return ScalabilityMetrics(
            dataset_sizes=dataset_sizes,
            latencies_by_size=latencies,
            throughputs_by_size=throughputs,
            memory_by_size=memories,
            index_build_times=build_times,
        )


class ComparativeBenchmarker:
    """
    Comparative benchmarking across different configurations

    Allows comparing different index types, parameters, etc.
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.output_dir = Path("benchmark_results")

    def run_comprehensive_benchmark(
        self,
        configurations: List[Dict[str, Any]],
        dataset: Dict[str, Any],
        benchmark_types: List[str] = ["performance", "accuracy", "scalability"],
    ) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark across configurations

        Args:
            configurations: List of system configurations to test
            dataset: Benchmark dataset
            benchmark_types: Types of benchmarks to run

        Returns:
            Benchmark results for all configurations
        """
        self.output_dir.mkdir(exist_ok=True)
        results = []

        for config in configurations:
            print(f"Running benchmark for configuration: {config['name']}")

            result = self._run_single_configuration_benchmark(
                config, dataset, benchmark_types
            )
            results.append(result)

            # Save intermediate results
            self._save_result(result)

        # Generate comparative report
        self._generate_comparative_report(results)

        return results

    def _run_single_configuration_benchmark(
        self,
        config: Dict[str, Any],
        dataset: Dict[str, Any],
        benchmark_types: List[str],
    ) -> BenchmarkResult:
        """
        Run benchmark for single configuration

        Args:
            config: Configuration to test
            dataset: Benchmark dataset
            benchmark_types: Types of benchmarks

        Returns:
            Benchmark result
        """
        start_time = time.time()

        # Initialize system with configuration
        index_manager = MultiIndexManager(dataset["dimension"])

        # Create index
        index_name = f"benchmark_{config['name']}"
        index_manager.create_index(
            index_name,
            config.get("index_type", "HNSW"),
            **config.get("index_params", {}),
        )

        # Add data
        index_manager.add_to_index(index_name, dataset["vectors"])

        # Run benchmarks
        metrics = {}

        if "performance" in benchmark_types:
            perf_benchmarker = PerformanceBenchmarker(index_manager)
            perf_metrics = perf_benchmarker.benchmark_latency(
                dataset["queries"], index_name, k=10
            )
            metrics.update(
                {
                    "avg_latency_ms": perf_metrics.avg_latency_ms,
                    "throughput_qps": perf_metrics.throughput_qps,
                    "p95_latency_ms": perf_metrics.p95_latency_ms,
                    "memory_usage_mb": perf_metrics.memory_usage_mb,
                }
            )

        if "accuracy" in benchmark_types:
            acc_benchmarker = AccuracyBenchmarker(index_manager)
            acc_metrics = acc_benchmarker.benchmark_accuracy(
                dataset["queries"], dataset.get("ground_truth", []), index_name
            )
            metrics.update(
                {
                    "recall_at_10": acc_metrics.recall_at_10,
                    "precision_at_10": acc_metrics.precision_at_10,
                    "ndcg_at_10": acc_metrics.ndcg_at_10,
                }
            )

        if "scalability" in benchmark_types:
            scale_benchmarker = ScalabilityBenchmarker(index_manager)
            scale_metrics = scale_benchmarker.benchmark_scalability(
                dataset["vectors"], dataset["queries"], config.get("index_type", "HNSW")
            )
            metrics.update(
                {
                    "scalability_latencies": scale_metrics.latencies_by_size,
                    "scalability_throughputs": scale_metrics.throughputs_by_size,
                }
            )

        duration = time.time() - start_time

        return BenchmarkResult(
            benchmark_name=f"{config['name']}_benchmark",
            configuration=config,
            metrics=metrics,
            raw_data={
                "index_stats": index_manager.get_all_stats(),
                "dataset_info": {
                    "n_vectors": len(dataset["vectors"]),
                    "n_queries": len(dataset["queries"]),
                    "dimension": dataset["dimension"],
                },
            },
            timestamp=datetime.now().isoformat(),
            duration=duration,
        )

    def _save_result(self, result: BenchmarkResult):
        """Save benchmark result to file"""
        filename = f"{result.benchmark_name}_{result.timestamp.replace(':', '-')}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(
                {
                    "benchmark_name": result.benchmark_name,
                    "configuration": result.configuration,
                    "metrics": result.metrics,
                    "raw_data": result.raw_data,
                    "timestamp": result.timestamp,
                    "duration": result.duration,
                },
                f,
                indent=2,
                default=str,
            )

    def _generate_comparative_report(self, results: List[BenchmarkResult]):
        """Generate comparative analysis report"""
        report = {
            "summary": {
                "num_configurations": len(results),
                "best_latency": min(
                    (r.metrics.get("avg_latency_ms", float("inf")) for r in results)
                ),
                "best_throughput": max(
                    (r.metrics.get("throughput_qps", 0) for r in results)
                ),
                "best_recall": max((r.metrics.get("recall_at_10", 0) for r in results)),
            },
            "configurations": [
                {
                    "name": r.configuration["name"],
                    "metrics": r.metrics,
                    "duration": r.duration,
                }
                for r in results
            ],
            "recommendations": self._generate_recommendations(results),
        }

        report_path = self.output_dir / "comparative_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate human-readable summary
        self._generate_human_readable_report(results)

    def _generate_recommendations(
        self, results: List[BenchmarkResult]
    ) -> Dict[str, str]:
        """Generate recommendations based on results"""
        recommendations = {}

        # Find best configurations for different metrics
        latencies = [
            (r.configuration["name"], r.metrics.get("avg_latency_ms", float("inf")))
            for r in results
        ]
        best_latency = min(latencies, key=lambda x: x[1])

        throughputs = [
            (r.configuration["name"], r.metrics.get("throughput_qps", 0))
            for r in results
        ]
        best_throughput = max(throughputs, key=lambda x: x[1])

        recalls = [
            (r.configuration["name"], r.metrics.get("recall_at_10", 0)) for r in results
        ]
        best_recall = max(recalls, key=lambda x: x[1])

        recommendations.update(
            {
                "lowest_latency": f"{best_latency[0]} ({best_latency[1]:.2f}ms)",
                "highest_throughput": f"{best_throughput[0]} ({best_throughput[1]:.0f} QPS)",
                "best_accuracy": f"{best_recall[0]} ({best_recall[1]:.3f} recall@10)",
                "overall_recommendation": self._get_overall_recommendation(results),
            }
        )

        return recommendations

    def _get_overall_recommendation(self, results: List[BenchmarkResult]) -> str:
        """Get overall recommendation based on balanced metrics"""
        # Simple scoring: balance latency, throughput, and accuracy
        scores = {}
        for result in results:
            latency_score = 1.0 / (
                1.0 + result.metrics.get("avg_latency_ms", 100)
            )  # Lower is better
            throughput_score = min(
                result.metrics.get("throughput_qps", 0) / 1000, 1.0
            )  # Higher is better
            recall_score = result.metrics.get("recall_at_10", 0)  # Higher is better

            overall_score = (latency_score + throughput_score + recall_score) / 3
            scores[result.configuration["name"]] = overall_score

        best_config = max(scores.items(), key=lambda x: x[1])
        return f"{best_config[0]} (balanced score: {best_config[1]:.3f})"

    def _generate_human_readable_report(self, results: List[BenchmarkResult]):
        """Generate human-readable benchmark report"""
        report_path = self.output_dir / "benchmark_report.txt"

        with open(report_path, "w") as f:
            f.write("RAG System Benchmark Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total configurations tested: {len(results)}\n")
            f.write(
                f"Benchmark completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # Performance comparison
            f.write("PERFORMANCE COMPARISON\n")
            f.write("-" * 30 + "\n")
            for result in sorted(
                results, key=lambda x: x.metrics.get("avg_latency_ms", float("inf"))
            ):
                f.write(
                    f"{result.configuration['name']:<20} | "
                    f"Latency: {result.metrics.get('avg_latency_ms', 0):>6.2f}ms | "
                    f"Throughput: {result.metrics.get('throughput_qps', 0):>6.0f} QPS | "
                    f"Memory: {result.metrics.get('memory_usage_mb', 0):>6.1f} MB\n"
                )

            f.write("\nACCURACY COMPARISON\n")
            f.write("-" * 30 + "\n")
            for result in sorted(
                results, key=lambda x: x.metrics.get("recall_at_10", 0), reverse=True
            ):
                f.write(
                    f"{result.configuration['name']:<20} | "
                    f"Recall@10: {result.metrics.get('recall_at_10', 0):>6.3f} | "
                    f"NDCG@10: {result.metrics.get('ndcg_at_10', 0):>6.3f}\n"
                )

            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            recommendations = self._generate_recommendations(results)
            for key, value in recommendations.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")


# Utility functions for benchmark data generation
def generate_synthetic_dataset(
    n_vectors: int = 10000,
    n_queries: int = 1000,
    dimension: int = 768,
    similarity_ratio: float = 0.1,
) -> Dict[str, Any]:
    """
    Generate synthetic benchmark dataset

    Args:
        n_vectors: Number of vectors in dataset
        n_queries: Number of query vectors
        dimension: Vector dimension
        similarity_ratio: Ratio of similar vectors for ground truth

    Returns:
        Benchmark dataset
    """
    np.random.seed(42)

    # Generate base vectors
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize

    # Generate queries
    queries = np.random.randn(n_queries, dimension).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    # Generate ground truth (approximate nearest neighbors)
    ground_truth = []
    n_similar = int(n_vectors * similarity_ratio)

    for query in queries:
        # Calculate similarities
        similarities = np.dot(vectors, query)

        # Get top similar indices
        top_indices = np.argsort(similarities)[-n_similar:][::-1]
        ground_truth.append(top_indices.tolist())

    return {
        "vectors": vectors,
        "queries": queries,
        "ground_truth": ground_truth,
        "dimension": dimension,
        "n_vectors": n_vectors,
        "n_queries": n_queries,
    }


def load_standard_benchmark_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Load standard benchmark dataset

    Args:
        dataset_name: Name of dataset ('sift', 'gist', 'glove', etc.)

    Returns:
        Benchmark dataset
    """
    try:
        # Intentar cargar dataset real si está disponible
        if dataset_name.lower() == 'sift':
            return _load_sift_dataset()
        elif dataset_name.lower() == 'gist':
            return _load_gist_dataset()
        elif dataset_name.lower() == 'glove':
            return _load_glove_dataset()
        elif dataset_name.lower() == 'random':
            return _load_random_dataset()
        else:
            print(f"Dataset '{dataset_name}' no reconocido, generando sintético")
            return generate_synthetic_dataset(10000, 1000, 128)

    except Exception as e:
        print(f"Error cargando dataset '{dataset_name}': {e}, usando datos sintéticos")
        return generate_synthetic_dataset(10000, 1000, 128)


def _load_sift_dataset() -> Dict[str, Any]:
    """Load SIFT dataset (ANN-Benchmarks format)"""
    try:
        import urllib.request
        import gzip
        import os

        # URLs para SIFT dataset
        base_url = "http://ann-benchmarks.com/"
        learn_url = f"{base_url}sift-128-euclidean.hdf5"
        query_url = f"{base_url}sift-128-euclidean-queries.hdf5"

        # Crear directorio para datasets
        dataset_dir = Path("benchmark_datasets")
        dataset_dir.mkdir(exist_ok=True)

        # Descargar si no existe
        learn_file = dataset_dir / "sift-128-euclidean.hdf5"
        query_file = dataset_dir / "sift-128-euclidean-queries.hdf5"

        if not learn_file.exists():
            print("Descargando SIFT learn dataset...")
            urllib.request.urlretrieve(learn_url, learn_file)

        if not query_file.exists():
            print("Descargando SIFT query dataset...")
            urllib.request.urlretrieve(query_url, query_file)

        # Cargar usando h5py si está disponible
        try:
            import h5py

            with h5py.File(learn_file, 'r') as f:
                vectors = np.array(f['train'])

            with h5py.File(query_file, 'r') as f:
                queries = np.array(f['test'])

            # Ground truth no disponible, generar aproximado
            ground_truth = _generate_ground_truth_approximation(vectors, queries)

            return {
                "vectors": vectors.astype(np.float32),
                "queries": queries.astype(np.float32),
                "ground_truth": ground_truth,
                "dimension": vectors.shape[1],
                "n_vectors": len(vectors),
                "n_queries": len(queries),
            }

        except ImportError:
            print("h5py no disponible, usando datos sintéticos")
            return generate_synthetic_dataset(10000, 1000, 128)

    except Exception as e:
        print(f"Error cargando SIFT dataset: {e}")
        return generate_synthetic_dataset(10000, 1000, 128)


def _load_gist_dataset() -> Dict[str, Any]:
    """Load GIST dataset"""
    try:
        # Similar a SIFT pero con URLs diferentes
        print("GIST dataset loading not implemented yet")
        return generate_synthetic_dataset(50000, 1000, 960)
    except Exception as e:
        print(f"Error cargando GIST dataset: {e}")
        return generate_synthetic_dataset(50000, 1000, 960)


def _load_glove_dataset() -> Dict[str, Any]:
    """Load GloVe embeddings dataset"""
    try:
        import urllib.request
        import os

        # GloVe 100D
        glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
        dataset_dir = Path("benchmark_datasets")
        dataset_dir.mkdir(exist_ok=True)

        glove_file = dataset_dir / "glove.6B.100d.txt"

        if not glove_file.exists():
            print("Descargando GloVe embeddings...")
            # Nota: En producción descargaría y extraería el zip
            print("GloVe download not implemented, using synthetic data")
            return generate_synthetic_dataset(100000, 1000, 100)

        # Cargar embeddings
        vectors = []
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                vector = np.array([float(x) for x in values[1:]])
                vectors.append(vector)

        vectors = np.array(vectors)
        queries = vectors[:1000]  # Usar primeros 1000 como queries

        return {
            "vectors": vectors.astype(np.float32),
            "queries": queries.astype(np.float32),
            "ground_truth": [],  # No ground truth disponible
            "dimension": vectors.shape[1],
            "n_vectors": len(vectors),
            "n_queries": len(queries),
        }

    except Exception as e:
        print(f"Error cargando GloVe dataset: {e}")
        return generate_synthetic_dataset(100000, 1000, 100)


def _load_random_dataset() -> Dict[str, Any]:
    """Load or generate random dataset"""
    return generate_synthetic_dataset(50000, 1000, 128)


def _generate_ground_truth_approximation(vectors: np.ndarray, queries: np.ndarray, k: int = 100) -> List[List[int]]:
    """Generate approximate ground truth for evaluation"""
    ground_truth = []

    # Para datasets grandes, usar aproximación con submuestreo
    if len(vectors) > 10000:
        # Submuestrear para hacer más rápido
        sample_indices = np.random.choice(len(vectors), size=min(10000, len(vectors)), replace=False)
        sample_vectors = vectors[sample_indices]
    else:
        sample_vectors = vectors
        sample_indices = np.arange(len(vectors))

    for query in queries:
        # Calcular similitudes con submuestra
        similarities = np.dot(sample_vectors, query)
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Mapear de vuelta a índices originales
        original_indices = sample_indices[top_k_indices]
        ground_truth.append(original_indices.tolist())

    return ground_truth
