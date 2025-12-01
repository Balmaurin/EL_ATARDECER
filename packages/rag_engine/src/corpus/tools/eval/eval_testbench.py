#!/usr/bin/env python3
"""
Evaluador del Testbench de Queries
===================================

Evalúa el testbench de 200 queries y calcula métricas de baseline
(recall@5, precision@k, MRR) para CI/CD.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

# Agregar paths necesarios
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from corpus.tools.eval.testbench_queries import TestbenchGenerator, TestQuery
from corpus.tools.retrieval.search_unified import unified_search


def evaluate_testbench(
    base_path: Path,
    testbench_path: Path = Path("corpus/_registry/testbench_queries.jsonl"),
    top_k: int = 5,
    mode: str = "hybrid",
) -> Dict:
    """Evaluar testbench y calcular métricas"""
    
    # Cargar queries
    generator = TestbenchGenerator(testbench_path)
    queries = generator.load_queries()
    
    if not queries:
        print("⚠️  No se encontraron queries. Generando testbench por defecto...")
        queries = generator.generate_and_save()
    
    print(f"📊 Evaluando {len(queries)} queries con recall@{top_k}...")
    
    # Métricas acumuladas
    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_queries = 0
    
    results_by_category = {}
    results_by_difficulty = {}
    
    for i, query in enumerate(queries):
        try:
            # Ejecutar búsqueda
            hits = unified_search(
                "universal", base_path, query.question, top_k=top_k, mode=mode
            )
            
            # Calcular recall y precision
            relevant_found = 0
            first_relevant_rank = None
            
            hit_texts = [h.get("text", "").lower() for h in hits]
            combined_text = " ".join(hit_texts).lower()
            
            # Verificar qué expected chunks están presentes
            for expected in query.expected_chunks:
                if expected.lower() in combined_text:
                    relevant_found += 1
                    if first_relevant_rank is None:
                        # Encontrar rank del primer chunk relevante
                        for rank, text in enumerate(hit_texts, 1):
                            if expected.lower() in text:
                                first_relevant_rank = rank
                                break
            
            # Calcular métricas
            recall = relevant_found / len(query.expected_chunks) if query.expected_chunks else 0.0
            precision = relevant_found / len(hits) if hits else 0.0
            mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            
            total_recall += recall
            total_precision += precision
            total_mrr += mrr
            total_queries += 1
            
            # Agregar por categoría
            if query.category not in results_by_category:
                results_by_category[query.category] = {
                    "recall": 0.0,
                    "precision": 0.0,
                    "mrr": 0.0,
                    "count": 0,
                }
            results_by_category[query.category]["recall"] += recall
            results_by_category[query.category]["precision"] += precision
            results_by_category[query.category]["mrr"] += mrr
            results_by_category[query.category]["count"] += 1
            
            # Agregar por dificultad
            if query.difficulty not in results_by_difficulty:
                results_by_difficulty[query.difficulty] = {
                    "recall": 0.0,
                    "precision": 0.0,
                    "mrr": 0.0,
                    "count": 0,
                }
            results_by_difficulty[query.difficulty]["recall"] += recall
            results_by_difficulty[query.difficulty]["precision"] += precision
            results_by_difficulty[query.difficulty]["mrr"] += mrr
            results_by_difficulty[query.difficulty]["count"] += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Procesadas {i + 1}/{len(queries)} queries...")
        
        except Exception as e:
            print(f"⚠️  Error procesando query {query.query_id}: {e}")
            continue
    
    # Calcular promedios
    avg_recall = total_recall / total_queries if total_queries > 0 else 0.0
    avg_precision = total_precision / total_queries if total_queries > 0 else 0.0
    avg_mrr = total_mrr / total_queries if total_queries > 0 else 0.0
    
    # Promedios por categoría
    for cat in results_by_category:
        count = results_by_category[cat]["count"]
        results_by_category[cat]["recall"] /= count
        results_by_category[cat]["precision"] /= count
        results_by_category[cat]["mrr"] /= count
    
    # Promedios por dificultad
    for diff in results_by_difficulty:
        count = results_by_difficulty[diff]["count"]
        results_by_difficulty[diff]["recall"] /= count
        results_by_difficulty[diff]["precision"] /= count
        results_by_difficulty[diff]["mrr"] /= count
    
    results = {
        "total_queries": total_queries,
        "top_k": top_k,
        "mode": mode,
        "metrics": {
            "recall@5": avg_recall,
            "precision@5": avg_precision,
            "mrr": avg_mrr,
        },
        "by_category": results_by_category,
        "by_difficulty": results_by_difficulty,
    }
    
    return results


def print_results(results: Dict) -> None:
    """Imprimir resultados formateados"""
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DE EVALUACIÓN DEL TESTBENCH")
    print("=" * 60)
    
    metrics = results["metrics"]
    print(f"\n🎯 Métricas Globales (recall@{results['top_k']}):")
    print(f"  Recall@5:    {metrics['recall@5']:.3f}")
    print(f"  Precision@5: {metrics['precision@5']:.3f}")
    print(f"  MRR:         {metrics['mrr']:.3f}")
    
    print(f"\n📂 Por Categoría:")
    for cat, res in sorted(results["by_category"].items()):
        print(f"  {cat:15s} - Recall: {res['recall']:.3f}, Precision: {res['precision']:.3f}, MRR: {res['mrr']:.3f} ({res['count']} queries)")
    
    print(f"\n🎯 Por Dificultad:")
    for diff, res in sorted(results["by_difficulty"].items()):
        print(f"  {diff:10s} - Recall: {res['recall']:.3f}, Precision: {res['precision']:.3f}, MRR: {res['mrr']:.3f} ({res['count']} queries)")
    
    print("\n" + "=" * 60)


def save_baseline(results: Dict, output_path: Path = Path("corpus/_registry/baseline_recall.json")) -> None:
    """Guardar baseline para CI/CD"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Baseline guardado en: {output_path}")


def main():
    """Ejecutar evaluación del testbench"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluar testbench de queries")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("corpus/universal/latest"),
        help="Path base del índice",
    )
    parser.add_argument(
        "--testbench",
        type=Path,
        default=Path("corpus/_registry/testbench_queries.jsonl"),
        help="Path al testbench de queries",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K para recall",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "dense", "bm25"],
        help="Modo de búsqueda",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Guardar baseline para CI/CD",
    )
    
    args = parser.parse_args()
    
    # Evaluar
    results = evaluate_testbench(
        args.base_path,
        args.testbench,
        args.top_k,
        args.mode,
    )
    
    # Imprimir resultados
    print_results(results)
    
    # Guardar baseline si se solicita
    if args.save_baseline:
        save_baseline(results)
    
    # Exit code basado en recall mínimo
    if results["metrics"]["recall@5"] < 0.5:
        print("\n⚠️  RECALL@5 BAJO (< 0.5). Revisar configuración del índice.")
        sys.exit(1)
    else:
        print("\n✅ Baseline aceptable (recall@5 >= 0.5)")
        sys.exit(0)


if __name__ == "__main__":
    main()

