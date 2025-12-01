#!/usr/bin/env python3
"""
Final Weights Analysis - Análisis Real de Pesos Neuronales Finales
====================================================================

Analiza los pesos neuronales generados y proporciona métricas reales
de calidad, distribución y utilidad para entrenamiento.
"""

import json
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FinalWeightsAnalyzer:
    """Analizador real de pesos neuronales finales"""
    
    def __init__(self, weights_dir: Optional[Path] = None):
        if weights_dir is None:
            # Buscar en directorio de pesos neuronales
            project_root = Path(__file__).parent.parent.parent
            weights_dir = project_root / "tools" / "analysis" / "real_neural_weights"
        
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Weights directory: {self.weights_dir}")

    def analyze_all_weights(self) -> Dict[str, Any]:
        """Analizar todos los pesos neuronales disponibles"""
        logger.info("🔍 Starting analysis of neural weights...")
        
        analysis_results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "weights_directory": str(self.weights_dir),
            "architectures": {},
            "summary": {}
        }
        
        # Buscar archivos de pesos por arquitectura
        architectures = ["feedforward", "lstm", "transformer", "cnn"]
        
        for arch in architectures:
            arch_dir = self.weights_dir / arch
            if not arch_dir.exists():
                logger.warning(f"⚠️ Architecture directory not found: {arch}")
                continue
            
            arch_analysis = self._analyze_architecture(arch_dir, arch)
            if arch_analysis:
                analysis_results["architectures"][arch] = arch_analysis
        
        # Calcular resumen
        analysis_results["summary"] = self._calculate_summary(analysis_results["architectures"])
        
        # Guardar análisis
        output_file = self.weights_dir / f"weights_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Analysis completed")
        logger.info(f"💾 Saved to: {output_file}")
        
        return analysis_results

    def _analyze_architecture(self, arch_dir: Path, arch_name: str) -> Optional[Dict[str, Any]]:
        """Analizar pesos de una arquitectura específica"""
        # Buscar archivos .npz más recientes
        weight_files = list(arch_dir.glob("weights_*.npz"))
        metadata_files = list(arch_dir.glob("metadata_*.json"))
        
        if not weight_files:
            logger.warning(f"⚠️ No weight files found for {arch_name}")
            return None
        
        # Usar el archivo más reciente
        latest_weights = max(weight_files, key=lambda p: p.stat().st_mtime)
        latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime) if metadata_files else None
        
        logger.info(f"📊 Analyzing {arch_name}: {latest_weights.name}")
        
        analysis = {
            "weights_file": latest_weights.name,
            "metadata_file": latest_metadata.name if latest_metadata else None,
            "layers": {},
            "statistics": {},
            "quality_metrics": {}
        }
        
        try:
            # Cargar pesos
            weights_data = np.load(latest_weights, allow_pickle=True)
            
            # Analizar cada capa
            layer_stats = {}
            total_params = 0
            total_memory_mb = 0
            
            for key in weights_data.files:
                weight_array = weights_data[key]
                
                # Estadísticas de la capa
                layer_stats[key] = {
                    "shape": list(weight_array.shape),
                    "dtype": str(weight_array.dtype),
                    "size": weight_array.size,
                    "mean": float(np.mean(weight_array)),
                    "std": float(np.std(weight_array)),
                    "min": float(np.min(weight_array)),
                    "max": float(np.max(weight_array)),
                    "memory_mb": weight_array.nbytes / (1024 * 1024),
                    "zero_ratio": float(np.sum(weight_array == 0) / weight_array.size),
                    "sparsity": float(np.sum(np.abs(weight_array) < 1e-6) / weight_array.size)
                }
                
                total_params += weight_array.size
                total_memory_mb += layer_stats[key]["memory_mb"]
            
            analysis["layers"] = layer_stats
            analysis["statistics"] = {
                "total_parameters": total_params,
                "total_memory_mb": total_memory_mb,
                "num_layers": len(layer_stats),
                "avg_params_per_layer": total_params / max(len(layer_stats), 1)
            }
            
            # Métricas de calidad
            analysis["quality_metrics"] = self._calculate_quality_metrics(layer_stats)
            
            # Cargar metadata si existe
            if latest_metadata:
                try:
                    with open(latest_metadata, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    analysis["metadata"] = metadata
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")
        
        except Exception as e:
            logger.error(f"Error analyzing {arch_name}: {e}")
            analysis["error"] = str(e)
            return analysis
        
        return analysis

    def _calculate_quality_metrics(self, layer_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calcular métricas de calidad de los pesos"""
        if not layer_stats:
            return {}
        
        # Calcular métricas agregadas
        all_means = [stats["mean"] for stats in layer_stats.values()]
        all_stds = [stats["std"] for stats in layer_stats.values()]
        all_sparsities = [stats["sparsity"] for stats in layer_stats.values()]
        
        # Detectar problemas comunes
        issues = []
        
        # Pesos muy grandes (posible explosión de gradientes)
        if any(abs(m) > 10 for m in all_means):
            issues.append("large_weight_values")
        
        # Pesos muy pequeños (posible desvanecimiento)
        if any(abs(m) < 1e-6 for m in all_means if abs(m) > 0):
            issues.append("vanishing_weights")
        
        # Alta varianza (inestabilidad)
        if any(std > 5 for std in all_stds):
            issues.append("high_variance")
        
        # Esparsidad extrema
        if any(sparsity > 0.9 for sparsity in all_sparsities):
            issues.append("extreme_sparsity")
        
        # Score de calidad (0-100)
        quality_score = 100.0
        
        # Penalizar por problemas
        quality_score -= len(issues) * 10
        
        # Penalizar por varianza alta
        avg_std = np.mean(all_stds) if all_stds else 0
        if avg_std > 2:
            quality_score -= (avg_std - 2) * 5
        
        # Bonificar por distribución normal
        if 0.5 < np.mean([abs(m) for m in all_means]) < 2.0:
            quality_score += 10
        
        quality_score = max(0, min(100, quality_score))
        
        return {
            "overall_quality_score": quality_score,
            "mean_weight_value": float(np.mean(all_means)),
            "mean_std": float(np.mean(all_stds)),
            "mean_sparsity": float(np.mean(all_sparsities)),
            "issues_detected": issues,
            "distribution_quality": "good" if quality_score >= 70 else "needs_improvement"
        }

    def _calculate_summary(self, architectures: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular resumen de todas las arquitecturas"""
        if not architectures:
            return {"error": "No architectures analyzed"}
        
        total_params = 0
        total_memory_mb = 0
        quality_scores = []
        
        for arch_name, arch_data in architectures.items():
            stats = arch_data.get("statistics", {})
            total_params += stats.get("total_parameters", 0)
            total_memory_mb += stats.get("total_memory_mb", 0)
            
            quality = arch_data.get("quality_metrics", {})
            quality_scores.append(quality.get("overall_quality_score", 0))
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        return {
            "total_architectures": len(architectures),
            "total_parameters": int(total_params),
            "total_memory_mb": round(total_memory_mb, 2),
            "average_quality_score": round(avg_quality, 2),
            "architectures_analyzed": list(architectures.keys())
        }

    def compare_weights(self, weights_file1: Path, weights_file2: Path) -> Dict[str, Any]:
        """Comparar dos archivos de pesos"""
        logger.info(f"🔍 Comparing weights: {weights_file1.name} vs {weights_file2.name}")
        
        try:
            weights1 = np.load(weights_file1, allow_pickle=True)
            weights2 = np.load(weights_file2, allow_pickle=True)
            
            comparison = {
                "file1": weights_file1.name,
                "file2": weights_file2.name,
                "common_layers": [],
                "differences": {}
            }
            
            # Comparar capas comunes
            common_keys = set(weights1.files) & set(weights2.files)
            
            for key in common_keys:
                w1 = weights1[key]
                w2 = weights2[key]
                
                if w1.shape != w2.shape:
                    comparison["differences"][key] = {
                        "shape_mismatch": True,
                        "shape1": list(w1.shape),
                        "shape2": list(w2.shape)
                    }
                    continue
                
                # Calcular diferencias
                diff = np.abs(w1 - w2)
                mse = float(np.mean(diff ** 2))
                max_diff = float(np.max(diff))
                mean_diff = float(np.mean(diff))
                
                comparison["common_layers"].append(key)
                comparison["differences"][key] = {
                    "mse": mse,
                    "max_difference": max_diff,
                    "mean_difference": mean_diff,
                    "relative_change": mean_diff / (np.mean(np.abs(w1)) + 1e-10)
                }
            
            return comparison
        
        except Exception as e:
            logger.error(f"Error comparing weights: {e}")
            return {"error": str(e)}


def main():
    """Función principal"""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    analyzer = FinalWeightsAnalyzer()
    results = analyzer.analyze_all_weights()
    
    print("\n" + "=" * 80)
    print("📊 FINAL WEIGHTS ANALYSIS RESULTS")
    print("=" * 80)
    
    summary = results.get("summary", {})
    print(f"\n✅ Architectures analyzed: {summary.get('total_architectures', 0)}")
    print(f"📈 Total parameters: {summary.get('total_parameters', 0):,}")
    print(f"💾 Total memory: {summary.get('total_memory_mb', 0):.2f} MB")
    print(f"⭐ Average quality score: {summary.get('average_quality_score', 0):.1f}/100")
    
    for arch_name, arch_data in results.get("architectures", {}).items():
        print(f"\n🏗️ {arch_name.upper()}:")
        stats = arch_data.get("statistics", {})
        quality = arch_data.get("quality_metrics", {})
        print(f"   Parameters: {stats.get('total_parameters', 0):,}")
        print(f"   Quality: {quality.get('overall_quality_score', 0):.1f}/100")
        if quality.get("issues_detected"):
            print(f"   ⚠️ Issues: {', '.join(quality['issues_detected'])}")
    
    print("\n" + "=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
