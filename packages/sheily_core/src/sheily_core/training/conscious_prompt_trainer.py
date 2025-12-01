"""
Entrenamiento REAL de ConsciousPromptGenerator
Optimiza templates y adaptadores emocionales con datos reales
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class ConsciousPromptTrainer:
    """Entrenador REAL para optimizar ConsciousPromptGenerator"""
    
    def __init__(self):
        self.optimization_history = []
        self.best_templates = {}
        self.best_emotional_adapters = {}
    
    def analyze_prompt_effectiveness(self, qa_data: List[Dict]) -> Dict[str, Any]:
        """Analizar efectividad REAL de prompts generados"""
        effectiveness_data = []
        
        for qa in qa_data:
            quality_score = qa.get("quality_score", 0.5)
            accepted = qa.get("accepted_for_training", False)
            tokens_used = qa.get("tokens_used", 0)
            response_length = len(qa.get("response", ""))
            
            # Calcular efectividad basada en múltiples factores
            effectiveness = {
                "question": qa.get("question", ""),
                "response_quality": quality_score,
                "accepted": accepted,
                "tokens_efficiency": quality_score / max(tokens_used, 1) * 1000,  # Calidad por token
                "response_length": response_length,
                "effectiveness_score": (
                    quality_score * 0.4 +
                    (1.0 if accepted else 0.0) * 0.3 +
                    min(response_length / 500, 1.0) * 0.2 +
                    (1.0 / max(tokens_used / 100, 1.0)) * 0.1
                )
            }
            effectiveness_data.append(effectiveness)
        
        avg_effectiveness = sum(e["effectiveness_score"] for e in effectiveness_data) / len(effectiveness_data) if effectiveness_data else 0
        
        return {
            "total_qa": len(qa_data),
            "avg_effectiveness": avg_effectiveness,
            "accepted_count": sum(1 for e in effectiveness_data if e["accepted"]),
            "high_quality_count": sum(1 for e in effectiveness_data if e["response_quality"] >= 0.7),
            "effectiveness_data": effectiveness_data
        }
    
    def optimize_templates(self, analysis: Dict[str, Any], current_templates: Dict[str, str]) -> Dict[str, str]:
        """Optimizar templates REAL basado en análisis"""
        logger.info("🔧 Optimizando templates de prompts...")
        
        effectiveness_data = analysis.get("effectiveness_data", [])
        
        # Agrupar por estilo/categoría
        style_performance = {}
        
        for item in effectiveness_data:
            # Inferir estilo basado en pregunta
            style = self._infer_style(item["question"])
            if style not in style_performance:
                style_performance[style] = []
            style_performance[style].append(item["effectiveness_score"])
        
        # Calcular promedio por estilo
        style_avg = {
            style: sum(scores) / len(scores)
            for style, scores in style_performance.items()
        }
        
        # Optimizar templates basado en rendimiento
        optimized_templates = current_templates.copy()
        
        for style, avg_score in style_avg.items():
            if style in optimized_templates:
                # Si el rendimiento es bajo, mejorar template
                if avg_score < 0.6:
                    optimized_templates[style] = self._improve_template(
                        optimized_templates[style],
                        style,
                        avg_score
                    )
                    logger.info(f"✅ Template '{style}' optimizado (score: {avg_score:.2f})")
        
        return optimized_templates
    
    def optimize_emotional_adapters(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Optimizar adaptadores emocionales REAL"""
        logger.info("🔧 Optimizando adaptadores emocionales...")
        
        effectiveness_data = analysis.get("effectiveness_data", [])
        
        # Analizar correlación entre tono emocional y efectividad
        emotional_performance = {}
        
        for item in effectiveness_data:
            # Inferir tono emocional de la pregunta
            emotional_tone = self._infer_emotional_tone(item["question"])
            if emotional_tone not in emotional_performance:
                emotional_performance[emotional_tone] = []
            emotional_performance[emotional_tone].append(item["effectiveness_score"])
        
        # Calcular ajustes para adaptadores
        optimized_adapters = {}
        
        for tone, scores in emotional_performance.items():
            avg_score = sum(scores) / len(scores)
            # Ajustar adaptador basado en rendimiento
            # Si rendimiento es bajo, aumentar sensibilidad
            adjustment = 1.0 + (0.6 - avg_score) * 0.5  # Ajuste proporcional
            optimized_adapters[tone] = max(0.1, min(2.0, adjustment))
            logger.info(f"✅ Adaptador '{tone}' ajustado: {optimized_adapters[tone]:.2f}")
        
        return optimized_adapters
    
    def optimize_thresholds(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Optimizar thresholds REAL (BasalGangliaGate, etc.)"""
        logger.info("🔧 Optimizando thresholds...")
        
        effectiveness_data = analysis.get("effectiveness_data", [])
        accepted_count = analysis.get("accepted_count", 0)
        total_count = analysis.get("total_qa", 1)
        
        acceptance_rate = accepted_count / total_count
        
        # Ajustar threshold basado en tasa de aceptación
        # Si aceptación es muy baja (<30%), bajar threshold
        # Si aceptación es muy alta (>80%), subir threshold
        current_threshold = 0.5
        if acceptance_rate < 0.3:
            new_threshold = current_threshold * 0.8
        elif acceptance_rate > 0.8:
            new_threshold = current_threshold * 1.2
        else:
            new_threshold = current_threshold
        
        new_threshold = max(0.1, min(0.9, new_threshold))
        
        logger.info(f"✅ Threshold optimizado: {current_threshold:.2f} → {new_threshold:.2f}")
        
        return {
            "basal_ganglia_threshold": new_threshold,
            "acceptance_rate": acceptance_rate
        }
    
    def train(
        self,
        qa_data: List[Dict],
        current_templates: Dict[str, str],
        output_dir: str = "data/training/prompt_optimization"
    ) -> Dict[str, Any]:
        """Entrenamiento REAL completo de ConsciousPromptGenerator"""
        logger.info(f"🚀 Iniciando entrenamiento REAL de ConsciousPromptGenerator...")
        
        # 1. Analizar efectividad
        analysis = self.analyze_prompt_effectiveness(qa_data)
        
        # 2. Optimizar templates
        optimized_templates = self.optimize_templates(analysis, current_templates)
        
        # 3. Optimizar adaptadores emocionales
        optimized_adapters = self.optimize_emotional_adapters(analysis)
        
        # 4. Optimizar thresholds
        optimized_thresholds = self.optimize_thresholds(analysis)
        
        # 5. Guardar resultados
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {
            "analysis": analysis,
            "optimized_templates": optimized_templates,
            "optimized_emotional_adapters": optimized_adapters,
            "optimized_thresholds": optimized_thresholds,
            "timestamp": datetime.now().isoformat()
        }
        
        output_file = Path(output_dir) / f"prompt_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Entrenamiento completado")
        logger.info(f"   - Templates optimizados: {len(optimized_templates)}")
        logger.info(f"   - Adaptadores optimizados: {len(optimized_adapters)}")
        logger.info(f"   - Resultados guardados en: {output_file}")
        
        return results
    
    def _infer_style(self, text: str) -> str:
        """Inferir estilo de prompt basado en texto"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["técnico", "técnica", "código", "algoritmo", "implementar"]):
            return "technical"
        elif any(word in text_lower for word in ["creativo", "idea", "diseño", "artístico"]):
            return "creative"
        elif any(word in text_lower for word in ["hola", "gracias", "por favor", "ayuda"]):
            return "casual"
        else:
            return "professional"
    
    def _infer_emotional_tone(self, text: str) -> str:
        """Inferir tono emocional basado en texto"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["feliz", "alegre", "genial", "excelente"]):
            return "positive"
        elif any(word in text_lower for word in ["triste", "problema", "error", "fallo"]):
            return "negative"
        elif any(word in text_lower for word in ["urgente", "rápido", "inmediato"]):
            return "aroused"
        else:
            return "neutral"
    
    def _improve_template(self, template: str, style: str, current_score: float) -> str:
        """Mejorar template basado en score actual"""
        # Estrategias de mejora según estilo
        improvements = {
            "professional": "\n[NOTA: Sé preciso y profesional en tu respuesta]",
            "casual": "\n[NOTA: Responde de forma natural y amigable]",
            "technical": "\n[NOTA: Proporciona detalles técnicos y ejemplos concretos]",
            "creative": "\n[NOTA: Sé creativo e innovador en tu respuesta]"
        }
        
        improvement = improvements.get(style, "")
        return template + improvement

