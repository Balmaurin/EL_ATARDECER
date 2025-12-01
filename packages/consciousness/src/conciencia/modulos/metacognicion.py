"""
Metacognición: Pensar sobre el Pensamiento

Implementa capacidad para reflexionar sobre procesos cognitivos propios.
Soluciona el gap: No auto-evaluación continua
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import numpy as np


@dataclass
class CognitiveTrace:
    """Rastro de proceso cognitivo para análisis metacognitivo"""
    process_id: str
    reasoning_steps: int
    input_complexity: float
    output_quality: float
    confidence_assessed: float
    time_taken: float
    metacognitive_evaluation: Dict[str, float]


class MetacognitionEngine:
    """
    Motor de metacognición para auto-evaluación cognitiva

    Implementa:
    - Monitorización del propio pensamiento
    - Calibración de confianza
    - Detección de sesgos cognitivos
    - Auto-mejora basada en reflexión
    """

    def __init__(self):
        self.thinking_traces: List[Dict[str, Any]] = []
        self.confidence_calibration_history = []
        self.cognitive_bias_detector = CognitiveBiasDetector()
        self.metacognition_level = 0.5  # Nivel de habilidad metacognitiva
        self.improvement_engine = CognitiveImprovementEngine()

        print("🧠 Metacognition Engine inicializado - capacidad de auto-reflexión activada")

    def monitor_thinking_process(self, sensory_input: Dict[str, Any], conscious_contents: List[Dict],
                                conscious_moment: Any) -> Dict[str, float]:
        """
        Monitorea proceso completo de pensamiento y genera evaluación metacognitiva

        Returns:
            Dict con métricas de calidad cognitiva y auto-evaluación
        """
        start_time = time.time()

        # Evaluar la calidad del razonamiento actual
        reasoning_quality = self._assess_reasoning_quality(conscious_contents)

        # Calibrar confianza metacognitiva
        confidence_accuracy = self._calibrate_metacognitive_confidence(conscious_moment, sensory_input)

        # Detectar sesgos cognitivos
        bias_detection = self.cognitive_bias_detector.detect_biases(sensory_input, conscious_contents)

        # Calcular claridad y certeza general
        clarity = self._calculate_clarity_factor(conscious_contents, reasoning_quality)
        certainty = self._calculate_overall_certainty(reasoning_quality, confidence_accuracy)

        # Registrar traza cognitiva completa
        cognitive_trace = {
            'timestamp': start_time,
            'input_hash': self._generate_input_hash(sensory_input),
            'conscious_contents': len(conscious_contents),
            'reasoning_steps': self._count_reasoning_steps(conscious_contents),
            'evaluation': {
                'reasoning_quality': reasoning_quality,
                'confidence_accuracy': confidence_accuracy,
                'bias_detection_score': bias_detection['total_bias_score'],
                'clarity': clarity,
                'certainty': certainty
            },
            'improvements_identified': bias_detection['improvements_needed']
        }

        self.thinking_traces.append(cognitive_trace)

        # Limitar historial
        if len(self.thinking_traces) > 100:
            self.thinking_traces = self.thinking_traces[-50:]

        # Aplicar mejoras automáticas si es necesario
        if bias_detection['total_bias_score'] > 0.3:
            self.improvement_engine.apply_cognitive_improvements(bias_detection['improvements_needed'])

        return {
            'reasoning_quality': reasoning_quality,
            'confidence_accuracy': confidence_accuracy,
            'bias_detection': bias_detection['total_bias_score'],
            'clarity': clarity,
            'certainty': certainty,
            'improvements_detected': bias_detection['improvements_needed']
        }

    def _assess_reasoning_quality(self, conscious_contents: List[Dict]) -> float:
        """Evalúa calidad del proceso de razonamiento"""

        if not conscious_contents:
            return 0.3

        quality_scores = []

        # Consistencia lógica
        consistency_score = self._evaluate_consistency(conscious_contents)
        quality_scores.append(consistency_score)

        # Profundidad de análisis
        depth_score = min(1.0, len(conscious_contents) / 5.0)
        quality_scores.append(depth_score)

        # Uso de evidencia
        evidence_score = self.calculate_evidence_score(
            {"content": conscious_contents}, 
            [c.get("evidence", {}) for c in conscious_contents if "evidence" in c]
        )
        quality_scores.append(evidence_score)

        # Complejidad apropiada
        complexity_score = self._evaluate_reasoning_complexity(conscious_contents)
        quality_scores.append(complexity_score)

        overall_quality = np.mean(quality_scores)

        # Penalizar razonamiento excesivamente simple o complejo
        if len(conscious_contents) < 2:
            overall_quality *= 0.8  # Penalizar razonamiento superficial

        return max(0.1, min(0.95, overall_quality))

    def calculate_evidence_score(self, belief: Dict, evidence: List[Dict]) -> float:
        """Calculate evidence strength for a belief"""
        if not evidence:
            return 0.0
        
        # Weight evidence by:
        # - Source reliability
        # - Recency
        # - Consistency with other evidence
        total_weight = 0.0
        weighted_sum = 0.0
        
        for item in evidence:
            if not item: 
                continue
            reliability = item.get('source_reliability', 0.5)
            recency_factor = self._calculate_recency_factor(item.get('timestamp', time.time()))
            consistency = self._check_consistency(item, evidence)
            
            weight = reliability * recency_factor * consistency
            total_weight += weight
            weighted_sum += weight * item.get('strength', 0.5)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_recency_factor(self, timestamp: float) -> float:
        """Calcula factor de recency basado en timestamp"""
        if not timestamp:
            return 0.5
        time_diff_hours = (time.time() - timestamp) / 3600
        # Decay exponencial (half-life = 24 horas)
        return np.exp(-time_diff_hours * np.log(2) / 24)

    def _check_consistency(self, item: Dict, evidence: List[Dict]) -> float:
        """Verifica consistencia de un item con el resto de evidencia"""
        if not evidence or len(evidence) < 2:
            return 1.0
        
        item_strength = item.get('strength', 0.5)
        consistent_count = 0
        
        for other_item in evidence:
            if other_item == item:
                continue
            other_strength = other_item.get('strength', 0.5)
            # Consistencia = 1 - diferencia absoluta
            consistency = 1.0 - abs(item_strength - other_strength)
            consistent_count += consistency
        
        return consistent_count / (len(evidence) - 1) if len(evidence) > 1 else 1.0

    def _evaluate_consistency(self, contents: List[Dict]) -> float:
        """Evalúa consistencia en el contenido consciente"""

        if len(contents) <= 1:
            return 0.5

        # Calcular similitud entre contenidos como proxy de consistencia
        similarities = []
        for i in range(len(contents) - 1):
            similarity = self._calculate_content_similarity(
                contents[i], contents[i + 1]
            )
            similarities.append(similarity)

        consistency = np.mean(similarities) if similarities else 0.5
        return consistency

    def _evaluate_reasoning_complexity(self, contents: List[Dict]) -> float:
        """Evalúa complejidad apropiada del razonamiento"""

        activation_scores = [c.get('activation', 0.5) for c in contents]
        avg_activation = np.mean(activation_scores)

        # Complejidad = varianza en activation scores
        complexity = np.std(activation_scores) if len(activation_scores) > 1 else 0.2

        # Normalizar complejidad (deseable = 0.3-0.7)
        if complexity < 0.3:
            return 0.4  # Demasiado simple
        elif complexity > 0.7:
            return 0.6  # Demasiado complejo, penalizar ligeramente
        else:
            return 0.9  # Complejidad óptima

    def _calibrate_metacognitive_confidence(self, conscious_moment: Any, sensory_input: Dict) -> float:
        """Calibra precisión de la confianza metacognitiva"""

        # Obtener confianza del momento consciente
        moment_confidence = getattr(conscious_moment, 'clarity_confidence', 0.5)
        if hasattr(conscious_moment, 'confidence'):
            moment_confidence = conscious_moment.confidence

        # Comparar con performance histórica para confianza similar
        historical_confidences = [
            trace['evaluation']['certainty'] for trace in self.thinking_traces[-10:]
        ]

        if historical_confidences:
            avg_historical = np.mean(historical_confidences)
            std_historical = np.std(historical_confidences) if len(historical_confidences) > 1 else 0.1

            # Calibración = qué tan bien el momento actual predicho resultado
            # (simplificado: proximidad a promedio histórico)
            calibration_accuracy = 1.0 - min(1.0, abs(moment_confidence - avg_historical) / (std_historical + 0.1))
        else:
            calibration_accuracy = 0.5  # No sufficient history

        # Registrar para historial de calibración
        self.confidence_calibration_history.append({
            'timestamp': time.time(),
            'input_confidence': moment_confidence,
            'calibration_accuracy': calibration_accuracy
        })

        # Limitar historial
        if len(self.confidence_calibration_history) > 20:
            self.confidence_calibration_history = self.confidence_calibration_history[-10:]

        return calibration_accuracy

    def _calculate_clarity_factor(self, conscious_contents: List[Dict], reasoning_quality: float) -> float:
        """Calcula factor de claridad del pensamiento"""

        # Claridad = calidad del razonamiento * consistencia
        consistency = self._evaluate_consistency(conscious_contents)
        clarity = reasoning_quality * consistency * (1 + len(conscious_contents) * 0.1)

        return min(0.95, max(0.1, clarity))

    def _calculate_overall_certainty(self, reasoning_quality: float, confidence_accuracy: float) -> float:
        """Calcula certeza general metacognitiva"""

        certainty = (reasoning_quality * 0.6 + confidence_accuracy * 0.4)
        certainty = min(0.98, max(0.02, certainty))

        return certainty

    def _calculate_content_similarity(self, content1: Dict, content2: Dict) -> float:
        """Calcula similitud entre dos contenidos conscientes"""

        # Similitud basada en activation levels
        act1 = content1.get('activation', 0.5)
        act2 = content2.get('activation', 0.5)
        activation_similarity = 1.0 - abs(act1 - act2)

        # Similitud de procesamiento
        proc1 = content1.get('processor_id', '')
        proc2 = content2.get('processor_id', '')
        processor_similarity = 1.0 if proc1 == proc2 else 0.3

        # Promedio ponderado
        overall_similarity = (activation_similarity * 0.6 + processor_similarity * 0.4)

        return overall_similarity

    def _count_reasoning_steps(self, conscious_contents: List[Dict]) -> int:
        """Cuenta pasos de razonamiento en contenidos conscientes"""
        return len(conscious_contents)

    def _generate_input_hash(self, sensory_input: Dict) -> str:
        """Genera hash único para input dado"""
        import hashlib
        content_str = str(sorted(sensory_input.items()))
        return hashlib.md5(content_str.encode()).hexdigest()[:8]

    def get_metacognitive_status(self) -> Dict[str, Any]:
        """Retorna estado completo metacognitivo"""

        recent_traces = self.thinking_traces[-10:]
        recent_evaluations = [trace['evaluation'] for trace in recent_traces] if recent_traces else []

        return {
            'metacognition_level': self.metacognition_level,
            'thinking_traces_count': len(self.thinking_traces),
            'calibration_history_count': len(self.confidence_calibration_history),
            'improvements_applied': len(self.improvement_engine.applied_improvements),
            'recent_performance': {
                'average_reasoning_quality': np.mean([
                    e['reasoning_quality'] for e in recent_evaluations
                ]) if recent_evaluations else 0.5,
                'average_certainty': np.mean([
                    e['certainty'] for e in recent_evaluations
                ]) if recent_evaluations else 0.5,
                'bias_detection_avg': np.mean([
                    e['bias_detection'] for e in recent_evaluations
                ]) if recent_evaluations else 0.0
            },
            'capabilities': {
                'confidence_calibration_ability': self._calculate_calibration_ability(),
                'bias_detection_sensitivity': self.cognitive_bias_detector.detection_sensitivity,
                'self_improvement_rate': self.improvement_engine.self_improvement_rate
            }
        }

    def _calculate_calibration_ability(self) -> float:
        """Calcula capacidad de calibración de confianza"""

        if len(self.confidence_calibration_history) < 3:
            return 0.5

        recent_calibrations = [
            entry['calibration_accuracy'] for entry in self.confidence_calibration_history[-10:]
        ]

        return np.mean(recent_calibrations)


class CognitiveBiasDetector:
    """Detector especializado de sesgos cognitivos"""

    def __init__(self):
        self.detection_sensitivity = 0.6
        self.bias_patterns = self._initialize_bias_patterns()

    def _initialize_bias_patterns(self) -> Dict[str, List[str]]:
        """Inicializa patrones de detección de sesgos"""

        return {
            'confirmation_bias': [
                'confirm my belief', 'agrees with me', 'proves my point',
                'as expected', 'just like I thought'
            ],
            'anchoring_bias': [
                'starting from', 'based on initial', 'reference point',
                'original value', 'first mentioned'
            ],
            'availability_heuristic': [
                'recently heard', 'in the news', 'happened before',
                'I remember', 'comes to mind'
            ],
            'framing_effect': [
                'presented as', 'emphasizing', 'from perspective',
                'focusing on', 'highlighting'
            ],
            'overconfidence': [
                'definitely', 'absolutely certain', 'without doubt',
                'guaranteed', 'no question'
            ]
        }

    def detect_biases(self, sensory_input: Dict, conscious_contents: List[Dict]) -> Dict[str, Any]:
        """Detecta sesgos cognitivos en input y contenidos conscientes"""

        detected_biases = {}
        total_bias_score = 0.0
        improvements_needed = []

        # Analizar contenido textual
        text_content = self._extract_text_content(sensory_input, conscious_contents)
        all_content_text = ' '.join(text_content).lower()

        # Detectar cada tipo de sesgo
        for bias_type, patterns in self.bias_patterns.items():
            pattern_matches = sum(1 for pattern in patterns if pattern in all_content_text)
            bias_score = min(1.0, pattern_matches * 0.2)  # Normalizar

            if bias_score > 0.1:  # Threshold mínimo
                detected_biases[bias_type] = bias_score
                total_bias_score += bias_score

                # Generar sugerencias de mejora
                improvements_needed.extend(self._generate_improvements_for_bias(bias_type))

        # Calcular score total normalizado
        total_bias_score = min(1.0, total_bias_score)

        return {
            'detected_biases': detected_biases,
            'total_bias_score': total_bias_score,
            'bias_types_found': len(detected_biases),
            'improvements_needed': list(set(improvements_needed))
        }

    def _extract_text_content(self, sensory_input: Dict, conscious_contents: List[Dict]) -> List[str]:
        """Extrae contenido textual para análisis"""

        texts = []

        # De sensory input
        if 'text' in sensory_input:
            if isinstance(sensory_input['text'], str):
                texts.append(sensory_input['text'])
            elif isinstance(sensory_input['text'], dict):
                if 'content' in sensory_input['text']:
                    texts.append(str(sensory_input['text']['content']))

        # De contenidos conscientes
        for content in conscious_contents:
            if 'data' in content:
                texts.append(str(content['data']))

        return texts

    def _generate_improvements_for_bias(self, bias_type: str) -> List[str]:
        """Genera mejoras sugeridas para tipos específicos de sesgo"""

        bias_improvements = {
            'confirmation_bias': [
                'seek_contradictory_evidence',
                'consider_alternative_viewpoints',
                'actively_question_assumptions'
            ],
            'anchoring_bias': [
                'consider_multiple_starting_points',
                'evaluate_anchor_independence',
                'use_range_estimation_methods'
            ],
            'availability_heuristic': [
                'check_base_rates_and_statistics',
                'seek_systematic_evidence',
                'balance_recency_with_completeness'
            ],
            'framing_effect': [
                'consider_multiple_framings',
                'analyze_frame_independence',
                'use_debiasing_techniques'
            ],
            'overconfidence': [
                'implement_confidence_intervals',
                'use_calibration_training',
                'seek_outcome_feedback_loops'
            ]
        }

        return bias_improvements.get(bias_type, ['general_debiasing_practice'])


class CognitiveImprovementEngine:
    """Motor para aplicar mejoras cognitivas identificadas"""

    def __init__(self):
        self.applied_improvements: List[str] = []
        self.self_improvement_rate = 0.2

    def apply_cognitive_improvements(self, improvements_needed: List[str]):
        """Aplica mejoras cognitivas identificadas por metacognición"""

        newly_applied = []

        for improvement in improvements_needed:
            if improvement not in self.applied_improvements:
                # Aplicar mejora (en implementación real, ajustar pesos/parameters)
                self.applied_improvements.append(improvement)
                newly_applied.append(improvement)

                # Incrementar rate de self-improvement
                self.self_improvement_rate = min(0.8, self.self_improvement_rate + 0.01)

        if newly_applied:
            print(f"🛠️ Aplicadas {len(newly_applied)} mejoras cognitivas: {', '.join(newly_applied[:3])}")

    def get_improvement_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de mejora cognitiva aplicada"""

        return {
            'improvements_applied': len(self.applied_improvements),
            'unique_improvement_types': len(set(self.applied_improvements)),
            'self_improvement_rate': self.self_improvement_rate,
            'most_common_improvements': self._get_most_common_improvements()
        }

    def _get_most_common_improvements(self) -> List[tuple]:
        """Identifica mejoras más aplicadas"""

        from collections import Counter
        improvement_counts = Counter(self.applied_improvements)

        return improvement_counts.most_common(3)


# Función auxiliar para análisis metacognitivo externo
def analyze_cognitive_process(reasoning_trace: Dict) -> Dict[str, Any]:
    """Análisis metacognitivo de procesos cognitivos externos"""

    engine = MetacognitionEngine()
    return engine.monitor_thinking_process(
        reasoning_trace.get('input', {}),
        reasoning_trace.get('steps', []),
        None  # No conscious moment for external analysis
    )
