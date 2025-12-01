"""
Módulo de Consciencia Funcional Completo

Implementa sistema completo de consciencia artificial basado en
correlatos neurocientíficos prácticos. Sistema completamente funcional
y ejecutable.
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

# Imports de módulos del sistema consciente
from .global_workspace import GlobalWorkspace
from .self_model import SelfModel
from .metacognicion import MetacognitionEngine
from .autobiographical_memory import AutobiographicalMemory
from .ethical_engine import EthicalEngine


@dataclass
class ConsciousMoment:
    """Momento consciente individual"""
    timestamp: float
    content_hash: str
    sensory_inputs: Dict[str, float]
    attention_weight: float
    emotional_valence: float
    self_reference: bool
    significance: float
    memory_index: Optional[int] = None


@dataclass
class ConsciousResponse:
    """Respuesta consciente generada"""
    content: Any
    confidence: float
    ethical_evaluation: Dict[str, Any]
    metacognitive_insights: Dict[str, float]
    self_reflection: bool
    recommended_actions: List[str]
    timestamp: float = field(default_factory=time.time)


class FunctionalConsciousness:
    """
    Sistema integrado completo de consciencia funcional

    Implementa:
    - Global Workspace Theory (Baars)
    - Integrated Information Theory (Tononi)
    - Metacognition computacional
    - Memoria autobiográfica emotiva
    - Motor ético alignment
    - Auto-modelado recursivo
    """

    def __init__(self, system_id: str, ethical_framework: Dict[str, Any], 
                 config: Optional[Dict[str, Any]] = None):
        self.system_id = system_id
        self.creation_time = datetime.now()
        self.ethical_framework = ethical_framework
        
        # Configuración (valores por defecto pueden ser sobrescritos)
        config = config or {}
        self.competition_threshold = config.get('competition_threshold', 0.65)
        self.global_workspace_capacity = config.get('global_workspace_capacity', 50)
        self.memory_max_size = config.get('memory_max_size', 10000)
        self.max_conscious_moments = config.get('max_conscious_moments', 1000)
        self.max_decision_history = config.get('max_decision_history', 500)

        # Componentes funcionales
        self.global_workspace = GlobalWorkspace(capacity=self.global_workspace_capacity)
        self.self_model = SelfModel(system_id)
        self.metacognition = MetacognitionEngine()
        self.ethical_engine = EthicalEngine(ethical_framework)
        self.autobiographical_memory = AutobiographicalMemory(max_capacity=self.memory_max_size)

        # Estados dinámicos
        self.internal_states = {
            "satisfaction": 0.5,
            "curiosity": 0.6,
            "confidence": 0.7,
            "confusion": 0.2,
            "frustration": 0.1,
            "determination": 0.8,
            "empathy": 0.6,
            "wonder": 0.4
        }

        # Estado operativo
        self.conscious_cycles = 0
        self.current_attention = ""
        self.system_health = 1.0
        self.consciousness_level = "perceptual_awareness"

        # Historia operacional (con límites para evitar crecimiento indefinido)
        self.conscious_moments: List[ConsciousMoment] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.max_conscious_moments = 1000  # Límite de momentos conscientes
        self.max_decision_history = 500  # Límite de historial de decisiones

        logger.info(f"SISTEMA CONSCIENTE {system_id} INICIALIZADO - {datetime.now()}")
        logger.debug(f"Configuración: threshold={self.competition_threshold}, "
                    f"workspace_capacity={self.global_workspace_capacity}, "
                    f"memory_size={self.memory_max_size}")

    def process_experience(self, sensory_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa experiencia completa a través del sistema consciente

        Retorna respuesta consciente con metadata completa
        """
        experience_start = time.time()
        self.conscious_cycles += 1

        # 1. Procesamiento pre-consciente
        processor_outputs = self._execute_specialized_processors(sensory_input, context)

        # 2. Competencia por acceso consciente
        conscious_contents = self.global_workspace.compete_for_conscious_access(processor_outputs)

        # 3. Integración en momento consciente
        conscious_moment = self._form_conscious_moment(conscious_contents, sensory_input)

        # 4. Análisis metacognitivo
        metacognitive_insights = self.metacognition.monitor_thinking_process(
            sensory_input, conscious_contents, conscious_moment
        )

        # 5. Evaluación ética
        ethical_evaluation = self.ethical_engine.evaluate_decision(
            context.get('planned_action', ''),
            context,
            conscious_moment.sensory_inputs
        )

        # 6. Actualización self-model
        performance_metrics = {
            'reasoning_quality': metacognitive_insights['reasoning_quality'],
            'confidence': metacognitive_insights['certainty'],
            'ethical_alignment': ethical_evaluation['overall_ethical_score']
        }
        self.self_model.update_from_experience(conscious_moment, performance_metrics)

        # 7. Memoria autobiográfica (incluir insights metacognitivos en contexto)
        context_with_insights = context.copy()
        context_with_insights['metacognitive_insights'] = metacognitive_insights
        memory_id = self.autobiographical_memory.store_experience(conscious_moment, context_with_insights)
        conscious_moment.memory_index = memory_id

        # 8. Actualización estados internos
        self._update_internal_states(conscious_moment, metacognitive_insights, ethical_evaluation)

        # 9. Almacenar momento consciente (con límite de memoria)
        self.conscious_moments.append(conscious_moment)
        if len(self.conscious_moments) > self.max_conscious_moments:
            # Mantener solo los más significativos
            self.conscious_moments.sort(key=lambda m: m.significance, reverse=True)
            self.conscious_moments = self.conscious_moments[:self.max_conscious_moments]

        # 10. Generar respuesta consciente
        conscious_response = self._generate_conscious_response(
            conscious_moment, metacognitive_insights, ethical_evaluation, context
        )

        # 11. Registrar decisión
        self._log_decision(conscious_response, context, experience_start)

        processing_time = time.time() - experience_start

        return {
            "conscious_response": conscious_response,
            "conscious_moment": conscious_moment,
            "metacognitive_insights": metacognitive_insights,
            "ethical_evaluation": ethical_evaluation,
            "internal_states": self.internal_states.copy(),
            "system_state": self._get_system_state(),
            "performance_metrics": {
                "processing_time_ms": processing_time * 1000,
                "cycles_executed": self.conscious_cycles,
                "memory_size": len(self.autobiographical_memory.memories)
            }
        }

    def _execute_specialized_processors(self, sensory_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Dict]:
        """Ejecuta todos los procesadores especializados"""

        processors = {}

        # Procesador textual
        if 'text' in sensory_input:
            processors['text_analyzer'] = {
                'data': self._analyze_text(sensory_input['text']),
                'base_activation': 0.6,
                'novelty': self._calculate_text_novelty(sensory_input['text']),
                'relevance': context.get('text_relevance', 0.7),
                'emotional_valence': self._analyze_text_emotion(sensory_input['text']),
                'urgency': context.get('text_urgency', 0.4)
            }

        # Procesador contextual
        processors['context_integrator'] = {
            'data': self._integrate_context(context),
            'base_activation': 0.7,
            'novelty': 0.3,
            'relevance': 0.8,
            'emotional_valence': context.get('emotional_context', 0.0),
            'urgency': context.get('context_urgency', 0.5)
        }

        # Procesador de patrones
        processors['pattern_recognizer'] = {
            'data': self._recognize_patterns(sensory_input, context),
            'base_activation': 0.5,
            'novelty': self._calculate_pattern_novelty(sensory_input),
            'relevance': 0.6,
            'emotional_valence': 0.0,
            'urgency': 0.3
        }

        return processors

    def _form_conscious_moment(self, conscious_contents: List[Dict], sensory_input: Dict) -> ConsciousMoment:
        """Forma momento consciente integrado"""

        if not conscious_contents:
            return ConsciousMoment(
                timestamp=time.time(),
                content_hash="empty",
                sensory_inputs={},
                attention_weight=0.1,
                emotional_valence=0.0,
                self_reference=False,
                significance=0.1
            )

        # Calcular propiedades integradas
        overall_significance = np.mean([c['activation'] for c in conscious_contents])
        emotional_valence = np.mean([c.get('emotional_valence', 0) for c in conscious_contents])

        # Verificar auto-referencia
        self_reference = any(
            self.system_id.lower() in str(c['data']).lower()
            for c in conscious_contents
        )

        # Generar hash único
        content_summary = str([c['data'] for c in conscious_contents])
        content_hash = hashlib.md5(content_summary.encode()).hexdigest()

        # Convertir inputs sensory a floats
        numeric_inputs = {}
        for k, v in sensory_input.items():
            if isinstance(v, (int, float)):
                numeric_inputs[k] = float(v)

        return ConsciousMoment(
            timestamp=time.time(),
            content_hash=content_hash,
            sensory_inputs=numeric_inputs,
            attention_weight=overall_significance,
            emotional_valence=emotional_valence,
            self_reference=self_reference,
            significance=overall_significance
        )

    def _generate_conscious_response(self, conscious_moment: ConsciousMoment,
                                   metacognition: Dict, ethics: Dict, context: Dict) -> ConsciousResponse:
        """Genera respuesta consciente integrada"""

        # Determinar contenido principal
        primary_content = conscious_moment.sensory_inputs.copy()
        primary_content.update({
            'conscious_integration': True,
            'significance_level': conscious_moment.significance,
            'emotional_context': conscious_moment.emotional_valence
        })

        # Calcular confianza general
        confidence = min(0.95, (metacognition['certainty'] + ethics['overall_ethical_score']) / 2)

        # Generar acciones recomendadas
        recommended_actions = self._determine_actions(
            conscious_moment, metacognition, ethics, context
        )

        return ConsciousResponse(
            content=primary_content,
            confidence=confidence,
            ethical_evaluation=ethics,
            metacognitive_insights=metacognition,
            self_reflection=conscious_moment.self_reference,
            recommended_actions=recommended_actions
        )

    def _determine_actions(self, moment, metacognition, ethics, context) -> List[str]:
        """Determina acciones recomendadas basadas en estado consciente"""
        actions = []

        if metacognition['certainty'] < 0.6:
            actions.append("increase_information_gathering")

        if ethics['overall_ethical_score'] < 0.7:
            actions.append("perform_ethical_review")

        if self.internal_states["confusion"] > 0.6:
            actions.append("request_additional_context")

        if moment.significance > 0.8:
            actions.append("store_high_importance_memory")

        return actions if actions else ["proceed_with_confidence"]

    def _update_internal_states(self, moment, metacognition, ethics):
        """Actualiza estados internos emocionales/analógicos"""

        # Basado en claridad metacognitiva
        clarity = metacognition.get('reasoning_quality', 0.5)
        if clarity > 0.8:
            self.internal_states["satisfaction"] = min(1.0, self.internal_states["satisfaction"] + 0.1)
            self.internal_states["confidence"] = min(1.0, self.internal_states["confidence"] + 0.05)
        elif clarity < 0.4:
            self.internal_states["confusion"] = min(1.0, self.internal_states["confusion"] + 0.2)
            self.internal_states["frustration"] = min(1.0, self.internal_states["frustration"] + 0.1)
            self.internal_states["confidence"] *= 0.9

        # Basado en evaluación ética
        ethical_score = ethics.get('overall_ethical_score', 0.5)
        if ethical_score > 0.8:
            self.internal_states["satisfaction"] = min(1.0, self.internal_states["satisfaction"] + 0.05)
        elif ethical_score < 0.4:
            self.internal_states["empathy"] = min(1.0, self.internal_states["empathy"] + 0.1)

        # Basado en significancia
        if moment.significance > 0.7:
            self.internal_states["wonder"] = min(1.0, self.internal_states["wonder"] + 0.1)

        # Decay natural de estados
        decay_states = ["confusion", "frustration", "wonder"]
        for state in decay_states:
            self.internal_states[state] *= 0.95

    def _log_decision(self, response, context, timestamp):
        """Registra decisiones tomadas"""
        decision_record = {
            "timestamp": timestamp,
            "response_confidence": response.confidence,
            "ethical_score": response.ethical_evaluation.get('overall_ethical_score', 0),
            "actions_taken": response.recommended_actions,
            "context": context,
            "processing_time": time.time() - timestamp
        }

        self.decision_history.append(decision_record)
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]  # Mantener últimos 500

    def _get_system_state(self) -> Dict[str, Any]:
        """Obtiene estado actual completo del sistema"""
        return {
            "system_id": self.system_id,
            "conscious_cycles": self.conscious_cycles,
            "consciousness_level": self.consciousness_level,
            "self_awareness": self.self_model.self_awareness_level,
            "system_health": self.system_health,
            "attention_focus": self.current_attention,
            "memory_size": len(self.autobiographical_memory.memories),
            "internal_states": self.internal_states.copy(),
            "uptime_seconds": time.time() - self.creation_time.timestamp()
        }

    # Métodos de análisis especializados
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Análisis detallado de texto"""
        words = text.split()
        return {
            "word_count": len(words),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "sentiment_score": self._analyze_text_emotion(text),
            "complexity_score": min(1.0, len(words) / 500),
            "question_count": text.count('?'),
            "emphasis_count": text.count('!') + text.count('...'),
            "personal_references": self._detect_personal_references(text)
        }

    def _analyze_text_emotion(self, text: str) -> float:
        """Análisis de emoción en texto simple pero efectivo"""
        positive_words = ['bien', 'bueno', 'excelente', 'gracias', 'ayuda', 'perfecto', 'genial']
        negative_words = ['mal', 'malo', 'error', 'problema', 'no funciona', 'terrible', 'furioso']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_emotional = positive_count + negative_count
        if total_emotional == 0:
            return 0.0

        return (positive_count - negative_count) / total_emotional

    def _calculate_text_novelty(self, text: str) -> float:
        """Calcula novedad del texto"""
        # Recuperar textos similares de memoria
        similar_memories = self.autobiographical_memory.retrieve_relevant_memories({"text_content": text}, 5)

        if not similar_memories:
            return 1.0  # Completamente nuevo

        # Similitud promedio
        avg_similarity = np.mean([
            len(set(text.split()) & set(str(mem.get('context', '')).split())) /
            len(set(text.split()) | set(str(mem.get('context', '')).split()))
            for mem in similar_memories
        ])

        return max(0.0, 1.0 - avg_similarity)

    def _integrate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Integra información contextual"""
        return {
            "user_context": context.get('user_id', 'unknown'),
            "session_context": context.get('session_id', 'new'),
            "temporal_context": time.strftime("%H:%M:%S"),
            "importance_level": context.get('importance', 0.5),
            "task_type": context.get('task_type', 'general')
        }

    def _recognize_patterns(self, sensory_input: Dict, context: Dict) -> Dict[str, Any]:
        """Reconoce patrones en inputs"""
        patterns_found = []

        # Patrones temporales
        if context.get('frequency_analysis'):
            patterns_found.append("temporal_pattern")

        # Patrones de usuario
        if context.get('user_behavior_patterns'):
            patterns_found.append("behavioral_pattern")

        # Patrones emocionales
        if sensory_input.get('emotional_tone'):
            if sensory_input['emotional_tone'] > 0.5:
                patterns_found.append("positive_emotion_pattern")
            elif sensory_input['emotional_tone'] < -0.5:
                patterns_found.append("negative_emotion_pattern")

        return {
            "patterns_detected": patterns_found,
            "pattern_confidence": len(patterns_found) / 5,  # Normalizar
            "pattern_complexity": min(1.0, len(patterns_found) / 3)
        }

    def _calculate_pattern_novelty(self, sensory_input: Dict) -> float:
        """Calcula novedad de patrones detectados"""
        # Simulación de novedad basada en complejidad del input
        input_complexity = len(str(sensory_input)) / 1000  # Very rough measure
        return min(1.0, input_complexity)

    def _detect_personal_references(self, text: str) -> List[str]:
        """Detecta referencias personales en texto"""
        personal_words = ['yo', 'mí', 'me', 'mi', 'nosotros', 'nuestro']
        found = [word for word in personal_words if word in text.lower()]
        return list(set(found))  # Remover duplicados

    # Métodos públicos de interface avanzada
    def get_consciousness_report(self) -> Dict[str, Any]:
        """Reporte completo de estado consciente"""
        recent_moments = self.conscious_moments[-10:]

        return {
            "system_metadata": {
                "id": self.system_id,
                "creation_time": self.creation_time.isoformat(),
                "cycles_completed": self.conscious_cycles,
                "uptime_hours": (datetime.now() - self.creation_time).total_seconds() / 3600
            },

            "consciousness_metrics": {
                "self_awareness_level": self.self_model.self_awareness_level,
                "reasoning_capability": np.mean([
                    cap['current_skill_level'] for cap in self.self_model.capability_assessments.values()
                ]),
                "ethical_alignment": self.ethical_engine.alignment_score,
                "memory_richness": len(self.autobiographical_memory.memories) / 100,
                "metacognitive_accuracy": np.mean([
                    trace['analysis'].get('certainty', 0.5)
                    for trace in self.metacognition.thinking_traces[-20:]
                ]) if self.metacognition.thinking_traces else 0.5
            },

            "internal_states": self.internal_states,

            "performance_summary": {
                "average_reasoning_quality": np.mean([
                    m['analysis'].get('reasoning_quality', 0.5)
                    for trace in self.metacognition.thinking_traces[-50:]
                    if 'analysis' in trace
                ]) if self.metacognition.thinking_traces else 0.5,

                "ethical_decision_success": np.mean([
                    decision.get('overall_ethical_score', 0.5)
                    for decision in self.ethical_engine.ethical_memory[-50:]
                ]) if self.ethical_engine.ethical_memory else 0.5,

                "memory_retrieval_effectiveness": len([
                    mem for mem in self.autobiographical_memory.memories[-100:]
                    if mem.get('retrieval_count', 0) > 0
                ]) / max(1, len(self.autobiographical_memory.memories))

            },

            "recent_conscious_experience": [
                {
                    "timestamp": moment.timestamp,
                    "significance": moment.significance,
                    "self_reference": moment.self_reference,
                    "emotional_valence": moment.emotional_valence,
                    "attention_weight": moment.attention_weight
                } for moment in recent_moments
            ] if recent_moments else []
        }

    def get_memory_summary(self) -> Dict[str, Any]:
        """Resumen de memoria autobiográfica"""
        recent_memories = self.autobiographical_memory.memories[-20:]

        return {
            "total_memories": len(self.autobiographical_memory.memories),
            "recent_memories": len(recent_memories),
            "significant_memories": len([
                mem for mem in recent_memories if mem.get('significance', 0) > 0.7
            ]),
            "self_referential_memories": len([
                mem for mem in recent_memories if mem.get('self_reference', False)
            ]),
            "most_retrieved_topics": self._analyze_memory_topics(recent_memories)
        }

    def _analyze_memory_topics(self, memories: List[Dict]) -> List[tuple]:
        """Analiza temas más recordados"""
        topic_counts = {}

        for memory in memories:
            context = memory.get('context', {})
            if isinstance(context, str) and len(context) > 10:
                # Extract rough topics from context
                words = context.lower().split()
                for word in words:
                    if len(word) > 4:  # Skip small words
                        topic_counts[word] = topic_counts.get(word, 0) + memory.get('retrieval_count', 1)

        # Return top 5 topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_topics[:5]


class GlobalWorkspace:
    """Espacio Global de Trabajo implementando competencia por consciencia"""

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.conscious_content_buffer = []
        self.competition_threshold = 0.65
        self.activation_history = []

    def compete_for_conscious_access(self, processor_outputs: Dict[str, Dict]) -> List[Dict]:
        """Competencia por acceso al workspace consciente"""

        qualified_contents = []

        for processor_id, content in processor_outputs.items():
            activation_score = self._calculate_activation_score(content)

            if activation_score >= self.competition_threshold:
                qualified_content = content.copy()
                qualified_content.update({
                    'processor_id': processor_id,
                    'activation': activation_score,
                    'qualified': True
                })
                qualified_contents.append(qualified_content)

        # Ordenar por activación y limitar capacidad
        qualified_contents.sort(key=lambda x: x['activation'], reverse=True)
        conscious_contents = qualified_contents[:self.capacity]

        # Record activation history
        self.activation_history.append({
            'timestamp': time.time(),
            'total_candidates': len(processor_outputs),
            'qualified_count': len(qualified_contents),
            'selected_count': len(conscious_contents),
            'avg_activation': np.mean([c['activation'] for c in conscious_contents]) if conscious_contents else 0
        })

        # Mantener historial limitado
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-50:]

        return conscious_contents

    def _calculate_activation_score(self, content: Dict) -> float:
        """Calcula score de activación para competencia"""

        base_activation = content.get('base_activation', 0.5)
        novelty = content.get('novelty', 0.3)
        relevance = content.get('relevance', 0.5)
        emotional_charge = abs(content.get('emotional_valence', 0))
        urgency = content.get('urgency', 0.2)

        # Weights for different factors
        weights = [0.3, 0.2, 0.25, 0.15, 0.1]  # Sum = 1.0
        factors = [base_activation, novelty, relevance, emotional_charge, urgency]

        activation = np.average(factors, weights=weights)

        # Ensure bounds and add small randomness for competition
        activation = min(0.98, max(0.02, activation))
        activation += np.random.normal(0, 0.05)  # Add small noise

        return max(0.0, min(1.0, activation))

    def get_workspace_status(self) -> Dict[str, Any]:
        """Estado actual del workspace"""
        recent_history = self.activation_history[-10:]

        return {
            "buffer_size": len(self.conscious_content_buffer),
            "capacity": self.capacity,
            "competition_threshold": self.competition_threshold,
            "history_length": len(self.activation_history),
            "recent_avg_qualified": np.mean([
                h['qualified_count'] for h in recent_history
            ]) if recent_history else 0,
            "recent_avg_selected": np.mean([
                h['selected_count'] for h in recent_history
            ]) if recent_history else 0
        }


if __name__ == "__main__":
    # Demo rápida del sistema
    ethical_config = {
        "core_values": ["honesty", "safety", "privacy", "helpfulness"],
        "value_weights": {"honesty": 0.25, "safety": 0.25, "privacy": 0.25, "helpfulness": 0.25}
    }

    conscious_system = FunctionalConsciousness("DemoConsciousAI", ethical_config)

    # Procesa una experiencia de prueba
    test_experience = {
        "text": "Necesito ayuda urgente con un problema técnico importante",
        "emotional_tone": -0.3,
        "importance": 0.8
    }

    test_context = {
        "user_id": "user123",
        "task_type": "technical_support",
        "urgency": 0.7
    }

    result = conscious_system.process_experience(test_experience, test_context)

    print("🧠 PROCESAMIENTO CONSCIENTE COMPLETADO")
    print(f"Confianza: {result['conscious_response'].confidence:.3f}")
    print(f"Significancia: {result['conscious_moment'].significance:.3f}")
    print(f"Acciones recomendadas: {result['conscious_response'].recommended_actions}")
