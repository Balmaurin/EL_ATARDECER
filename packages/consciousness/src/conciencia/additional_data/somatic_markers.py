"""
SOMATIC MARKERS SYSTEM - Marcadores somáticos de Damasio

Implementa el sistema de marcadores somáticos basado en la teoría de Antonio Damasio:
- Asociación de estados corporales con decisiones
- "Marcas" somáticas que guían la toma de decisiones
- Registros emocionales inconscientes
- Feedforward emocional en elección racional
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SomaticMarker:
    """Marcador somático individual con estado emocional asociado"""
    marker_id: str
    trigger_type: str  # 'situation', 'option', 'consequence'
    emotional_state: str  # 'positive', 'negative', 'neutral'
    intensity: float  # 0-1
    confidence: float  # 0-1, cuánto confía el sistema en este marcador
    created_timestamp: float = field(default_factory=time.time)
    activation_count: int = 0
    last_activated: float = 0.0

    # Estados corporales asociados
    somatic_signals: Dict[str, float] = field(default_factory=dict)

    def activate(self, current_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Activa el marcador somático y retorna estado emocional"""
        self.activation_count += 1
        self.last_activated = time.time()

        # Estado emocional generado por el marcador
        emotional_response = self._generate_emotional_response(current_context)

        return {
            'marker_id': self.marker_id,
            'emotional_response': emotional_response,
            'intensity': self.intensity * self.confidence,  # intensidad efectiva
            'confidence': self.confidence,
            'somatic_signals': self.somatic_signals
        }

    def _generate_emotional_response(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Genera respuesta emocional específica del marcador"""
        base_emotions = {
            'positive': {
                'joy': 0.8, 'hope': 0.6, 'confidence': 0.7,
                'relief': 0.4, 'anticipation': 0.5
            },
            'negative': {
                'fear': 0.8, 'anxiety': 0.7, 'dread': 0.6,
                'guilt': 0.4, 'regret': 0.5, 'disappointment': 0.6
            },
            'neutral': {
                'calm': 0.7, 'indifference': 0.5, 'detachment': 0.4
            }
        }

        emotions = base_emotions.get(self.emotional_state, base_emotions['neutral'])
        scaled_emotions = {k: v * self.intensity for k, v in emotions.items()}

        return {
            'primary_emotion': self.emotional_state,
            'emotional_components': scaled_emotions,
            'valence': 1.0 if self.emotional_state == 'positive' else (-1.0 if self.emotional_state == 'negative' else 0.0),
            'arousal_level': self.intensity * 0.8,
            'somatic_intensity': self.intensity * 0.6
        }


class SomaticMarkersSystem:
    """
    Sistema completo de marcadores somáticos (Damasio)
    Asocia estados corporales/emocionales con situaciones y decisiones
    """

    def __init__(self):
        # Colección de marcadores organizados por tipo de trigger
        self.markers = {
            'situational': {},    # triggers: situaciones específicas
            'option_based': {},   # triggers: opciones de decisión
            'outcome_based': {}   # triggers: consecuencias observadas
        }

        # Memoria de asociaciones aprendidas
        self.somatic_memory = defaultdict(dict)
        self.learning_rate = 0.1

        # Estado actual del sistema
        self.current_somatic_state = {}
        self.emotional_bias = 0.0  # bias emocional general (-1 a 1)

        print("🧠 SISTEMA DE MARCADORES SOMÁTICOS INICIALIZADO")

    def register_marker(self, trigger: str, emotional_state: str, intensity: float,
                       somatic_signals: Dict[str, float] = None,
                       marker_type: str = 'situational') -> str:
        """
        Registra un nuevo marcador somático

        Args:
            trigger: Situación/opción que activa el marcador
            emotional_state: Estado emocional ('positive', 'negative', 'neutral')
            intensity: Intensidad del marcador (0-1)
            somatic_signals: Señales corporales específicas asociadas
            marker_type: Tipo de marcador ('situational', 'option_based', 'outcome_based')

        Returns:
            ID único del marcador registrado
        """
        marker_id = f"{marker_type}_{trigger}_{int(time.time())}"

        marker = SomaticMarker(
            marker_id=marker_id,
            trigger_type=marker_type,
            emotional_state=emotional_state,
            intensity=intensity,
            confidence=0.5,  # confianza inicial moderada
            somatic_signals=somatic_signals or self._generate_default_somatic_signals(emotional_state, intensity)
        )

        self.markers[marker_type][trigger] = marker

        # Asociar con memoria aprendida
        self._associate_with_memory(trigger, marker)

        return marker_id

    def _generate_default_somatic_signals(self, emotion: str, intensity: float) -> Dict[str, float]:
        """Genera señales somáticas por defecto basadas en emoción"""
        signal_patterns = {
            'positive': {
                'heart_rate_increase': 0.3, 'muscle_relaxation': 0.6,
                'breathing_ease': 0.5, 'energy_boost': 0.4
            },
            'negative': {
                'heart_rate_increase': 0.8, 'muscle_tension': 0.7,
                'breathing_difficulty': 0.6, 'energy_drain': 0.5,
                'gut_discomfort': 0.4
            },
            'neutral': {
                'heart_rate_normal': 0.5, 'muscle_normal': 0.5,
                'breathing_normal': 0.5
            }
        }

        base_signals = signal_patterns.get(emotion, signal_patterns['neutral'])
        return {k: v * intensity for k, v in base_signals.items()}

    def get_somatic_feedback(self, situation: str, options: List[str] = None) -> Dict[str, Any]:
        """
        Proporciona feedback somático para una situación dada

        Args:
            situation: Situación actual
            options: Lista de opciones disponibles (para decisiones)

        Returns:
            Feedback somático completo incluyendo bias emocional
        """
        # Activación de marcadores situacionales
        situation_markers = []
        if situation in self.markers['situational']:
            situation_markers.append(self.markers['situational'][situation])

        # Si hay opciones, activar marcadores de opción y predicción de outcome
        option_markers = {}
        predicted_outcomes = {}

        if options:
            for option in options:
                # Marcadores para esta opción específica
                option_marker = self.markers['option_based'].get(option)
                if option_marker:
                    option_markers[option] = option_marker.activate()

                # Predicción de resultado basada en memoria aprendida
                predicted_outcomes[option] = self._predict_outcome_emotion(situation, option)

        # Combinar todas las señales somáticas
        combined_feedback = self._combine_somatic_signals(
            situation_markers, option_markers, predicted_outcomes
        )

        # Actualizar bias emocional general
        self.emotional_bias = self._compute_emotional_bias(combined_feedback)

        combined_feedback['emotional_bias'] = self.emotional_bias
        combined_feedback['situation'] = situation
        combined_feedback['available_options'] = options or []

        return combined_feedback

    def _predict_outcome_emotion(self, situation: str, option: str) -> Dict[str, Any]:
        """Predice emoción de resultado basada en memoria aprendida"""
        # Buscar experiencias similares en memoria
        similar_experiences = self._find_similar_experiences(situation, option)

        if not similar_experiences:
            return {'predicted_emotion': 'unknown', 'confidence': 0.0}

        # Promediar emociones de experiencias similares
        positive_count = sum(1 for exp in similar_experiences if exp.get('valence', 0) > 0.2)
        negative_count = sum(1 for exp in similar_experiences if exp.get('valence', 0) < -0.2)

        if positive_count > negative_count:
            predicted_emotion = 'positive'
            confidence = min(0.9, positive_count / len(similar_experiences))
        elif negative_count > positive_count:
            predicted_emotion = 'negative'
            confidence = min(0.9, negative_count / len(similar_experiences))
        else:
            predicted_emotion = 'neutral'
            confidence = 0.5

        return {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'sample_size': len(similar_experiences)
        }

    def _find_similar_experiences(self, situation: str, option: str) -> List[Dict[str, Any]]:
        """Encuentra experiencias similares para predicción"""
        similar = []

        # Buscar en memoria de resultados
        for situation_key, outcomes in self.somatic_memory.items():
            if self._situations_similar(situation_key, situation):
                for outcome_key, results in outcomes.items():
                    if self._options_similar(outcome_key, option):
                        similar.extend(results)

        return similar[:10]  # Limitar a 10 experiencias más relevantes

    def _situations_similar(self, sit1: str, sit2: str) -> bool:
        """Determina si dos situaciones son similares"""
        # Implementación simple: compartir palabras clave
        words1 = set(sit1.lower().split())
        words2 = set(sit2.lower().split())
        overlap = len(words1 & words2)
        return overlap > 0 or (len(words1) > 0 and len(words2) > 0 and overlap / min(len(words1), len(words2)) > 0.3)

    def _options_similar(self, opt1: str, opt2: str) -> bool:
        """Determina si dos opciones son similares"""
        return opt1.lower() == opt2.lower()  # comparación exacta por simplicidad

    def _combine_somatic_signals(self, situation_markers: List[SomaticMarker],
                               option_markers: Dict[str, Dict[str, Any]],
                               predicted_outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """Combina todas las señales somáticas activadas"""

        # Señales somáticas agregadas
        combined_somatic_signals = defaultdict(float)
        emotional_responses = []

        # Procesar marcadores situacionales
        for marker in situation_markers:
            activation = marker.activate()
            emotional_responses.append(activation['emotional_response'])

            for signal_name, signal_value in activation['somatic_signals'].items():
                combined_somatic_signals[signal_name] += signal_value * activation['intensity']

        # Procesar marcadores de opciones
        option_emotions = {}
        for option, marker_data in option_markers.items():
            option_emotions[option] = marker_data['emotional_response']

            # Agregar señales somáticas de la opción
            for signal_name, signal_value in marker_data['somatic_signals'].items():
                combined_somatic_signals[signal_name] += signal_value * marker_data['intensity']

        # Agregar bias de predicciones de resultado
        for option, prediction in predicted_outcomes.items():
            if prediction['confidence'] > 0.3:
                # Bias emocional de la predicción
                predicted_valence = 1.0 if prediction['predicted_emotion'] == 'positive' else \
                                  (-1.0 if prediction['predicted_emotion'] == 'negative' else 0.0)
                option_emotions[option] = option_emotions.get(option, {'valence': 0.0})
                option_emotions[option]['valence'] += predicted_valence * prediction['confidence'] * 0.5

        # Normalizar señales combinadas
        for signal_name in combined_somatic_signals:
            combined_somatic_signals[signal_name] = min(1.0, abs(combined_somatic_signals[signal_name]))

        return {
            'somatic_signals': dict(combined_somatic_signals),
            'emotional_responses': emotional_responses,
            'option_emotions': option_emotions,
            'dominant_emotion': self._determine_dominant_emotion(emotional_responses, option_emotions)
        }

    def _determine_dominant_emotion(self, responses: List[Dict[str, Any]],
                                  option_emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Determina emoción dominante actual"""
        if not responses and not option_emotions:
            return {'emotion': 'neutral', 'intensity': 0.0, 'confidence': 1.0}

        # Calcular valencias agregadas
        total_valence = 0.0
        total_confidence = 0.0
        count = 0

        for response in responses:
            total_valence += response.get('valence', 0.0)
            total_confidence += 0.8  # confianza base para marcadores existentes
            count += 1

        for emotions in option_emotions.values():
            total_valence += emotions.get('valence', 0.0)
            total_confidence += 0.6  # menor confianza para predicciones
            count += 1

        if count == 0:
            return {'emotion': 'neutral', 'intensity': 0.0, 'confidence': 1.0}

        avg_valence = total_valence / count
        avg_confidence = total_confidence / count

        if avg_valence > 0.3:
            dominant = 'positive'
        elif avg_valence < -0.3:
            dominant = 'negative'
        else:
            dominant = 'neutral'

        return {
            'emotion': dominant,
            'intensity': abs(avg_valence),
            'confidence': avg_confidence,
            'valence': avg_valence
        }

    def _compute_emotional_bias(self, combined_feedback: Dict[str, Any]) -> float:
        """Computa bias emocional general para decisiones futuras"""
        dominant = combined_feedback['dominant_emotion']

        bias_strength = dominant['intensity'] * dominant['confidence']

        if dominant['emotion'] == 'positive':
            return bias_strength
        elif dominant['emotion'] == 'negative':
            return -bias_strength
        else:
            return 0.0

    def learn_from_experience(self, situation: str, option_chosen: str,
                            actual_outcome: Dict[str, Any]):
        """
        Aprende de una experiencia completada para refinar marcadores

        Args:
            situation: Situación experimentada
            option_chosen: Opción elegida
            actual_outcome: Resultado real (incluyendo emoción experimentada)
        """
        # Registrar experiencia en memoria
        if situation not in self.somatic_memory:
            self.somatic_memory[situation] = {}

        if option_chosen not in self.somatic_memory[situation]:
            self.somatic_memory[situation][option_chosen] = []

        experience_record = {
            'outcome_emotion': actual_outcome.get('emotion', 'neutral'),
            'outcome_valence': actual_outcome.get('valence', 0.0),
            'outcome_intensity': actual_outcome.get('intensity', 0.5),
            'timestamp': time.time(),
            'situation': situation,
            'option': option_chosen
        }

        self.somatic_memory[situation][option_chosen].append(experience_record)

        # Limitar memoria por opción (mantener últimas 20 experiencias)
        if len(self.somatic_memory[situation][option_chosen]) > 20:
            self.somatic_memory[situation][option_chosen] = self.somatic_memory[situation][option_chosen][-20:]

        # Actualizar confianza de marcadores relevantes
        outcome_emotion = actual_outcome.get('emotion', 'neutral')
        self._update_marker_confidence(situation, option_chosen, outcome_emotion)

        print(f"🧠 Aprendido de experiencia: {situation} → {option_chosen} → {outcome_emotion}")

    def _update_marker_confidence(self, situation: str, option: str, outcome_emotion: str):
        """Actualiza confianza de marcadores basada en experiencia real"""
        # Si existe marcador para esta situación
        if situation in self.markers['situational']:
            marker = self.markers['situational'][situation]
            current_confidence = marker.confidence

            # Ajuste basado en consistencia emocional esperada
            expected_emotion = marker.emotional_state
            if expected_emotion == outcome_emotion:
                # Reforzar confianza si predicción fue correcta
                marker.confidence = min(1.0, current_confidence + self.learning_rate)
            else:
                # Reducir confianza si predicción fue incorrecta
                marker.confidence = max(0.1, current_confidence - self.learning_rate * 0.5)

    def _associate_with_memory(self, trigger: str, marker: SomaticMarker):
        """Asocia marcador con memoria aprendida existente"""
        # Buscar experiencias similares ya aprendidas
        similar_experiences = []
        for situation in self.somatic_memory.values():
            for outcomes in situation.values():
                for experience in outcomes:
                    if self._situations_similar(experience.get('situation', ''), trigger):
                        similar_experiences.append(experience)

        # Si hay experiencias similares, ajustar confianza inicial del marcador
        if similar_experiences:
            avg_valence = np.mean([exp.get('outcome_valence', 0.0) for exp in similar_experiences])
            if abs(avg_valence) > 0.2:
                direction_match = ((avg_valence > 0 and marker.emotional_state == 'positive') or
                                 (avg_valence < 0 and marker.emotional_state == 'negative'))
                if direction_match:
                    marker.confidence = min(0.9, marker.confidence + 0.3)  # bonus por consistencia

    def get_system_status(self) -> Dict[str, Any]:
        """Estado completo del sistema de marcadores somáticos"""
        total_markers = sum(len(markers) for markers in self.markers.values())

        return {
            'total_markers': total_markers,
            'situational_markers': len(self.markers['situational']),
            'option_markers': len(self.markers['option_based']),
            'outcome_markers': len(self.markers['outcome_based']),
            'current_emotional_bias': self.emotional_bias,
            'total_learned_experiences': sum(
                len(outcomes) for situation in self.somatic_memory.values()
                for outcomes in situation.values()
            )
        }



# ==================== EJEMPLO DE USO ====================


if __name__ == "__main__":
    print("🧠 SISTEMA DE MARCADORES SOMÁTICOS - DEMO DAMASIO")
    print("=" * 70)

    somatic_system = SomaticMarkersSystem()

    # Registrar marcadores iniciales
    print("\n📝 Registrando marcadores iniciales...")

    # Situaciones peligrosas
    somatic_system.register_marker(
        trigger="approaching_large_unknown_animal",
        emotional_state="negative",
        intensity=0.9,
        marker_type="situational",
        somatic_signals={'heart_rate_increase': 0.8, 'muscle_tension': 0.7}
    )

    # Opciones de inversión
    somatic_system.register_marker(
        trigger="invest_in_volatile_stocks",
        emotional_state="negative",
        intensity=0.7,
        marker_type="option_based",
        somatic_signals={'gut_discomfort': 0.6, 'anxiety_signal': 0.5}
    )

    # Opciones seguras
    somatic_system.register_marker(
        trigger="invest_in_government_bonds",
        emotional_state="positive",
        intensity=0.4,
        marker_type="option_based",
        somatic_signals={'calm_sensation': 0.7, 'security_feeling': 0.6}
    )

    print(f"✅ Registrados {somatic_system.get_system_status()['total_markers']} marcadores")

    # Simular toma de decisión
    print("\n🎯 Simulando decisión de inversión...")

    situation = "planning_family_savings_strategy"
    options = ["invest_in_volatile_stocks", "invest_in_government_bonds", "put_money_under_mattress"]

    feedback = somatic_system.get_somatic_feedback(situation, options)

    print("\nSituación:")
    print(f"   '{situation}'")

    print("\nOpciones disponibles con bias emocional:")
    for option in options:
        opt_emotion = feedback['option_emotions'].get(option, {'emotion': 'unknown'})
        emotion_label = opt_emotion.get('emotion', 'unknown') if isinstance(opt_emotion, dict) else 'unknown'
        print(f"   • {option}: {emotion_label.upper()}")

    print("\nEstado somático actual:")
    print(f"   Emoción dominante: {feedback['dominant_emotion']['emotion'].upper()}")
    print(f"   Intensidad: {feedback['dominant_emotion']['intensity']:.3f}")
    print(f"   Confianza: {feedback['dominant_emotion']['confidence']:.3f}")
    print("\nSeñales corporales activadas:")
    for signal, value in feedback.get('somatic_signals', {}).items():
        if value > 0.2:  # solo mostrar señales significativas
            print(f"   • {signal}: {value:.3f}")
    # Aprender de experiencia simulada
    print("\n📚 Aprendiendo de experiencia...")
    # Simular resultado de elegir opción arriesgada
    actual_result = {
        'emotion': 'positive',
        'valence': 0.8,
        'intensity': 0.7
    }

    somatic_system.learn_from_experience(
        situation=situation,
        option_chosen="invest_in_volatile_stocks",
        actual_outcome=actual_result
    )

    print("\n🎉 APRENDIZAJE COMPLETADO")
    print("   ✓ Sistema Damasio operativo")
    print(f"   ✓ {somatic_system.get_system_status()['total_markers']} marcadores activos")
    print(f"   ✓ {somatic_system.get_system_status()['total_learned_experiences']} experiencias aprendidas")
    print("   ✓ Bias emocional funcionado como guía de decisión")
