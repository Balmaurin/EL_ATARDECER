"""
EMBODIED SIGNALS SYSTEM - Señales corporales simuladas

Implementa señales corporales realistas que afectan consciencia:
- Señales de dolor y placer
- Estados de hambre y saciedad
- Temperatura corporal extrema
- Fatiga y necesidad de descanso
- Estados de comodidad física
"""

import numpy as np
import time
import random
from typing import Dict, List, Any, Optional, Tuple


class EmbodiedSignal:
    """Señal corporal individual con características fisiológicas"""

    def __init__(self, signal_type: str, name: str, base_intensity: float = 0.0):
        self.signal_type = signal_type  # 'pain', 'pleasure', 'hunger', 'fatigue', etc.
        self.name = name
        self.base_intensity = base_intensity
        self.current_intensity = base_intensity
        self.urgency_level = 0.0
        self.last_triggered = time.time()
        self.accumulated_exposure = 0.0
        self.adaptation_level = 0.0  # Adaptación a la señal

        # Umbrales específicos por tipo
        self.thresholds = self._set_thresholds_by_type()

        # Efectos corporales asociados
        self.corporeal_effects = self._generate_corporeal_effects()

    def _set_thresholds_by_type(self) -> Dict[str, float]:
        """Establece umbrales específicos según tipo de señal"""
        thresholds = {
            'pain': {
                'noticeable': 0.2, 'uncomfortable': 0.5, 'severe': 0.8, 'intolerable': 0.95
            },
            'pleasure': {
                'mild': 0.1, 'enjoyable': 0.3, 'intense': 0.6, 'ecstatic': 0.9
            },
            'hunger': {
                'peckish': 0.2, 'hungry': 0.5, 'famished': 0.8, 'starving': 0.95
            },
            'fatigue': {
                'tired': 0.3, 'exhausted': 0.7, 'debilitated': 0.9
            },
            'thirst': {
                'dry': 0.2, 'thirsty': 0.5, 'dehydrated': 0.8, 'severe_dehydration': 0.95
            },
            'temperature': {
                'cool': 0.2, 'cold': 0.5, 'freezing': 0.8, 'hypothermic': 0.95
            }
        }
        return thresholds.get(self.signal_type, {'mild': 0.2, 'moderate': 0.5, 'severe': 0.8})

    def _generate_corporeal_effects(self) -> Dict[str, float]:
        """Genera efectos corporales típicos para esta señal"""
        effects_patterns = {
            'pain': {
                'muscle_tension': 0.7, 'heart_rate_increase': 0.6,
                'sweating': 0.4, 'gritting_teeth': 0.5
            },
            'pleasure': {
                'muscle_relaxation': 0.8, 'blushing': 0.3,
                'smiling': 0.6, 'warm_sensation': 0.5
            },
            'hunger': {
                'stomach_gurgling': 0.6, 'weakness': 0.4, 'irritability': 0.3
            },
            'fatigue': {
                'heavy_eyelids': 0.8, 'yawning': 0.7, 'slow_movement': 0.6
            },
            'thirst': {
                'dry_mouth': 0.8, 'headache': 0.4, 'concentration_difficulty': 0.3
            },
            'temperature': {
                'shivering': 0.7, 'goosebumps': 0.5, 'teeth_chattering': 0.6
            }
        }

        base_effects = effects_patterns.get(self.signal_type, {'general_discomfort': 0.5})
        return {effect: intensity * self.base_intensity for effect, intensity in base_effects.items()}

    def trigger_signal(self, intensity: float, duration: float = 5.0) -> Dict[str, Any]:
        """
        Activa señal corporal con intensidad específica

        Args:
            intensity: Intensidad de la señal (0-1)
            duration: Duración en segundos

        Returns:
            Estado de activación de la señal
        """
        self.current_intensity = min(1.0, max(0.0, intensity - self.adaptation_level))
        self.last_triggered = time.time()
        self.accumulated_exposure += self.current_intensity * duration

        # Calcular urgencia y nivel de consciencia
        urgency, consciousness_level = self._calculate_consciousness_impact()

        # Efectos sobre consciencia
        consciousness_effects = {
            'attention_grab': urgency * 0.8,
            'emotional_influence': self._get_emotional_valence() * self.current_intensity,
            'behavioral_urgency': urgency,
            'cognitive_impairment': self._calculate_cognitive_impact()
        }

        activation_state = {
            'signal_type': self.signal_type,
            'intensity': self.current_intensity,
            'urgency': urgency,
            'consciousness_level': consciousness_level,
            'consciousness_effects': consciousness_effects,
            'corporeal_effects': self._get_current_corporeal_effects(),
            'duration_estimated': duration,
            'timestamp': self.last_triggered
        }

        # Adaptación gradual a señales repetidas
        self._update_adaptation(urgency)

        return activation_state

    def _calculate_consciousness_impact(self) -> Tuple[float, float]:
        """Calcula impacto de la señal en consciencia"""
        # Urgencia basada en thresholds
        if self.signal_type in self.thresholds:
            thresholds = self.thresholds[self.signal_type]
            if self.current_intensity >= list(thresholds.values())[-1]:
                urgency = 0.95
            elif self.current_intensity >= list(thresholds.values())[-2]:
                urgency = 0.7
            elif self.current_intensity >= list(thresholds.values())[-3]:
                urgency = 0.4
            else:
                urgency = 0.1
        else:
            urgency = self.current_intensity * 0.6

        # Nivel de consciencia: señales intensas son más conscientes
        consciousness_level = min(1.0, urgency * 1.2)

        return urgency, consciousness_level

    def _get_emotional_valence(self) -> float:
        """Obtiene valencia emocional de la señal (-1 a 1)"""
        valence_map = {
            'pain': -0.9, 'hunger': -0.6, 'fatigue': -0.4,
            'thirst': -0.7, 'temperature': -0.8,
            'pleasure': 0.9, 'comfort': 0.7
        }
        return valence_map.get(self.signal_type, 0.0)

    def _calculate_cognitive_impact(self) -> float:
        """Calcula impacto en funcionamiento cognitivo (0-1 deterioro)"""
        cognitive_impairment = self.current_intensity * self.urgency_level

        # Algunos tipos afectan más al funcionamiento cognitivo
        cognitive_multipliers = {
            'pain': 0.8, 'fatigue': 0.6, 'hunger': 0.4,
            'thirst': 0.5, 'temperature': 0.7, 'pleasure': 0.2
        }

        multiplier = cognitive_multipliers.get(self.signal_type, 0.5)
        return min(1.0, cognitive_impairment * multiplier)

    def _get_current_corporeal_effects(self) -> Dict[str, float]:
        """Obtiene efectos corporales actuales"""
        # Intensificar efectos basados en intensidad actual
        current_effects = {}
        for effect, base_intensity in self.corporeal_effects.items():
            current_effects[effect] = base_intensity * self.current_intensity

        return current_effects

    def _update_adaptation(self, urgency: float):
        """Actualiza nivel de adaptación a la señal"""
        # Más urgencia = menos adaptación
        adaptation_rate = 0.02 * (1 - urgency)
        self.adaptation_level = min(0.5, self.adaptation_level + adaptation_rate)

    def decay_signal(self, time_passed: float):
        """Hace decaer intensidad de la señal con el tiempo"""
        if self.current_intensity > self.base_intensity:
            decay_rate = 0.1 * time_passed  # 10% por segundo base
            self.current_intensity = max(self.base_intensity,
                                       self.current_intensity - decay_rate)

    def get_signal_state(self) -> Dict[str, Any]:
        """Estado completo de la señal"""
        return {
            'name': self.name,
            'type': self.signal_type,
            'current_intensity': self.current_intensity,
            'urgency_level': self.urgency_level,
            'adaptation_level': self.adaptation_level,
            'accumulated_exposure': self.accumulated_exposure,
            'last_triggered': self.last_triggered,
            'active_status': self.current_intensity > self.base_intensity + 0.1
        }


class EmbodiedSignalsSystem:
    """
    Sistema completo de señales corporales para consciencia embodied
    Maneja múltiples señales corporales simultáneamente
    """

    def __init__(self):
        # Señales corporales principales
        self.body_signals = self._initialize_body_signals()

        # Estados sistémicos
        self.overall_discomfort = 0.0
        self.attention_load = 0.0
        self.behavioral_drive = {}

        # Historial de estados corporales
        self.signal_history = []
        self.last_update = time.time()

    def _initialize_body_signals(self) -> Dict[str, EmbodiedSignal]:
        """Inicializa sistema de señales corporales principal"""
        signals = {}

        # Señales de dolor/placer
        signals['headache'] = EmbodiedSignal('pain', 'Headache', 0.0)
        signals['back_pain'] = EmbodiedSignal('pain', 'Back Pain', 0.0)
        signals['stomach_pain'] = EmbodiedSignal('pain', 'Stomach Discomfort', 0.0)
        signals['muscle_soreness'] = EmbodiedSignal('pleasure', 'Muscle Relaxation', 0.0)  # Inverso de dolor

        # Señales metabólicas
        signals['hunger'] = EmbodiedSignal('hunger', 'Hunger', 0.1)  # Nivel basal de hambre
        signals['thirst'] = EmbodiedSignal('thirst', 'Thirst', 0.05)  # Nivel basal de sed
        signals['fullness'] = EmbodiedSignal('pleasure', 'Satiety', 0.0)

        # Señales de fatiga
        signals['physical_fatigue'] = EmbodiedSignal('fatigue', 'Physical Fatigue', 0.0)
        signals['mental_fatigue'] = EmbodiedSignal('fatigue', 'Mental Fatigue', 0.0)
        signals['sleep_deprivation'] = EmbodiedSignal('fatigue', 'Sleep Deprivation', 0.0)

        # Señales térmicas
        signals['cold_discomfort'] = EmbodiedSignal('temperature', 'Cold Discomfort', 0.0)
        signals['heat_discomfort'] = EmbodiedSignal('temperature', 'Heat Discomfort', 0.0)

        # Señales de confort general
        signals['physical_comfort'] = EmbodiedSignal('pleasure', 'Physical Comfort', 0.2)
        signals['posture_discomfort'] = EmbodiedSignal('pain', 'Poor Posture', 0.0)

        return signals

    def update_body_signals(self, body_state_changes: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Actualiza estado de todas las señales corporales

        Args:
            body_state_changes: Cambios específicos en estado corporal

        Returns:
            Estado completo de señales corporales
        """
        current_time = time.time()
        time_delta = current_time - self.last_update
        self.last_update = current_time

        if body_state_changes is None:
            body_state_changes = {}

        # Aplicar cambios externos
        for signal_name, change in body_state_changes.items():
            if signal_name in self.body_signals:
                signal = self.body_signals[signal_name]
                # Convertir cambio relativo a intensidad absoluta
                target_intensity = max(0.0, min(1.0, signal.current_intensity + change))
                signal.trigger_signal(target_intensity)

        # Aplicar decadencia temporal
        for signal in self.body_signals.values():
            signal.decay_signal(time_delta)

        # Calcular estados agregados
        overall_state = self._compute_overall_embodied_state()

        # Registrar en historial
        self.signal_history.append({
            'timestamp': current_time,
            'signals': {name: sig.get_signal_state() for name, sig in self.body_signals.items()},
            'overall_state': overall_state
        })

        # Mantener historial limitado
        if len(self.signal_history) > 50:
            self.signal_history = self.signal_history[-30:]

        return overall_state

    def trigger_specific_signal(self, signal_name: str, intensity: float,
                              duration: float = 5.0) -> Dict[str, Any]:
        """
        Activa señal corporal específica

        Args:
            signal_name: Nombre de la señal a activar
            intensity: Intensidad de activación
            duration: Duración en segundos

        Returns:
            Resultado de activación
        """
        if signal_name not in self.body_signals:
            return {'error': f'Signal {signal_name} not found'}

        signal = self.body_signals[signal_name]
        activation = signal.trigger_signal(intensity, duration)

        # Actualizar estados agregados
        self._update_overall_states()

        return activation

    def _compute_overall_embodied_state(self) -> Dict[str, Any]:
        """Computa estado general de señales corporales"""

        # Señales activas (por encima del threshold)
        active_signals = []
        total_urgency = 0.0
        total_negative_valence = 0.0
        total_positive_valence = 0.0

        active_count = 0

        for name, signal in self.body_signals.items():
            state = signal.get_signal_state()
            if state['active_status']:
                active_signals.append({
                    'name': name,
                    'type': state['type'],
                    'intensity': state['current_intensity'],
                    'urgency': state['urgency_level']
                })

                total_urgency += state['urgency_level']

                valence = signal._get_emotional_valence()
                if valence > 0:
                    total_positive_valence += valence * state['current_intensity']
                else:
                    total_negative_valence += abs(valence) * state['current_intensity']

                active_count += 1

        # Estado general
        if active_count == 0:
            general_state = 'neutral'
            discomfort_level = 0.0
        elif total_urgency / max(1, active_count) > 0.7:
            general_state = 'crisis'
            discomfort_level = 0.9
        elif total_urgency / max(1, active_count) > 0.4:
            general_state = 'distressed'
            discomfort_level = 0.6
        elif total_negative_valence > total_positive_valence:
            general_state = 'uncomfortable'
            discomfort_level = 0.4
        else:
            general_state = 'comfortable'
            discomfort_level = 0.1

        # Carga atencional corporales
        attention_load = min(1.0, total_urgency / max(1, len(self.body_signals)))

        # Drivas comportamentales
        behavioral_drives = self._compute_behavioral_drives(active_signals)

        return {
            'general_state': general_state,
            'discomfort_level': discomfort_level,
            'attention_load': attention_load,
            'active_signals': active_signals,
            'total_urgency': total_urgency,
            'valence_balance': total_positive_valence - total_negative_valence,
            'behavioral_drives': behavioral_drives,
            'homeostatic_stress': min(1.0, total_urgency / 5.0),  # Normalizado
            'signal_count': {'total': len(self.body_signals), 'active': active_count}
        }

    def _compute_behavioral_drives(self, active_signals: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcula drives comportamentales basados en señales activas"""

        drives = {
            'seek_food': 0.0,
            'seek_water': 0.0,
            'seek_rest': 0.0,
            'seek_medical_attention': 0.0,
            'seek_temperature_comfort': 0.0,
            'reduce_pain': 0.0,
            'maintain_comfort': 0.0
        }

        # Sumar contribución de cada señal activa
        drive_mapping = {
            'hunger': 'seek_food',
            'thirst': 'seek_water',
            'physical_fatigue': 'seek_rest',
            'mental_fatigue': 'seek_rest',
            'sleep_deprivation': 'seek_rest',
            'headache': 'seek_medical_attention',
            'back_pain': 'seek_medical_attention',
            'stomach_pain': 'seek_medical_attention',
            'cold_discomfort': 'seek_temperature_comfort',
            'heat_discomfort': 'seek_temperature_comfort',
            'physical_comfort': 'maintain_comfort',
            'posture_discomfort': 'maintain_comfort'
        }

        for signal_info in active_signals:
            signal_name = signal_info['name']
            intensity = signal_info['intensity']
            urgency = signal_info['urgency']

            target_drive = drive_mapping.get(signal_name)
            if target_drive:
                drives[target_drive] = min(1.0, drives[target_drive] + intensity * urgency)

            # Drive genérico de reducción de dolor
            if signal_info['type'] == 'pain':
                drives['reduce_pain'] = min(1.0, drives['reduce_pain'] + intensity * urgency)

        return drives

    def _update_overall_states(self):
        """Actualiza estados agregados del sistema"""
        overall = self._compute_overall_embodied_state()
        self.overall_discomfort = overall['discomfort_level']
        self.attention_load = overall['attention_load']
        self.behavioral_drive = overall['behavioral_drives']

    def simulate_body_state_change(self, scenario: str, intensity: float = 0.7) -> Dict[str, Any]:
        """
        Simula cambio de estado corporal común

        Args:
            scenario: Tipo de escenario ('exercise', 'eating', 'sleeping', 'injury', etc.)
            intensity: Intensidad del cambio
        """
        scenario_effects = {
            'heavy_exercise': {
                'physical_fatigue': intensity * 0.8,
                'muscle_soreness': intensity * 0.6,
                'heat_discomfort': intensity * 0.4,
                'thirst': intensity * 0.5
            },
            'mental_work': {
                'mental_fatigue': intensity * 0.7,
                'headache': intensity * 0.3
            },
            'eating': {
                'hunger': -intensity * 0.9,  # Reduce hambre
                'thirst': intensity * 0.2,   # Puede causar sed
                'fullness': intensity * 0.8
            },
            'sleeping': {
                'physical_fatigue': -intensity * 0.8,
                'mental_fatigue': -intensity * 0.7,
                'sleep_deprivation': -intensity * 0.9
            },
            'injury': {
                'back_pain': intensity * 0.7,
                'physical_fatigue': intensity * 0.3
            },
            'cold_environment': {
                'cold_discomfort': intensity * 0.9,
                'physical_fatigue': intensity * 0.2
            },
            'comfortable_position': {
                'physical_comfort': intensity * 0.8,
                'posture_discomfort': -intensity * 0.6
            }
        }

        if scenario not in scenario_effects:
            return {'error': f'Unknown scenario: {scenario}'}

        # Aplicar cambios del escenario
        changes = scenario_effects[scenario]
        result = self.update_body_signals(changes)

        result['scenario_applied'] = scenario
        result['intensity_used'] = intensity

        return result

    def get_embodied_feedback(self) -> Dict[str, Any]:
        """
        Proporciona feedback completo de estado corporal para consciencia
        Este feedback integra con sistemas emocional y de decisión
        """
        overall_state = self._compute_overall_embodied_state()

        # Feedback emocional basado en señales corporales
        emotional_feedback = {
            'bodily_mood': overall_state['valence_balance'],
            'physical_wellbeing': 1.0 - overall_state['discomfort_level'],
            'urge_satisfaction_level': min(1.0,
                sum(drive for drive in overall_state['behavioral_drives'].values()) / 3.0
            )
        }

        # Señales conscientes principales
        conscious_signals = {
            signal_name: signal.get_signal_state()
            for signal_name, signal in self.body_signals.items()
            if signal.get_signal_state()['active_status']
        }

        # Estado homeostático general
        homeostatic_status = {
            'physical_stability': 1.0 - overall_state['homeostatic_stress'],
            'bodily_demands': len(overall_state['active_signals']),
            'comfort_index': overall_state['general_state'] == 'comfortable',
            'needs_attention': overall_state['attention_load'] > 0.3
        }

        return {
            'overall_embodied_state': overall_state,
            'emotional_feedback': emotional_feedback,
            'conscious_signals': conscious_signals,
            'homeostatic_status': homeostatic_status,
            'behavioral_imperatives': overall_state['behavioral_drives'],
            'attention_demands': overall_state['attention_load'],
            'system_health': {
                'signal_integrity': len(self.body_signals),
                'active_monitoring': len(conscious_signals),
                'adaptation_levels': {
                    name: sig.adaptation_level for name, sig in self.body_signals.items()
                }
            }
        }

    def reset_to_baseline(self):
        """Resetea todas las señales a estado basal"""
        for signal in self.body_signals.values():
            signal.current_intensity = signal.base_intensity
            signal.adaptation_level = 0.0
            signal.accumulated_exposure = 0.0

        self.signal_history.clear()
        self._update_overall_states()

        print("🏥 Estado corporal reseteado a baseline")


# ==================== EJEMPLO DE USO ====================

if __name__ == "__main__":
    print("🏥 SISTEMA DE SEÑALES CORPORALES EMBODIED")
    print("=" * 60)

    embodied_system = EmbodiedSignalsSystem()

    print("\n📊 Estado inicial:")
    initial_state = embodied_system.get_embodied_feedback()
    print(f"   Estado general: {initial_state['overall_embodied_state']['general_state'].upper()}")
    print(f"   Nivel de discomfort: {initial_state['overall_embodied_state']['discomfort_level']:.2f}")
    print(f"   Señales activas: {initial_state['overall_embodied_state']['signal_count']['active']}")

    # Simulaciones de escenarios corporales
    print("\n⚡ Simulando escenarios corporales...")    # Ejercicio intenso
    print("   • Después de ejercicio pesado:")
    exercise_result = embodied_system.simulate_body_state_change('heavy_exercise', 0.8)
    print(f"     Estado: {exercise_result['general_state'].upper()}")
    print(f"     Molestias activas: {exercise_result['signal_count']['active']}")

    # Comer
    print("   • Después de comer:")
    eating_result = embodied_system.simulate_body_state_change('eating', 0.9)
    print(f"     Estado: {eating_result['general_state'].upper()}")
    print(f"     Nivel de discomfort: {eating_result['discomfort_level']:.2f}")

    # Descansar un poco
    print("   • Después de dormir:")
    sleep_result = embodied_system.simulate_body_state_change('sleeping', 0.7)
    print(f"     Estado: {sleep_result['general_state'].upper()}")
    print(f"     Señales activas: {sleep_result['signal_count']['active']}")

    # Lesión menor
    print("   • Después de torcer el tobillo:")
    injury_result = embodied_system.simulate_body_state_change('injury', 0.6)
    print(f"     Estado: {injury_result['general_state'].upper()}")
    print(f"     Urgencia total: {injury_result['total_urgency']:.2f}")

    print("\n🌡️ Feedback emocional corporal:")
    final_feedback = embodied_system.get_embodied_feedback()
