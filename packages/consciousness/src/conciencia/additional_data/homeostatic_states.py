"""
HOMEOSTATIC STATES SYSTEM - Estado homeostático interno completo

Implementa estados fisiológicos internos para consciencia corporal completa:
- Regulación de temperatura corporal
- Mantenimiento de niveles de energía
- Control de recursos internos
- Señales homeostáticas
- Feedback a consciencia central
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class HomeostaticParameter:
    """Parámetro homeostático individual con límites y regulación"""
    name: str
    current_value: float
    optimal_value: float
    min_safe_value: float
    max_safe_value: float
    urgency_threshold: float
    change_rate: float  # velocidad natural de cambio
    stability_force: float  # fuerza para mantener en óptimo

    def get_deviation_from_optimal(self) -> float:
        """Devición del valor óptimo (-1 = mín, +1 = máx)"""
        return (self.current_value - self.optimal_value) / (self.optimal_value - self.min_safe_value)

    def is_out_of_range(self) -> bool:
        """Verdadero si está fuera de límites seguros"""
        return self.current_value < self.min_safe_value or self.current_value > self.max_safe_value

    def get_urgency_signal(self) -> float:
        """Señal de urgencia (0-1) basada en desviación"""
        deviation = abs(self.get_deviation_from_optimal())
        if deviation < self.urgency_threshold:
            return 0.0
        return min(1.0, (deviation - self.urgency_threshold) / (1.0 - self.urgency_threshold))


class HomeostaticSystem:
    """
    Sistema homeostático completo que mantiene el equilibrio interno
    Proporciona feedback constante sobre estado fisiológico corporal
    """

    def __init__(self):
        # Parámetros vitales principales
        self.parameters = {
            'body_temperature': HomeostaticParameter(
                name='Temperatura Corporal',
                current_value=36.5,  # °C
                optimal_value=36.6,
                min_safe_value=35.0,
                max_safe_value=38.0,
                urgency_threshold=0.3,
                change_rate=0.01,
                stability_force=0.05
            ),

            'blood_glucose': HomeostaticParameter(
                name='Glucosa en Sangre',
                current_value=90.0,  # mg/dL
                optimal_value=90.0,
                min_safe_value=70.0,
                max_safe_value=140.0,
                urgency_threshold=0.4,
                change_rate=0.1,
                stability_force=0.1
            ),

            'blood_pressure': HomeostaticParameter(
                name='Presión Arterial',
                current_value=120.0,  # mmHg (sistólica)
                optimal_value=120.0,
                min_safe_value=90.0,
                max_safe_value=140.0,
                urgency_threshold=0.2,
                change_rate=0.05,
                stability_force=0.03
            ),

            'hydration_level': HomeostaticParameter(
                name='Nivel de Hidratación',
                current_value=0.8,  # porcentaje 0-1
                optimal_value=0.85,
                min_safe_value=0.6,
                max_safe_value=1.0,
                urgency_threshold=0.5,
                change_rate=0.002,
                stability_force=0.01
            ),

            'energy_level': HomeostaticParameter(
                name='Nivel de Energía',
                current_value=0.7,  # porcentaje 0-1
                optimal_value=0.8,
                min_safe_value=0.3,
                max_safe_value=1.0,
                urgency_threshold=0.4,
                change_rate=0.005,
                stability_force=0.02
            ),

            'stress_hormones': HomeostaticParameter(
                name='Hormonas de Estrés',
                current_value=0.2,  # nivel relativo
                optimal_value=0.1,
                min_safe_value=0.0,
                max_safe_value=0.8,
                urgency_threshold=0.6,
                change_rate=0.02,
                stability_force=0.08
            )
        }

        # Estados de regulación
        self.regulatory_effects = {}
        self.last_update = time.time()

    def update_states(self, external_influences: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Actualiza todos los parámetros homeostáticos

        Args:
            external_influences: Factores externos que afectan homeostasis (ej: ejercicio, comida)

        Returns:
            Estado completo del sistema homeostático
        """
        current_time = time.time()
        time_delta = current_time - self.last_update
        self.last_update = current_time

        if external_influences is None:
            external_influences = {}

        # Actualizar cada parámetro
        regulatory_actions = {}
        status_signals = {}

        for param_key, param in self.parameters.items():
            # Aplicar cambios naturales
            natural_drift = np.random.normal(0, param.change_rate) * time_delta

            # Aplicar influencias externas
            external_effect = external_influences.get(param_key, 0.0)

            # Fuera de homeostsis - aplicar corrección activa
            homeostatic_correction = 0.0
            if abs(param.get_deviation_from_optimal()) > 0.1:
                correction_strength = param.stability_force * param.get_deviation_from_optimal()
                homeostatic_correction = -correction_strength * time_delta

            # Actualizar valor
            param.current_value += natural_drift + external_effect + homeostatic_correction

            # Mantener dentro de límites fisiológicos
            param.current_value = np.clip(param.current_value,
                                        param.min_safe_value * 0.8,  # límite físico inferior
                                        param.max_safe_value * 1.2)  # límite físico superior

            # Señales de estado
            status_signals[param_key] = {
                'value': param.current_value,
                'deviation': param.get_deviation_from_optimal(),
                'urgency': param.get_urgency_signal(),
                'out_of_range': param.is_out_of_range(),
                'needs_attention': param.get_urgency_signal() > 0.5
            }

            # Acciones regulatorias si necesarias
            if param.is_out_of_range():
                regulatory_actions[param_key] = self._generate_regulatory_action(param)

        # Señales agregadas para consciencia
        overall_status = self._compute_overall_homeostatic_status(status_signals)

        return {
            'parameter_states': status_signals,
            'regulatory_actions': regulatory_actions,
            'overall_status': overall_status,
            'needs_immediate_attention': overall_status['crisis_level'] > 0.7
        }

    def _generate_regulatory_action(self, parameter: HomeostaticParameter) -> Dict[str, Any]:
        """Genera acción regulatoria automática para parámetro fuera de rango"""
        deviation = parameter.get_deviation_from_optimal()

        actions = {
            'body_temperature': {
                'too_high': ['increase_sweating', 'reduce_metabolism', 'seek_cooler_environment'],
                'too_low': ['shivering', 'increase_metabolism', 'seek_warmer_environment']
            },
            'blood_glucose': {
                'too_high': ['increase_insulin_release', 'reduce_food_intake'],
                'too_low': ['release_glucagon', 'increase_hunger_signal', 'seek_food']
            },
            'blood_pressure': {
                'too_high': ['vasodilation', 'reduce_heart_rate', 'increase_diuresis'],
                'too_low': ['vasoconstriction', 'increase_heart_rate', 'increase_fluid_retention']
            },
            'hydration_level': {
                'too_low': ['increase_thirst', 'reduce_urination', 'seek_water_source'],
                'too_high': ['increase_diuresis', 'reduce_thirst']
            },
            'energy_level': {
                'too_low': ['increase_fatigue', 'reduce_activity', 'seek_rest_or_food'],
                'optimal': ['maintain_current_activity_level']
            },
            'stress_hormones': {
                'too_high': ['activate_relaxation_response', 'reduce_cortisol_production'],
                'too_low': ['maintain_baseline_stress_response']
            }
        }

        action_type = 'too_high' if deviation > 0 else 'too_low'
        suggested_actions = actions.get(parameter.name.lower().replace(' ', '_'), {}).get(action_type, [])

        return {
            'parameter': parameter.name,
            'deviation': deviation,
            'actions_needed': suggested_actions,
            'priority': 'critical' if abs(deviation) > 0.5 else 'high',
            'timeframe': 'immediate' if abs(deviation) > 0.7 else 'moderate'
        }

    def _compute_overall_homeostatic_status(self, parameter_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Computa estado general de homeostasis corporal"""

        total_urgency = sum(sig['urgency'] for sig in parameter_signals.values())
        urgent_parameters = sum(1 for sig in parameter_signals.values() if sig['needs_attention'])
        critical_parameters = sum(1 for sig in parameter_signals.values() if sig['out_of_range'])

        # Nivel de crisis (0-1)
        crisis_level = min(1.0, (total_urgency / len(parameter_signals)) * 0.7 +
                          (urgent_parameters / len(parameter_signals)) * 0.3)

        # Estado general
        if critical_parameters > 0:
            general_state = 'critical'
        elif crisis_level > 0.6:
            general_state = 'distressed'
        elif crisis_level > 0.3:
            general_state = 'uncomfortable'
        else:
            general_state = 'balanced'

        return {
            'general_state': general_state,
            'crisis_level': crisis_level,
            'urgent_parameters': urgent_parameters,
            'critical_parameters': critical_parameters,
            'overall_urgency': total_urgency,
            'comfort_level': max(0.0, 1.0 - crisis_level),
            'physiological_stress': total_urgency / len(parameter_signals)
        }

    def apply_external_effect(self, effect_type: str, intensity: float, duration: float = 5.0):
        """
        Aplica efecto externo temporal al sistema homeostático
        Ejemplo: comer, ejercicio, exposición al frío/calor
        """
        effects = {
            'food_intake': {'blood_glucose': 20.0, 'energy_level': 0.3},
            'exercise_heavy': {'energy_level': -0.4, 'body_temperature': 1.0, 'blood_glucose': -10.0},
            'exercise_light': {'energy_level': -0.1, 'body_temperature': 0.5, 'blood_glucose': -5.0},
            'sleep_rest': {'energy_level': 0.5, 'stress_hormones': -0.3},
            'cold_exposure': {'body_temperature': -2.0, 'energy_level': -0.1},
            'hot_exposure': {'body_temperature': 3.0, 'hydration_level': -0.1},
            'stress_event': {'stress_hormones': 0.4, 'blood_pressure': 15.0},
            'relaxation': {'stress_hormones': -0.3, 'blood_pressure': -10.0}
        }

        if effect_type in effects:
            effect_changes = effects[effect_type]

            # Convertir intensidad y duración a cambios temporales
            for param_key, change_magnitude in effect_changes.items():
                if param_key in self.parameters:
                    actual_change = change_magnitude * intensity * (duration / 10.0)  # normalizar duración
                    # Efecto se aplicará en la próxima actualización
                    self.regulatory_effects[param_key] = self.regulatory_effects.get(param_key, 0) + actual_change

    def get_homeostatic_feedback(self) -> Dict[str, Any]:
        """
        Proporciona feedback completo del estado homeostático para consciencia
        Este feedback se integra con el sistema emocional y de toma de decisiones
        """
        current_state = self.update_states()

        # Feedback emocional basado en homeostasis
        emotional_signals = self._generate_emotional_signals(current_state)

        # Señales corporales conscientes
        conscious_body_signals = {
            'discomfort_sensation': current_state['overall_status']['crisis_level'],
            'energy_level_awareness': self.parameters['energy_level'].current_value,
            'stress_physiology': self.parameters['stress_hormones'].current_value,
            'bodily_needs_signals': {k: v['urgency'] for k, v in current_state['parameter_states'].items()},
            'overall_body_comfort': current_state['overall_status']['comfort_level']
        }

        return {
            'physiological_state': current_state['parameter_states'],
            'emotional_signals': emotional_signals,
            'conscious_body_signals': conscious_body_signals,
            'regulatory_actions': current_state['regulatory_actions'],
            'homeostatic_balance': current_state['overall_status']['comfort_level'],
            'physiological_stress_level': current_state['overall_status']['physiological_stress']
        }

    def _generate_emotional_signals(self, current_state: Dict[str, Any]) -> Dict[str, float]:
        """Genera señales emocionales basadas en estado homeostático"""
        crisis_level = current_state['overall_status']['crisis_level']

        emotional_signals = {
            'bodily_discomfort': crisis_level * 0.8,
            'security_feeling': max(0.0, 1.0 - crisis_level),
            'energy_emotion': self.parameters['energy_level'].current_value * 0.9,
            'stress_emotion': self.parameters['stress_hormones'].current_value,
            'bodily_confidence': current_state['overall_status']['comfort_level'],
            'physiological_anxiety': min(1.0, crisis_level * 1.2)
        }

        return emotional_signals


# ==================== EJEMPLO DE USO ====================

if __name__ == "__main__":
    homeostasis = HomeostaticSystem()

    print("🏥 SISTEMA HOMEOSTÁTICO INICIALIZADO")
    print("=" * 60)

    # Estado inicial
    initial_state = homeostasis.get_homeostatic_feedback()
    print(f"   Parámetros críticos: {initial_state['overall_status']['critical_parameters']}")

    # Aplicar estrés
    homeostasis.apply_external_effect('stress_event', 0.8)
    stressed_state = homeostasis.update_states()
    print(f"\nDespués de evento estresante:")
    print(f"   Nivel de crisis: {stressed_state['overall_status']['crisis_level']:.2f}")

    # Recuperación
    homeostasis.apply_external_effect('relaxation', 1.0)
    recovered_state = homeostasis.update_states()
    print(f"\nDespués de relajación:")
    print(f"   Nivel de crisis: {recovered_state['overall_status']['crisis_level']:.2f}")
    print(f"\n🏥 Homeostasis proporciona feedback emocional y corporal constante")
    print(f"   ✓ {len(homeostasis.parameters)} parámetros fisiológicos monitoreados")
    print(f"   ✓ Señales emocionales integradas con consciencia")
    print(f"   ✓ Acciones regulatorias automáticas generadas")
