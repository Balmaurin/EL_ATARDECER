"""
Espacio Global de Trabajo - Global Workspace

Implementa la teoría Global Workspace (Baars) para integración consciente
de información multimodal. Es el centro de integración donde información
compite por acceso a la consciencia.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import defaultdict
import time
import numpy as np


@dataclass
class WorkspaceEntry:
    """Entrada en el espacio global de trabajo"""
    processor_id: str
    content: Any
    activation_level: float
    timestamp: float = field(default_factory=time.time)
    coalition_members: List[str] = field(default_factory=list)
    priority: float = 0.5


class GlobalWorkspace:
    """
    Espacio Global de Trabajo implementando la teoría de Bernard Baars

    Características principales:
    - Competencia entre procesadores especializados por acceso
    - Umbral de consciencia para integrar información
    - Broadcasting a todos los procesadores
    - Atención selectiva basada en salience y contexto
    """

    def __init__(self,
                 competition_threshold: float = 0.6,
                 max_entries: int = 100,
                 decay_rate: float = 0.9,
                 capacity: Optional[int] = None):
        # Compatibilidad: capacity puede venir como parámetro alternativo
        if capacity is not None:
            max_entries = capacity
        self.competition_threshold = competition_threshold
        self.max_entries = max_entries
        self.capacity = max_entries  # Alias para compatibilidad
        self.decay_rate = decay_rate

        # Buffer de workspace - entradas activas
        self.workspace_buffer = []
        self.competition_history = []

        # Procesadores especializados registrados
        self.specialized_processors = {}
        self.processing_queue = []

        # Estadísticas del workspace
        self.workspace_stats = {
            'total_competitions': 0,
            'successful_integrations': 0,
            'average_activation': 0.0,
            'conscious_overload_events': 0
        }

    def integrate(self, pre_conscious_inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa entradas pre-conscientes y genera contenido consciente integrado

        Args:
            pre_conscious_inputs: Procesos especializados (visión, lenguaje, memoria, etc.)
            context: Contexto de la situación (urgencia, tarea, estado emocional)

        Returns:
            Dict con contenido consciente integrado y metadata
        """
        # 1. Obtener activaciones de cada procesador especializado
        processor_activations = self._compute_processor_activations(
            pre_conscious_inputs, context
        )

        # 2. Competencia por acceso al workspace
        competition_result = self._run_competition_round(
            processor_activations, context
        )

        # 3. Forma la integración consciente si supera umbral
        if competition_result['max_activation'] >= self.competition_threshold:
            conscious_content = self._form_conscious_content(
                competition_result, pre_conscious_inputs, context
            )

            # 4. Broadcasting a todos los procesadores especializados
            self._broadcast_to_specialists(conscious_content)

            # 5. Actualizar estadísticas
            self.workspace_stats['successful_integrations'] += 1

            return conscious_content
        else:
            # Contenido permanece pre-consciente (sub-threshold)
            return {
                'conscious_content': None,
                'status': 'sub-threshold',
                'max_activation': competition_result['max_activation'],
                'activations': processor_activations
            }

    def _compute_processor_activations(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """Calcula nivel de activación para cada procesador especializado"""

        activations = {}
        context_factors = self._extract_context_factors(context)

        for processor_id, content in inputs.items():
            # Activación base del procesador
            base_activation = self._compute_base_activation(content)

            # Factores contextuales
            contextual_multiplier = self._compute_contextual_boosts(
                processor_id, content, context_factors
            )

            # Activar ruido (modela atención no-perfecta)
            activation_noise = np.random.normal(0, 0.1)

            # Activación final
            final_activation = min(1.0, max(0.0,
                base_activation * contextual_multiplier + activation_noise))

            activations[processor_id] = final_activation

        return activations

    def _compute_base_activation(self, content: Any) -> float:
        """Computa activación base basada en diferentes tipos de contenido"""

        if isinstance(content, str):
            # Procesamiento de texto
            activation = min(1.0, len(content.split()) / 100)  # Más palabras = más visible
            if any(word in content.lower() for word in ['urgente', 'importante', 'crítico']):
                activation *= 1.3
            return activation

        elif isinstance(content, (list, tuple)):
            # Procesamiento de listas
            return min(1.0, len(content) / 50)

        elif isinstance(content, dict):
            # Procesamiento de diccionarios (estructurados)
            confidence = content.get('confidence', 0.5)
            novelty = content.get('novelty', 0.5)
            emotional_weight = content.get('emotional_weight', 0.0)

            return (confidence + novelty + emotional_weight) / 3.0

        else:
            # Otros tipos
            return 0.3  # Activación por defecto

    def _extract_context_factors(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extrae factores contextuales que influyen en activaciones"""

        factors = {
            'urgency': context.get('urgency', 0.5),
            'task_relevance': context.get('task_relevance', 0.5),
            'emotional_state': context.get('emotional_state', 0.5),
            'attention_mode': context.get('attention_mode', 'balanced'),  # focused, diffuse, divided
            'vigilance_level': context.get('vigilance_level', 0.7)
        }

        return factors

    def _compute_contextual_boosts(self, processor_id: str, content: Any,
                                  context_factors: Dict[str, float]) -> float:
        """Calcula multiplicadores contextuales para activaciones"""

        multiplier = 1.0

        # Urgencia aumenta saliencia
        multiplier *= (1.0 + context_factors['urgency'] * 0.5)

        # Relevancia de tarea específica
        if processor_id == 'visual' and context_factors.get('visual_required', False):
            multiplier *= 1.8
        elif processor_id == 'language' and context_factors.get('language_required', False):
            multiplier *= 1.8
        elif processor_id == 'memory' and context_factors.get('memory_required', False):
            multiplier *= 1.8

        # Estado emocional
        if context_factors['emotional_state'] > 0.7:  # Alto arousal emocional
            multiplier *= 1.3
        elif context_factors['emotional_state'] < 0.3:  # Bajo arousal
            multiplier *= 0.9

        # Nivel de vigilancia
        multiplier *= context_factors['vigilance_level']

        return multiplier

    def _run_competition_round(self, activations: Dict[str, float],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una ronda de competencia entre procesadores"""

        self.workspace_stats['total_competitions'] += 1

        if not activations:
            return {'max_activation': 0.0, 'winners': [], 'coalition_size': 0}

        # Encontrar los procesadores más activados
        sorted_processors = sorted(activations.items(), key=lambda x: x[1], reverse=True)
        max_activation = sorted_processors[0][1]

        # Identificar ganadores (superan umbral)
        winners = [pid for pid, act in sorted_processors if act >= self.competition_threshold]

        # Record competition
        self.competition_history.append({
            'timestamp': time.time(),
            'activations': activations.copy(),
            'max_activation': max_activation,
            'winner_count': len(winners),
            'threshold': self.competition_threshold
        })

        return {
            'max_activation': max_activation,
            'winners': winners[:5],  # Top 5 máximo
            'coalition_size': len(winners),
            'all_activations': activations
        }

    def _form_conscious_content(self, competition_result: Dict[str, Any],
                               inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Forma contenido consciente integrado desde ganadores"""

        winners = competition_result['winners']

        if not winners:
            return {'conscious_content': None, 'status': 'no_winners'}

        # Integrar contenido de los ganadores
        integrated_content = {
            'primary_focus': inputs.get(winners[0]),  # El más activo
            'supporting_content': {pid: inputs[pid] for pid in winners[1:] if pid in inputs},
            'coalition': winners,
            'context': context,
            'integration_strength': competition_result['max_activation'],
            'diversity_score': len(set(str(type(inputs[pid])) for pid in winners if pid in inputs))
        }

        # Metadata consciente
        conscious_content = {
            'conscious_content': integrated_content,
            'status': 'integrated',
            'confidence': competition_result['max_activation'],
            'integration_time': time.time(),
            'processor_count': len(winners),
            'attention_stability': self._compute_attention_stability(winners),
            'context_alignment': self._compute_context_alignment(integrated_content, context)
        }

        return conscious_content

    def _broadcast_to_specialists(self, conscious_content: Dict[str, Any]):
        """Difunde contenido consciente a todos los procesadores especializados"""

        broadcast_message = {
            'conscious_focus': conscious_content.get('conscious_content', {}).get('primary_focus'),
            'context': conscious_content,
            'timestamp': time.time(),
            'authority': 'global_workspace'
        }

        # Enviar a cada procesador registrado
        for processor_id, processor in self.specialized_processors.items():
            try:
                processor.receive_conscious_broadcast(broadcast_message)
            except Exception as e:
                # Log error pero continúa
                self._log_broadcast_failure(processor_id, str(e))

    def _compute_attention_stability(self, winners: List[str]) -> float:
        """Computa estabilidad de atención basada en historia reciente"""

        if len(self.competition_history) < 3:
            return 0.5  # No suficiente historia

        # Verificar si ganadores recientes son similares
        recent_winners = [comp['winners'][:3] for comp in self.competition_history[-5:]]
        recent_consistent = len(set(tuple(w) for w in recent_winners))

        stability = 1.0 - (recent_consistent / len(recent_winners))
        stability = min(1.0, max(0.0, stability))

        return stability

    def _compute_context_alignment(self, integrated_content: Dict[str, Any],
                                 context: Dict[str, Any]) -> float:
        """Computa qué tan bien se alinea el contenido con el contexto"""

        # Simple implementation - puede ser más sofisticado
        context_keys = set(context.keys())
        content_keys = set()

        for content_item in integrated_content.values():
            if isinstance(content_item, dict):
                content_keys.update(content_item.keys())

        overlap = len(context_keys.intersection(content_keys))
        total = len(context_keys.union(content_keys))

        return overlap / total if total > 0 else 0.0

    def register_specialized_processor(self, processor_id: str, processor: Any):
        """Registra un procesador especializado para broadcasts"""

        self.specialized_processors[processor_id] = processor

    def get_workspace_status(self) -> Dict[str, Any]:
        """Retorna estado completo del workspace"""

        avg_activation = np.mean([
            comp['max_activation'] for comp in self.competition_history[-50:]
        ]) if self.competition_history else 0.0

        return {
            'workspace_entries': len(self.workspace_buffer),
            'registered_processors': len(self.specialized_processors),
            'competition_threshold': self.competition_threshold,
            'total_competitions': self.workspace_stats['total_competitions'],
            'successful_integrations': self.workspace_stats['successful_integrations'],
            'average_activation': avg_activation,
            'attention_stability': self._compute_attention_stability([]),
            'conscious_overload_events': self.workspace_stats['conscious_overload_events']
        }

    def _log_broadcast_failure(self, processor_id: str, error: str):
        """Log errores de broadcasting (para debugging)"""
        print(f"Warning: Broadcast to {processor_id} failed: {error}")

    def clear_workspace(self):
        """Limpia el workspace (útil para reset)"""
        self.workspace_buffer.clear()
        self.competition_history.clear()
