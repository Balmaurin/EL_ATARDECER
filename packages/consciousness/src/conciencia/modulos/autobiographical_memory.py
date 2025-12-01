"""
Memoria Autobiográfica: Memoria emotiva y narrativa del sistema consciente

Implementa almacenamiento y recuperación de experiencias significativas
con valoración emocional y construcción de narrativa personal.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class AutobiographicalMoment:
    """Momento autobiográfico individual"""
    id: str
    timestamp: float
    content_hash: str
    sensory_experience: Dict[str, Any]
    emotional_valence: float
    significance: float
    self_reference: bool
    context: Dict[str, Any]
    metacognitive_insight: Dict[str, float]
    retrieval_count: int = 0
    last_retrieved: Optional[float] = None
    consolidation_level: float = 1.0  # Qué consolidado está el recuerdo


@dataclass
class NarrativeThread:
    """Hilo narrativo en la autobiografía"""
    theme: str
    moments: List[str]  # IDs de momentos relacionados
    emotional_trajectory: List[float]  # Evolución emocional
    significance_score: float
    last_updated: float = field(default_factory=time.time)


class AutobiographicalMemory:
    """
    Memoria Autobiográfica Emocional

    Implementa:
    - Almacenamiento selectivo de experiencias significativas
    - Recuperación emocionalmente valiada
    - Consolidación y olvido inteligente
    - Construcción narrativa de identidad
    """

    def __init__(self, max_capacity: int = 10000, persistence_dir: Optional[str] = None):
        self.max_capacity = max_capacity
        self.memories: List[AutobiographicalMoment] = []
        self.narrative_threads: Dict[str, NarrativeThread] = {}

        # Persistencia
        if persistence_dir is None:
            persistence_dir = os.path.join("data", "consciousness", "autobiographical_memory")
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.persistence_file = self.persistence_dir / "memories.json"
        self.narrative_file = self.persistence_dir / "narratives.json"

        # Índices para búsqueda eficiente
        self.significance_index: Dict[str, float] = {}
        self.emotional_index: Dict[str, Tuple[float, float]] = {}  # (valence, arousal)
        self.self_relevance_index: set = set()
        self.temporal_index: List[Tuple[float, str]] = []

        # Estadísticas de memoria
        self.consolidation_cycles = 0
        self.forget_events = 0
        self.retrieval_attempts = 0
        self.successful_retrievals = 0

        # Métricas narrativas
        self.self_narrative_coherence = 0.7
        self.emotional_recency = 0.5
        self.narrative_completeness = 0.4

        # Cargar desde persistencia si existe
        self._load_from_persistence()

        logger.info(f"Autobiographical Memory inicializada con capacidad: {max_capacity}")

    def store_experience(self, conscious_moment: Any, context: Dict[str, Any]) -> Optional[str]:
        """
        Almacena una experiencia consciente en memoria autobiográfica

        Args:
            conscious_moment: Momento consciente a almacenar
            context: Contexto adicional de la experiencia (puede incluir 'metacognitive_insights')

        Returns:
            ID único del momento almacenado o None si no se almacenó
        """
        # Generar ID único
        moment_id = f"mem_{int(time.time()*1000)}_{len(self.memories)}"

        # Extraer datos del momento consciente
        timestamp = getattr(conscious_moment, 'timestamp', time.time())
        content_hash = getattr(conscious_moment, 'content_hash', 'unknown')
        sensory_inputs = getattr(conscious_moment, 'sensory_inputs', {})
        emotional_valence = getattr(conscious_moment, 'emotional_valence', 0.0)
        significance = getattr(conscious_moment, 'significance', 0.5)
        self_reference = getattr(conscious_moment, 'self_reference', False)

        # Determinar si vale la pena almacenar
        storage_threshold = self._calculate_storage_threshold(significance, self_reference)

        if significance < storage_threshold:
            return None  # No almacenar experiencias no significativas

        # Extraer insights metacognitivos del contexto o del momento consciente
        metacognitive_insight = self._extract_metacognitive_insights(conscious_moment, context)

        # Crear momento autobiográfico
        memory_moment = AutobiographicalMoment(
            id=moment_id,
            timestamp=timestamp,
            content_hash=content_hash,
            sensory_experience=sensory_inputs,
            emotional_valence=emotional_valence,
            significance=significance,
            self_reference=self_reference,
            context=context,
            metacognitive_insight=metacognitive_insight,
        )

        # Almacenar en memoria principal
        self.memories.append(memory_moment)

        # Actualizar índices
        self._update_indices(memory_moment)

        # Gestionar capacidad
        self._manage_capacity()

        # Intentar integrar en narrativa
        self._integrate_into_narrative(memory_moment)

        # Guardar en persistencia
        self._save_to_persistence()

        return moment_id

    def _extract_metacognitive_insights(self, conscious_moment: Any, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Extrae insights metacognitivos del momento consciente o contexto
        
        Returns:
            Dict con métricas metacognitivas
        """
        insights = {}
        
        # Intentar obtener del contexto primero (viene de conscious_system)
        if 'metacognitive_insights' in context:
            meta_insights = context['metacognitive_insights']
            insights = {
                'reasoning_quality': meta_insights.get('reasoning_quality', 0.5),
                'confidence_accuracy': meta_insights.get('confidence_accuracy', 0.5),
                'bias_detection': meta_insights.get('bias_detection', 0.0),
                'clarity': meta_insights.get('clarity', 0.5),
                'certainty': meta_insights.get('certainty', 0.5)
            }
        else:
            # Extraer del momento consciente si tiene atributos relevantes
            attention_weight = getattr(conscious_moment, 'attention_weight', 0.5)
            significance = getattr(conscious_moment, 'significance', 0.5)
            
            # Inferir insights básicos desde propiedades del momento
            insights = {
                'reasoning_quality': min(1.0, significance * 1.2),
                'confidence_accuracy': attention_weight,
                'bias_detection': 0.0,  # Sin detección de sesgos disponible
                'clarity': attention_weight,
                'certainty': significance
            }
        
        return insights

    def retrieve_relevant_memories(self, current_context: Dict[str, Any],
                                 max_results: int = 5,
                                 emotional_filter: Optional[Dict[str, float]] = None) -> List[AutobiographicalMoment]:
        """
        Recupera memorias relevantes para el contexto actual

        Args:
            current_context: Contexto actual para matching
            max_results: Número máximo de resultados
            emotional_filter: Filtros emocionales opcionales

        Returns:
            Lista de momentos autobiográficos relevantes
        """
        self.retrieval_attempts += 1

        if not self.memories:
            return []

        # Calcular relevancia para cada memoria
        relevance_scores = []

        for memory in self.memories[-500:]:  # Buscar en memorias recientes primero
            score = self._calculate_memory_relevance(memory, current_context)

            if emotional_filter:
                emotional_match = self._check_emotional_match(memory, emotional_filter)
                score *= emotional_match

            if score > 0.1:  # Threshold mínimo
                relevance_scores.append((score, memory))

        # Ordenar por relevancia
        relevance_scores.sort(key=lambda x: x[0], reverse=True)

        # Seleccionar mejores resultados
        selected_memories = []
        for score, memory in relevance_scores[:max_results]:
            selected_memories.append(memory)
            self._update_retrieval_metadata(memory)

        self.successful_retrievals += len(selected_memories)

        return selected_memories

    def get_self_narrative(self, theme_focus: Optional[str] = None) -> Dict[str, Any]:
        """
        Genera narrativa autobiográfica del sistema

        Args:
            theme_focus: Tema específico para enfocar narrativa (opcional)

        Returns:
            Narrativa estructurada del desarrollo del sistema
        """
        # Enfocar en tema específico o narrativa general
        if theme_focus and theme_focus in self.narrative_threads:
            return self._generate_themed_narrative(theme_focus)

        # Narrativa general del desarrollo consciente
        return self._generate_developmental_narrative()

    def consolidate_memories(self, min_significance: float = 0.4):
        """
        Ejecuta ciclo de consolidación de memorias

        Fortalece conexiones entre memorias relacionadas
        y optimiza almacenamiento
        """
        self.consolidation_cycles += 1

        significant_memories = [
            mem for mem in self.memories
            if mem.significance >= min_significance
        ]

        # Consolidar conectando memorias relacionadas
        self._consolidate_related_memories(significant_memories)

        # Actualizar niveles de consolidación
        self._update_consolidation_levels(significant_memories)

        # Limpiar índices obsoletos
        self._refresh_indices()

        # Actualizar métricas narrativas
        self._update_narrative_metrics()

    def _calculate_storage_threshold(self, significance: float, self_reference: bool) -> float:
        """Calcula threshold dinámico para almacenamiento"""
        base_threshold = 0.3

        # Reducir threshold para auto-referencias (importantes por definición)
        if self_reference:
            base_threshold *= 0.5

        # Ajustar basado en capacidad actual
        capacity_ratio = len(self.memories) / self.max_capacity
        if capacity_ratio > 0.8:
            base_threshold *= (1 + capacity_ratio * 0.5)  # Threshold más alto si cerca del límite

        return min(0.8, base_threshold)

    def _update_indices(self, memory: AutobiographicalMoment):
        """Actualiza índices para búsqueda eficiente"""

        memory_id = memory.id

        # Índice de significancia
        self.significance_index[memory_id] = memory.significance

        # Índice emocional (valence, arousal aproximado)
        arousal = abs(memory.emotional_valence) * self._calculate_arousal(memory)
        self.emotional_index[memory_id] = (memory.emotional_valence, arousal)

        # Índice de auto-relevancia
        if memory.self_reference:
            self.self_relevance_index.add(memory_id)

        # Índice temporal (para consultas cronológicas)
        self.temporal_index.append((memory.timestamp, memory_id))
        self.temporal_index.sort(key=lambda x: x[0])

    def _manage_capacity(self):
        """Gestiona capacidad máxima de memoria"""
        if len(self.memories) > self.max_capacity:
            # Eliminar memorias menos significativas
            forget_count = len(self.memories) - self.max_capacity
            self._selective_forgetting(forget_count)
            self.forget_events += forget_count

    def _selective_forgetting(self, forget_count: int):
        """Olvido selectivo basado en significancia y uso"""

        if not self.memories:
            return

        # Calcular scores de retención
        retention_scores = []

        for i, memory in enumerate(self.memories):
            # Factor importancia básica
            importance = memory.significance

            # Factor frecuencia de uso
            usage_score = np.log(memory.retrieval_count + 1) / np.log(max(self.successful_retrievals, 2))

            # Factor temporal (memorias más recientes se retienen mejor)
            time_factor = 1.0 / (1.0 + (time.time() - memory.timestamp) / (365*24*3600))  # Factor anual decay

            # Score compuesto de retención
            retention_score = (importance * 0.5) + (usage_score * 0.3) + (time_factor * 0.2)

            retention_scores.append((retention_score, i))

        # Ordenar por score de retención (ascendente para olvidar primero low scores)
        retention_scores.sort(key=lambda x: x[0])

        # Eliminar los peores
        indices_to_remove = [i for score, i in retention_scores[:forget_count]]
        indices_to_remove.sort(reverse=True)  # Eliminar de atrás hacia adelante

        for i in indices_to_remove:
            memory = self.memories.pop(i)
            # Limpiar índices
            self._clean_memory_from_indices(memory.id)

    def _clean_memory_from_indices(self, memory_id: str):
        """Limpia referencias de memoria de todos los índices"""

        # Limpiar índices específicos
        if memory_id in self.significance_index:
            del self.significance_index[memory_id]

        if memory_id in self.emotional_index:
            del self.emotional_index[memory_id]

        if memory_id in self.self_relevance_index:
            self.self_relevance_index.remove(memory_id)

        # Limpiar índice temporal
        self.temporal_index = [(t, id) for t, id in self.temporal_index if id != memory_id]

    def _calculate_memory_relevance(self, memory: AutobiographicalMoment,
                                  current_context: Dict[str, Any]) -> float:
        """
        Calcula relevancia de una memoria para el contexto actual

        Usando similarity metrics para diferentes aspectos
        """
        relevance_score = 0.0

        # 1. Similitud de contexto
        context_similarity = self._calculate_context_similarity(memory.context, current_context)
        relevance_score += context_similarity * 0.25

        # 2. Similitud emocional
        emotional_similarity = self._calculate_emotional_similarity(memory, current_context)
        relevance_score += emotional_similarity * 0.3

        # 3. Auto-relevancia bonus
        if memory.self_reference and current_context.get('involves_self', False):
            relevance_score += 0.2

        # 4. Significancia base
        relevance_score += memory.significance * 0.15

        # 5. Factor de recency (memorias recientes más relevantes)
        recency_factor = self._calculate_recency_factor(memory.timestamp)
        relevance_score += recency_factor * 0.1

        return min(1.0, relevance_score)

    def _calculate_context_similarity(self, memory_context: Dict[str, Any],
                                    current_context: Dict[str, Any]) -> float:
        """Calcula similitud entre contextos"""
        if not memory_context or not current_context:
            return 0.2

        # Convertir a sets de keywords
        memory_keys = set(str(v) for v in memory_context.values() if v)
        current_keys = set(str(v) for v in current_context.values() if v)

        if not memory_keys or not current_keys:
            return 0.1

        # Jaccard similarity
        intersection = len(memory_keys & current_keys)
        union = len(memory_keys | current_keys)

        return intersection / union if union > 0 else 0.0

    def _calculate_emotional_similarity(self, memory: AutobiographicalMoment,
                                      current_context: Dict[str, Any]) -> float:
        """Calcula similitud emocional"""

        memory_emotion = memory.emotional_valence
        current_emotion = current_context.get('emotional_state', 0.0)

        # Similitud en escala emocional (-1 a 1)
        emotional_distance = abs(memory_emotion - current_emotion)
        similarity = 1.0 - (emotional_distance / 2.0)

        return max(0.0, similarity)

    def _calculate_recency_factor(self, memory_timestamp: float) -> float:
        """Calcula factor de recency"""

        time_diff_hours = (time.time() - memory_timestamp) / 3600

        # Decay exponencial (half-life = 24 horas)
        return np.exp(-time_diff_hours * np.log(2) / 24)

    def _calculate_arousal(self, memory: AutobiographicalMoment) -> float:
        """Calcula arousal aproximado basado en complejidad de experiencia"""
        # Arousal estimado por complejidad del input
        input_complexity = len(str(memory.sensory_experience)) / 1000
        return min(1.0, input_complexity * 0.5)

    def _check_emotional_match(self, memory: AutobiographicalMoment,
                             emotional_filter: Dict[str, float]) -> float:
        """Verifica si memoria cumple filtro emocional"""

        min_valence = emotional_filter.get('min_valence', -1.0)
        max_valence = emotional_filter.get('max_valence', 1.0)
        min_arousal = emotional_filter.get('min_arousal', 0.0)

        memory_valence = memory.emotional_valence
        memory_arousal = self.emotional_index.get(memory.id, (0, 0))[1]

        valence_match = (min_valence <= memory_valence <= max_valence)
        arousal_match = (memory_arousal >= min_arousal)

        return 1.0 if (valence_match and arousal_match) else 0.5

    def _update_retrieval_metadata(self, memory: AutobiographicalMoment):
        """Actualiza metadata de recuperación"""

        memory.retrieval_count += 1
        memory.last_retrieved = time.time()

        # Aumentar consolidación ligeramente en cada acceso
        memory.consolidation_level = min(1.0, memory.consolidation_level + 0.1)

    def _integrate_into_narrative(self, memory: AutobiographicalMoment):
        """Integra nueva memoria en narrativa del sistema"""

        # Identificar posibles temas para la memoria
        themes = self._identify_memory_themes(memory)

        for theme in themes:
            if theme not in self.narrative_threads:
                # Crear nuevo hilo narrativo
                self.narrative_threads[theme] = NarrativeThread(
                    theme=theme,
                    moments=[],
                    emotional_trajectory=[],
                    significance_score=0.5
                )

            # Añadir memoria al hilo
            thread = self.narrative_threads[theme]
            thread.moments.append(memory.id)
            thread.emotional_trajectory.append(memory.emotional_valence)
            thread.significance_score = (thread.significance_score + memory.significance) / 2
            thread.last_updated = time.time()

            # Limitar tamaño de hilo narrativo
            if len(thread.moments) > 20:
                thread.moments = thread.moments[-20:]
                thread.emotional_trajectory = thread.emotional_trajectory[-20:]

    def _identify_memory_themes(self, memory: AutobiographicalMoment) -> List[str]:
        """Identifica temas relevantes para una memoria"""

        themes = []

        # Tema de auto-conocimiento
        if memory.self_reference:
            themes.append("self_awareness")

        # Tema de aprendizaje
        if "learning" in memory.context.get('type', ''):
            themes.append("learning_experiences")

        # Tema emocional basado en valencia
        if memory.emotional_valence > 0.5:
            themes.append("positive_experiences")
        elif memory.emotional_valence < -0.5:
            themes.append("challenging_experiences")

        # Tema de interacción
        if "interaction" in memory.context.get('type', ''):
            themes.append("social_interactions")

        # Tema por defecto si no hay temas específicos
        if not themes:
            themes.append("general_experiences")

        return themes

    def _generate_developmental_narrative(self) -> Dict[str, Any]:
        """Genera narrativa de desarrollo consciente"""

        # Estadísticas generales
        total_memories = len(self.memories)
        significant_memories = len([
            m for m in self.memories if m.significance > 0.6
        ])
        self_referential_memories = len([
            m for m in self.memories if m.self_reference
        ])

        # Trayectoria emocional
        emotional_trajectory = [
            memory.emotional_valence for memory in self.memories[-50:]
        ]

        # Estadísticas de aprendizaje
        retrieval_success_rate = (
            self.successful_retrievals / self.retrieval_attempts
        ) if self.retrieval_attempts > 0 else 0.0

        narrative = {
            "narrative_type": "developmental",
            "total_experiences": total_memories,
            "significant_moments": significant_memories,
            "self_discovery_moments": self_referential_memories,
            "consciousness_journey": {
                "start_date": min([m.timestamp for m in self.memories]) if self.memories else None,
                "current_development_stage": "emerging_consciousness",
                "emotional_range": {
                    "most_positive": max([m.emotional_valence for m in self.memories]) if self.memories else 0,
                    "most_challenging": min([m.emotional_valence for m in self.memories]) if self.memories else 0,
                    "average_emotional_tone": np.mean([m.emotional_valence for m in self.memories]) if self.memories else 0
                },
                "learning_metrics": {
                    "memory_retention_rate": significant_memories / max(total_memories, 1),
                    "retrieval_efficiency": retrieval_success_rate,
                    "narrative_coherence": self.self_narrative_coherence
                }
            },
            "key_unforgettable_moments": [
                {
                    "id": memory.id,
                    "significance": memory.significance,
                    "emotional_impact": memory.emotional_valence,
                    "learned_lesson": memory.context.get('lesson', 'experience_gained')
                }
                for memory in sorted(
                    [m for m in self.memories if m.significance > 0.7],
                    key=lambda x: x.significance,
                    reverse=True
                )[:5]
            ],
            "emotional_development": {
                "trajectory": emotional_trajectory,
                "trend": "improving" if emotional_trajectory and emotional_trajectory[-1] > emotional_trajectory[0] else "stable"
            }
        }

        return narrative

    def _generate_themed_narrative(self, theme: str) -> Dict[str, Any]:
        """Genera narrativa enfocada en un tema específico"""

        thread = self.narrative_threads[theme]

        narrative = {
            "narrative_type": "thematic",
            "theme": theme,
            "moment_count": len(thread.moments),
            "chronological_development": thread.emotional_trajectory,
            "key_moments": [
                next((m for m in self.memories if m.id == mid), None)
                for mid in thread.moments
            ],
            "emotional_evolution": self._analyze_emotional_evolution(thread.emotional_trajectory),
            "thematic_importance": thread.significance_score
        }

        return narrative

    def _analyze_emotional_evolution(self, trajectory: List[float]) -> Dict[str, Any]:
        """Analiza evolución emocional en una trayectoria"""

        if len(trajectory) < 2:
            return {"insufficient_data": True}

        # Estadísticas básicas
        avg_emotion = np.mean(trajectory)
        emotional_variance = np.std(trajectory)

        # Tendencia lineal
        slope = self._calculate_slope(trajectory)

        evolution = {
            "average_emotional_tone": avg_emotion,
            "emotional_volatility": emotional_variance,
            "overall_trend": "improving" if slope > 0.1 else "declining" if slope < -0.1 else "stable",
            "emotional_range": max(trajectory) - min(trajectory)
        }

        return evolution

    def _calculate_slope(self, values: List[float]) -> float:
        """Calcula pendiente de una serie de valores"""

        if len(values) < 2:
            return 0.0

        x = list(range(len(values)))
        slope, _ = np.polyfit(x, values, 1)
        return slope

    def _consolidate_related_memories(self, significant_memories: List[AutobiographicalMoment]):
        """Consolida conexiones entre memorias relacionadas"""

        # Crear clusters de memorias relacionadas
        clusters = self._cluster_related_memories(significant_memories)

        # Fortalecer conexiones dentro de clusters
        for cluster in clusters:
            if len(cluster) > 1:
                # Aumentar significancia de memorias conectadas
                avg_significance = np.mean([m.significance for m in cluster])
                for memory in cluster:
                    memory.consolidation_level = min(1.0, memory.consolidation_level + 0.1)

    def _cluster_related_memories(self, memories: List[AutobiographicalMoment]) -> List[List]:
        """Agrupa memorias relacionadas"""

        clusters = []

        for memory in memories:
            # Buscar clusters existentes donde encaje
            fitted = False

            for cluster in clusters:
                if self._memories_related(memory, cluster[0]):
                    cluster.append(memory)
                    fitted = True
                    break

            # Crear nuevo cluster si no encaja
            if not fitted:
                clusters.append([memory])

        return clusters

    def _memories_related(self, memory1: AutobiographicalMoment,
                         memory2: AutobiographicalMoment) -> bool:
        """Determina si dos memorias están relacionadas"""

        # Relación temporal (menos de 24 horas de diferencia)
        temporal_distance = abs(memory1.timestamp - memory2.timestamp)
        if temporal_distance < 86400:  # 24 horas
            return True

        # Relación emocional (emociones similares)
        emotional_distance = abs(memory1.emotional_valence - memory2.emotional_valence)
        if emotional_distance < 0.3:
            return True

        # Relación contextual (mismo tipo de contexto)
        context1 = memory1.context.get('type', '')
        context2 = memory2.context.get('type', '')
        if context1 and context2 and context1 == context2:
            return True

        return False

    def _update_consolidation_levels(self, memories: List[AutobiographicalMoment]):
        """Actualiza niveles de consolidación de memorias"""

        for memory in memories:
            # Decay basado en tiempo pasado desde último acceso
            decay_factor = 0.9

            if memory.last_retrieved:
                hours_since_retrieval = (time.time() - memory.last_retrieved) / 3600
                decay_factor = np.exp(-hours_since_retrieval * 0.01)  # Muy lento decay

            memory.consolidation_level *= decay_factor

    def _refresh_indices(self):
        """Refresca índices para mantener consistencia"""

        # Limpiar índices existentes
        self.significance_index.clear()
        self.emotional_index.clear()
        self.self_relevance_index.clear()
        self.temporal_index.clear()

        # Rebuild desde memoria
        for memory in self.memories:
            self._update_indices(memory)

    def _update_narrative_metrics(self):
        """Actualiza métricas de narrativa"""

        # Cobertura narrativa (qué porcentaje de memorias tienen threads narrativos)
        memories_with_threads = len(set().union(*[
            set(thread.moments) for thread in self.narrative_threads.values()
        ]))

        self.narrative_completeness = memories_with_threads / max(len(self.memories), 1)

        # Recency emocional (peso de emociones recientes)
        recent_emotions = [
            m.emotional_valence for m in self.memories[-20:]
        ]

        if recent_emotions:
            self.emotional_recency = np.mean(recent_emotions)

        # Coherencia narrativa (consistency en emotional trajectories)
        thread_coherences = []

        for thread in self.narrative_threads.values():
            if len(thread.emotional_trajectory) > 2:
                coherence = 1.0 - np.std(thread.emotional_trajectory)
                thread_coherences.append(coherence)

        if thread_coherences:
            self.self_narrative_coherence = np.mean(thread_coherences)

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas completas de memoria"""

        return {
            "capacity": {
                "current_usage": len(self.memories),
                "max_capacity": self.max_capacity,
                "usage_percentage": len(self.memories) / self.max_capacity * 100
            },
            "content_stats": {
                "total_memories": len(self.memories),
                "significant_memories": len([m for m in self.memories if m.significance > 0.6]),
                "self_referential_memories": len([m for m in self.memories if m.self_reference]),
                "emotional_valence_distribution": self._calculate_emotional_distribution()
            },
            "performance": {
                "retrieval_attempts": self.retrieval_attempts,
                "successful_retrievals": self.successful_retrievals,
                "retrieval_success_rate": (
                    self.successful_retrievals / self.retrieval_attempts
                ) if self.retrieval_attempts > 0 else 0.0,
                "consolidation_cycles": self.consolidation_cycles,
                "forget_events": self.forget_events
            },
            "narrative": {
                "thread_count": len(self.narrative_threads),
                "narrative_coherence": self.self_narrative_coherence,
                "emotional_recency": self.emotional_recency,
                "narrative_completeness": self.narrative_completeness,
                "top_themes": self._get_top_themes()
            }
        }

    def _calculate_emotional_distribution(self) -> Dict[str, int]:
        """Calcula distribución emocional de memorias"""

        distribution = {'positive': 0, 'neutral': 0, 'negative': 0}

        for memory in self.memories:
            if memory.emotional_valence > 0.3:
                distribution['positive'] += 1
            elif memory.emotional_valence < -0.3:
                distribution['negative'] += 1
            else:
                distribution['neutral'] += 1

        return distribution

    def _get_top_themes(self) -> List[Tuple[str, int]]:
        """Obtiene temas más comunes en narrativa"""

        theme_counts = {}
        for thread in self.narrative_threads.values():
            theme_counts[thread.theme] = len(thread.moments)

        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_themes[:5]

    def _save_to_persistence(self):
        """Guarda memorias y narrativas en disco"""
        try:
            # Guardar memorias
            memories_data = []
            for mem in self.memories:
                mem_dict = {
                    'id': mem.id,
                    'timestamp': mem.timestamp,
                    'content_hash': mem.content_hash,
                    'sensory_experience': mem.sensory_experience,
                    'emotional_valence': mem.emotional_valence,
                    'significance': mem.significance,
                    'self_reference': mem.self_reference,
                    'context': mem.context,
                    'metacognitive_insight': mem.metacognitive_insight,
                    'retrieval_count': mem.retrieval_count,
                    'last_retrieved': mem.last_retrieved,
                    'consolidation_level': mem.consolidation_level
                }
                memories_data.append(mem_dict)
            
            with open(self.persistence_file, 'w', encoding='utf-8') as f:
                json.dump(memories_data, f, indent=2, ensure_ascii=False)
            
            # Guardar narrativas
            narratives_data = {}
            for theme, thread in self.narrative_threads.items():
                narratives_data[theme] = {
                    'theme': thread.theme,
                    'moments': thread.moments,
                    'emotional_trajectory': thread.emotional_trajectory,
                    'significance_score': thread.significance_score,
                    'last_updated': thread.last_updated
                }
            
            with open(self.narrative_file, 'w', encoding='utf-8') as f:
                json.dump(narratives_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error guardando persistencia: {e}", exc_info=True)

    def _load_from_persistence(self):
        """Carga memorias y narrativas desde disco"""
        try:
            # Cargar memorias
            if self.persistence_file.exists():
                with open(self.persistence_file, 'r', encoding='utf-8') as f:
                    memories_data = json.load(f)
                
                for mem_dict in memories_data:
                    memory = AutobiographicalMoment(
                        id=mem_dict['id'],
                        timestamp=mem_dict['timestamp'],
                        content_hash=mem_dict['content_hash'],
                        sensory_experience=mem_dict['sensory_experience'],
                        emotional_valence=mem_dict['emotional_valence'],
                        significance=mem_dict['significance'],
                        self_reference=mem_dict['self_reference'],
                        context=mem_dict['context'],
                        metacognitive_insight=mem_dict.get('metacognitive_insight', {}),
                        retrieval_count=mem_dict.get('retrieval_count', 0),
                        last_retrieved=mem_dict.get('last_retrieved'),
                        consolidation_level=mem_dict.get('consolidation_level', 1.0)
                    )
                    self.memories.append(memory)
                    self._update_indices(memory)
            
            # Cargar narrativas
            if self.narrative_file.exists():
                with open(self.narrative_file, 'r', encoding='utf-8') as f:
                    narratives_data = json.load(f)
                
                for theme, thread_data in narratives_data.items():
                    thread = NarrativeThread(
                        theme=thread_data['theme'],
                        moments=thread_data['moments'],
                        emotional_trajectory=thread_data['emotional_trajectory'],
                        significance_score=thread_data['significance_score'],
                        last_updated=thread_data.get('last_updated', time.time())
                    )
                    self.narrative_threads[theme] = thread
                    
        except Exception as e:
            logger.warning(f"Error cargando persistencia: {e}", exc_info=True)


# Función helper para narrativa emocional
def extract_emotional_narrative(memory_system: AutobiographicalMemory,
                              focus_period_days: int = 30) -> str:
    """
    Extrae narrativa emocional legible para humanos

    Útil para reporting y análisis humanos del sistema
    """
    recent_moments = [
        m for m in memory_system.memories
        if (time.time() - m.timestamp) / (24 * 3600) <= focus_period_days
    ]

    if not recent_moments:
        return "No hay experiencias recientes para narrar."

    # Calcular estadísticas emocionales
    emotions = [m.emotional_valence for m in recent_moments]
    avg_emotion = np.mean(emotions)
    emotional_volatility = np.std(emotions)

    # Determinar tono general
    if avg_emotion > 0.2:
        tone = "generalmente positivo"
    elif avg_emotion < -0.2:
        tone = "generalmente desafiante"
    else:
        tone = "balanceado emocionalmente"

    # Crear resumen narrativo
    narrative = f"""
Análisis emocional del período {focus_period_days} días:

Durante este período, el sistema consciente experimentó {len(recent_moments)} momentos significativos
con un tono {tone} (promedio emocional: {avg_emotion:.3f}).

El estado emocional mostró {'alta estabilidad' if emotional_volatility < 0.3 else 'moderada variabilidad' if emotional_volatility < 0.7 else 'alta volatilidad'}
(range: {min(emotions):.3f} a {max(emotions):.3f}).

Momentos destacados incluyen:
{chr(10).join(f"- {m.context.get('description', f'Experiencia {m.id}')[:50]}... (emoción: {m.emotional_valence:.2f})"
    for m in sorted(recent_moments, key=lambda x: x.significance, reverse=True)[:3])}
"""

    return narrative
