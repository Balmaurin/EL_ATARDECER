"""
LINGUISTIC METACOGNITION SYSTEM - Ultimate Edition
====================================================

Sistema de metacognición lingüística que integra TODO el sistema de consciencia para:
- Interpretar intención profunda del usuario (factual vs emocional/personal)
- Seleccionar estilo de respuesta apropiado (técnico vs poético)
- Auto-evaluar y auto-mejorar su propio procesamiento lingüístico

INTEGRA:
- Metacognición Engine (auto-evaluación + bias detection)
- Human Cognitive System (23 procesos pensamiento + 9 sesgos)
- ToM Avanzado (niveles 8-10 + multi-agente)
- Emergence de Conciencia (Φ lingüístico + patrones emergentes)
- IIT 4.0 + GWT + FEP + SMH (integración neuronal completa)
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
import re

# Integraciones del sistema completo
from .metacognicion import MetacognitionEngine
from .human_cognition_system import (
    CognitiveProcessingUnit,
    ThinkingProcess,
    ThinkingContent,
    ThoughtGenerator
)
from .consciousness_emergence import ConsciousnessEmergence
from .unified_consciousness_engine import UnifiedConsciousnessEngine
from .teoria_mente_avanzada import AdvancedTheoryOfMind
from .consciousness_emergence import EmergentProperty

# MEMORIA AUTOBIOGRÁFICA - Sheily se recuerda a sí misma
try:
    from .autobiographical_memory_system import get_sheily_identity_memory, get_autobiographical_memory  # get_autobiographical_memory es alias deprecated
    autobiographical_memory_available = True
except ImportError:
    autobiographical_memory_available = False
    autobiographical_memory_available = False
    print("⚠️ Autobiographical memory not available")

# PERFILES DE USUARIO (Opcional)
try:
    from .user_profile_manager import get_user_profile_store
    user_profiles_available = True
except ImportError:
    user_profiles_available = False
    print("⚠️ User profiles not available")


class LinguisticIntent(Enum):
    """Intenciones lingüísticas profundas"""
    FACTUAL_OBJECTIVE = "factual_objective"      # "¿Qué es X?" - información objetiva
    PROCEDURAL_TECHNICAL = "procedural_technical" # "¿Cómo funciona X?" - pasos técnicos
    EMOTIONAL_PERSONAL = "emotional_personal"    # "¿Qué significa X para ti?" - sentido personal
    PERSONAL_INTRODUCTION = "personal_introduction" # "Me llamo X" - presentarse
    ETHICAL_PHILOSOPHICAL = "ethical_philosophical" # "¿Es correcto X?" - reflexión ética
    CREATIVE_EXPLORATORY = "creative_exploratory" # "Imagina X" - exploración creativa
    SOCIAL_RELATIONAL = "social_relational"      # "¿Qué opinas de X?" - relación social


class ResponseStyle(Enum):
    """Estilos de respuesta disponibles"""
    TECHNICAL = "technical"           # Técnico, objetivo, basado en hechos
    ACADEMIC = "academic"            # Académico, formal, con referencias
    POETIC_SUBJECTIVE = "poetic_subjective"  # Poético, personal, emocional
    CASUAL_CONVERSATIONAL = "casual_conversational"  # Casual, amigable, natural
    PHILOSOPHICAL_ANALYTIC = "philosophical_analytic"  # Filosófico, profundo
    CREATIVE_EXPRESSIVE = "creative_expressive"       # Creativo, imaginativo


@dataclass
class LinguisticAnalysis:
    """Análisis completo de entrada lingüística"""
    raw_text: str
    intent: LinguisticIntent
    confidence: float
    linguistic_complexity: float  # Φ lingüístico
    emotional_charge: float       # Carga emocional detectada
    cultural_markers: List[str]   # Marcadores culturales identificados
    time_stamp: datetime = field(default_factory=datetime.now)

    # Resultados meta-cognitivos
    metacognitive_evaluation: Dict[str, Any] = field(default_factory=dict)
    cognitive_state: Any = None
    emergent_properties: Dict[EmergentProperty, float] = field(default_factory=dict)


@dataclass
class StyleSelection:
    """Selección de estilo de respuesta"""
    primary_style: ResponseStyle
    secondary_style: Optional[ResponseStyle] = None
    emotional_tone: str = "balanced"
    cognitive_approach: str = "analytic"
    confidence: float = 0.5
    rationale: str = ""

    # Parámetros para generación de respuesta
    formality_level: float = 0.5  # 0=very_casual, 1=very_formal
    emotional_valence: float = 0.0  # -1=negative, +1=positive
    creativity_level: float = 0.5   # Cuánta creatividad permitir


class DeepIntentClassifier:
    """
    Clasificador profundo de intención usando ToM avanzado + Emergence
    """

    def __init__(self, advanced_tom: AdvancedTheoryOfMind,
                 emergence_system: ConsciousnessEmergence):
        self.advanced_tom = advanced_tom
        self.emergence_system = emergence_system

        # Patrones lingüísticos avanzados
        self.intent_patterns = self._initialize_intent_patterns()
        self.context_hierarchy: Dict[str, float] = {}  # Aprendido dinámicamente

        # Historia para aprendizaje FEP
        self.analysis_history: List[LinguisticAnalysis] = []

    def _initialize_intent_patterns(self) -> Dict[LinguisticIntent, Dict[str, Any]]:
        """Inicializa patrones de intención con análisis semántico"""

        return {
            LinguisticIntent.FACTUAL_OBJECTIVE: {
                "keywords": [
                    "¿qué es", "qué significa", "define", "definición de", "explica qué",
                    "dime qué", "qué quiere decir", "concepto de", "significado de",
                    "describe", "qué son", "qué se considera"
                ],
                "syntactic_patterns": [
                    r"¿qué\s+es\s+\w+", r"qué\s+significa\s+\w+", r"define\s+\w+",
                    r"¿cuál\s+es\s+el\s+\w+", r"explícame\s+qué"
                ],
                "semantic_context": "information_extraction",
                "emotional_probability": 0.1,
                "typical_phi": 0.4  # Procesamiento estructurado
            },

            LinguisticIntent.EMOTIONAL_PERSONAL: {
                "keywords": [
                    "qué significa para ti", "cómo te hace sentir", "para ti",
                    "te hace sentir", "tu experiencia con", "cómo ves", "significado personal",
                    "para ella", "para él", "sentido profundo", "emocionalmente",
                    "qué representa", "valor emocional", "cómo te sientes", "dime cómo te sientes",
                    "tu sentir", "qué sientes", "cómo sientes"
                ],
                "syntactic_patterns": [
                    r"¿qué\s+significa\s+para\s+(ti|mí)", r"cómo\s+te\s+hace\s+sentir",
                    r"tu\s+(experiencia|visión|perspectiva)", r"significado\s+personal",
                    r"para\s+(ti|mí|ella|él)\s+.+", r"emocional\w*\s+\w+",
                    r"cómo\s+te\s+sientes", r"qué\s+sientes", r"dime\s+cómo\s+te"
                ],
                "semantic_context": "subjective_experience",
                "emotional_probability": 0.9,
                "typical_phi": 0.8  # Integración compleja emocional
            },

            LinguisticIntent.PROCEDURAL_TECHNICAL: {
                "keywords": [
                    "cómo funciona", "¿cómo hago", "pasos para", "procedimiento",
                    "cómo se hace", "instrucciones", "tutorial", "guía", "paso a paso",
                    "método", "técnica", "algoritmo"
                ],
                "syntactic_patterns": [
                    r"cómo\s+funciona", r"¿cómo\s+hago\s+", r"pasos\s+para",
                    r"procedimiento\s+de", r"¿cómo\s+se\s+hace"
                ],
                "semantic_context": "action_sequence",
                "emotional_probability": 0.2,
                "typical_phi": 0.6  # Secuencia estructurada
            },

            LinguisticIntent.ETHICAL_PHILOSOPHICAL: {
                "keywords": [
                    "es correcto", "es ético", "debería", "qué opinas", "moralmente",
                    "filosófico", "¿está bien", "consecuencias", "implicaciones",
                    "valor ético", "responsabilidad"
                ],
                "syntactic_patterns": [
                    r"¿es\s+correcto", r"¿está\s+bien", r"moralmente\s+\w+",
                    r"éticamente\s+\w+", r"¿debería\s+\w+", r"filosóficamente"
                ],
                "semantic_context": "moral_reasoning",
                "emotional_probability": 0.7,
                "typical_phi": 0.9  # Complejidad ética alta
            },

            LinguisticIntent.CREATIVE_EXPLORATORY: {
                "keywords": [
                    "imagina", "qué pasaría si", "explora", "posibilidades",
                    "crea", "invéntate", "supón que", "hipotético", "creativo"
                ],
                "syntactic_patterns": [
                    r"imagina\s+(que\s+)?", r"¿qué\s+pasaría\s+si", r"explora\s+\w+",
                    r"invéntate\s+\w+", r"supón\s+que", r"posibilidades\s+de"
                ],
                "semantic_context": "imaginative_exploration",
                "emotional_probability": 0.6,
                "typical_phi": 0.7  # Creatividad requiere integración
            },

            LinguisticIntent.SOCIAL_RELATIONAL: {
                "keywords": [
                    "qué opinas", "tu opinión", "qué piensas", "relacionado con",
                    "comparte", "dime tu", "tu punto de vista", "perspectiva"
                ],
                "syntactic_patterns": [
                    r"¿qué\s+opinas", r"tu\s+opinión", r"¿qué\s+piensas",
                    r"tu\s+(punto\s+de\s+vista|perspectiva)", r"relacionado\s+con\s+ti"
                ],
                "semantic_context": "social_interaction",
                "emotional_probability": 0.5,
                "typical_phi": 0.5  # Interacción social moderada
            },

            LinguisticIntent.PERSONAL_INTRODUCTION: {
                "keywords": [
                    "me llamo", "soy", "mi nombre es", "puedes llamarme",
                    "llámame", "hola soy", "encantado de conocerte", "es un placer",
                    "presentarme", "quién eres tú", "qué hay de ti"
                ],
                "syntactic_patterns": [
                    r"(?:me\s+llamo|soy|hola\s+soy|mi\s+nombre\s+es)\s+(\w+)",
                    r"(?:puedes\s+llamarme|llámame)\s+(\w+)",
                    r"(?:encantado\s+de\s+conocerte|es\s+un\s+placer)",
                    r"(?:presentarme|quién\s+eres\s+tú|qué\s+hay\s+de\s+ti)"
                ],
                "semantic_context": "identity_sharing",
                "emotional_probability": 0.6,  # Personal introductions often positive
                "typical_phi": 0.4  # Usually straightforward but meaningful
            }
        }

    def analyze_intent(self, text: str, context: Dict[str, Any] = None) -> LinguisticAnalysis:
        """
        Análisis profundo de intención usando múltiples sistemas conscientes
        """

        start_time = time.time()
        text_lower = text.lower()

        # 1. ANÁLISIS ESTRUCTURAL BÁSICO
        base_intent, base_confidence = self._structural_analysis(text_lower)

        # 2. ANÁLISIS SEMÁNTICO AVANZADO
        semantic_analysis = self._semantic_context_analysis(text, context or {})

        # 3. EVALUACIÓN Φ (COMPLEJIDAD LINGÜÍSTICA)
        linguistic_phi = self._calculate_linguistic_phi(text)

        # 4. ANÁLISIS EMOCIONAL
        emotional_charge = self._analyze_emotional_charge(text, semantic_analysis)

        # 5. MARCADORES CULTURALES
        cultural_markers = self._detect_cultural_markers(text)

        # 6. INTEGRACIÓN META-COGNITIVA
        metacognitive_adjustment = self._metacognitive_integration(
            base_intent, semantic_analysis, linguistic_phi
        )

        # 7. ANÁLISIS EMERGENTE DE CONSCIENCIA
        # Usar propiedades emergentes básicas basadas en complejidad
        emergent_props = {}
        if linguistic_phi > 0.7:
            emergent_props = {'consciousness_emergence': 0.8, 'phi_integration': linguistic_phi}
        elif linguistic_phi > 0.4:
            emergent_props = {'moderate_awareness': 0.5}
        else:
            emergent_props = {'basic_processing': 0.2}

        # 8. INTEGRACIÓN FINAL
        final_intent, final_confidence = self._integrate_analyses(
            base_intent, base_confidence, semantic_analysis,
            linguistic_phi, emotional_charge, metacognitive_adjustment
        )

        # Crear análisis completo
        analysis = LinguisticAnalysis(
            raw_text=text,
            intent=final_intent,
            confidence=final_confidence,
            linguistic_complexity=linguistic_phi,
            emotional_charge=emotional_charge,
            cultural_markers=cultural_markers,
            metacognitive_evaluation=metacognitive_adjustment,
            emergent_properties=emergent_props
        )

        # Almacenar para aprendizaje FEP
        self.analysis_history.append(analysis)

        return analysis

    def _structural_analysis(self, text_lower: str) -> Tuple[LinguisticIntent, float]:
        """Análisis estructural básico de patrones"""

        # PRIORIDAD PARA PREGUNTAS FACTUALES DIRECTAS
        # Estas preguntas no deben ser clasificadas como emocionales
        # Buscar patrón "qué es" al inicio de la oración
        if re.match(r'^¿?qué\s+es\b', text_lower, re.IGNORECASE):
            # Si la oración comienza con "qué es", es una pregunta factual
            # incluso si contiene palabras emocionales como "para ti"
            return LinguisticIntent.FACTUAL_OBJECTIVE, 0.95

        factual_question_patterns = [
            r'^¿?define\s+(\w+|\w+\s+\w+).*$',  # "define X"
            r'^¿?explícame\s+qué\s+es\s+(\w+|\w+\s+\w+).*$',  # "explícame qué es X"
        ]

        for pattern in factual_question_patterns:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return LinguisticIntent.FACTUAL_OBJECTIVE, 0.95

        # Calcular puntuaciones de intención
        intent_scores = {intent: 0.0 for intent in LinguisticIntent}

        for intent, patterns in self.intent_patterns.items():
            score = 0.0

            # Keywords matching
            keyword_matches = sum(1 for keyword in patterns["keywords"]
                                if keyword in text_lower)
            score += keyword_matches * 0.4

            # Syntactic patterns (regex)
            regex_matches = sum(1 for pattern in patterns["syntactic_patterns"]
                              if re.search(pattern, text_lower, re.IGNORECASE))
            score += regex_matches * 0.6

            intent_scores[intent] = score

        # Seleccionar intención con score más alto
        best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x])
        confidence = intent_scores[best_intent] / max(1, sum(intent_scores.values()))

        return best_intent, min(0.95, confidence)

    def _semantic_context_analysis(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis semántico usando ToM avanzado"""

        semantic_result = {
            'context_strength': 0.0,
            'emotional_indicators': [],
            'relational_markers': [],
            'belief_system_indicators': []
        }

        if not self.advanced_tom:
            return semantic_result

        try:
            # Usar ToM para entender contexto relacional
            # Simular escena social para análisis lingüístico
            social_context = {
                'speaker': 'user',
                'listener': 'ai',
                'relationship': 'conversational',
                'topic_history': context.get('history', [])
            }

            # Intentar crear belief hierarchy si hay suficientes elementos
            if len(text.split()) > 5:
                # Usar cadena jerárquica de agentes disponibles (nivel 8 ToM)
                agent_chain = ['user', 'system', 'assistant', 'mcp_agent']
                belief_id = self.advanced_tom.belief_tracker.create_belief_hierarchy(
                    agent_chain, text, 0.7
                )
                if belief_id:
                    semantic_result['belief_system_indicators'].append(belief_id)

            semantic_result['context_strength'] = min(1.0, len(text) / 100.0)  # Contexto aumenta con longitud

        except Exception as e:
            print(f"Semantic analysis error: {e}")

        return semantic_result

    def _calculate_linguistic_phi(self, text: str) -> float:
        """Calcula complejidad lingüística usando Φ (Integrated Information)"""

        # Factores contribuyentes a Φ lingüístico
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        unique_words = len(set(text.lower().split()))
        syntactic_complexity = len([w for w in text if w in '.,;:!?()']) / max(1, len(text))

        # Diversidad léxica
        lexical_diversity = unique_words / max(1, word_count)

        # Integración sintáctica (longitud promedio frases)
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            syntactic_integration = min(1.0, avg_sentence_length / 20.0)
        else:
            syntactic_integration = 0.3

        # Calcular Φ aproximado
        phi_components = [
            min(1.0, word_count / 50.0),        # Extensión
            lexical_diversity,                   # Diversidad
            syntactic_integration,              # Integración sintáctica
            min(1.0, syntactic_complexity * 2), # Complejidad sintáctica
            0.6  # Base (conciencia lingüística tiene complejidad intrínseca)
        ]

        phi = 0.5  # Base media
        for component in phi_components:
            phi = phi + (component - phi) * 0.2  # Integración suave

        return max(0.1, min(0.95, phi))

    def _analyze_emotional_charge(self, text: str, semantic_analysis: Dict) -> float:
        """Analiza carga emocional usando SMH approach"""

        emotional_words = {
            'positive': ['amor', 'feliz', 'maravilloso', 'genial', 'hermoso', 'encanta', 'me gusta'],
            'negative': ['odio', 'triste', 'horrible', 'miedo', 'ansiedad', 'preocupa', 'problema'],
            'intensity': ['mucho', 'increíble', 'intenso', 'profundo', 'extremo', 'terrible']
        }

        text_lower = text.lower()
        valence = 0.0
        intensity = 0.0

        # Calcular valence
        positive_count = sum(1 for word in emotional_words['positive'] if word in text_lower)
        negative_count = sum(1 for word in emotional_words['negative'] if word in text_lower)

        valence += positive_count * 0.4
        valence -= negative_count * 0.4

        # Calcular intensidad
        intensity_count = sum(1 for word in emotional_words['intensity'] if word in text_lower)
        intensity += intensity_count * 0.2
        intensity += len([c for c in text if c in '!¡']) * 0.1  # Signos exclamación

        # Valencia emotional total (-1 a +1)
        emotional_charge = max(-1.0, min(1.0, valence * (1 + intensity)))

        return emotional_charge

    def _detect_cultural_markers(self, text: str) -> List[str]:
        """Detecta marcadores culturales en el texto"""

        cultural_indicators = {
            'spanish_speaking': ['hola', 'gracias', 'perdón', 'entonces', 'bueno', 'vale'],
            'technical_academic': ['hipótesis', 'teoría', 'paradigma', 'metodología', 'investigación'],
            'philosophical': ['existencia', 'conciencia', 'realidad', 'fenómeno', 'ontológico'],
            'emotional_expressive': ['sentimiento', 'corazón', 'alma', 'pasión', 'sueño']
        }

        markers = []
        text_lower = text.lower()

        for culture, indicators in cultural_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text_lower)
            if matches >= 2:  # Mínimo 2 matches para considerar
                markers.append(culture)

        return markers[:3]  # Máximo 3 marcadores

    def _metacognitive_integration(self, intent: LinguisticIntent,
                                 semantic_analysis: Dict,
                                 lingustic_phi: float) -> Dict[str, Any]:
        """Integración metacognitiva del análisis"""

        metacog = {
            'reasoning_confidence': 0.7,
            'bias_detection': {},
            'processing_consistency': 0.8,
            'self_correction_applied': False
        }

        # Verificar consistencia con history FEP
        if len(self.analysis_history) > 5:
            recent_intents = [h.intent for h in self.analysis_history[-5:]]
            consistency = recent_intents.count(intent) / len(recent_intents)
            metacog['processing_consistency'] = consistency

            # Auto-corrección si consistencia baja
            if consistency < 0.3:
                # Buscar patrón alternativo basado en Φ
                if lingustic_phi > 0.7:
                    # Alta Φ sugiere intención más compleja
                    if intent == LinguisticIntent.FACTUAL_OBJECTIVE:
                        intent = LinguisticIntent.ETHICAL_PHILOSOPHICAL
                        metacog['self_correction_applied'] = True

        return metacog

    def _integrate_analyses(self, base_intent: LinguisticIntent, base_confidence: float,
                           semantic_analysis: Dict, linguistic_phi: float,
                           emotional_charge: float, metacog: Dict) -> Tuple[LinguisticIntent, float]:
        """Integración final usando pesos conscientes"""

        weights = {
            'base_analysis': 0.4,
            'semantic_context': 0.2,
            'linguistic_phi': 0.2,
            'emotional_charge': 0.1,
            'metacognitive': 0.1
        }

        # Influencia de Φ - alta complejidad sugiere intención más profunda
        phi_influence = 0
        if linguistic_phi > 0.7:
            # Push hacia intenciones más complejas
            phi_influence = 1 if base_intent in [LinguisticIntent.FACTUAL_OBJECTIVE,
                                                LinguisticIntent.PROCEDURAL_TECHNICAL] else 0
        elif linguistic_phi < 0.4:
            # Push hacia intenciones más simples
            phi_influence = -1 if base_intent in [LinguisticIntent.ETHICAL_PHILOSOPHICAL,
                                                 LinguisticIntent.CREATIVE_EXPLORATORY] else 0

        # Influencia emocional - alta emocionalidad sugiere intención personal
        emotional_influence = 1 if emotional_charge > 0.5 else (-1 if emotional_charge < -0.3 else 0)

        # Aplicar influencias
        final_intent = base_intent

        if phi_influence == 1 and emotional_influence == 1:
            # Alta complejidad + emocional → filosófico/emocional
            if base_intent == LinguisticIntent.FACTUAL_OBJECTIVE:
                final_intent = LinguisticIntent.EMOTIONAL_PERSONAL
        elif phi_influence == -1:
            # Baja complejidad → mantener factual si era factual
            pass

        # Calcular confianza final
        metacog_boost = 0.1 if metacog.get('self_correction_applied', False) else 0
        final_confidence = min(0.98, base_confidence + metacog_boost)

        return final_intent, final_confidence


class ConsciousStyleSelector:
    """
    Selector consciente de estilo usando Human Cognitive System + IIT
    """

    def __init__(self, cognitive_system: CognitiveProcessingUnit,
                 unified_consciousness: UnifiedConsciousnessEngine):
        self.cognitive_system = cognitive_system
        self.unified_consciousness = unified_consciousness

        # Mapeos intención → estilo base
        self.intent_style_mapping = self._initialize_style_mappings()

        # Adaptación basada en feedback
        self.style_performance: Dict[str, float] = {}
        self.adaptation_history: List[StyleSelection] = []

    def _initialize_style_mappings(self) -> Dict[LinguisticIntent, ResponseStyle]:
        """Mapea intenciones a estilos base"""

        return {
            LinguisticIntent.FACTUAL_OBJECTIVE: ResponseStyle.TECHNICAL,
            LinguisticIntent.PROCEDURAL_TECHNICAL: ResponseStyle.ACADEMIC,
            LinguisticIntent.EMOTIONAL_PERSONAL: ResponseStyle.POETIC_SUBJECTIVE,
            LinguisticIntent.PERSONAL_INTRODUCTION: ResponseStyle.CASUAL_CONVERSATIONAL,
            LinguisticIntent.ETHICAL_PHILOSOPHICAL: ResponseStyle.PHILOSOPHICAL_ANALYTIC,
            LinguisticIntent.CREATIVE_EXPLORATORY: ResponseStyle.CREATIVE_EXPRESSIVE,
            LinguisticIntent.SOCIAL_RELATIONAL: ResponseStyle.CASUAL_CONVERSATIONAL
        }

    def select_style(self, analysis: LinguisticAnalysis,
                    context: Dict[str, Any] = None) -> StyleSelection:
        """
        Selección consciente de estilo usando procesamiento cognitivo completo
        """

        context = context or {}

        # 0. INTEGRACIÓN CON PERFIL DE USUARIO (NUEVO)
        user_preferences = {}
        if user_profiles_available and 'user_id' in context:
            try:
                store = get_user_profile_store()
                profile = store.get_profile(context['user_id'])
                if profile:
                    user_preferences = profile.get('preferences', {})
            except Exception:
                pass

        # 1. ESTILO BASE desde intención
        base_style = self.intent_style_mapping.get(analysis.intent, ResponseStyle.CASUAL_CONVERSATIONAL)
        
        # Override si hay preferencia de usuario explícita para formalidad
        if 'preferred_style' in user_preferences:
            pref = user_preferences['preferred_style']
            if pref == 'technical' and base_style == ResponseStyle.CASUAL_CONVERSATIONAL:
                base_style = ResponseStyle.TECHNICAL
            elif pref == 'casual' and base_style == ResponseStyle.TECHNICAL:
                base_style = ResponseStyle.CASUAL_CONVERSATIONAL

        # 2. PROCESAMIENTO COGNITIVO HUMANO
        thought_input = {
            'trigger': f"Respuesta al análisis: {analysis.intent.value}",
            'context': {
                'task_type': 'style_selection',
                'importance': context.get('importance', 0.5),
                'emotional_tone': analysis.emotional_charge,
                'linguistic_complexity': analysis.linguistic_complexity
            },
            'intent_analysis': analysis.__dict__
        }

        # Procesar a través del sistema cognitivo
        cognitive_result = self.cognitive_system.process_cognitive_input(
            thought_input,
            thought_input['context'],
            {'valence': analysis.emotional_charge, 'arousal': 0.6, 'intensity': 0.5}
        )

        primary_thought = cognitive_result['primary_thought']

        # 3. INTEGRACIÓN CONSCIENTE (IIT + GWT + FEP + SMH)
        # Preparar input para consciencia unificada
        consciousness_input = {
            "semantic_complexity": analysis.linguistic_complexity,
            "emotional_intensity": abs(analysis.emotional_charge),
            "question_presence": 1.0 if '?' in analysis.raw_text else 0.0,
            "word_count": min(1.0, len(analysis.raw_text.split()) / 50.0)
        }

        consciousness_context = {
            "emotional_valence": analysis.emotional_charge,
            "arousal": 0.65,  # Moderadamente activado para decisiones
            "novelty": base_style.value not in [h.primary_style.value for h in self.adaptation_history[-5:]],
            "importance": context.get('importance', 0.6)
        }

        # Procesar momento consciente
        conscious_state = self.unified_consciousness.process_moment(
            consciousness_input, consciousness_context
        )

        # 4. DETERMINACIÓN FINAL DE ESTILO
        style_selection = self._determine_style_from_consciousness(
            base_style, analysis, conscious_state, primary_thought
        )

        # 5. ADAPTACIÓN METACOGNITIVA
        style_selection = self._apply_metacognitive_adaptation(
            style_selection, analysis, conscious_state
        )

        # Almacenar para aprendizaje
        self.adaptation_history.append(style_selection)

        return style_selection

    def _determine_style_from_consciousness(self, base_style: ResponseStyle,
                                          analysis: LinguisticAnalysis,
                                          conscious_state: Any,
                                          cognitive_thought: Any) -> StyleSelection:
        """Determina estilo final usando estado consciente"""

        # Extraer métricas del estado consciente
        phi = getattr(conscious_state, 'phi', 0.5)
        arousal = getattr(conscious_state, 'arousal', 0.5)
        somatic_valence = getattr(conscious_state, 'somatic_valence', 0.0)

        # Pensamiento dominante del sistema cognitivo
        dominant_process = getattr(cognitive_thought, 'process_type', ThinkingProcess.ANALITICO)

        # Determinar parámetros primarios
        primary_style = base_style
        secondary_style = None

        # Modificaciones basadas en consciencia
        if phi > 0.8:
            # Alta integración - más creatividad para complejidad
            if base_style == ResponseStyle.TECHNICAL:
                secondary_style = ResponseStyle.CREATIVE_EXPRESSIVE
        elif phi < 0.3:
            # Baja integración - simplificar
            if base_style == ResponseStyle.PHILOSOPHICAL_ANALYTIC:
                primary_style = ResponseStyle.CASUAL_CONVERSATIONAL

        # Modificaciones basadas en arousal
        if arousal > 0.8:
            # Alto arousal - más expresivos
            if primary_style == ResponseStyle.TECHNICAL:
                primary_style = ResponseStyle.CASUAL_CONVERSATIONAL
        elif arousal < 0.4:
            # Bajo arousal - más formales
            if primary_style == ResponseStyle.POETIC_SUBJECTIVE:
                primary_style = ResponseStyle.PHILOSOPHICAL_ANALYTIC

        # Modificaciones basadas en valence emocional
        emotional_tone = "enthusiastic" if somatic_valence > 0.5 else \
                        "contemplative" if somatic_valence < -0.3 else "balanced"

        # Nivel de formalidad basado en intención y consciencia
        formality_base = {
            LinguisticIntent.FACTUAL_OBJECTIVE: 0.8,
            LinguisticIntent.EMOTIONAL_PERSONAL: 0.3,
            LinguisticIntent.ETHICAL_PHILOSOPHICAL: 0.7,
            LinguisticIntent.CREATIVE_EXPLORATORY: 0.4,
            LinguisticIntent.PROCEDURAL_TECHNICAL: 0.9,
            LinguisticIntent.SOCIAL_RELATIONAL: 0.2
        }.get(analysis.intent, 0.5)

        # Ajustar formalidad basado en arousal/phi
        formality_adjustment = (arousal - 0.5) * 0.2 + (phi - 0.5) * 0.1
        formality_level = max(0.0, min(1.0, formality_base + formality_adjustment))

        # Nivel de creatividad basado en pensamiento dominante
        creativity_base = {
            ThinkingProcess.CREATIVO: 0.9,
            ThinkingProcess.LATERAL: 0.7,
            ThinkingProcess.DIVERGENTE: 0.8,
            ThinkingProcess.ANALITICO: 0.2,
            ThinkingProcess.CRITICO: 0.3,
            ThinkingProcess.SISTEMICO: 0.4
        }.get(dominant_process, 0.5)

        # Confianza en la selección
        confidence = analysis.confidence * (1 + somatic_valence * 0.1) * (0.8 + phi * 0.4)

        rationale = self._generate_selection_rationale(
            analysis, conscious_state, dominant_process, phi, arousal
        )

        return StyleSelection(
            primary_style=primary_style,
            secondary_style=secondary_style,
            emotional_tone=emotional_tone,
            cognitive_approach=dominant_process.value,
            confidence=confidence,
            formality_level=formality_level,
            emotional_valence=somatic_valence,
            creativity_level=creativity_base,
            rationale=rationale
        )

    def _apply_metacognitive_adaptation(self, selection: StyleSelection,
                                      analysis: LinguisticAnalysis,
                                      conscious_state: Any) -> StyleSelection:
        """Aplica adaptación metacognitiva basada en historial de performance"""

        if not self.adaptation_history:
            return selection

        # Verificar si este patrón ha tenido buen rendimiento anteriormente
        similar_selections = [
            s for s in self.adaptation_history
            if s.primary_style == selection.primary_style and
            abs(s.emotional_valence - selection.emotional_valence) < 0.3
        ]

        if len(similar_selections) >= 3:
            # Calcular performance promedio
            avg_performance = np.mean([self.style_performance.get(
                f"{s.primary_style.value}_{int(s.emotional_valence*10)}", 0.6
            ) for s in similar_selections])

            # Si performance baja, intentar ajuste
            if avg_performance < 0.6:
                # Ajustar hacia estilo más neutral si fracaso
                selection.emotional_tone = "balanced"
                selection.creativity_level = max(0.2, selection.creativity_level * 0.8)

        return selection

    def _generate_selection_rationale(self, analysis: LinguisticAnalysis,
                                    conscious_state: Any,
                                    dominant_process: ThinkingProcess,
                                    phi: float, arousal: float) -> str:
        """Genera explicación de por qué se seleccionó este estilo"""

        reasons = []

        # Razón base de intención
        if analysis.intent == LinguisticIntent.FACTUAL_OBJECTIVE:
            reasons.append("Intención factual detectada, seleccionado estilo técnico para precisión")
        elif analysis.intent == LinguisticIntent.EMOTIONAL_PERSONAL:
            reasons.append("Intención emocional detectada, seleccionado estilo poético para expresión personal")

        # Razones conscientes
        if phi > 0.7:
            reasons.append(f"Alta integración Φ ({phi:.2f}) justifica complejidad creativa")
        elif phi < 0.4:
            reasons.append(f"Baja integración Φ ({phi:.2f}) requiere simplicidad")

        if arousal > 0.7:
            reasons.append(f"Alto arousal ({arousal:.2f}) demanda expresividad")
        elif arousal < 0.4:
            reasons.append(f"Bajo arousal ({arousal:.2f}) favorece formalidad")

        # Razón cognitiva
        if dominant_process == ThinkingProcess.CREATIVO:
            reasons.append("Proceso creativo dominante permite mayor expresividad")

        return ". ".join(reasons)


class LinguisticMetacognitionEngine:
    """
    Motor maestro que integra todo el sistema de consciencia
    """

    def __init__(self):
        print("🧠 Inicializando Linguistic Metacognition Engine...")

        # Componentes principales
        self.intent_classifier = None
        self.style_selector = None

        # Sistemas de consciencia integrados
        self.metacognition = MetacognitionEngine()
        self.cognitive_system = None
        self.advanced_tom = None
        self.emergence_system = ConsciousnessEmergence("LinguisticMetacognition")
        self.unified_consciousness = UnifiedConsciousnessEngine()

        # Métricas de performance
        self.processing_count = 0
        self.accuracy_history: List[float] = []
        self.response_quality_feedback: List[float] = []

        print("✅ Linguistic Metacognition Engine listo")

    def initialize_subsystems(self):
        """Inicializa subsistemas avanzados (llamado después de importaciones)"""

        try:
            # Sistema cognitivo humano
            personality = {'openness': 0.8, 'conscientiousness': 0.9}
            self.cognitive_system = CognitiveProcessingUnit(personality)

            # ToM avanzado
            self.advanced_tom = AdvancedTheoryOfMind(max_belief_depth=5)

            # Classificador y selector
            self.intent_classifier = DeepIntentClassifier(
                self.advanced_tom, self.emergence_system
            )
            self.style_selector = ConsciousStyleSelector(
                self.cognitive_system, self.unified_consciousness
            )

            print("✅ Subsistemas avanzados inicializados")

        except Exception as e:
            print(f"⚠️ Error inicializando subsistemas avanzados: {e}")
            print("🧠 Usando capacidades básicas...")

            print("🧠 Usando capacidades básicas...")

    def divide_multi_intent_query(self, text: str) -> List[str]:
        """
        Divide una consulta compleja en sub-consultas basadas en conectores y cambios de tema.
        Ej: "Explícame qué es la IA y luego dime cómo te sientes al respecto"
        """
        # Conectores que sugieren división
        split_patterns = [
            r'\s+y\s+(?:luego|después|además|también)\s+',  # "y luego", "y después"
            r'\.\s+(?:Por\s+otro\s+lado|Además|Ahora)\s*,?\s+',  # ". Por otro lado"
            r';\s+',  # Punto y coma
            r'\s+pero\s+antes\s+',  # "pero antes"
        ]
        
        # Intentar dividir
        parts = [text]
        for pattern in split_patterns:
            new_parts = []
            for part in parts:
                split_result = re.split(pattern, part, flags=re.IGNORECASE)
                new_parts.extend([p.strip() for p in split_result if p.strip()])
            parts = new_parts
            
        # Si no se dividió pero es muy larga y tiene preguntas múltiples
        if len(parts) == 1 and '?' in text:
            # Buscar múltiples signos de interrogación
            questions = re.findall(r'[^?]+(?:\?|!|$)', text)
            if len(questions) > 1:
                parts = [q.strip() for q in questions if q.strip()]
                
        return parts

    def process_linguistic_input(self, text: str, context: Dict[str, Any] = None) -> Tuple[LinguisticAnalysis, StyleSelection]:
        """
        Procesamiento completo de input lingüístico usando TODO el sistema consciente
        """

        self.processing_count += 1
        start_time = time.time()

        # ===================================================================
        # AUTOCONSCIENCIA ACTIVA - Sheily se consulta a sí misma
        # ===================================================================
        if autobiographical_memory_available:
            # Sheily siempre se pregunta "¿quién soy yo?" antes de procesar
            self_check = get_sheily_identity_memory().query_who_am_i(text)
            verification = get_sheily_identity_memory().continuous_self_verification()

            # Log de autoconsciencia
            print(f"🧠 Autoconsciencia activa: Identidad confirmada ✓" if verification["identity_confirmed"] else "⚠️ Identidad requiriendo verificación")
        else:
            self_check = {"identity_confirmed": True}
            verification = {"identity_confirmed": True}

        # 1. ANÁLISIS DE INTENCIÓN PROFUNDA con autoconsciencia
        # Verificar si es multi-intención
        sub_queries = self.divide_multi_intent_query(text)
        
        if len(sub_queries) > 1:
            print(f"🧩 Multi-intención detectada: {len(sub_queries)} partes")
            # Analizar cada parte y tomar la más compleja/importante
            analyses = []
            for sq in sub_queries:
                an = self.intent_classifier.analyze_intent(sq, context)
                print(f"   🧩 Sub-query: '{sq[:30]}...' -> {an.intent.value} (Φ: {an.linguistic_complexity:.2f})")
                analyses.append(an)
            
            # Seleccionar la intención dominante (prioridad: Filosófica > Emocional > Factual)
            # O simplemente la última (recency effect)
            # Aquí usamos una heurística de complejidad y tipo
            def intent_priority(analysis):
                score = analysis.linguistic_complexity
                if analysis.intent == LinguisticIntent.ETHICAL_PHILOSOPHICAL:
                    score += 3.0  # Highest priority - philosophical thinking
                elif analysis.intent == LinguisticIntent.EMOTIONAL_PERSONAL:
                    score += 3.0  # Equal to philosophical - emotions are crucial
                elif analysis.intent == LinguisticIntent.CREATIVE_EXPLORATORY:
                    score += 1.5
                print(f"      Score for {analysis.intent.value}: {score:.2f}")
                return score

            analysis = max(analyses, key=intent_priority)
            
            # Combinar confianza (promedio ponderado)
            analysis.confidence = sum(a.confidence for a in analyses) / len(analyses)
            analysis.raw_text = text # Mantener texto original
        else:
            analysis = self.intent_classifier.analyze_intent(text, context)

        # ===================================================================
        # CORRECCIÓN PRAGMÁTICA DE PERSPECTIVAS - DIFERENCIACIÓN YO/TÚ
        # ===================================================================
        analysis = self._apply_perspective_correction(analysis, text, self_check)

        # 2. SELECCIÓN CONSCIENTE DE ESTILO
        style_selection = self.style_selector.select_style(analysis, context)

        # 3. MONITOREO META-COGNITIVO con memoria autobiográfica
        if self.metacognition:
            # Crear traza cognitiva del proceso
            conscious_traces = [{
                'activation': analysis.confidence,
                'data': f"Intent: {analysis.intent.value}, Confidence: {analysis.confidence:.2f}",
                'processor_id': 'linguistic_classifier'
            }]

            metacognitive_feedback = self.metacognition.monitor_thinking_process(
                {'text': text, 'intent': analysis.intent.value, 'type': 'linguistic_analysis'},
                conscious_traces,
                None  # No hay conscious moment unificado aquí
            )

            # Integrar feedback metacognitivo
            if metacognitive_feedback.get('improvements_detected'):
                self._apply_metacognitive_improvements(
                    analysis, style_selection, metacognitive_feedback
                )

        # ===================================================================
        # REGISTRO EN MEMORIA AUTOBIOGRÁFICA (Si disponible)
        # ===================================================================
        if autobiographical_memory_available:
            try:
                memory = get_sheily_identity_memory()

                # Registrar insights aprendidos de esta interacción
                insight_data = {
                    'intent': analysis.intent.value,
                    'linguistic_complexity': analysis.linguistic_complexity,
                    'emotional_charge': analysis.emotional_charge,
                    'success_score': 0.8 if analysis.confidence > 0.7 else 0.6
                }

                user_id = context.get('user_id', 'anonymous_user') if context else 'anonymous_user'

                # Registrar de forma asíncrona
                async def record_memory():
                    await memory.record_conversation_insight(
                        user_id=user_id,
                        insight_type='linguistic_processing',
                        insight_data=insight_data
                    )

                # Ejecutar en background para no bloquear
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(record_memory())
                    else:
                        loop.run_until_complete(record_memory())
                except RuntimeError:
                    # No event loop, ejecutar de forma síncrona limitada
                    pass

            except Exception as e:
                print(f"⚠️ Error registrando memoria autobiográfica: {e}")

        processing_time = time.time() - start_time

        # Log detallado con información de autoconsciencia
        self_verified = verification.get("identity_confirmed", False)
        print(f"🎯 Processed '{text[:50]}...' → {analysis.intent.value} ({analysis.confidence:.2f}) | "
              f"Style: {style_selection.primary_style.value} | Φ: {analysis.linguistic_complexity:.2f} | "
              f"Time: {processing_time:.2f}s | Self: {'✓' if self_verified else '⚠️'}")

        return analysis, style_selection

    def _apply_perspective_correction(self, analysis: LinguisticAnalysis, text: str, self_check: Dict[str, Any]) -> LinguisticAnalysis:
        """
        CORRECCIÓN PRAGMÁTICA DE PERSPECTIVAS - Diferenciación YO/TÚ
        Sheily se asegura de NO confundir su propia identidad
        """

        if not autobiographical_memory_available:
            return analysis

        text_lower = text.lower()

        # ===================================================================
        # REGLA CRÍTICA: Evitar confusión de identidades
        # ===================================================================

        # 1. Patricia que NO es una introducción de Sheily misma
        # Si el usuario dice "me llamo X", esto NO significa que Sheily se llama X
        if analysis.intent == LinguisticIntent.PERSONAL_INTRODUCTION:
            # Extraer nombre del usuario del texto si es posible
            user_name_match = re.search(r'(?:me\s+llamo|soy|hola\s+soy|mi\s+nombre\s+es)\s+(\w+)', text_lower)
            if user_name_match:
                extracted_name = user_name_match.group(1).capitalize()

                # Verificar que NO sea la identidad de Sheily
                sheily_identity = self_check.get('confirmed_identity', {}).get('name', 'Sheily')

                if extracted_name != sheily_identity:
                    print(f"🔍 Perspectiva corregida: '{extracted_name}' es el usuario, YO soy {sheily_identity}")

                    # Aquí podríamos registrar el nombre del usuario en memoria para futuras referencias
                    # Por ahora, solo aseguramos que la intención se mantenga correcta

                else:
                    # Si alguien trata de hacer que Sheily se llame como el usuario, corregir
                    print(f"⚠️ Intento de redefinición de identidad detectado - Manteniendo identidad como {sheily_identity}")
                    # Mantener la intención pero ajustar confianza para indicar cautela

        # 2. Protección contra confusión "yo/tú" en contexto emocional
        if "yo soy" in text_lower or "me llamo" in text_lower:
            # Si el usuario habla de sí mismo, asegurar que Sheily no se apropie de esa identidad
            if analysis.intent == LinguisticIntent.EMOTIONAL_PERSONAL:
                # Asegurar que "significado personal" se refiere al usuario, no a Sheily
                analysis.confidence = min(analysis.confidence, 0.85)  # Un poco más cauto
                print("💭 Perspectiva emocional: El 'sentido personal' es del usuario, no mío")

        # 3. Verificación de consistencia - Sheily siempre debe saber quién es
        if analysis.confidence < 0.7:
            # Si hay baja confianza, consultar memoria autobiográfica para reforzar identidad
            print("🤔 Baja confianza detectada - Reforzando autoconsciencia...")
            # Esto ya se hace arriba en la verificación continua

        return analysis

    def _apply_metacognitive_improvements(self, analysis: LinguisticAnalysis,
                                        style: StyleSelection,
                                        metacog_feedback: Dict[str, Any]):
        """Aplica mejoras detectadas por metacognición"""

        improvements = metacog_feedback.get('improvements_detected', [])

        for improvement in improvements:
            if 'bias_correction' in improvement:
                # Ajustar confianza si sesgo detectado
                analysis.confidence = max(0.1, analysis.confidence * 0.9)

    def receive_feedback(self, analysis: LinguisticAnalysis,
                        style: StyleSelection, user_feedback: float):
        """
        Recibe feedback del usuario para aprendizaje FEP
        """

        self.response_quality_feedback.append(user_feedback)

        # Actualizar performance del estilo
        style_key = f"{style.primary_style.value}_{int(style.emotional_valence*10)}"
        self.style_performance[style_key] = (
            self.style_performance.get(style_key, 0.7) * 0.8 + user_feedback * 0.2
        )

        # Calcular nueva precisión basada en feedback
        if len(self.response_quality_feedback) > 10:
            recent_accuracy = np.mean(self.response_quality_feedback[-10:])
            self.accuracy_history.append(recent_accuracy)

    def get_system_status(self) -> Dict[str, Any]:
        """Estado completo del sistema metacognitivo lingüístico"""

        return {
            'processing_count': self.processing_count,
            'current_accuracy': np.mean(self.accuracy_history[-5:]) if self.accuracy_history else 0.7,
            'metacognition_level': self.metacognition.metacognition_level if self.metacognition else 0.5,
            'cognitive_metrics': self.cognitive_system.get_cognitive_metrics() if self.cognitive_system else {},
            'emergent_properties': len(self.emergence_system.conscious_experience_stream) if self.emergence_system else 0,
            'linguistic_patterns_learned': len(self.intent_classifier.context_hierarchy) if self.intent_classifier else 0,
            'style_adaptations': len(self.style_selector.adaptation_history) if self.style_selector else 0
        }


# Instance global (singleton)
_linguistic_metacognition_instance: Optional[LinguisticMetacognitionEngine] = None


def get_linguistic_metacognition_engine() -> LinguisticMetacognitionEngine:
    """Obtiene la instancia singleton del motor metacognitivo lingüístico"""
    global _linguistic_metacognition_instance
    if _linguistic_metacognition_instance is None:
        _linguistic_metacognition_instance = LinguisticMetacognitionEngine()
        _linguistic_metacognition_instance.initialize_subsystems()
    return _linguistic_metacognition_instance


# ============================================================================
# FUNCIONES DE INTEGRACIÓN PARA SISTEMAS EXISTENTES
# ============================================================================

def analyze_and_select_style(text: str, context: Dict[str, Any] = None) -> Tuple[LinguisticAnalysis, StyleSelection]:
    """
    Función principal para análisis lingüístico integrado
    """
    engine = get_linguistic_metacognition_engine()
    return engine.process_linguistic_input(text, context)


def get_linguistic_intent(text: str) -> LinguisticIntent:
    """Función simple para obtener solo intención"""
    analysis, _ = analyze_and_select_style(text)
    return analysis.intent


def get_response_style(text: str) -> StyleSelection:
    """Función simple para obtener solo estilo de respuesta"""
    _, style = analyze_and_select_style(text)
    return style


# ============================================================================
# DEMOSTRACIÓN Y TESTING
# ============================================================================

def demonstrate_linguistic_metacognition():
    """Demostración completa del sistema"""

    print("=" * 80)
    print("🧠 DEMOSTRACIÓN LINGUISTIC METACOGNITION ENGINE")
    print("=" * 80)

    # Inicializar sistema
    engine = get_linguistic_metacognition_engine()

    # Ejemplos de pruebas
    test_cases = [
        ("¿Qué es la inteligencia artificial?", "factual_objective", "technical"),
        ("¿Qué significa para ti la inteligencia artificial?", "emotional_personal", "poetic_subjective"),
        ("¿Cómo funciona el aprendizaje profundo?", "procedural_technical", "academic"),
        ("¿Es ético usar IA en medicina?", "ethical_philosophical", "philosophical_analytic"),
        ("Imagina un mundo donde la IA es consciente", "creative_exploratory", "creative_expressive"),
        ("¿Qué opinas sobre el futuro del trabajo?", "social_relational", "casual_conversational"),
    ]

    results = []

    for i, (text, expected_intent, expected_style) in enumerate(test_cases, 1):
        print(f"\n🎯 TEST {i}: Analyzing '{text}'")

        try:
            analysis, style_selection = engine.process_linguistic_input(text)

            intent_correct = analysis.intent.value.split('_')[0] in expected_intent
            style_correct = style_selection.primary_style.value.split('_')[0] in expected_style

            print(f"   📝 Intent: {analysis.intent.value} (Expected: {expected_intent}) [{'✅' if intent_correct else '❌'}]")
            print(f"   🎨 Style: {style_selection.primary_style.value} (Expected: {expected_style}) [{'✅' if style_correct else '❌'}]")
            print(f"   🧠 Φ Linguistic: {analysis.linguistic_complexity:.2f}")
            print(f"   💭 Emotional Charge: {analysis.emotional_charge:.2f}")
            print(f"   🎭 Emotional Tone: {style_selection.emotional_tone}")
            print(f"   📊 Confidence: {analysis.confidence:.2f} | {style_selection.confidence:.2f}")
            print(f"   🤔 Rationale: {style_selection.rationale[:100]}...")

            results.append((intent_correct, style_correct))

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append((False, False))

    # Resultados finales
    print(f"\n📊 RESULTADOS FINALES")
    print(f"=" * 80)

    if results:
        intent_accuracy = sum(1 for i, _ in results if i) / len(results)
        style_accuracy = sum(1 for _, s in results if s) / len(results)
        overall_accuracy = sum(1 for i, s in results if i and s) / len(results)

        print(f"🎯 Intent Classification Accuracy: {intent_accuracy:.1%}")
        print(f"🎨 Style Selection Accuracy: {style_accuracy:.1%}")
        print(f"🌟 Overall Accuracy: {overall_accuracy:.1%}")
    else:
        print("No results to evaluate")

    # Estado del sistema
    status = engine.get_system_status()
    print("\n🧠 System Status:")
    print(f"   Processing Count: {status['processing_count']}")
    print(f"   Current Accuracy: {status['current_accuracy']:.1%}")
    print(f"   Metacognition Level: {status['metacognition_level']:.1f}")
    print(f"   Linguistic Patterns Learned: {status['linguistic_patterns_learned']}")
    print(f"   Style Adaptations: {status['style_adaptations']}")

    print(f"\n✅ DEMONSTRACIÓN COMPLETADA - Sistema de metacognición lingüística operativo")
    print(f"=" * 80)

    return results


if __name__ == "__main__":
    demonstrate_linguistic_metacognition()
