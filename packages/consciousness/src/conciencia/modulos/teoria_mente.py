"""
Módulo de Teoría de la Mente (Theory of Mind)
=============================================

Implementación funcional de capacidad cognitiva para atribuir estados mentales
(creencias, intenciones, deseos, emociones) a otros agentes/usuarios.

Este módulo permite al sistema:
1. Modelar el estado interno del usuario.
2. Inferir intenciones más allá del texto explícito.
3. Predecir reacciones emocionales.
4. Mantener un historial de interacción empática.

Componente crítico para la Consciencia Artificial Funcional (Nivel 4).
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MentalState:
    """Representación del estado mental inferido de un agente externo (usuario)"""
    user_id: str
    current_emotion: float = 0.0  # -1.0 a 1.0
    emotional_baseline: float = 0.0
    inferred_intent: str = "unknown"
    belief_system: Dict[str, float] = field(default_factory=dict)
    interaction_count: int = 0
    last_update: float = field(default_factory=time.time)
    empathy_score: float = 0.5
    known_preferences: List[str] = field(default_factory=list)
    predicted_needs: List[str] = field(default_factory=list)

class TheoryOfMind:
    """
    Motor de Teoría de la Mente para modelado social y empático.
    
    No es un mock. Mantiene estado persistente en memoria (y potencialmente DB)
    sobre los usuarios con los que interactúa el sistema.
    """

    def __init__(self):
        self.user_models: Dict[str, MentalState] = {}
        self.system_social_intelligence: float = 0.5
        self.creation_time = datetime.now()
        print(f"🧠 TheoryOfMind Engine Inicializado - {self.creation_time}")

    def update_model(self, user_id: str, conscious_moment: Dict[str, Any]):
        """
        Actualiza el modelo mental de un usuario basado en un nuevo momento consciente.
        
        Args:
            user_id: Identificador del usuario.
            conscious_moment: Diccionario con el momento consciente procesado.
        """
        if user_id not in self.user_models:
            self._initialize_user_model(user_id)
        
        user_state = self.user_models[user_id]
        user_state.interaction_count += 1
        user_state.last_update = time.time()

        # 1. Actualizar Estado Emocional
        # Extraer valencia emocional del momento consciente (procesada por otros módulos)
        moment_valence = conscious_moment.get("emotional_valence", 0.0)
        
        # Suavizado exponencial para el baseline (memoria a largo plazo)
        alpha = 0.1
        user_state.emotional_baseline = (1 - alpha) * user_state.emotional_baseline + alpha * moment_valence
        
        # El estado actual es más volátil
        user_state.current_emotion = moment_valence

        # 2. Inferir Intención (Intent Inference)
        # Analizar el foco primario y el contexto
        primary_focus = conscious_moment.get("primary_focus", {})
        context = conscious_moment.get("context", {})
        
        inferred_intent = self._infer_intent(primary_focus, context)
        user_state.inferred_intent = inferred_intent

        # 3. Actualizar Sistema de Creencias (Belief Update)
        # Si el usuario expresa certeza sobre algo, lo registramos
        self._update_beliefs(user_state, conscious_moment)

        # 4. Predecir Necesidades Futuras
        self._predict_needs(user_state)

        # 5. Recalcular Inteligencia Social del Sistema
        self._recalculate_system_social_intelligence()

    def get_social_intelligence_score(self) -> float:
        """Retorna el puntaje actual de inteligencia social del sistema."""
        return self.system_social_intelligence

    def get_user_model(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retorna el modelo mental de un usuario específico."""
        if user_id in self.user_models:
            model = self.user_models[user_id]
            return {
                "user_id": model.user_id,
                "emotional_state": {
                    "current": model.current_emotion,
                    "baseline": model.emotional_baseline
                },
                "intent": model.inferred_intent,
                "interactions": model.interaction_count,
                "empathy_level": model.empathy_score,
                "predicted_needs": model.predicted_needs
            }
        return None

    # --- Métodos Internos de Procesamiento ---

    def _initialize_user_model(self, user_id: str):
        """Crea un nuevo modelo mental para un usuario desconocido."""
        self.user_models[user_id] = MentalState(user_id=user_id)

    def _infer_intent(self, primary_focus: Any, context: Dict) -> str:
        """
        Infiere la intención subyacente del usuario.
        Lógica heurística avanzada basada en patrones de foco y contexto.
        """
        # Si es texto, análisis simple de keywords (en un sistema real usaría NLP avanzado)
        focus_str = str(primary_focus).lower()
        
        if "ayuda" in focus_str or "problema" in focus_str or "error" in focus_str:
            return "seeking_resolution"
        elif "explicar" in focus_str or "qué es" in focus_str or "cómo" in focus_str:
            return "seeking_knowledge"
        elif "crear" in focus_str or "generar" in focus_str:
            return "creative_expression"
        elif "gracias" in focus_str or "bueno" in focus_str:
            return "social_bonding"
        elif "no" in focus_str or "mal" in focus_str:
            return "expressing_dissatisfaction"
        
        # Fallback al contexto
        task_type = context.get("task_type", "general")
        return f"executing_{task_type}"

    def _update_beliefs(self, user_state: MentalState, moment: Dict):
        """Actualiza el mapa de creencias del usuario."""
        # Detectar afirmaciones fuertes
        content = str(moment.get("integrated_content", "")).lower()
        
        # Heurística simple: Si el usuario dice "creo que X" o "X es Y"
        # En producción real, esto requeriría parsing semántico
        if "importante" in content:
            # El usuario valora el tema actual
            topic = self._extract_topic(content)
            if topic:
                user_state.belief_system[f"values_{topic}"] = 0.8

    def _extract_topic(self, text: str) -> Optional[str]:
        """Intenta extraer un tópico simple del texto."""
        words = text.split()
        if len(words) > 3:
            return words[2] # Muy simple, solo para demo funcional
        return None

    def _predict_needs(self, user_state: MentalState):
        """Predice qué podría necesitar el usuario a continuación."""
        user_state.predicted_needs = []
        
        if user_state.current_emotion < -0.3:
            user_state.predicted_needs.append("emotional_support")
            user_state.predicted_needs.append("problem_resolution")
        
        if user_state.inferred_intent == "seeking_knowledge":
            user_state.predicted_needs.append("clear_explanation")
            user_state.predicted_needs.append("examples")
            
        if user_state.interaction_count > 10 and user_state.empathy_score < 0.6:
            user_state.predicted_needs.append("rapport_building")

    def _recalculate_system_social_intelligence(self):
        """
        Calcula una métrica global de qué tan bien el sistema está modelando a los usuarios.
        Basado en la cantidad de modelos activos y la profundidad de los mismos.
        """
        if not self.user_models:
            self.system_social_intelligence = 0.1
            return

        total_interactions = sum(u.interaction_count for u in self.user_models.values())
        avg_interactions = total_interactions / len(self.user_models)
        
        # Complejidad del modelo: cuántas creencias/necesidades hemos inferido
        complexity_score = np.mean([
            len(u.belief_system) + len(u.predicted_needs) 
            for u in self.user_models.values()
        ])

        # Normalizar score entre 0 y 1
        # Asumimos que >50 interacciones promedio y >5 items de complejidad es "experto"
        interaction_factor = min(1.0, avg_interactions / 50.0)
        complexity_factor = min(1.0, complexity_score / 5.0)
        
        self.system_social_intelligence = (interaction_factor * 0.4) + (complexity_factor * 0.6)


# =============================================================================
# INTEGRATION WITH ADVANCED THEORY OF MIND (LEVELS 8-10)
# =============================================================================

try:
    from .teoria_mente_avanzada import (
        AdvancedTheoryOfMind,
        BeliefType,
        SocialStrategy
    )
    ADVANCED_TOM_AVAILABLE = True
    print("[OK] Advanced Theory of Mind (Levels 8-10) LOADED successfully")
except ImportError as e:
    ADVANCED_TOM_AVAILABLE = False
    print(f"⚠️  Advanced Theory of Mind not available: {e}")


class UnifiedTheoryOfMind:
    """
    Unified Theory of Mind system combining basic (Levels 1-7) and advanced (Levels 8-10).
    
    Provides seamless integration between:
    - Basic ToM: Single-agent mental state modeling
    - Advanced ToM: Multi-agent belief hierarchies, strategic reasoning, cultural context
    
    Usage:
        tom = UnifiedTheoryOfMind()
        
        # Basic update (Level 1-7)
        tom.update_model("user123", conscious_moment)
        
        # Advanced multi-agent interaction (Level 8-10)
        if tom.has_advanced_capabilities:
            result = await tom.process_social_interaction(
                actor="agent_a",
                target="agent_b",
                interaction_type="negotiation",
                content={"text": "Let's collaborate"}
            )
    """
    
    def __init__(self, enable_advanced: bool = True, max_belief_depth: int = 5):
        """
        Initialize Unified Theory of Mind.
        
        Args:
            enable_advanced: Enable advanced ToM (Levels 8-10) if available
            max_belief_depth: Maximum depth for belief hierarchies (Level 8)
        """
        # Basic ToM (Levels 1-7)
        self.basic_tom = TheoryOfMind()
        
        # Advanced ToM (Levels 8-10)
        self.advanced_tom = None
        self.has_advanced_capabilities = False
        
        if enable_advanced and ADVANCED_TOM_AVAILABLE:
            try:
                self.advanced_tom = AdvancedTheoryOfMind(max_belief_depth=max_belief_depth)
                self.has_advanced_capabilities = True
                print(f"🧠 Unified ToM initialized with FULL CAPABILITIES (Levels 1-10)")
            except Exception as e:
                print(f"⚠️  Advanced ToM initialization failed: {e}")
                print(f"🧠 Unified ToM initialized with basic capabilities (Levels 1-7)")
        else:
            print(f"🧠 Unified ToM initialized with basic capabilities (Levels 1-7)")
    
    def update_model(self, user_id: str, conscious_moment: Dict[str, Any]):
        """
        Update user model with basic ToM (Levels 1-7).
        
        This is the primary interface for single-user mental state tracking.
        """
        self.basic_tom.update_model(user_id, conscious_moment)
    
    def get_user_model(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get basic mental state model for a user"""
        return self.basic_tom.get_user_model(user_id)
    
    def get_social_intelligence_score(self) -> float:
        """Get overall social intelligence score"""
        basic_score = self.basic_tom.get_social_intelligence_score()
        
        if self.has_advanced_capabilities and self.advanced_tom:
            # Get advanced ToM level
            status = self.advanced_tom.get_comprehensive_status()
            advanced_level = status.get("overall_tom_level", 6.0)
            
            # Combine scores (weighted average)
            # BasicToM contributes 40%, AdvancedToM contributes 60%
            return (basic_score * 0.4) + ((advanced_level / 10.0) * 0.6)
        
        return basic_score
    
    def get_tom_level(self) -> Tuple[float, str]:
        """
        Get current Theory of Mind level according to ConsScale.
        
        Returns:
            (level, description) where level is 1-10 and description explains capabilities
        """
        if not self.has_advanced_capabilities:
            # Basic ToM: Levels 1-7 based on social intelligence
            score = self.basic_tom.get_social_intelligence_score()
            if score < 0.2:
                return (1.0, "Level 1: Basic user modeling")
            elif score < 0.4:
                return (4.0, "Level 4: Attentional - can track user focus")
            elif score < 0.6:
                return (6.0, "Level 6: Emotional - models emotions and intent")
            else:
                return (7.0, "Level 7: Self-Conscious - advanced single-user ToM")
        
        # Advanced ToM: Levels 8-10
        status = self.advanced_tom.get_comprehensive_status()
        level = status.get("overall_tom_level", 6.0)
        
        if level >= 10.0:
            return (10.0, "Level 10: Human-Like - cultural modeling, Turing-capable")
        elif level >= 9.0:
            return (9.0, "Level 9: Social - Machiavellian strategic reasoning")
        elif level >= 8.0:
            return (8.0, "Level 8: Empathic - multi-agent belief hierarchies")
        else:
            return (7.0, "Level 7: Self-Conscious - transitioning to multi-agent")
    
    async def process_social_interaction(
        self,
        actor: str,
        target: str,
        interaction_type: str,
        content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process multi-agent social interaction (Level 8-10).
        
        Requires advanced ToM capabilities.
        
        Args:
            actor: Agent initiating interaction
            target: Target agent
            interaction_type: Type of interaction (request, offer, challenge, etc.)
            content: Interaction content
            context: Additional context
            
        Returns:
            Complete ToM analysis including beliefs, strategy, cultural assessment
            
        Raises:
            RuntimeError: If advanced ToM not available
        """
        if not self.has_advanced_capabilities:
            raise RuntimeError(
                "Advanced ToM (Levels 8-10) not available. "
                "Enable with enable_advanced=True and install teoria_mente_avanzada.py"
            )
        
        return await self.advanced_tom.process_social_interaction(
            actor, target, interaction_type, content, context
        )
    
    def create_belief_hierarchy(
        self,
        agent_chain: List[str],
        final_content: str,
        confidence: float = 0.8
    ) -> Optional[str]:
        """
        Create hierarchical belief (Level 8): "A believes that B believes that C believes X"
        
        Args:
            agent_chain: List of agents [A, B, C, ...]
            final_content: The deepest belief content
            confidence: Confidence level
            
        Returns:
            belief_id or None if advanced ToM not available
        """
        if not self.has_advanced_capabilities:
            logger.warning("Belief hierarchies require Advanced ToM (Level 8+)")
            return None
        
        return self.advanced_tom.belief_tracker.create_belief_hierarchy(
            agent_chain, final_content, confidence
        )
    
    def evaluate_strategic_action(
        self,
        actor: str,
        target: str,
        strategy_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate strategic social action (Level 9).
        
        Args:
            actor: Agent performing action
            target: Target agent
            strategy_type: Strategy type (cooperation, deception, alliance, etc.)
            context: Additional context
            
        Returns:
            Strategic action evaluation or None if not available
        """
        if not self.has_advanced_capabilities:
            logger.warning("Strategic reasoning requires Advanced ToM (Level 9+)")
            return None
        
        try:
            strategy = SocialStrategy[strategy_type.upper()]
        except KeyError:
            logger.error(f"Invalid strategy type: {strategy_type}")
            return None
        
        action = self.advanced_tom.strategic_reasoner.evaluate_strategic_action(
            actor, target, strategy, context or {}
        )
        
        return {
            "strategy": action.action_type.value,
            "expected_payoff": action.expected_payoff,
            "risk_level": action.risk_level,
            "ethical_score": action.ethical_score,
            "description": action.description,
            "predicted_responses": action.predicted_responses
        }
    
    def assign_culture(self, agent_id: str, cultures: List[str]):
        """
        Assign cultural background to agent (Level 10).
        
        Args:
            agent_id: Agent identifier
            cultures: List of culture identifiers
        """
        if not self.has_advanced_capabilities:
            logger.warning("Cultural modeling requires Advanced ToM (Level 10)")
            return
        
        self.advanced_tom.cultural_engine.assign_culture_to_agent(agent_id, cultures)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get complete status of all ToM subsystems"""
        basic_status = {
            "users_modeled": len(self.basic_tom.user_models),
            "social_intelligence": self.basic_tom.get_social_intelligence_score()
        }
        
        if self.has_advanced_capabilities:
            advanced_status = self.advanced_tom.get_comprehensive_status()
            advanced_status["basic_tom"] = basic_status
            return advanced_status
        
        tom_level, description = self.get_tom_level()
        basic_status["tom_level"] = tom_level
        basic_status["capabilities"] = description
        basic_status["advanced_available"] = False
        
        return basic_status


# Module-level convenience instance
_unified_tom_instance: Optional[UnifiedTheoryOfMind] = None


def get_unified_tom(enable_advanced: bool = True) -> UnifiedTheoryOfMind:
    """
    Get singleton instance of Unified Theory of Mind.
    
    Args:
        enable_advanced: Enable advanced ToM capabilities
        
    Returns:
        UnifiedTheoryOfMind instance
    """
    global _unified_tom_instance
    if _unified_tom_instance is None:
        _unified_tom_instance = UnifiedTheoryOfMind(enable_advanced=enable_advanced)
    return _unified_tom_instance
