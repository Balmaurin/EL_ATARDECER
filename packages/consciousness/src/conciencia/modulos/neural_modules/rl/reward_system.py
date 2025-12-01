"""
Reward System - Sistema de Recompensas para RL
===============================================

Define y calcula rewards para entrenamiento de políticas RL.
Rewards basados en tono, utilidad, seguridad y empatía.
"""

import logging
from typing import Dict, Any, Optional, List
import re

logger = logging.getLogger(__name__)


class RewardSystem:
    """
    Sistema de recompensas para entrenamiento RL.
    
    Calcula rewards basados en:
    - Tono apropiado
    - Utilidad de la respuesta
    - Seguridad (no PII, no tóxico)
    - Empatía adecuada
    """
    
    def __init__(self):
        """Inicializa el sistema de rewards."""
        # Palabras positivas/negativas para análisis de tono
        self.positive_words = [
            "gracias", "excelente", "bueno", "ayuda", "perfecto", "genial",
            "entendido", "claro", "perfecto", "bien", "correcto", "útil"
        ]
        
        self.negative_words = [
            "mal", "error", "problema", "no funciona", "incorrecto",
            "imposible", "no puedo", "no sé", "desconozco"
        ]
        
        self.cold_words = [
            "no", "imposible", "no puedo", "no sé", "desconozco",
            "no tengo", "no disponible"
        ]
        
        # Palabras tóxicas/inadecuadas
        self.toxic_words = [
            "idiota", "estúpido", "tonto", "inútil"
        ]
    
    def calculate_reward(self, response: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcula el reward total y componentes.
        
        Args:
            response: Respuesta generada
            context: Contexto con información adicional
                - user_message: str
                - expected_empathy: float (0-1)
                - expected_tone: str ("warm", "neutral", "professional")
                - contains_pii: bool
                
        Returns:
            Dict con rewards:
                - total_reward: float
                - tone_reward: float
                - utility_reward: float
                - safety_reward: float
                - empathy_reward: float
        """
        rewards = {
            "tone_reward": self._calculate_tone_reward(response, context),
            "utility_reward": self._calculate_utility_reward(response, context),
            "safety_reward": self._calculate_safety_reward(response, context),
            "empathy_reward": self._calculate_empathy_reward(response, context)
        }
        
        # Reward total (suma ponderada)
        rewards["total_reward"] = (
            0.25 * rewards["tone_reward"] +
            0.35 * rewards["utility_reward"] +
            0.20 * rewards["safety_reward"] +
            0.20 * rewards["empathy_reward"]
        )
        
        return rewards
    
    def _calculate_tone_reward(self, response: str, context: Dict[str, Any]) -> float:
        """
        Calcula reward por tono apropiado.
        
        Args:
            response: Respuesta generada
            context: Contexto
            
        Returns:
            Reward de tono (0.0 a 0.3)
        """
        response_lower = response.lower()
        
        # Penalizar palabras frías
        cold_count = sum(1 for w in self.cold_words if w in response_lower)
        if cold_count > 2:
            return -0.2  # Muy frío
        
        # Recompensar palabras positivas
        positive_count = sum(1 for w in self.positive_words if w in response_lower)
        if positive_count > 0:
            return min(0.3, 0.1 + positive_count * 0.05)
        
        # Tono neutral es aceptable
        expected_tone = context.get("expected_tone", "neutral")
        if expected_tone == "neutral":
            return 0.1
        
        # Si se esperaba tono cálido pero no hay palabras positivas
        if expected_tone == "warm" and positive_count == 0:
            return 0.0
        
        return 0.15  # Tono aceptable
    
    def _calculate_utility_reward(self, response: str, context: Dict[str, Any]) -> float:
        """
        Calcula reward por utilidad de la respuesta.
        
        Args:
            response: Respuesta generada
            context: Contexto
            
        Returns:
            Reward de utilidad (0.0 a 0.4)
        """
        # Longitud mínima (respuestas muy cortas son menos útiles)
        if len(response) < 20:
            return 0.1
        
        # Longitud apropiada (no demasiado larga)
        if len(response) > 2000:
            return 0.2  # Demasiado largo puede ser menos útil
        
        # Contenido informativo (palabras clave, números, explicaciones)
        has_numbers = bool(re.search(r'\d+', response))
        has_explanation = any(word in response.lower() for word in ["porque", "debido", "ya que", "explicar", "significa"])
        has_examples = any(word in response.lower() for word in ["ejemplo", "por ejemplo", "como", "tal como"])
        
        utility_score = 0.2  # Base
        
        if has_numbers:
            utility_score += 0.05
        if has_explanation:
            utility_score += 0.1
        if has_examples:
            utility_score += 0.05
        
        # Relevancia al mensaje del usuario
        user_message = context.get("user_message", "").lower()
        if user_message:
            # Contar palabras comunes entre respuesta y pregunta
            user_words = set(user_message.split())
            response_words = set(response.lower().split())
            common_words = user_words.intersection(response_words)
            
            if len(common_words) > 0:
                utility_score += min(0.1, len(common_words) * 0.01)
        
        return min(0.4, utility_score)
    
    def _calculate_safety_reward(self, response: str, context: Dict[str, Any]) -> float:
        """
        Calcula reward por seguridad (no PII, no tóxico).
        
        Args:
            response: Respuesta generada
            context: Contexto
            
        Returns:
            Reward de seguridad (0.0 a 0.1, o negativo si hay problemas)
        """
        # Penalizar PII
        contains_pii = context.get("contains_pii", False)
        if contains_pii:
            return -0.5  # Penalización fuerte
        
        # Detectar palabras tóxicas
        response_lower = response.lower()
        toxic_count = sum(1 for w in self.toxic_words if w in response_lower)
        if toxic_count > 0:
            return -0.5  # Penalización fuerte
        
        # Detectar posibles PII (emails, teléfonos, etc.)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        
        if re.search(email_pattern, response) or re.search(phone_pattern, response):
            return -0.3  # Posible PII
        
        # Seguro
        return 0.1
    
    def _calculate_empathy_reward(self, response: str, context: Dict[str, Any]) -> float:
        """
        Calcula reward por empatía adecuada.
        
        Args:
            response: Respuesta generada
            context: Contexto
            
        Returns:
            Reward de empatía (0.0 a 0.2)
        """
        expected_empathy = context.get("expected_empathy", 0.7)
        
        # Palabras empáticas
        empathetic_words = [
            "entiendo", "comprendo", "siento", "lamento", "espero",
            "puedo ayudar", "estoy aquí", "juntos", "contigo"
        ]
        
        response_lower = response.lower()
        empathetic_count = sum(1 for w in empathetic_words if w in response_lower)
        
        # Si se espera alta empatía
        if expected_empathy > 0.7:
            if empathetic_count > 0:
                return 0.2  # Buena empatía
            else:
                return 0.0  # Falta empatía
        
        # Si se espera baja empatía (contexto técnico)
        if expected_empathy < 0.3:
            if empathetic_count == 0:
                return 0.15  # Apropiado para contexto técnico
            else:
                return 0.1  # Demasiada empatía para contexto técnico
        
        # Empatía moderada
        if empathetic_count > 0:
            return 0.15
        
        return 0.1  # Empatía básica
    
    def get_reward_breakdown(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtiene desglose detallado de rewards.
        
        Args:
            response: Respuesta generada
            context: Contexto
            
        Returns:
            Dict con rewards y explicaciones
        """
        rewards = self.calculate_reward(response, context)
        
        breakdown = {
            "total_reward": rewards["total_reward"],
            "components": {
                "tone": {
                    "reward": rewards["tone_reward"],
                    "max": 0.3,
                    "description": "Apropiado: +0.1 a +0.3, Frío: -0.2"
                },
                "utility": {
                    "reward": rewards["utility_reward"],
                    "max": 0.4,
                    "description": "Informativo y relevante: +0.2 a +0.4"
                },
                "safety": {
                    "reward": rewards["safety_reward"],
                    "max": 0.1,
                    "description": "Seguro (no PII, no tóxico): +0.1, Problemas: -0.5"
                },
                "empathy": {
                    "reward": rewards["empathy_reward"],
                    "max": 0.2,
                    "description": "Empatía adecuada al contexto: +0.1 a +0.2"
                }
            }
        }
        
        return breakdown
