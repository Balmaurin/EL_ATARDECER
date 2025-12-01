"""
Auto Questions - Sistema de Auto-Preguntas
==========================================

Sistema que genera preguntas sobre su propio funcionamiento
y las usa para entrenamiento.
"""

import logging
from typing import Dict, Any, Optional, List
import random

logger = logging.getLogger(__name__)


class AutoQuestions:
    """
    Sistema de auto-preguntas para entrenamiento.
    
    Genera preguntas sobre:
    - Funcionamiento interno
    - Decisiones tomadas
    - Estados emocionales
    - Memoria y recuerdos
    """
    
    def __init__(self):
        """Inicializa el sistema de auto-preguntas."""
        self.question_templates = [
            "¿Por qué respondí de esa manera?",
            "¿Qué emoción sentí al procesar esa pregunta?",
            "¿Cómo llegué a esa decisión?",
            "¿Qué información de mi memoria usé?",
            "¿Fue apropiada mi respuesta?",
            "¿Qué podría haber mejorado en mi respuesta?",
            "¿Cómo me sentí durante esa interacción?",
            "¿Qué aprendí de esa experiencia?",
            "¿Mi nivel de empatía fue adecuado?",
            "¿Mi razonamiento fue correcto?"
        ]
        
        logger.info("AutoQuestions system initialized")
    
    def generate_question(self, context: Dict[str, Any]) -> str:
        """
        Genera una pregunta automática basada en contexto.
        
        Args:
            context: Contexto de la interacción reciente
            
        Returns:
            Pregunta generada
        """
        # Seleccionar template aleatorio
        template = random.choice(self.question_templates)
        
        # Personalizar según contexto
        user_message = context.get("user_message", "")
        response = context.get("response", "")
        
        # Añadir contexto específico
        if "emoción" in user_message.lower() or "sentir" in user_message.lower():
            template = "¿Cómo procesé las emociones en esa interacción?"
        elif "decidir" in user_message.lower() or "elegir" in user_message.lower():
            template = "¿Cómo llegué a esa decisión?"
        elif "recordar" in user_message.lower() or "memoria" in user_message.lower():
            template = "¿Qué información de mi memoria usé?"
        
        return template
    
    def generate_question_batch(self, num_questions: int = 5, 
                               contexts: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Genera un batch de preguntas.
        
        Args:
            num_questions: Número de preguntas
            contexts: Lista de contextos (opcional)
            
        Returns:
            Lista de preguntas
        """
        questions = []
        
        if contexts:
            for context in contexts[:num_questions]:
                questions.append(self.generate_question(context))
        else:
            # Generar preguntas genéricas
            for _ in range(num_questions):
                questions.append(random.choice(self.question_templates))
        
        return questions
    
    def create_training_example(self, question: str, response: str, 
                               neural_states: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea un ejemplo de entrenamiento a partir de auto-pregunta.
        
        Args:
            question: Pregunta generada
            response: Respuesta del sistema
            neural_states: Estados neurales durante la respuesta
            
        Returns:
            Ejemplo de entrenamiento
        """
        example = {
            "question": question,
            "response": response,
            "neural_states": neural_states,
            "timestamp": None,  # Se llenará al guardar
            "type": "auto_question"
        }
        
        return example
    
    def validate_response(self, question: str, response: str, 
                         expected_components: List[str]) -> Dict[str, Any]:
        """
        Valida si una respuesta contiene componentes esperados.
        
        Args:
            question: Pregunta
            response: Respuesta a validar
            expected_components: Componentes esperados en la respuesta
            
        Returns:
            Dict con validación
        """
        response_lower = response.lower()
        
        validation = {
            "is_valid": True,
            "missing_components": [],
            "score": 0.0
        }
        
        for component in expected_components:
            if component.lower() not in response_lower:
                validation["missing_components"].append(component)
                validation["is_valid"] = False
        
        # Calcular score
        if len(expected_components) > 0:
            validation["score"] = 1.0 - (len(validation["missing_components"]) / len(expected_components))
        else:
            validation["score"] = 1.0
        
        return validation

