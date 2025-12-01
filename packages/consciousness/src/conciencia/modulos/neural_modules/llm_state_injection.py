"""
LLM State Injection - Inyección de Tokens de Estado
====================================================

Sistema para inyectar estados del cerebro como tokens especiales
en el prompt del LLM.
"""

import logging
from typing import Dict, Any, Optional, List
import json

logger = logging.getLogger(__name__)


class LLMStateInjection:
    """
    Sistema de inyección de tokens de estado al LLM.
    
    Convierte estados de módulos neurales en tokens especiales
    que el LLM puede interpretar para modular su comportamiento.
    """
    
    def __init__(self):
        """Inicializa el sistema de inyección."""
        self.state_tokens = []
    
    def create_state_tokens(self, neural_states: Dict[str, Any]) -> str:
        """
        Crea tokens de estado a partir de estados neurales.
        
        Args:
            neural_states: Dict con estados de módulos neurales
                - ras: Dict con arousal, neurotransmisores
                - vmpfc: Dict con empathy_score, emotional_bias, tone_modulation
                - ofc: Dict con decision_confidence, requires_planning
                - ecn: Dict con control_mode, gating_weights
                - brain_state: Dict con estado global
                
        Returns:
            String con tokens de estado formateados
        """
        tokens = []
        
        # RAS tokens
        if "ras" in neural_states:
            ras = neural_states["ras"]
            arousal = ras.get("arousal", 0.5)
            tokens.append(f"<STATE:r_as={arousal:.2f}>")
            tokens.append(f"<STATE:ne={ras.get('norepinephrine', 0.5):.2f}>")
            tokens.append(f"<STATE:da={ras.get('dopamine', 0.5):.2f}>")
        
        # vmPFC tokens
        if "vmpfc" in neural_states:
            vmpfc = neural_states["vmpfc"]
            tokens.append(f"<STATE:emotion={vmpfc.get('emotional_bias', 0.0):.2f}>")
            tokens.append(f"<STATE:empathy={vmpfc.get('empathy_score', 0.7):.2f}>")
            tokens.append(f"<STATE:tone={vmpfc.get('tone_modulation', 0.5):.2f}>")
        
        # OFC tokens
        if "ofc" in neural_states:
            ofc = neural_states["ofc"]
            tokens.append(f"<STATE:decision_confidence={ofc.get('decision_confidence', 0.5):.2f}>")
            if ofc.get("requires_planning", False):
                tokens.append("<STATE:requires_planning=true>")
        
        # ECN tokens
        if "ecn" in neural_states:
            ecn = neural_states["ecn"]
            control_mode = ecn.get("control_mode", "automatic")
            tokens.append(f"<STATE:control_mode={control_mode}>")
        
        # Brain state tokens
        if "brain_state" in neural_states:
            brain_state = neural_states["brain_state"]
            mood = brain_state.get("mood", "neutral")
            goals = brain_state.get("goals", [])
            tokens.append(f"<STATE:mood={mood}>")
            if goals:
                goals_str = ",".join(goals[:3])  # Primeros 3 objetivos
                tokens.append(f'<STATE:goal="{goals_str}">')
        
        return " ".join(tokens)
    
    def inject_into_prompt(self, prompt: str, neural_states: Dict[str, Any], 
                          position: str = "prepend") -> str:
        """
        Inyecta tokens de estado en un prompt.
        
        Args:
            prompt: Prompt original
            neural_states: Estados neurales
            position: Posición de inyección ("prepend", "append", "both")
            
        Returns:
            Prompt con tokens inyectados
        """
        state_tokens = self.create_state_tokens(neural_states)
        
        if position == "prepend":
            return f"{state_tokens}\n\n{prompt}"
        elif position == "append":
            return f"{prompt}\n\n{state_tokens}"
        elif position == "both":
            return f"{state_tokens}\n\n{prompt}\n\n{state_tokens}"
        else:
            return prompt
    
    def parse_state_tokens(self, text: str) -> Dict[str, Any]:
        """
        Parsea tokens de estado de un texto.
        
        Args:
            text: Texto con tokens de estado
            
        Returns:
            Dict con estados parseados
        """
        states = {}
        
        # Buscar tokens <STATE:key=value>
        import re
        pattern = r'<STATE:(\w+)=([^>]+)>'
        matches = re.findall(pattern, text)
        
        for key, value in matches:
            # Intentar convertir a número
            try:
                if '.' in value:
                    states[key] = float(value)
                else:
                    states[key] = int(value)
            except ValueError:
                # Mantener como string
                states[key] = value.strip('"')
        
        return states

