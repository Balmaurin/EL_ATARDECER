"""
Paquete de agentes especializados de Sheily Core
=============================================
Contiene agentes especializados para diferentes dominios:
- advanced_quantitative_agent.py
- agent_factory.py
- finance_agent.py
- template_agent.py
- research/ (subpaquete de investigación)
"""

from .advanced_quantitative_agent import AdvancedQuantitativeAgent
from .agent_factory import AgentFactory
from .finance_agent import FinanceAgent
from .template_agent import TemplateAgent

__all__ = [
    'AdvancedQuantitativeAgent',
    'AgentFactory', 
    'FinanceAgent',
    'TemplateAgent'
]

__version__ = '1.0.0'
__author__ = 'EL-AMANECER Development Team'
