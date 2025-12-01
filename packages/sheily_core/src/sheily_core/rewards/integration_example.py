"""
Sheily Rewards Integration Example
==================================

Ejemplo de integración del sistema de recompensas con otros componentes.
"""

from typing import Dict, Any, Optional
from .reward_system import SheilyRewardSystem


class SheilyRewardsIntegration:
    """
    Clase de integración para conectar el sistema de recompensas
    con otros componentes de Sheily AI.
    """

    def __init__(self, reward_system: Optional[SheilyRewardSystem] = None):
        self.reward_system = reward_system or SheilyRewardSystem()
        self.integrations = {}

    def register_integration(self, name: str, integration_func):
        """
        Registrar una nueva integración.

        Args:
            name: Nombre de la integración
            integration_func: Función de integración
        """
        self.integrations[name] = integration_func

    def process_reward_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar un evento de recompensa y ejecutar integraciones.

        Args:
            event_data: Datos del evento de recompensa

        Returns:
            Dict con resultados del procesamiento
        """
        try:
            # Procesar recompensa principal
            reward_result = self.reward_system.record_reward(event_data)

            # Ejecutar integraciones
            integration_results = {}
            for name, integration_func in self.integrations.items():
                try:
                    integration_results[name] = integration_func(event_data, reward_result)
                except Exception as e:
                    integration_results[name] = {"error": str(e)}

            return {
                "success": True,
                "reward_result": reward_result,
                "integration_results": integration_results,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_integration_status(self) -> Dict[str, Any]:
        """
        Obtener estado de todas las integraciones.

        Returns:
            Dict con estado de integraciones
        """
        return {
            "active_integrations": list(self.integrations.keys()),
            "total_integrations": len(self.integrations),
        }


def create_integration_example() -> SheilyRewardsIntegration:
    """
    Crear una instancia de ejemplo de integración.

    Returns:
        SheilyRewardsIntegration: Instancia configurada
    """
    integration = SheilyRewardsIntegration()

    # Registrar integración de ejemplo
    def example_integration(event_data, reward_result):
        return {
            "processed": True,
            "event_type": event_data.get("domain", "unknown"),
            "reward_id": reward_result.get("reward_id"),
        }

    integration.register_integration("example", example_integration)

    return integration
