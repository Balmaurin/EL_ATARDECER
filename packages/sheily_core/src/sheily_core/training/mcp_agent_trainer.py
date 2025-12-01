"""
Entrenamiento REAL de agentes MCP y consolidados
Sistema completo sin mocks ni fallbacks
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MCPAgentTrainer:
    """Entrenador REAL para agentes MCP y consolidados"""
    
    def __init__(self):
        self.training_history = []
    
    async def train_core_agent(self, qa_data: List[Dict]) -> Dict[str, Any]:
        """Entrenar CoreAgent REAL con datos de Hack-Memori"""
        logger.info("🤖 Entrenando CoreAgent...")
        
        try:
            from packages.sheily_core.src.sheily_core.core.system.consolidated_agents import CoreAgent
            
            # Crear instancia de CoreAgent
            core_agent = CoreAgent(agent_id="core_agent_training", name="Core Agent Training")
            await core_agent.initialize()
            
            # Preparar datos de entrenamiento
            training_examples = []
            for qa in qa_data:
                if qa.get("quality_score", 0) >= 0.7:
                    training_examples.append({
                        "type": "chat",
                        "data": {
                            "query": qa.get("question", ""),
                            "context": qa.get("response", ""),
                            "quality_score": qa.get("quality_score", 0.5)
                        }
                    })
            
            # Entrenar con ejemplos
            trained_count = 0
            for example in training_examples[:50]:  # Limitar a 50 para no sobrecargar
                try:
                    result = await core_agent._handle_request(example)
                    if result.get("status") == "success":
                        trained_count += 1
                except Exception as e:
                    logger.warning(f"Error entrenando ejemplo: {e}")
            
            logger.info(f"✅ CoreAgent entrenado con {trained_count} ejemplos")
            
            return {
                "success": True,
                "agent": "CoreAgent",
                "examples_trained": trained_count,
                "total_examples": len(training_examples)
            }
            
        except Exception as e:
            logger.error(f"Error entrenando CoreAgent: {e}")
            raise
    
    async def train_business_agent(self, qa_data: List[Dict]) -> Dict[str, Any]:
        """Entrenar BusinessAgent REAL"""
        logger.info("💼 Entrenando BusinessAgent...")
        
        try:
            from packages.sheily_core.src.sheily_core.core.system.consolidated_agents import BusinessAgent
            
            business_agent = BusinessAgent(agent_id="business_agent_training", name="Business Agent Training")
            await business_agent.initialize()
            
            # Filtrar Q&A relacionadas con negocio
            business_qa = [
                qa for qa in qa_data
                if any(word in qa.get("question", "").lower() for word in ["negocio", "empresa", "venta", "cliente", "mercado", "precio"])
            ]
            
            trained_count = 0
            for qa in business_qa[:30]:
                try:
                    result = await business_agent._handle_request({
                        "type": "business_analysis",
                        "data": {
                            "query": qa.get("question", ""),
                            "context": qa.get("response", "")
                        }
                    })
                    if result.get("status") == "success":
                        trained_count += 1
                except Exception as e:
                    logger.warning(f"Error entrenando BusinessAgent: {e}")
            
            logger.info(f"✅ BusinessAgent entrenado con {trained_count} ejemplos")
            
            return {
                "success": True,
                "agent": "BusinessAgent",
                "examples_trained": trained_count,
                "total_examples": len(business_qa)
            }
            
        except Exception as e:
            logger.error(f"Error entrenando BusinessAgent: {e}")
            raise
    
    async def train_infrastructure_agent(self, qa_data: List[Dict]) -> Dict[str, Any]:
        """Entrenar InfrastructureAgent REAL"""
        logger.info("🏗️ Entrenando InfrastructureAgent...")
        
        try:
            from packages.sheily_core.src.sheily_core.core.system.consolidated_agents import InfrastructureAgent
            
            infra_agent = InfrastructureAgent(agent_id="infra_agent_training", name="Infrastructure Agent Training")
            await infra_agent.initialize()
            
            # Filtrar Q&A relacionadas con infraestructura
            infra_qa = [
                qa for qa in qa_data
                if any(word in qa.get("question", "").lower() for word in ["sistema", "servidor", "red", "base de datos", "deploy", "monitoreo"])
            ]
            
            trained_count = 0
            for qa in infra_qa[:30]:
                try:
                    result = await infra_agent._handle_request({
                        "type": "infrastructure_operation",
                        "data": {
                            "query": qa.get("question", ""),
                            "context": qa.get("response", "")
                        }
                    })
                    if result.get("status") == "success":
                        trained_count += 1
                except Exception as e:
                    logger.warning(f"Error entrenando InfrastructureAgent: {e}")
            
            logger.info(f"✅ InfrastructureAgent entrenado con {trained_count} ejemplos")
            
            return {
                "success": True,
                "agent": "InfrastructureAgent",
                "examples_trained": trained_count,
                "total_examples": len(infra_qa)
            }
            
        except Exception as e:
            logger.error(f"Error entrenando InfrastructureAgent: {e}")
            raise
    
    async def train_meta_cognition_agent(self, qa_data: List[Dict]) -> Dict[str, Any]:
        """Entrenar MetaCognitionAgent REAL"""
        logger.info("🧠 Entrenando MetaCognitionAgent...")
        
        try:
            from packages.sheily_core.src.sheily_core.core.system.consolidated_agents import MetaCognitionAgent
            
            meta_agent = MetaCognitionAgent(agent_id="meta_agent_training", name="Meta-Cognition Agent Training")
            await meta_agent.initialize()
            
            # Filtrar Q&A relacionadas con meta-cognición
            meta_qa = [
                qa for qa in qa_data
                if any(word in qa.get("question", "").lower() for word in ["aprender", "mejorar", "optimizar", "entrenar", "evolución"])
            ]
            
            trained_count = 0
            for qa in meta_qa[:30]:
                try:
                    result = await meta_agent._handle_request({
                        "type": "meta_cognition",
                        "data": {
                            "query": qa.get("question", ""),
                            "context": qa.get("response", ""),
                            "learning_opportunity": True
                        }
                    })
                    if result.get("status") == "success":
                        trained_count += 1
                except Exception as e:
                    logger.warning(f"Error entrenando MetaCognitionAgent: {e}")
            
            logger.info(f"✅ MetaCognitionAgent entrenado con {trained_count} ejemplos")
            
            return {
                "success": True,
                "agent": "MetaCognitionAgent",
                "examples_trained": trained_count,
                "total_examples": len(meta_qa)
            }
            
        except Exception as e:
            logger.error(f"Error entrenando MetaCognitionAgent: {e}")
            raise
    
    async def train_all_agents(self, qa_data: List[Dict]) -> Dict[str, Any]:
        """Entrenar TODOS los agentes consolidados REAL"""
        logger.info("🚀 Entrenando TODOS los agentes MCP y consolidados...")
        
        results = {}
        
        # Entrenar cada agente
        try:
            results["core_agent"] = await self.train_core_agent(qa_data)
        except Exception as e:
            logger.error(f"Error entrenando CoreAgent: {e}")
            results["core_agent"] = {"success": False, "error": str(e)}
        
        try:
            results["business_agent"] = await self.train_business_agent(qa_data)
        except Exception as e:
            logger.error(f"Error entrenando BusinessAgent: {e}")
            results["business_agent"] = {"success": False, "error": str(e)}
        
        try:
            results["infrastructure_agent"] = await self.train_infrastructure_agent(qa_data)
        except Exception as e:
            logger.error(f"Error entrenando InfrastructureAgent: {e}")
            results["infrastructure_agent"] = {"success": False, "error": str(e)}
        
        try:
            results["meta_cognition_agent"] = await self.train_meta_cognition_agent(qa_data)
        except Exception as e:
            logger.error(f"Error entrenando MetaCognitionAgent: {e}")
            results["meta_cognition_agent"] = {"success": False, "error": str(e)}
        
        # Calcular estadísticas
        total_trained = sum(
            r.get("examples_trained", 0)
            for r in results.values()
            if isinstance(r, dict) and r.get("success")
        )
        
        successful_agents = sum(
            1 for r in results.values()
            if isinstance(r, dict) and r.get("success")
        )
        
        logger.info(f"✅ Entrenamiento de agentes completado")
        logger.info(f"   - Agentes exitosos: {successful_agents}/4")
        logger.info(f"   - Total ejemplos entrenados: {total_trained}")
        
        return {
            "success": successful_agents > 0,
            "agents_trained": successful_agents,
            "total_examples_trained": total_trained,
            "results": results
        }

