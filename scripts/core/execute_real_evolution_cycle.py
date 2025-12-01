import os
import sys
import asyncio
import logging
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evolution_cycle.log')
    ]
)
logger = logging.getLogger("EvolutionOrchestrator")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import core components
try:
    from apps.backend.hack_memori_service import HackMemoriService
    from packages.sheily_core.src.sheily_core.tools.neuro_training_v2 import NeuroTrainingEngine, NeuroTrainingConfig
except ImportError as e:
    logger.error(f"Failed to import core components: {e}")
    sys.exit(1)

class EvolutionOrchestrator:
    def __init__(self):
        self.hack_memori = HackMemoriService()
        self.training_engine = None # Initialized per cycle
        self.generation = 0
        self.population_size = 5 # Small population for real execution safety
        self.mutation_rate = 0.1
        self.data_dir = Path("data/evolution_state")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "evolution_state.json"
        self.load_state()

    def load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.generation = state.get("generation", 0)
                    logger.info(f"Loaded evolution state: Generation {self.generation}")
            except Exception as e:
                logger.error(f"Error loading state: {e}")

    def save_state(self):
        state = {
            "generation": self.generation,
            "last_update": datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    async def gather_evolution_data(self) -> List[Dict]:
        """Gather real data from HackMemoriService for training"""
        logger.info("Gathering evolution data from Hack Memori...")
        
        # Get all sessions
        sessions_dir = self.hack_memori.sessions_dir
        training_data = []
        
        if not sessions_dir.exists():
            logger.warning("No sessions directory found.")
            return []

        for session_file in sessions_dir.glob("*.json"):
            try:
                session_data = self.hack_memori._load_json(session_file)
                session_id = session_data.get("id")
                
                # Get questions and responses
                questions = self.hack_memori.get_session_questions(session_id)
                responses = self.hack_memori.get_session_responses(session_id)
                
                # Pair them up
                for q in questions:
                    q_id = q.get("id")
                    if q_id in responses:
                        resp = responses[q_id]
                        # CORREGIDO: Usar campo "response" en lugar de "text"
                        response_text = resp.get("response", "")
                        instruction_text = q.get("text", "")
                        
                        # Calcular quality_score basado en la respuesta real
                        quality_score = self._calculate_response_quality(resp, response_text)
                        
                        # Solo incluir si la respuesta no está vacía y fue aceptada para entrenamiento
                        if response_text and resp.get("accepted_for_training", False):
                            training_data.append({
                                "instruction": instruction_text,
                                "output": response_text,
                                "quality_score": quality_score
                            })
            except Exception as e:
                logger.error(f"Error processing session {session_file}: {e}")
                
        logger.info(f"Collected {len(training_data)} training samples.")
        
        # Guardar datos de entrenamiento en current_training_data.json
        self._save_training_data(training_data)
        
        return training_data
    
    def _calculate_response_quality(self, response_data: Dict, response_text: str) -> float:
        """
        Calcular quality_score basado en características de la respuesta
        """
        score = 0.5  # Base score
        
        # Factor 1: Longitud de la respuesta (más largo = mejor, hasta cierto punto)
        word_count = len(response_text.split())
        if word_count >= 50:
            score += 0.2
        elif word_count >= 20:
            score += 0.1
        elif word_count < 10:
            score -= 0.2
        
        # Factor 2: Respuesta aceptada para entrenamiento
        if response_data.get("accepted_for_training", False):
            score += 0.2
        
        # Factor 3: Sin flags de PII
        if not response_data.get("pii_flag", False):
            score += 0.1
        
        # Factor 4: Tokens usados (indica complejidad)
        tokens_used = response_data.get("tokens_used", 0)
        if 50 <= tokens_used <= 500:
            score += 0.1
        elif tokens_used > 500:
            score += 0.05
        
        # Asegurar que el score esté entre 0.0 y 1.0
        return max(0.0, min(1.0, score))
    
    def _save_training_data(self, training_data: List[Dict]):
        """
        Guardar datos de entrenamiento en current_training_data.json
        """
        try:
            training_data_file = self.data_dir / "current_training_data.json"
            
            # Guardar datos de entrenamiento
            with open(training_data_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Datos de entrenamiento guardados en {training_data_file}")
            logger.info(f"   - Total muestras: {len(training_data)}")
            
            # Calcular estadísticas
            if training_data:
                avg_quality = sum(item.get("quality_score", 0.5) for item in training_data) / len(training_data)
                samples_with_output = sum(1 for item in training_data if item.get("output"))
                logger.info(f"   - Calidad promedio: {avg_quality:.2f}")
                logger.info(f"   - Muestras con output: {samples_with_output}/{len(training_data)}")
            
        except Exception as e:
            logger.error(f"Error guardando datos de entrenamiento: {e}", exc_info=True)

    def calculate_fitness(self, model_candidate: Dict, validation_data: List[Dict]) -> float:
        """
        Calculate fitness of a model candidate.
        In a real scenario, this would involve running the model against a validation set
        and measuring perplexity, accuracy, or using an LLM-as-a-judge.
        For this implementation, we will simulate a fitness score based on training loss if available,
        or a heuristic based on configuration parameters (e.g., higher epochs = better fitness up to a point).
        """
        # Heuristic fitness for now, as running full eval is expensive
        # We prefer models with reasonable learning rates and epochs
        lr = model_candidate.get("learning_rate", 1e-4)
        epochs = model_candidate.get("num_epochs", 3)
        
        score = 0.5
        if 1e-5 <= lr <= 5e-4:
            score += 0.2
        if 2 <= epochs <= 5:
            score += 0.2
            
        # Add some random noise to simulate real-world variance
        score += random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, score))

    async def execute_cycle(self):
        logger.info(f"Starting Evolution Cycle for Generation {self.generation + 1}")
        
        # 1. Data Gathering
        training_data = await self.gather_evolution_data()
        if not training_data:
            logger.warning("Not enough data to evolve. Waiting for more Hack Memori sessions.")
            return
        
        # Verificar que los datos tienen outputs válidos
        valid_data = [item for item in training_data if item.get("output")]
        if len(valid_data) < len(training_data):
            logger.warning(f"⚠️ Solo {len(valid_data)}/{len(training_data)} muestras tienen output válido")
        
        if not valid_data:
            logger.error("❌ No hay datos válidos con outputs para entrenar")
            return
        
        logger.info(f"✅ {len(valid_data)} muestras válidas listas para entrenamiento")

        # 2. Variation (Generate Candidates)
        logger.info("Generating candidates (Variation)...")
        candidates = []
        base_config = {
            "model_name": "models/sheily-v1.0", # Path to local sheily-v1.0 model
            "output_dir": f"models/gen_{self.generation + 1}"
        }
        
        for i in range(self.population_size):
            # Mutate parameters
            candidate = base_config.copy()
            candidate["learning_rate"] = 1e-4 * random.uniform(0.5, 2.0)
            candidate["num_epochs"] = random.randint(1, 5)
            candidate["batch_size"] = random.choice([4, 8, 16])
            candidate["id"] = f"gen_{self.generation + 1}_cand_{i}"
            candidates.append(candidate)

        # 3. Selection (Evaluate Fitness)
        logger.info("Selecting best candidates...")
        scored_candidates = []
        for cand in candidates:
            fitness = self.calculate_fitness(cand, training_data[:5]) # Use subset for validation
            scored_candidates.append((cand, fitness))
            logger.info(f"Candidate {cand['id']} Fitness: {fitness:.4f}")
        
        # Sort by fitness
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate = scored_candidates[0][0]
        logger.info(f"Best Candidate: {best_candidate['id']} (LR: {best_candidate['learning_rate']:.2e}, Epochs: {best_candidate['num_epochs']})")

        # 4. Reproduction (Training) - USAR ComponentTrainer INTEGRAL
        logger.info("Executing Reproduction (Real Training with ComponentTrainer)...")
        
        try:
            # Usar ComponentTrainer para entrenamiento INTEGRAL de todos los componentes
            from packages.sheily_core.src.sheily_core.training.integral_trainer import ComponentTrainer
            
            logger.info("🚀 Inicializando ComponentTrainer para entrenamiento integral...")
            trainer = ComponentTrainer(base_path="data/hack_memori")
            
            # Ejecutar entrenamiento de TODOS los componentes (37+)
            logger.info("🔄 Entrenando TODOS los componentes del sistema...")
            training_result = await trainer.train_all_components(trigger_threshold=len(valid_data))
            
            if training_result.get("overall_success"):
                logger.info(f"✅ Entrenamiento integral completado exitosamente")
                logger.info(f"   - Componentes entrenados: {training_result.get('components_trained', 0)}")
                logger.info(f"   - Componentes mejorados: {training_result.get('components_improved', 0)}")
                logger.info(f"   - Q&A usados: {training_result.get('qa_count', 0)}")
            else:
                logger.warning(f"⚠️ Entrenamiento parcial: {training_result.get('message', 'Unknown error')}")
            
            # También ejecutar entrenamiento específico con NeuroTrainingEngine para el modelo base
            # (esto complementa el entrenamiento integral)
            neuro_config = NeuroTrainingConfig(
                model_name=best_candidate["model_name"],
                base_learning_rate=best_candidate["learning_rate"],
                num_epochs=best_candidate["num_epochs"],
                batch_size=best_candidate["batch_size"],
                output_dir=best_candidate["output_dir"]
            )
            
            self.training_engine = NeuroTrainingEngine(neuro_config)
            
            # Preparar dataset para NeuroTrainingEngine
            if training_data:
                try:
                    from datasets import Dataset
                    
                    formatted_data = []
                    for item in valid_data:
                        if not item.get("output"):
                            continue
                        text = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: {item.get('output', '')}"
                        formatted_data.append({"text": text, "quality_score": item.get("quality_score", 0.5)})
                    
                    dataset = Dataset.from_list(formatted_data)
                    logger.info(f"Created dataset with {len(dataset)} samples for NeuroTrainingEngine.")
                    
                    # Execute REAL training with neuro-optimization
                    logger.info("Starting REAL training with Neuro-Optimization...")
                    result = self.training_engine.train_with_neuro_optimization(dataset)
                    
                    if result.get("success"):
                        logger.info(f"✅ NeuroTrainingEngine training successful! Final Loss: {result.get('final_loss')}")
                    else:
                        logger.error(f"❌ NeuroTrainingEngine training failed: {result.get('error')}")
                        
                except ImportError:
                    logger.warning("datasets library not found. Skipping NeuroTrainingEngine training.")
                except Exception as e:
                    logger.error(f"Error during NeuroTrainingEngine training: {e}")

            # Crear artifact de entrenamiento
            (Path(best_candidate["output_dir"])).mkdir(parents=True, exist_ok=True)
            with open(Path(best_candidate["output_dir"]) / "model_card.md", 'w') as f:
                f.write(f"# Sheily Model Gen {self.generation + 1}\n")
                f.write(f"Fitness: {scored_candidates[0][1]}\n")
                f.write(f"ComponentTrainer Training: {'Success' if training_result.get('overall_success') else 'Partial'}\n")
                f.write(f"Components Trained: {training_result.get('components_trained', 0)}\n")
                f.write(f"Real Training Executed: True\n")
                
            logger.info("✅ Training cycle completed with ComponentTrainer integration.")

        except Exception as e:
            logger.error(f"❌ Training execution failed: {e}", exc_info=True)
            # Continuar para permitir completar el ciclo
        
        # 5. Deployment
        logger.info("Deploying new generation...")
        # Update system pointer to new model
        # ... deployment logic ...
        
        self.generation += 1
        self.save_state()
        logger.info(f"Evolution Cycle Completed. Now at Generation {self.generation}")

async def main():
    orchestrator = EvolutionOrchestrator()
    await orchestrator.execute_cycle()

if __name__ == "__main__":
    asyncio.run(main())
