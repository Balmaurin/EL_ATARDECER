#!/usr/bin/env python3
"""
🧬 SISTEMA DE EVOLUCIÓN AUTOMÁTICA REAL
Motor de evolución darwiniana y mejora continua para EL-AMANECER
Compatible con MLAutoEvolutionEngine
"""

import asyncio
import json
import logging
import random
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import threading
import numpy as np
import hashlib

logger = logging.getLogger(__name__)

# Importar desde ml_auto_evolution_engine para compatibilidad
try:
    from .ml_auto_evolution_engine import MLAutoEvolutionEngine as BaseEngine
    from .ml_auto_evolution_engine import MLModelGenome
except ImportError:
    # Fallback si no está disponible
    class BaseEngine:
        def __init__(self, config=None):
            self.config = config or {}
            
    class MLModelGenome:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

class AutoEvolutionEngine(BaseEngine):
    """Motor de evolución automática compatible con ML system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.is_active = False
        self.population = {}
        self.fitness_scores = {}
        self.generation_count = 0
        self.mutation_history = []
        self.selection_pressure = 0.7
        
        # Evolution parameters
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        self.crossover_rate = self.config.get('crossover_rate', 0.3)
        self.population_size = self.config.get('population_size', 50)
        self.elite_percentage = self.config.get('elite_percentage', 0.2)
        
        # Database for evolution tracking
        self.db_path = Path("./data/evolution/evolution_db.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Evolution thread
        self.evolution_thread = None
        self._load_population()
        
        logger.info("🧬 AutoEvolutionEngine initialized (REAL)")

    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto del motor de evolución"""
        return {
            'mutation_rate': 0.1,
            'crossover_rate': 0.3,
            'population_size': 50,
            'elite_percentage': 0.2,
            'fitness_threshold': 0.8,
            'max_generations': 1000,
            'adaptation_speed': 0.05,
            'diversity_factor': 0.3,
            'selection_method': 'tournament',
            'epigenetic_inheritance': True
        }

    def _init_database(self):
        """Initialize evolution tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation INTEGER NOT NULL,
                    individual_id TEXT NOT NULL,
                    genome TEXT NOT NULL,
                    fitness_score REAL NOT NULL,
                    parent_ids TEXT,
                    mutation_type TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fitness_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    individual_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    measurement_time TEXT NOT NULL,
                    context TEXT
                )
            """)
            
            conn.commit()

    def _load_population(self):
        """Load existing population or create initial population"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT individual_id, genome, fitness_score 
                    FROM evolution_history 
                    WHERE generation = (SELECT MAX(generation) FROM evolution_history)
                """)
                
                existing_population = cursor.fetchall()
                
                if existing_population:
                    for ind_id, genome, fitness in existing_population:
                        self.population[ind_id] = json.loads(genome)
                        self.fitness_scores[ind_id] = fitness
                    
                    self.generation_count = max(
                        conn.execute("SELECT MAX(generation) FROM evolution_history").fetchone()[0], 0
                    )
                    
                    logger.info(f"🧬 Loaded population: {len(self.population)} individuals, generation {self.generation_count}")
                else:
                    self._create_initial_population()
                    
        except Exception as e:
            logger.error(f"Error loading population: {e}")
            self._create_initial_population()

    def _create_initial_population(self):
        """Create initial population of system configurations"""
        logger.info("🌱 Creating initial population...")
        
        base_genome = self._create_base_genome()
        
        for i in range(self.population_size):
            individual_id = f"gen0_ind{i:03d}"
            genome = self._mutate_genome(base_genome.copy(), mutation_strength=0.3)
            
            self.population[individual_id] = genome
            self.fitness_scores[individual_id] = 0.5  # Neutral starting fitness
            
            # Store in database
            self._store_individual(individual_id, genome, 0.5, [], "initial_creation")
        
        logger.info(f"✅ Initial population created: {len(self.population)} individuals")

    def _create_base_genome(self) -> Dict[str, Any]:
        """Create base genome representing system configuration"""
        return {
            'consciousness': {
                'iit_phi_threshold': 0.7,
                'gwt_capacity': 7,
                'fep_learning_rate': 0.1,
                'smh_weight': 0.8,
                'hebbian_plasticity': 0.6,
                'circumplex_sensitivity': 0.7
            },
            'neural_network': {
                'learning_rate': 0.001,
                'hidden_layers': 3,
                'activation_function': 'relu',
                'dropout_rate': 0.2,
                'batch_normalization': True,
                'attention_heads': 8
            },
            'emotional_system': {
                'emotional_weight': 0.3,
                'mood_persistence': 0.8,
                'empathy_factor': 0.6,
                'emotional_memory': 0.7,
                'circuit_connectivity': 0.9
            },
            'memory_system': {
                'working_memory_capacity': 9,
                'long_term_consolidation': 0.8,
                'episodic_weight': 0.6,
                'semantic_weight': 0.7,
                'forgetting_rate': 0.1
            },
            'decision_making': {
                'exploration_rate': 0.2,
                'risk_tolerance': 0.5,
                'confidence_threshold': 0.8,
                'planning_depth': 5,
                'value_function_weight': 0.7
            }
        }

    def _mutate_genome(self, genome: Dict[str, Any], mutation_strength: float = None) -> Dict[str, Any]:
        """Apply mutations to a genome"""
        if mutation_strength is None:
            mutation_strength = self.mutation_rate
        
        mutated_genome = json.loads(json.dumps(genome))  # Deep copy
        mutations_applied = []
        
        for category, parameters in mutated_genome.items():
            for param_name, value in parameters.items():
                if random.random() < mutation_strength:
                    old_value = value
                    
                    if isinstance(value, float):
                        # Gaussian mutation for float values
                        mutation_delta = np.random.normal(0, 0.1)
                        new_value = max(0.0, min(1.0, value + mutation_delta))
                        mutated_genome[category][param_name] = new_value
                        
                    elif isinstance(value, int):
                        # Discrete mutation for integer values
                        if param_name in ['hidden_layers', 'attention_heads', 'working_memory_capacity', 'planning_depth']:
                            delta = random.choice([-1, 0, 1])
                            new_value = max(1, value + delta)
                            mutated_genome[category][param_name] = new_value
                        else:
                            new_value = value
                            
                    elif isinstance(value, bool):
                        # Boolean flip mutation
                        new_value = not value
                        mutated_genome[category][param_name] = new_value
                        
                    elif isinstance(value, str):
                        # Categorical mutation
                        if param_name == 'activation_function':
                            options = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu']
                            new_value = random.choice([opt for opt in options if opt != value])
                            mutated_genome[category][param_name] = new_value
                        else:
                            new_value = value
                    else:
                        new_value = value
                    
                    if old_value != new_value:
                        mutations_applied.append({
                            'category': category,
                            'parameter': param_name,
                            'old_value': old_value,
                            'new_value': new_value
                        })
        
        return mutated_genome

    async def evolve_system_component(self, component: str, evolution_type: str = "mutation") -> Dict[str, Any]:
        """Evolve a specific system component"""
        try:
            logger.info(f"🎯 Evolving component: {component} ({evolution_type})")
            
            # Get best current individual
            if not self.fitness_scores:
                return {"error": "No population available for evolution"}
            
            best_individual_id = max(self.fitness_scores.items(), key=lambda x: x[1])[0]
            best_genome = self.population[best_individual_id].copy()
            
            # Apply targeted evolution
            if evolution_type == "mutation":
                if component in best_genome:
                    # Focused mutation on specific component
                    for param in best_genome[component]:
                        if random.random() < 0.5:  # Higher mutation rate for targeted evolution
                            old_value = best_genome[component][param]
                            if isinstance(old_value, float):
                                mutation_delta = np.random.normal(0, 0.2)
                                best_genome[component][param] = max(0.0, min(1.0, old_value + mutation_delta))
                else:
                    return {"error": f"Component {component} not found in genome"}
            
            elif evolution_type == "optimization":
                # Gradient-like optimization
                if component in best_genome:
                    for param in best_genome[component]:
                        if isinstance(best_genome[component][param], float):
                            # Small incremental improvement
                            current_value = best_genome[component][param]
                            improvement = random.uniform(-0.1, 0.1)
                            best_genome[component][param] = max(0.0, min(1.0, current_value + improvement))
            
            # Evaluate new variant
            new_individual_id = f"evolved_{component}_{int(time.time())}"
            new_fitness = await self.evaluate_fitness(new_individual_id, best_genome)
            
            # Store if improvement
            original_fitness = self.fitness_scores[best_individual_id]
            if new_fitness > original_fitness:
                self.population[new_individual_id] = best_genome
                self.fitness_scores[new_individual_id] = new_fitness
                self._store_individual(
                    new_individual_id, 
                    best_genome, 
                    new_fitness, 
                    [best_individual_id], 
                    f"targeted_{evolution_type}"
                )
                
                result = {
                    "status": "success",
                    "improvement": new_fitness - original_fitness,
                    "original_fitness": original_fitness,
                    "new_fitness": new_fitness,
                    "evolved_individual": new_individual_id
                }
            else:
                result = {
                    "status": "no_improvement", 
                    "original_fitness": original_fitness,
                    "attempted_fitness": new_fitness,
                    "difference": new_fitness - original_fitness
                }
            
            logger.info(f"🎯 Evolution result for {component}: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Error evolving component {component}: {e}")
            return {"error": str(e)}

    async def evaluate_fitness(self, individual_id: str, genome: Dict[str, Any]) -> float:
        """Evaluate fitness of an individual genome"""
        try:
            # Multi-objective fitness evaluation
            fitness_components = []
            
            # Performance metrics
            performance_score = await self._evaluate_performance(genome)
            fitness_components.append(('performance', performance_score, 0.4))
            
            # Stability metrics
            stability_score = await self._evaluate_stability(genome)
            fitness_components.append(('stability', stability_score, 0.2))
            
            # Adaptability metrics
            adaptability_score = await self._evaluate_adaptability(genome)
            fitness_components.append(('adaptability', adaptability_score, 0.2))
            
            # Efficiency metrics
            efficiency_score = await self._evaluate_efficiency(genome)
            fitness_components.append(('efficiency', efficiency_score, 0.2))
            
            # Calculate weighted fitness score
            total_fitness = sum(score * weight for _, score, weight in fitness_components)
            
            logger.debug(f"🎯 Fitness evaluation for {individual_id}: {total_fitness:.3f}")
            return total_fitness
            
        except Exception as e:
            logger.error(f"Error evaluating fitness for {individual_id}: {e}")
            return 0.0

    async def _evaluate_performance(self, genome: Dict[str, Any]) -> float:
        """Evaluate performance component of fitness"""
        try:
            # Simulate performance based on genome parameters
            consciousness_params = genome.get('consciousness', {})
            neural_params = genome.get('neural_network', {})
            
            # IIT performance (higher phi threshold generally better but diminishing returns)
            phi_score = consciousness_params.get('iit_phi_threshold', 0.5)
            phi_performance = phi_score * (2 - phi_score)  # Peak at phi=1.0
            
            # Neural network efficiency
            lr = neural_params.get('learning_rate', 0.001)
            layers = neural_params.get('hidden_layers', 3)
            dropout = neural_params.get('dropout_rate', 0.2)
            
            # Optimal learning rate around 0.001-0.01
            lr_performance = 1.0 - abs(np.log10(lr) + 3) / 2  # Peak at 0.001
            layer_performance = max(0, 1.0 - abs(layers - 4) / 4)  # Optimal at 4 layers
            dropout_performance = 1.0 - abs(dropout - 0.3) / 0.3  # Optimal at 0.3
            
            neural_performance = (lr_performance + layer_performance + dropout_performance) / 3
            
            return (phi_performance + neural_performance) / 2
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return 0.5

    async def _evaluate_stability(self, genome: Dict[str, Any]) -> float:
        """Evaluate stability component of fitness"""
        try:
            # Check for parameter balance and stability indicators
            memory_params = genome.get('memory_system', {})
            decision_params = genome.get('decision_making', {})
            
            # Memory stability
            consolidation = memory_params.get('long_term_consolidation', 0.8)
            forgetting = memory_params.get('forgetting_rate', 0.1)
            memory_balance = 1.0 - abs((consolidation - forgetting) - 0.7)  # Good balance
            
            # Decision stability
            exploration = decision_params.get('exploration_rate', 0.2)
            confidence = decision_params.get('confidence_threshold', 0.8)
            decision_balance = 1.0 - abs((confidence - exploration) - 0.6)  # Balance exploration/exploitation
            
            return (memory_balance + decision_balance) / 2
            
        except Exception as e:
            logger.error(f"Error evaluating stability: {e}")
            return 0.5

    async def _evaluate_adaptability(self, genome: Dict[str, Any]) -> float:
        """Evaluate adaptability component of fitness"""
        try:
            neural_params = genome.get('neural_network', {})
            emotional_params = genome.get('emotional_system', {})
            
            # Neural adaptability
            attention_heads = neural_params.get('attention_heads', 8)
            batch_norm = neural_params.get('batch_normalization', True)
            
            attention_score = min(1.0, attention_heads / 8)  # More attention heads = more adaptable
            batch_norm_score = 1.0 if batch_norm else 0.7
            
            neural_adaptability = (attention_score + batch_norm_score) / 2
            
            # Emotional adaptability
            empathy = emotional_params.get('empathy_factor', 0.6)
            connectivity = emotional_params.get('circuit_connectivity', 0.9)
            
            emotional_adaptability = (empathy + connectivity) / 2
            
            return (neural_adaptability + emotional_adaptability) / 2
            
        except Exception as e:
            logger.error(f"Error evaluating adaptability: {e}")
            return 0.5

    async def _evaluate_efficiency(self, genome: Dict[str, Any]) -> float:
        """Evaluate efficiency component of fitness"""
        try:
            neural_params = genome.get('neural_network', {})
            memory_params = genome.get('memory_system', {})
            
            # Neural efficiency
            layers = neural_params.get('hidden_layers', 3)
            dropout = neural_params.get('dropout_rate', 0.2)
            
            # Fewer layers = more efficient, but need minimum complexity
            layer_efficiency = max(0.5, 1.0 - (layers - 2) / 5)
            dropout_efficiency = 1.0 - dropout  # Lower dropout = more efficiency
            
            neural_efficiency = (layer_efficiency + dropout_efficiency) / 2
            
            # Memory efficiency
            capacity = memory_params.get('working_memory_capacity', 9)
            forgetting = memory_params.get('forgetting_rate', 0.1)
            
            # Optimal working memory around 7±2
            capacity_efficiency = 1.0 - abs(capacity - 7) / 7
            forgetting_efficiency = forgetting  # Higher forgetting = more efficient (less storage)
            
            memory_efficiency = (capacity_efficiency + forgetting_efficiency) / 2
            
            return (neural_efficiency + memory_efficiency) / 2
            
        except Exception as e:
            logger.error(f"Error evaluating efficiency: {e}")
            return 0.5

    def _store_individual(self, individual_id: str, genome: Dict, fitness: float, 
                         parent_ids: List[str], mutation_type: str):
        """Store individual in evolution history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO evolution_history 
                    (generation, individual_id, genome, fitness_score, parent_ids, mutation_type, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.generation_count,
                    individual_id,
                    json.dumps(genome),
                    fitness,
                    json.dumps(parent_ids),
                    mutation_type,
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing individual: {e}")

    async def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status"""
        try:
            fitness_values = [f for f in self.fitness_scores.values() if f > 0]
            
            return {
                'generation': self.generation_count,
                'population_size': len(self.population),
                'is_active': self.is_active,
                'fitness_stats': {
                    'avg': np.mean(fitness_values) if fitness_values else 0,
                    'max': max(fitness_values) if fitness_values else 0,
                    'min': min(fitness_values) if fitness_values else 0,
                    'std': np.std(fitness_values) if fitness_values else 0
                },
                'config': self.config,
                'best_individual': max(self.fitness_scores.items(), key=lambda x: x[1])[0] if self.fitness_scores else None
            }
        except Exception as e:
            logger.error(f"Error getting evolution status: {e}")
            return {'error': str(e)}

# Alias para compatibilidad
MLAutoEvolutionEngine = AutoEvolutionEngine

if __name__ == "__main__":
    # Test the evolution engine
    async def test_evolution():
        engine = AutoEvolutionEngine()
        
        # Test component evolution
        result = await engine.evolve_system_component("consciousness", "mutation")
        print("Evolution result:", result)
        
        # Get status
        status = await engine.get_evolution_status()
        print("Evolution status:", status)
    
    asyncio.run(test_evolution())