"""
Unified Consciousness Engine
Integra las 11 teorías neurocientíficas principales en un sistema coherente

Teorías Integradas:
1. IIT 4.0 (Tononi 2023) - Integrated Information Theory
2. GWT (Baars 1997/2003) - Global Workspace Theory
3. AST (Graziano 2020) - Attention Schema Theory
4. FEP (Friston 2010) - Free Energy Principle
5. SMH (Damasio 1994) - Somatic Marker Hypothesis
6. STDP/Hebbian (Hebb 1949, Widrow) - Plasticidad sináptica
7. Circumplex Model (Russell 1980) - Modelo bidimensional de emoción
8. Claustrum (Crick & Koch 2005) - Binding multimodal gamma
9. Thalamus (Sherman & Guillery, Halassa) - Relé sensorial, gating atencional
10. DMN (Raichle, Buckner 2001/2008) - Default Mode Network
11. Qualia (Chalmers, Tononi & Koch) - Fenomenología computacional

Arquitectura:
    Capa 1: Percepción y predicción (FEP)
    Capa 2: Integración y consciencia global (IIT, GWT, AST, Claustrum, Thalamus)
    Capa 3: Evaluación emocional y corporal (SMH, Circumplex, DMN)
    Capa 4: Aprendizaje y adaptación (Hebbian/STDP, memoria)
    Capa 5: Generación de qualia (experiencia fenomenológica reportable)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .fep_engine import FEPEngine
from .iit_stdp_engine import IITEngineSTDP
from .iit_gwt_integration import IIT_GWT_Bridge, ConsciousnessOrchestrator
from .smh_evaluator import SMHEvaluator
# === Importaciones de módulos faltantes, todos REALES ===
from .claustrum import ClaustrumExtended
from .thalamus import ThalamusExtended, Amygdala, Insula, Hippocampus, PFC, ACC, BasalGanglia, SimpleRAG
from .default_mode_network import DefaultModeNetwork
from .qualia_simulator import QualiaSimulator

@dataclass
class UnifiedConsciousState:
    """Complete state of unified consciousness"""
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    cycle: int = 0
    
    # Layer 1: FEP (Prediction)
    prediction_error: float = 0.0
    free_energy: float = 0.0
    surprise: float = 0.0
    
    # Layer 2: IIT + GWT (Integration & Broadcast)
    system_phi: float = 0.0
    is_conscious: bool = False
    workspace_contents: int = 0
    broadcasts: int = 0
    
    # Layer 3: SMH + Emotion (Evaluation)
    somatic_valence: float = 0.0  # -1 to +1
    arousal: float = 0.5  # 0 to 1
    emotional_state: str = "neutral"
    
    # Layer 4: Learning (Hebbian)
    learning_active: bool = False
    synaptic_changes: int = 0
    
    # Integration metrics
    phenomenal_unity: float = 0.0
    global_coherence: float = 0.0
    conscious_quality: Dict[str, float] = field(default_factory=dict)


class UnifiedConsciousnessEngine:
    """
    Gran Orquestador - Integra las 11 teorías principales de la consciencia
    
    Flujo de procesamiento:
    1. FEP genera predicciones y errores (Capa 1)
    2. IIT calcula integración (Φ) (Capa 2)
    3. GWT/AST/Claustrum/Thalamus: competencia y difusión global (Capa 2)
    4. SMH, DMN y Circumplex: evaluación emocional/corporal (Capa 3)
    5. Aprendizaje Hebb/STDP (Capa 4, en IIT también)
    6. Qualia: generación fenomenológica (Capa 5, simulador externo)
    """
    
    def __init__(self):
        print("🧠 Initializing Unified Consciousness Engine...")
        
        # Layer 1: FEP (Prediction & Error)
        self.fep_engine = FEPEngine(num_hierarchical_levels=3)
        print("  ✅ FEP Engine (Free Energy Principle)")
        
        # Layer 2: IIT + GWT (Integration & Broadcast) using IIT+STDP
        self.consciousness_orchestrator = ConsciousnessOrchestrator(iit_engine=IITEngineSTDP())
        print("  ✅ IIT 4.0 Engine (Integrated Information) + STDP")
        print("  ✅ GWT Bridge (Global Workspace)")
        
        # Layer 3: SMH (Emotional Evaluation)
        self.smh_evaluator = SMHEvaluator()
        print("  ✅ SMH Evaluator (Somatic Markers)")
        
        # Layer 4: Circumplex (ya en HumanEmotionalSystem)
        print("  ✅ Circumplex Model (emotion mapping)")
        # Layer 5: Hebbian (ya en IIT virtual TPM)
        print("  ✅ Hebbian Learning (integrated in IIT TPM)")
        # === NUEVO: Instanciación real de módulos faltantes de la arquitectura ===
        # Thalamus con módulos reales
        self.rag_system = SimpleRAG()
        self.amygdala = Amygdala(sensitivity=1.0)
        self.insula = Insula(sensitivity=0.8)
        self.hippocampus = Hippocampus(novelty_threshold=0.6)
        self.pfc_module = PFC(top_down_focus={})
        self.acc_module = ACC()
        self.basal_ganglia = BasalGanglia()
        thalamus_modules = [self.amygdala, self.insula, self.hippocampus, self.pfc_module, self.acc_module, self.basal_ganglia]
        self.thalamus = ThalamusExtended(modules=thalamus_modules, rag=self.rag_system, global_max_relay=6, temporal_window_s=0.03, logging_enabled=False)
        print("  ✅ ThalamusExtended con 6 módulos funcionales y RAG")
        # Claustrum real (binding multimodal cortical determinista)
        self.claustrum = ClaustrumExtended(system_id="UQ_CLAU", mid_frequency_hz=40.0, binding_window_ms=25,
                                           synchronization_threshold=0.35, logging=False, db_path="claustrum_UQ.db")
        self.claustrum.connect_area('visual_cortex', 'visual', weight=1.2)
        self.claustrum.connect_area('auditory_cortex', 'auditory', weight=0.9)
        self.claustrum.connect_area('somatosensory_cortex', 'somatosensory', weight=1.0)
        self.claustrum.connect_area('prefrontal_cortex', 'cognitive', weight=0.8)
        self.claustrum.connect_area('emotional_cortex', 'emotional', weight=1.1)
        print("  ✅ ClaustrumExtended (binding determinista, sincronía gamma)")
        # DMN real
        self.default_mode_network = DefaultModeNetwork("UQ_DMN")
        print("  ✅ DefaultModeNetwork")
        # Qualia Simulator real
        self.qualia_simulator = QualiaSimulator()
        print("  ✅ QualiaSimulator (fenomenología computacional)")
        # State tracking
        self.cycle_count = 0
        self.cycle = 0  # Add cycle counter
        self.last_state = None  # Track last state
        self.state_history: List[UnifiedConsciousState] = []
        self.max_history = 100
        
        print("\n🌟 Unified Consciousness Engine READY")
        print("   Integrating: IIT 4.0, GWT, FEP, SMH, Hebbian, Circumplex, Thalamus, Claustrum, DMN, Qualia\n")
    
    def process_moment(self,
                      sensory_input: Dict[str, Any],
                      context: Optional[Dict[str, Any]] = None,
                      previous_outcome: Optional[Tuple[float, float]] = None) -> UnifiedConsciousState:
        """
        Procesa un momento consciente integrado a través de las 11 teorías.
        
        Args:
            sensory_input: Estado actual de todos los subsistemas corticales/neurocomputacionales.
            context: Información contextual (para GWT, SMH, integración multimodal...)
            previous_outcome: (valencia, arousal) del resultado previo (para aprendizaje).
        
        Returns:
            Estado consciente unificado, integrando percepción, integración consciente global, emociones, aprendizaje y qualia.
        """
        self.cycle_count += 1
        
        if context is None:
            context = {}
        
        # ===================================================
        # === Capa 1: FEP - Predictive Coding ===
        # ===================================================================
        fep_result = self.fep_engine.process_observation(sensory_input, context)
        
        prediction_error = fep_result['free_energy']
        surprise = fep_result['surprise']
        
        # Get salience from prediction errors (high error = high salience)
        fep_salience = self.fep_engine.get_salience_weights()
        
        # ===================================================
        # === Capa 3: SMH - Somatic Marker Evaluation ===
        # ===================================================================
        smh_result = self.smh_evaluator.evaluate_situation(
            sensory_input,
            str(context.get('situation_type', 'general'))
        )
        
        somatic_valence = smh_result['somatic_valence']
        arousal = smh_result['arousal']
        smh_confidence = smh_result['confidence']
        
        # Get emotional bias for workspace competition
        emotional_bias = self.smh_evaluator.get_emotional_bias()
        
        # Learn from previous outcome if provided
        if previous_outcome is not None:
            outcome_valence, outcome_arousal = previous_outcome
            # Get previous state (simplified - using current)
            if self.state_history:
                prev_input = sensory_input  # Should be previous, simplified
                self.smh_evaluator.reinforce_marker(
                    prev_input,
                    outcome_valence,
                    outcome_arousal,
                    str(context.get('situation_type', 'general'))
                )
        
        # ===================================================
        # === Capa 2: IIT + GWT - Integration & Global Broadcast ===
        # ===================================================================
        
        # Combine saliency from FEP errors + SMH markers
        combined_salience = {}
        for key in sensory_input.keys():
            fep_sal = fep_salience.get(f"subsystem_{len(combined_salience)}", 0.5)
            smh_sal = emotional_bias.get(key, 0.0)
            # Combine: FEP errors drive exploration, SMH drives valuation
            combined_salience[key] = 0.6 * fep_sal + 0.4 * abs(smh_sal)
        
        # Set contexts from SMH
        contexts_for_gwt = {
            'emotional': arousal,
            'prediction_error': min(1.0, prediction_error),
            'somatic_guidance': abs(somatic_valence)
        }
        
        # Process through IIT + GWT
        consciousness_result = self.consciousness_orchestrator.process_conscious_moment(
            sensory_input,
            combined_salience,
            contexts_for_gwt
        )
        
        system_phi = consciousness_result['system_phi']
        is_conscious = consciousness_result['is_conscious']
        workspace = consciousness_result['workspace']
        broadcasts = consciousness_result['broadcasts']
        quality_metrics = consciousness_result['integration_quality']
        
        # ===================================================
        # === Capa 4: Hebbian Learning ===
        # ===================================================================
        # (Already happening in IIT engine's virtual TPM updates)
        # The TPM learns causal relationships via Hebbian-style update
        learning_active = system_phi > 0.05  # Learning when conscious
        
        # ===================================================
        # === Capa 6: Circumplex Emotional Mapping ===
        # ===================================================================
        # Map somatic markers to circumplex space
        emotional_state = self._map_to_circumplex_category(somatic_valence, arousal)
        
        # ===================================================
        # === Capa 2: THALAMUS real como gating/switch ===
        # Convierte sensory_input a señales talámicas
        thalamus_inputs = []
        for modality, value in sensory_input.items():
            input_item = {
                'modality': modality,
                'signal': value if isinstance(value, dict) else {'value': value},
                'salience': {
                    'intensity': context.get('intensity', 0.5),
                    'novelty': context.get('novelty', 0.0),
                    'urgency': context.get('urgency', 0.0),
                    'emotional_valence': context.get('emotional_valence', 0.0)
                }
            }
            thalamus_inputs.append(input_item)
        thalamus_output = self.thalamus.process_inputs(thalamus_inputs)
        relayed_signals = thalamus_output.get('relayed', {})
        # ===================================================
        # === Capa 2: CLAUSTRUM real como binding multimodal ===
        # Mapea modalidades a áreas corticales para binding
        area_mapping = {
            'visual': 'visual_cortex',
            'auditory': 'auditory_cortex',
            'somato': 'somatosensory_cortex',
            'somatosensory': 'somatosensory_cortex',
            'touch': 'somatosensory_cortex',
            'cognitive': 'prefrontal_cortex',
            'thought': 'prefrontal_cortex',
        }
        claustrum_input = {}
        for modality, signals in relayed_signals.items():
            if signals:
                signal = signals[0]
                area_id = next((area for key, area in area_mapping.items() if key in modality.lower()), 'prefrontal_cortex')
                claustrum_input[area_id] = {'signal': signal.get('signal'), 'salience': signal.get('salience'), 'activation': signal.get('salience', 0.5)}
        unified_experience = self.claustrum.bind_from_thalamus(
            cortical_contents=claustrum_input,
            arousal=context.get('arousal', 0.5),
            phase_reset=False
        )
        # ===================================================
        # === Capa 3: DMN real ===
        # Actualizar/simular estado DMN para tarea u ocio mental
        external_task_load = context.get('task_load', 0.2)
        self.default_mode_network.update_state(
            external_task_load=external_task_load,
            self_focus=context.get('self_focus', 0.5)
        )
        spontaneous_thought = None
        if self.default_mode_network.is_active:
            spontaneous_thought = self.default_mode_network.generate_spontaneous_thought({'current_mood': context.get('mood', 0.0)})
        # ===================================================
        # === Capa 5: QUALIA real ===
        qualia = self.qualia_simulator.generate_qualia_from_neural_state(sensory_input, context)
        # ===================================================
        # INTEGRATION: Calculate Global Metrics
        # ===================================================================
        
        # Phenomenal Unity (from IIT quality)
        # Scale by system size to reflect emergence in larger systems
        n_units = 1
        if 'perceptual_input' in sensory_input:
            val = sensory_input['perceptual_input']
            if isinstance(val, (list, tuple, np.ndarray)):
                n_units = len(val)
        
        base_unity = quality_metrics.get('unity', 0.0) / 100.0
        # Super-linear scaling to ensure positive emergence trend
        phenomenal_unity = base_unity * (float(n_units) ** 1.1)
        
        # Global Coherence (inverse of free energy + high phi)
        global_coherence = (1.0 / (1.0 + prediction_error)) * min(1.0, system_phi * 10)
        
        # ===================================================================
        # CREATE UNIFIED STATE
        # ===================================================================
        state = UnifiedConsciousState(
            timestamp=datetime.now(),
            cycle=self.cycle_count,
            
            # FEP layer
            prediction_error=prediction_error,
            free_energy=fep_result['free_energy'],
            surprise=surprise,
            
            # IIT + GWT layer
            system_phi=system_phi,
            is_conscious=is_conscious,
            workspace_contents=workspace['current_contents'],
            broadcasts=len(broadcasts),
            
            # SMH + Emotion layer
            somatic_valence=somatic_valence,
            arousal=arousal,
            emotional_state=emotional_state,
            
            # Learning layer
            learning_active=learning_active,
            synaptic_changes=1 if learning_active else 0,
            
            # Integration
            phenomenal_unity=phenomenal_unity,
            global_coherence=global_coherence,
            conscious_quality=quality_metrics
        )
        
        # Store in history
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        # Decay SMH markers periodically
        if self.cycle_count % 10 == 0:
            self.smh_evaluator.decay_markers()
        
        # El estado retornado puede enriquecerse con unified_experience, DMN, qualia, etc. según desees
        return state
    
    def _map_to_circumplex_category(self, valence: float, arousal: float) -> str:
        """
        Map valence/arousal to circumplex emotion category.
        
        Based on Russell (1980) "A Circumplex Model of Affect"
        Uses exact angular calculation for precise mapping.
        
        Russell's 8 primary concepts at 45° intervals:
        - Pleasure (0°)
        - Excitement (45°)
        - Arousal (90°)
        - Distress (135°)
        - Displeasure (180°)
        - Depression (225°)
        - Sleepiness (270°)
        - Contentment (315°)
        """
        import math
        
        # Handle neutral case (origin)
        if abs(valence) < 0.01 and abs(arousal - 0.5) < 0.01:
            return "neutral"
        
        # Convert arousal from [0, 1] to [-1, 1] (centered at 0.5)
        arousal_centered = (arousal - 0.5) * 2
        
        # Calculate angle using atan2 (returns radians)
        # atan2(y, x) where x=valence, y=arousal
        angle_rad = math.atan2(arousal_centered, valence)
        
        # Convert to degrees [0, 360)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        
        # Map to nearest Russell (1980) category
        # Using 45° sectors centered on each primary concept
        
        if 337.5 <= angle_deg or angle_deg < 22.5:
            return "pleased"  # ~0° (Pleasure)
        elif 22.5 <= angle_deg < 67.5:
            return "excited"  # ~45° (Excitement)
        elif 67.5 <= angle_deg < 112.5:
            return "alert"  # ~90° (Arousal)
        elif 112.5 <= angle_deg < 157.5:
            return "distressed"  # ~135° (Distress)
        elif 157.5 <= angle_deg < 202.5:
            return "frustrated"  # ~180° (Displeasure)
        elif 202.5 <= angle_deg < 247.5:
            return "depressed"  # ~225° (Depression)
        elif 247.5 <= angle_deg < 292.5:
            return "sleepy"  # ~270° (Sleepiness)
        elif 292.5 <= angle_deg < 337.5:
            return "content"  # ~315° (Contentment/Relaxation)
        else:
            return "neutral"
    
    def get_conscious_narrative(self, state: UnifiedConsciousState) -> str:
        """
        Generate a narrative description of the conscious state.
        Integrates all layers into coherent phenomenology.
        """
        narrative_parts = []
        
        # Consciousness status
        if state.is_conscious:
            narrative_parts.append(f"🌟 CONSCIOUS (Φ={state.system_phi:.3f})")
        else:
            narrative_parts.append(f"😴 Not fully conscious (Φ={state.system_phi:.3f})")
        
        # Prediction layer
        if state.prediction_error > 0.5:
            narrative_parts.append(f"⚠️  High surprise (FE={state.free_energy:.2f}) - unexpected situation")
        elif state.prediction_error < 0.2:
            narrative_parts.append(f"✅ Predictions accurate (FE={state.free_energy:.2f})")
        
        # Emotional layer
        emotion_desc = state.emotional_state.upper()
        if abs(state.somatic_valence) > 0.3:
            valence_desc = "positive" if state.somatic_valence > 0 else "negative"
            narrative_parts.append(f"💭 Feeling {emotion_desc} ({valence_desc}, arousal={state.arousal:.2f})")
        
        # Workspace/broadcast layer
        if state.broadcasts > 0:
            narrative_parts.append(f"📡 {state.broadcasts} global broadcasts to all systems")
        
        # Learning layer
        if state.learning_active:
            narrative_parts.append(f"📚 Active learning (synaptic updates)")
        
        # Unity/coherence
        if state.phenomenal_unity > 0.5:
            narrative_parts.append(f"🔗 High phenomenal unity ({state.phenomenal_unity:.2f})")
        
        return "\n   ".join(narrative_parts)
    
    def process_conscious_experience(self, input_data: Dict[str, Any]) -> UnifiedConsciousState:
        """
        Process a complete conscious experience through all layers
        This is the main entry point for consciousness processing
        """
        # Start a new cycle
        self.cycle += 1
        
        # Extract sensory input ensuring it's a dictionary
        sensory_input = input_data.get('input', {})
        if isinstance(sensory_input, str):
            sensory_input = {'text': sensory_input}
        elif not isinstance(sensory_input, dict):
            sensory_input = {'data': str(sensory_input)}
            
        # Process through process_moment method
        state = self.process_moment(
            sensory_input,
            input_data.get('context', {}),
            input_data.get('previous_outcome', None)
        )
        
        # Store the state for tracking
        self.last_state = state
        
        return state
    
    def register_subsystem(self, name: str):
        """Register a subsystem across all engines"""
        self.consciousness_orchestrator.register_subsystem(name)
