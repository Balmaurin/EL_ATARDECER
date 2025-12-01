# -*- coding: utf-8 -*-
"""
VENTROMEDIAL PFC - ENTERPRISE (EXTENDED)
=======================================

Integración emocional-racional real con persistencia, aprendizaje counterfactual,
decisión basada en utilidad, regulación emocional y hooks para integración en Sheily-AI.

Autor: Adaptado y extendido para Sheily-AI / Kimi
Fecha: 2025-11-25
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
import math
import json
import sqlite3
import os
import logging
import numpy as np

logger = logging.getLogger("vmPFC")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


# --------------------------
# Somatic Marker (Bayesian-ish)
# --------------------------
@dataclass
class SomaticMarker:
    situation_id: str
    emotional_valence: float  # -1 .. +1
    arousal: float            # 0 .. 1
    confidence: float         # 0 .. 1
    experiences: int = 0
    last_update_ts: float = field(default_factory=lambda: float(datetime.utcnow().timestamp()))

    def strengthen(self, observed_valence: float, observed_arousal: Optional[float] = None, weight: float = 1.0):
        """
        Actualiza marcador usando una mezcla ponderada que aumenta confidence.
        Implementa un comportamiento similar a un update bayesiano simple.
        """
        # Clamp inputs
        observed_valence = float(np.clip(observed_valence, -1.0, 1.0))
        if observed_arousal is None:
            observed_arousal = abs(observed_valence)
        observed_arousal = float(np.clip(observed_arousal, 0.0, 1.0))

        # Learning rate depende de confianza: baja confianza → learning rate alto
        lr = float(min(0.9, 0.6 * (1.0 - self.confidence) + 0.1 * weight))
        # Update valence with weighted average
        self.emotional_valence = float(np.clip((1 - lr) * self.emotional_valence + lr * observed_valence, -1.0, 1.0))
        # Update arousal similarly
        self.arousal = float(np.clip((1 - lr) * self.arousal + lr * observed_arousal, 0.0, 1.0))
        # Increase experiences and update confidence (diminishing returns)
        self.experiences += 1
        self.confidence = float(min(1.0, self.confidence + 0.2 * (1 - self.confidence)))
        self.last_update_ts = float(datetime.utcnow().timestamp())

    def weaken(self, decay: float = 0.05):
        """Decay a largo plazo de marcador"""
        self.emotional_valence *= (1 - decay)
        self.arousal *= (1 - decay)
        self.confidence = max(0.0, self.confidence - decay)
        self.last_update_ts = float(datetime.utcnow().timestamp())


# --------------------------
# Simple persistence for markers
# --------------------------
class VMDatabase:
    def __init__(self, path: str = "vmPFC_markers.db"):
        self.path = path
        self._init_db_if_needed()

    def _init_db_if_needed(self):
        init_needed = not os.path.exists(self.path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        cur = self.conn.cursor()
        if init_needed:
            cur.execute("""
            CREATE TABLE markers (
                situation_id TEXT PRIMARY KEY,
                emotional_valence REAL,
                arousal REAL,
                confidence REAL,
                experiences INTEGER,
                last_update_ts REAL,
                payload TEXT
            )
            """)
            cur.execute("""
            CREATE TABLE decisions (
                id TEXT PRIMARY KEY,
                ts REAL,
                choice TEXT,
                expected_value REAL,
                integrated_value REAL,
                metadata TEXT
            )
            """)
            self.conn.commit()

    def save_marker(self, marker: SomaticMarker, payload: Optional[Dict[str, Any]] = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO markers (situation_id, emotional_valence, arousal, confidence, experiences, last_update_ts, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (marker.situation_id, float(marker.emotional_valence), float(marker.arousal), float(marker.confidence),
             int(marker.experiences), float(marker.last_update_ts), json.dumps(payload or {}))
        )
        self.conn.commit()

    def load_marker(self, situation_id: str) -> Optional[SomaticMarker]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM markers WHERE situation_id = ?", (situation_id,))
        row = cur.fetchone()
        if not row:
            return None
        return SomaticMarker(
            situation_id=row[0],
            emotional_valence=float(row[1]),
            arousal=float(row[2]),
            confidence=float(row[3]),
            experiences=int(row[4]),
            last_update_ts=float(row[5])
        )

    def save_decision(self, choice_id: str, choice_repr: str, expected_value: float, integrated_value: float, metadata: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO decisions (id, ts, choice, expected_value, integrated_value, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                    (choice_id, float(datetime.utcnow().timestamp()), choice_repr, float(expected_value), float(integrated_value), json.dumps(metadata)))
        self.conn.commit()

    def recent_decisions(self, limit: int = 10):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM decisions ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]


# --------------------------
# Real RAG using RealSemanticSearch
# --------------------------
class RealRAGWrapper:
    """
    Wrapper para RealSemanticSearch que mantiene la interfaz de SimpleRAG
    Usa búsqueda semántica real con embeddings y FAISS
    """
    def __init__(self):
        self._search = None
        self._initialized = False
        self._default_docs = [
            {"id": "ctx1", "text": "En situaciones pasadas, permanecer calmado funcionó."},
            {"id": "ctx2", "text": "Evitar riesgo es mejor cuando la confianza es baja."}
        ]
    
    def _initialize_real_search(self):
        """Inicializar RealSemanticSearch de forma lazy"""
        if self._initialized:
            return
        
        try:
            import sys
            from pathlib import Path
            
            # Agregar path para importar RealSemanticSearch
            root = Path(__file__).resolve().parents[6]
            sys.path.insert(0, str(root / "packages" / "sheily_core" / "src"))
            
            from sheily_core.search.real_semantic_search import get_real_semantic_search
            
            self._search = get_real_semantic_search()
            
            # Agregar documentos por defecto
            default_texts = [doc["text"] for doc in self._default_docs]
            self._search.add_documents(default_texts, metadata=self._default_docs)
            
            self._initialized = True
            logger.info("✅ RealSemanticSearch inicializado para vmPFC")
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo inicializar RealSemanticSearch: {e}. Usando búsqueda básica.")
            self._search = None
            self._initialized = True  # Marcar como inicializado para no reintentar
    
    def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Recupera documentos relevantes usando búsqueda semántica real.
        Si RealSemanticSearch no está disponible, usa búsqueda básica como fallback.
        """
        self._initialize_real_search()
        
        if self._search is not None:
            try:
                # Usar búsqueda semántica real
                results = self._search.search(query, k=top_k)
                
                # Convertir formato de RealSemanticSearch a formato esperado
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "id": result.get("metadata", {}).get("id", f"doc_{result.get('index', 0)}"),
                        "text": result.get("document", ""),
                        "similarity_score": result.get("score", 0.0)
                    })
                
                return formatted_results
                
            except Exception as e:
                logger.warning(f"Error en búsqueda semántica real: {e}. Usando fallback básico.")
        
        # Fallback básico si RealSemanticSearch no está disponible
        if not query:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_docs = []
        for doc in self._default_docs:
            doc_text_lower = doc["text"].lower()
            doc_words = set(doc_text_lower.split())
            
            common_words = query_words.intersection(doc_words)
            word_similarity = len(common_words) / max(len(query_words), 1) if query_words else 0
            
            if query_lower in doc_text_lower:
                word_similarity += 0.3
            
            scored_docs.append({
                **doc,
                'similarity_score': word_similarity
            })
        
        scored_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_docs[:top_k] if scored_docs else []

# Alias para compatibilidad
SimpleRAG = RealRAGWrapper


# --------------------------
# vmPFC enterprise class
# --------------------------
class VentromedialPFC:
    def __init__(self,
                 system_id: str,
                 persist: bool = True,
                 db_path: str = "vmPFC_markers.db",
                 rag: Optional[SimpleRAG] = None,
                 stochastic: bool = False):
        self.system_id = system_id
        self.created_at = datetime.utcnow().isoformat() + "Z"
        self.somatic_markers: Dict[str, SomaticMarker] = {}
        self.regulation_active = False
        self.regulation_strength = 0.5  # 0..1
        self.decisions_with_emotion = 0
        self.decisions_purely_rational = 0
        self.regulation_events = 0
        self.rag = rag
        self.stochastic = bool(stochastic)
        # persistence
        self._db = VMDatabase(db_path) if persist else None
        if self._db:
            # load markers present in DB into memory (optional - lazy load could be used)
            self._load_markers_from_db()
        # hooks
        self.on_decision: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_marker_update: Optional[Callable[[SomaticMarker, Dict[str, Any]], None]] = None
        # metrics
        self.choice_history: List[Dict[str, Any]] = []

        logger.info("vmPFC initialized id=%s persist=%s stochastic=%s", system_id, bool(self._db), self.stochastic)

    # ------------------
    # Persistence helpers
    # ------------------
    def _load_markers_from_db(self):
        # naive: query all markers
        try:
            cur = self._db.conn.cursor()
            cur.execute("SELECT situation_id FROM markers")
            rows = cur.fetchall()
            for r in rows:
                sid = r[0]
                mk = self._db.load_marker(sid)
                if mk:
                    self.somatic_markers[sid] = mk
        except Exception:
            logger.exception("Failed to load markers from DB")

    def _persist_marker(self, marker: SomaticMarker, payload: Optional[Dict[str, Any]] = None):
        if self._db:
            try:
                self._db.save_marker(marker, payload)
            except Exception:
                logger.exception("Failed to persist marker")

    def _persist_decision(self, choice_id: str, choice_repr: str, expected_value: float, integrated_value: float, metadata: Dict[str, Any]):
        if self._db:
            try:
                self._db.save_decision(choice_id, choice_repr, expected_value, integrated_value, metadata)
            except Exception:
                logger.exception("Failed to persist decision")

    # ------------------
    # Marker API
    # ------------------
    def create_somatic_marker(self, situation_id: str, emotional_valence: float, arousal: float = 0.5, confidence: float = 0.3):
        if situation_id in self.somatic_markers:
            self.somatic_markers[situation_id].strengthen(emotional_valence, observed_arousal=arousal)
            marker = self.somatic_markers[situation_id]
        else:
            marker = SomaticMarker(situation_id=situation_id,
                                   emotional_valence=float(np.clip(emotional_valence, -1.0, 1.0)),
                                   arousal=float(np.clip(arousal, 0.0, 1.0)),
                                   confidence=float(np.clip(confidence, 0.0, 1.0)),
                                   experiences=1)
            self.somatic_markers[situation_id] = marker
        self._persist_marker(marker, payload={"source": "create_somatic_marker"})
        if self.on_marker_update:
            try:
                self.on_marker_update(marker, {"reason": "create_or_strengthen"})
            except Exception:
                logger.exception("on_marker_update hook error")
        return marker

    def retrieve_somatic_marker(self, situation_id: str) -> Optional[SomaticMarker]:
        return self.somatic_markers.get(situation_id)

    def integrate_emotion_reason(self, params: Dict[str, Any]) -> float:
        """
        Meta-cognitive integration of emotion and reason for any experience.
        Returns a balance value representing the emotional-rational integration.
        """
        situation_id = params.get('situation_id')
        sensory_input = params.get('sensory_input', {})
        context = params.get('context', {})

        experience_value = params.get('experience_value', 0.0)
        cognitive_load = params.get('cognitive_load', 0.5)
        arousal = params.get('arousal', 0.5)

        # Get or create somatic marker for this situation type
        marker = self.retrieve_somatic_marker(situation_id) or \
                self.create_somatic_marker(situation_id,
                                         emotional_valence=context.get('emotional_valence', 0.0),
                                         arousal=arousal)

        # Emotional component (weight higher for conscious experience quality)
        emotional_weight = marker.emotional_valence * marker.confidence * 0.6
        emotional_weight += abs(context.get('emotional_valence', 0.0)) * 0.4

        # Rational component (experience value evaluation)
        rational_weight = experience_value * 0.5
        rational_weight += cognitive_load * 0.3
        rational_weight += arousal * 0.2

        # Integration (weighted sum biased toward emotion for consciousness)
        integration = (emotional_weight * 0.7) + (rational_weight * 0.3)

        return float(np.clip(integration, -1.0, 1.0))

    # ------------------
    # Decision utilities
    # ------------------
    @staticmethod
    def _expected_value_of_option(option: Dict[str, Any]) -> float:
        """
        Compute expected rational value given option spec:
        option = {
            'id': 'opt1',
            'outcomes': [{'value': float, 'prob': float}, ...]  OR
            'value': float  (deterministic)
        }
        """
        if not option:
            return 0.0
        if 'value' in option:
            return float(option['value'])
        outs = option.get('outcomes')
        if not outs:
            return 0.0
        ev = float(sum([o.get('value', 0.0) * o.get('prob', 0.0) for o in outs]))
        return ev

    @staticmethod
    def _risk_adjustment(value: float, risk_aversion: float) -> float:
        """Utility transform for risk aversion: concave utility"""
        # CRRA-like: u(x) = x^(1 - r) for x>=0; for negative values keep sign
        if value >= 0:
            return math.pow(value, max(1e-6, 1.0 - float(np.clip(risk_aversion, 0.0, 0.99))))
        else:
            # symmetrical transform for losses
            return -math.pow(abs(value), max(1e-6, 1.0 - float(np.clip(risk_aversion, 0.0, 0.99))))

    # ------------------
    # Main integration function
    # ------------------
    def integrate_emotion_and_reason(self,
                                    option: Dict[str, Any],
                                    situation_id: Optional[str],
                                    integration_weight: float = 0.5,
                                    risk_aversion: float = 0.2,
                                    use_marker: bool = True,
                                    rag_retrieve: bool = False) -> Tuple[float, Dict[str, Any]]:
        """
        Compute integrated value for a single option.
        Returns (integrated_value, debug_metadata)
        """
        # 1. Rational expected value
        ev = float(self._expected_value_of_option(option))

        # 2. Risk-adjusted utility
        util = float(self._risk_adjustment(ev, risk_aversion))

        # 3. Somatic marker signal (if available)
        marker_val = 0.0
        marker_conf = 0.0
        marker_arousal = 0.0
        marker: Optional[SomaticMarker] = None
        if use_marker and situation_id:
            marker = self.retrieve_somatic_marker(situation_id)
            if marker:
                marker_val = marker.emotional_valence
                marker_conf = marker.confidence
                marker_arousal = marker.arousal

        # 4. Optional RAG retrieval to enrich context that may affect integration
        rag_context = None
        if rag_retrieve and self.rag and situation_id:
            rag_context = self.rag.retrieve(situation_id, top_k=1)

        # 5. Integration weight dynamics:
        # effective_weight = integration_weight * (0.5 + 0.5 * marker_arousal * marker_conf)
        effective_weight = integration_weight
        if marker:
            effective_weight = float(np.clip(integration_weight * (0.5 + 0.5 * marker_arousal * marker_conf), 0.0, 1.0))

        # 6. Integrated value: linear combination (rational + somatic), then map to utility space
        integrated = (1 - effective_weight) * util + effective_weight * marker_val

        debug = {
            "ev": ev, "util": util, "marker_present": bool(marker), "marker_val": marker_val,
            "marker_conf": marker_conf, "marker_arousal": marker_arousal, "effective_weight": effective_weight,
            "rag_context": rag_context
        }
        return float(integrated), debug

    # ------------------
    # Decision API: choose among options
    # ------------------
    def make_decision_under_uncertainty(self,
                                       situation_id: str,
                                       options: List[Dict[str, Any]],
                                       integration_weight: float = 0.5,
                                       risk_aversion: float = 0.2,
                                       use_gut_feeling: bool = True,
                                       rag_retrieve: bool = False,
                                       counterfactual_learning: bool = True) -> Dict[str, Any]:
        """
        Select the best option using integrated value.
        Options: list of dicts with either 'value' or 'outcomes' array.
        Returns the chosen option dict plus diagnostics.
        """
        if not options:
            return {}

        values_debug = []
        chosen = None
        chosen_score = -math.inf

        for opt in options:
            if use_gut_feeling:
                integrated, debug = self.integrate_emotion_and_reason(
                    option=opt,
                    situation_id=situation_id,
                    integration_weight=integration_weight,
                    risk_aversion=risk_aversion,
                    use_marker=True,
                    rag_retrieve=rag_retrieve
                )
            else:
                # pure rational
                ev = self._expected_value_of_option(opt)
                integrated = float(self._risk_adjustment(ev, risk_aversion))
                debug = {"ev": ev, "util": integrated, "marker_present": False}

            values_debug.append({"option": opt, "integrated": integrated, "debug": debug})

            # deterministic tie-breaker: prefer higher integrated, then lower risk
            tie_breaker = -sum([o.get('prob', 0.0) for o in opt.get('outcomes', [])]) if 'outcomes' in opt else 0.0
            if integrated > chosen_score or (math.isclose(integrated, chosen_score) and tie_breaker > 0):
                chosen_score = integrated
                chosen = opt

        # optional stochasticity (exploration)
        if self.stochastic and len(options) > 1:
            # softmax sampling with temperature inverse to integration weight
            scores = np.array([vd["integrated"] for vd in values_debug], dtype=float)
            # avoid overflow: shift
            maxs = np.max(scores)
            exp_scores = np.exp((scores - maxs) * (1.0 + 1.0 * (1 - integration_weight)))
            probs = exp_scores / (np.sum(exp_scores) + 1e-12)
            idx = int(np.random.choice(len(options), p=probs))
            chosen = options[idx]
            chosen_score = float(scores[idx])

        # Register metrics & callbacks
        self.decisions_with_emotion += 1 if use_gut_feeling else 0
        choice_id = f"choice_{int(datetime.utcnow().timestamp()*1000)}"
        meta = {"situation_id": situation_id, "integration_weight": integration_weight, "risk_aversion": risk_aversion}
        self._persist_decision(choice_id, json.dumps(chosen, default=str), chosen_score, chosen_score, meta)
        entry = {"choice_id": choice_id, "chosen": chosen, "score": chosen_score, "values_debug": values_debug, "meta": meta}
        self.choice_history.append(entry)

        if self.on_decision:
            try:
                self.on_decision(entry)
            except Exception:
                logger.exception("on_decision hook failed")

        # Counterfactual learning: if enabled, we can update markers after outcome is known (external call)
        # For now, just return the chosen option and diagnostics
        return entry

    # ------------------
    # Evaluate outcome (to strengthen markers & counterfactuals)
    # ------------------
    def evaluate_decision_outcome(self, situation_id: str, chosen: Dict[str, Any], outcome: Dict[str, Any], counterfactuals: Optional[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = None):
        """
        After observing outcome, update somatic marker for situation_id and optionally apply counterfactual updates:
        - chosen: the chosen option
        - outcome: dict containing 'valence' (-1..1), 'arousal' (0..1)
        - counterfactuals: list of tuples (option, outcome_if_taken) for options not chosen
        """
        valence = float(np.clip(outcome.get('valence', 0.0), -1.0, 1.0))
        arousal = float(np.clip(outcome.get('arousal', abs(valence)), 0.0, 1.0))
        marker = self.somatic_markers.get(situation_id)
        if not marker:
            marker = SomaticMarker(situation_id=situation_id, emotional_valence=valence, arousal=arousal, confidence=0.4, experiences=1)
            self.somatic_markers[situation_id] = marker
        else:
            marker.strengthen(valence, observed_arousal=arousal, weight=1.0)

        # persist & callback
        self._persist_marker(marker, payload={"last_outcome": outcome})
        if self.on_marker_update:
            try:
                self.on_marker_update(marker, {"reason": "outcome_update"})
            except Exception:
                logger.exception("on_marker_update callback error")

        # Counterfactual regret learning: if an unchosen option had better outcome, we penalize marker or adjust
        if counterfactuals:
            for opt, cf_outcome in counterfactuals:
                cf_val = float(np.clip(cf_outcome.get('valence', 0.0), -1.0, 1.0))
                # if counterfactual better than chosen outcome, slightly reduce confidence (simulates regret)
                if cf_val > valence:
                    marker.confidence = max(0.0, marker.confidence - 0.05)
                    # weaken valence slightly (learn to be cautious)
                    marker.emotional_valence *= 0.98

        # Return updated marker for inspection
        return marker

    # ------------------
    # Regulation strategies
    # ------------------
    def regulate_emotion(self, emotion: Dict[str, Any], strategy: str = 'reappraisal') -> Dict[str, Any]:
        """
        Multiple regulation strategies implemented deterministically.
        """
        self.regulation_active = True
        self.regulation_events += 1
        s = strategy.lower()
        regulated = dict(emotion)
        if s == 'reappraisal':
            # reduce valence magnitude and arousal
            regulated['valence'] = float(regulated.get('valence', 0.0) * (1 - 0.5 * self.regulation_strength))
            regulated['arousal'] = float(regulated.get('arousal', 0.5) * (1 - 0.3 * self.regulation_strength))
        elif s == 'suppression':
            regulated['expression'] = regulated.get('expression', 1.0) * 0.5
            regulated['arousal'] = float(min(1.0, regulated.get('arousal', 0.5) * (1 + 0.1 * (1 - self.regulation_strength))))
        elif s == 'distancing':
            regulated['valence'] = float(regulated.get('valence', 0.0) * (1 - 0.4 * self.regulation_strength))
            regulated['arousal'] = float(regulated.get('arousal', 0.5) * (1 - 0.4 * self.regulation_strength))
        elif s == 'breathing':
            # gradual reduction in arousal (simulate with deterministic decay)
            regulated['arousal'] = float(max(0.0, regulated.get('arousal', 0.5) - 0.05 * self.regulation_strength))
        else:
            # noop
            pass

        # record
        self.regulation_active = False
        return regulated

    # ------------------
    # Export / metrics
    # ------------------
    def export_state(self) -> Dict[str, Any]:
        markers_summary = {
            sid: {"valence": m.emotional_valence, "arousal": m.arousal, "conf": m.confidence, "exp": m.experiences}
            for sid, m in self.somatic_markers.items()
        }
        return {
            "system_id": self.system_id,
            "created_at": self.created_at,
            "markers_count": len(self.somatic_markers),
            "markers": markers_summary,
            "decisions_recorded": len(self.choice_history),
            "regulation_strength": self.regulation_strength,
            "metrics": {
                "decisions_with_emotion": self.decisions_with_emotion,
                "decisions_purely_rational": self.decisions_purely_rational,
                "regulation_events": self.regulation_events
            }
        }

    def recent_decisions(self, limit: int = 10):
        if self._db:
            return self._db.recent_decisions(limit)
        return list(self.choice_history)[-limit:]


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Inicializar vmPFC con persistencia y RAG stub
    rag = SimpleRAG()
    vm = VentromedialPFC("vmPFC-ENTERPRISE", persist=True, db_path="vm_markers_demo.db", rag=rag, stochastic=False)

    # Registrar hooks
    def on_decision_hook(entry):
        print("ON_DECISION:", json.dumps(entry, ensure_ascii=False, indent=2))

    def on_marker_hook(marker, meta):
        print("ON_MARKER_UPDATE:", marker.situation_id, "valence=", marker.emotional_valence, "conf=", marker.confidence)

    vm.on_decision = on_decision_hook
    vm.on_marker_update = on_marker_hook

    # Crear algunos marcadores históricos
    vm.create_somatic_marker("street_dog", emotional_valence=-0.6, arousal=0.7, confidence=0.4)
    vm.create_somatic_marker("calm_office", emotional_valence=0.4, arousal=0.2, confidence=0.5)

    # Opciones: cada opción tiene outcomes with probabilities (value in util points)
    options = [
        {"id": "approach", "outcomes": [{"value": -10, "prob": 0.2}, {"value": 5, "prob": 0.8}]},
        {"id": "avoid", "outcomes": [{"value": 0, "prob": 1.0}]},
        {"id": "call_help", "outcomes": [{"value": -1, "prob": 0.9}, {"value": 10, "prob": 0.1}]}
    ]

    # Decisión usando marcador somático (situation 'street_dog')
    decision = vm.make_decision_under_uncertainty(situation_id="street_dog", options=options, integration_weight=0.6, risk_aversion=0.3, rag_retrieve=True)
    print("Decision:", json.dumps(decision, ensure_ascii=False, indent=2))

    # Simular outcome observado y evaluación (chosen outcome valence scaled into [-1,1])
    observed_outcome = {"valence": -0.5, "arousal": 0.6}
    updated_marker = vm.evaluate_decision_outcome("street_dog", decision["chosen"], observed_outcome, counterfactuals=None)
    print("Updated marker:", asdict(updated_marker))

    # Export state
    print("VM state snapshot:")
    import pprint
    pprint.pprint(vm.export_state())
