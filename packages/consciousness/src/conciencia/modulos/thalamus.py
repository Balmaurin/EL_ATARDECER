# -*- coding: utf-8 -*-
"""
TÁLAMO + MÓDULOS CORTICALES Y SUBCORTICALES
==========================================
Versión extendida: integra Amígdala, Hipocampo, Ínsula, PFC, ACC, Ganglios de la Base,
y un simple MemoryStore / RAG hook.

Principio de diseño:
- El Thalamus sigue filtrando y priorizando señales.
- Los módulos reciben las señales relayadas y devuelven
  influencias (arousal_delta, cortical_bias, salience_boost, actions).
- Módulos pueden disparar side-effects via callbacks (p. ej. RAG, DB, reward).
- Estructura modular: puedes añadir/retirar módulos fácilmente.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Iterable, Tuple
from datetime import datetime
import time
import math
import numpy as np
import logging

logger = logging.getLogger("thalamo_system")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


# ---------------------- Núcleos talámicos (simplificados) ----------------------

@dataclass
class ThalamicNucleus:
    nucleus_id: str
    sensory_modality: str
    base_threshold: float = 0.5
    excitability: float = 0.8
    refractory_window: float = 0.01
    last_relay_time: float = field(default_factory=lambda: -1e9)
    total_inputs: int = 0
    total_relayed: int = 0
    noise: float = 0.0

    def can_relay(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.time()
        return (now - self.last_relay_time) >= self.refractory_window

    def attempt_relay(self, saliency: float, arousal: float, cortical_bias: float = 0.0) -> bool:
        self.total_inputs += 1
        now = time.time()
        if not self.can_relay(now):
            return False

        arousal_factor = 1.0 - (arousal * 0.35)
        eff_threshold = self.base_threshold * arousal_factor - cortical_bias
        eff_threshold = np.clip(eff_threshold * (1.0 / max(0.01, self.excitability)), 0.0, 1.0)

        margin = saliency - eff_threshold
        if self.noise > 0:
            margin += np.random.normal(0.0, self.noise)

        prob = 1.0 / (1.0 + math.exp(-12.0 * (margin - 0.02)))
        do_relay = np.random.rand() < prob

        if do_relay:
            self.last_relay_time = now
            self.total_relayed += 1

        return do_relay

    def efficiency(self) -> float:
        return float(self.total_relayed) / max(1, self.total_inputs)


# ---------------------- Módulos (amígdala, hipocampo, ínsula, PFC, ACC, BG) ----------------------

@dataclass
class ModuleResult:
    arousal_delta: float = 0.0        # modifica arousal global
    cortical_bias: float = 0.0        # bias top-down para bajar thresholds
    salience_boost: float = 0.0       # incremento puntual de saliencia para señal
    action: Optional[Dict[str, Any]] = None  # p. ej. {"type": "retrieve", "query": "..."}
    notes: Optional[str] = None


class BaseModule:
    """Interfaz simple para módulos."""
    def __init__(self, name: str):
        self.name = name

    def process(self, relayed_signals: Dict[str, List[Dict[str, Any]]]) -> ModuleResult:
        """Recibe todas las señales relayadas en un ciclo. Devuelve influencia."""
        return ModuleResult()


class Amygdala(BaseModule):
    """
    Detecta saliencia emocional y urgencia.
    - Sube arousal cuando detecta valencia elevada o amenaza.
    - Puede solicitar recuperación de contexto/emoción.
    """
    def __init__(self, sensitivity: float = 1.0):
        super().__init__("Amygdala")
        self.sensitivity = sensitivity

    def process(self, relayed_signals: Dict[str, List[Dict[str, Any]]]) -> ModuleResult:
        max_emotional = 0.0
        urgent_count = 0
        for modality, arr in relayed_signals.items():
            for s in arr:
                sig = s.get("signal") or {}
                # asumimos que 'emotional_valence' podría venir en signal o salience dict
                ev = 0.0
                if isinstance(sig, dict) and "emotional_valence" in sig:
                    ev = sig["emotional_valence"]
                elif isinstance(s, dict) and "salience" in s and isinstance(s["salience"], dict):
                    ev = s["salience"].get("emotional_valence", 0.0)
                max_emotional = max(max_emotional, abs(ev))
                if isinstance(s.get("salience"), dict) and s["salience"].get("urgency", 0.0) > 0.6:
                    urgent_count += 1

        # Arousal aumenta con la emoción y urgencia
        arousal_delta = np.tanh(self.sensitivity * (max_emotional * 1.2 + urgent_count * 0.4))
        # Bias cortical pequeño: cuando hay emoción fuerte, PFC se "solicita"
        cortical_bias = min(0.2, max(0.0, max_emotional * 0.12))
        notes = f"amygdala:max_emotional={max_emotional:.2f}, urgent_count={urgent_count}"
        return ModuleResult(arousal_delta=arousal_delta, cortical_bias=cortical_bias, notes=notes)


class Insula(BaseModule):
    """
    Procesa interocepción y valencia visceral (sensaciones corporales).
    - Si detecta alto disconfort aumenta arousal y prioriza signals somatosensoriales.
    """
    def __init__(self, sensitivity: float = 0.8):
        super().__init__("Insula")
        self.sensitivity = sensitivity

    def process(self, relayed_signals: Dict[str, List[Dict[str, Any]]]) -> ModuleResult:
        somatic_sal = 0.0
        for modality, arr in relayed_signals.items():
            if "somato" in modality or "touch" in modality:
                for s in arr:
                    sal = s.get("salience", 0.0)
                    somatic_sal = max(somatic_sal, sal if isinstance(sal, (int, float)) else 0.0)

        arousal_delta = somatic_sal * 0.7 * self.sensitivity
        salience_boost = somatic_sal * 0.2
        return ModuleResult(arousal_delta=arousal_delta, salience_boost=salience_boost,
                            notes=f"insula:somatic={somatic_sal:.2f}")


class Hippocampus(BaseModule):
    """
    Detección de novedad y recuperación de contexto/episodios.
    - Si detecta novedad alta -> solicita 'retrieve' para contexto.
    - Almacena pequeños episodios en memoria interna (simple store).
    """
    def __init__(self, memory_store: Optional[List[Dict[str, Any]]] = None, novelty_threshold: float = 0.6):
        super().__init__("Hippocampus")
        self.memory_store = memory_store if memory_store is not None else []
        self.novelty_threshold = novelty_threshold

    def process(self, relayed_signals: Dict[str, List[Dict[str, Any]]]) -> ModuleResult:
        max_novelty = 0.0
        candidate = None
        for modality, arr in relayed_signals.items():
            for s in arr:
                sal = s.get("salience", 0.0)
                # si salience incluye 'novelty' dict
                if isinstance(s.get("salience"), dict):
                    nov = s["salience"].get("novelty", 0.0)
                else:
                    nov = 0.0
                if nov > max_novelty:
                    max_novelty = nov
                    candidate = s

        if max_novelty >= self.novelty_threshold and candidate:
            # pedir recuperación de contexto (RAG-like)
            query = f"novel_event_modality={candidate.get('nucleus', 'unknown')};desc={str(candidate.get('signal'))[:80]}"
            # almacenar el episodio (very light)
            self.memory_store.append({"time": time.time(), "event": candidate})
            return ModuleResult(arousal_delta=0.2, action={"type": "retrieve", "query": query},
                                notes=f"hippocampus:novelty={max_novelty:.2f}")
        return ModuleResult(notes="hippocampus:noop")


class PFC(BaseModule):
    """
    Prefrontal Cortex (top-down control).
    - Recibe relayed_signals y decide bias cortical (priorización) y reglas.
    - Puede bajar thresholds para tareas esperadas.
    """
    def __init__(self, top_down_focus: Optional[Dict[str, float]] = None):
        super().__init__("PFC")
        # top_down_focus: modality -> bias (0..0.5)
        self.top_down_focus = top_down_focus or {}

    def process(self, relayed_signals: Dict[str, List[Dict[str, Any]]]) -> ModuleResult:
        # Si hay foco explícito, aplicar bias para esa modalidad
        bias = 0.0
        for modality in relayed_signals.keys():
            for k, v in self.top_down_focus.items():
                if k in modality:
                    bias = max(bias, v)
        # PFC reduce arousal levemente si puede reappraise (regulación)
        arousal_delta = -0.05 if bias > 0 else 0.0
        return ModuleResult(arousal_delta=arousal_delta, cortical_bias=bias,
                            notes=f"pfc:bias={bias:.3f}")


class ACC(BaseModule):
    """
    Anterior Cingulate Cortex: monitor de conflicto / error.
    - Si hay señales contradictorias o muchas políticas activas, aumenta arousal y solicita PFC.
    """
    def __init__(self):
        super().__init__("ACC")

    def process(self, relayed_signals: Dict[str, List[Dict[str, Any]]]) -> ModuleResult:
        # detect conflict simply: múltiples señales de alta saliencia en diferentes modalities
        high = 0
        modalities = list(relayed_signals.keys())
        for mod in modalities:
            arr = relayed_signals.get(mod, [])
            for s in arr:
                if s.get("salience", 0.0) >= 0.7:
                    high += 1
        if high >= 2:
            return ModuleResult(arousal_delta=0.15, cortical_bias=0.05, notes=f"acc:conflict_high={high}")
        return ModuleResult()


class BasalGanglia(BaseModule):
    """
    Gating de acciones: decide si ejecutar una acción sugerida por módulos.
    - Si action tiene type 'act', BG puede permitir o inhibir.
    """
    def __init__(self, threshold: float = 0.6):
        super().__init__("BasalGanglia")
        self.threshold = threshold

    def process(self, relayed_signals: Dict[str, List[Dict[str, Any]]]) -> ModuleResult:
        # Aquí no tenemos acciones externas directas, retornamos noop
        return ModuleResult()

# ---------------------- Memory / RAG Hook (Real Implementation) ----------------------

class RealRAGWrapper:
    """
    Wrapper para RealSemanticSearch que mantiene la interfaz de SimpleRAG
    Usa búsqueda semántica real con embeddings y FAISS
    """
    def __init__(self):
        self._search = None
        self._initialized = False
        self._default_docs = [
            {"id": 1, "text": "Contexto: cuando suena una campana, suele ser anuncio horario."},
            {"id": 2, "text": "Contexto: una luz intensa suele indicar cámara/flash."},
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
            logger.info("✅ RealSemanticSearch inicializado para Thalamus")
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo inicializar RealSemanticSearch: {e}. Usando búsqueda básica.")
            self._search = None
            self._initialized = True
    
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
            similarity = len(common_words) / max(len(query_words), 1) if query_words else 0
            
            if query_lower in doc_text_lower:
                similarity += 0.3
            
            scored_docs.append({
                **doc,
                'similarity_score': similarity
            })
        
        scored_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_docs[:top_k]

# Alias para compatibilidad
SimpleRAG = RealRAGWrapper


# ---------------------- Thalamus extendido que integra módulos ----------------------

class ThalamusExtended:
    """
    Extensión del Thalamus que integra módulos funcionales.
    - modules: lista de BaseModule
    - rag: Optional RAG-like object con método retrieve(query)
    """
    def __init__(self, modules: Optional[List[BaseModule]] = None, rag: Optional[SimpleRAG] = None,
                 global_max_relay: int = 6, temporal_window_s: float = 0.03, logging_enabled: bool = False):
        self.nuclei = {
            "LGN": ThalamicNucleus("LGN", "visual", base_threshold=0.6),
            "MGN": ThalamicNucleus("MGN", "auditory", base_threshold=0.6),
            "VPL": ThalamicNucleus("VPL", "somatosensory", base_threshold=0.5),
            "MD": ThalamicNucleus("MD", "cognitive", base_threshold=0.45),
            "LP": ThalamicNucleus("LP", "associative", base_threshold=0.5),
            "CM": ThalamicNucleus("CM", "arousal", base_threshold=0.3),
        }
        self.modality_map = {
            "visual": "LGN", "sight": "LGN",
            "auditory": "MGN", "sound": "MGN",
            "touch": "VPL", "somato": "VPL",
            "cognitive": "MD", "thought": "MD",
            "associative": "LP", "arousal": "CM",
        }

        self.global_max_relay = max(1, global_max_relay)
        self.temporal_window_s = max(0.001, temporal_window_s)
        self.arousal = 0.5
        self.cortical_feedback = 0.0
        self.salience_history: List[float] = []
        self.total_inputs = 0
        self.total_relayed = 0
        self.logging_enabled = logging_enabled

        # módulos
        self.modules: List[BaseModule] = modules if modules is not None else []
        self.rag = rag
        # callback para cuando un módulo solicita una acción (ej. RAG retrieve)
        self.on_action_callback: Optional[Callable[[Dict[str, Any]], Any]] = None

    def register_on_action(self, cb: Callable[[Dict[str, Any]], Any]):
        self.on_action_callback = cb

    def _get_nucleus_for_modality(self, modality: str) -> ThalamicNucleus:
        m = modality.lower()
        for key, nid in self.modality_map.items():
            if key in m:
                return self.nuclei.get(nid, list(self.nuclei.values())[0])
        return self.nuclei.get("LP", list(self.nuclei.values())[0])

    def _normalize_saliency(self, raw: Any) -> float:
        if raw is None:
            return 0.0
        if isinstance(raw, (int, float)):
            return float(np.clip(raw, 0.0, 1.0))
        s = 0.0; w = 0.0
        if "intensity" in raw:
            s += raw["intensity"] * 0.30; w += 0.30
        if "novelty" in raw:
            s += raw["novelty"] * 0.25; w += 0.25
        if "emotional_valence" in raw:
            s += abs(raw["emotional_valence"]) * 0.20; w += 0.20
        if "urgency" in raw:
            s += raw["urgency"] * 0.35; w += 0.35
        if "duration" in raw:
            d = float(raw["duration"]); s += (min(1.0, d / 5.0) * 0.05); w += 0.05
        if "frequency" in raw:
            s += (min(1.0, raw["frequency"]) * 0.10); w += 0.10
        if w == 0:
            return 0.0
        return float(np.clip(s / w, 0.0, 1.0))

    def _temporal_batching(self, inputs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if not inputs:
            return []
        now = time.time()
        for inp in inputs:
            inp.setdefault("timestamp", now)
        inputs_sorted = sorted(inputs, key=lambda x: x["timestamp"])
        batches: List[List[Dict[str, Any]]] = []
        current_batch: List[Dict[str, Any]] = []
        window_start = inputs_sorted[0]["timestamp"]
        for inp in inputs_sorted:
            ts = inp["timestamp"]
            if ts - window_start <= self.temporal_window_s:
                current_batch.append(inp)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [inp]
                window_start = ts
        if current_batch:
            batches.append(current_batch)
        return batches

    def set_arousal(self, level: float):
        self.arousal = float(np.clip(level, 0.0, 1.0))

    def set_cortical_feedback(self, bias: float):
        self.cortical_feedback = float(bias)

    def process_inputs(self, sensory_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Procesa el pipeline completo: tálamo -> módulos -> acciones (RAG, callbacks)"""
        if not sensory_inputs:
            return {"relayed": {}, "modules": {}, "metrics": self._metrics_snapshot()}

        batches = self._temporal_batching(sensory_inputs)
        relayed_store: Dict[str, List[Dict[str, Any]]] = {}
        cycle_relay_count = 0

        for batch in batches:
            enriched = []
            for item in batch:
                sal_raw = item.get("salience", item.get("signal"))
                sal = self._normalize_saliency(sal_raw)
                self.salience_history.append(sal)
                enriched.append({**item, "salience": sal})

            enriched_sorted = sorted(enriched, key=lambda x: x["salience"], reverse=True)

            for item in enriched_sorted:
                if cycle_relay_count >= self.global_max_relay:
                    break

                modality = item.get("modality", "associative")
                nucleus = self._get_nucleus_for_modality(modality)
                sal = item["salience"]

                cortical_bias = self.cortical_feedback if nucleus.sensory_modality != "arousal" else 0.0

                if nucleus.attempt_relay(saliency=sal, arousal=self.arousal, cortical_bias=cortical_bias):
                    rel_entry = {
                        "signal": item.get("signal"),
                        "salience": sal,
                        "nucleus": nucleus.nucleus_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                    relayed_store.setdefault(modality, []).append(rel_entry)
                    cycle_relay_count += 1
                    self.total_relayed += 1
                self.total_inputs += 1

        # Ejecutar módulos con el conjunto de señales relayadas
        module_results: Dict[str, ModuleResult] = {}
        aggregated_influence = {"arousal_delta": 0.0, "cortical_bias": 0.0, "salience_boosts": {}}

        for mod in self.modules:
            try:
                res = mod.process(relayed_store)
                module_results[mod.name] = res
                aggregated_influence["arousal_delta"] += res.arousal_delta
                aggregated_influence["cortical_bias"] += res.cortical_bias
                # si hay salience boost, aplicar por modalidad simple: boost para todas somáticas si insula
                if res.salience_boost and res.salience_boost > 0:
                    aggregated_influence["salience_boosts"][mod.name] = res.salience_boost

                # acciones (p. ej. retrieve)
                if res.action:
                    if self.logging_enabled:
                        logger.info(f"Module {mod.name} requested action: {res.action}")
                    # si es retrieve y disponemos de RAG, ejecutarlo y devolver resultado por callback
                    if res.action.get("type") == "retrieve" and self.rag:
                        q = res.action.get("query", "")
                        retrieved = self.rag.retrieve(q)
                        # enviar a callback si existe
                        if self.on_action_callback:
                            try:
                                self.on_action_callback({"module": mod.name, "action": res.action, "retrieved": retrieved})
                            except Exception as e:
                                logger.exception("error in on_action_callback: %s", e)
            except Exception as e:
                logger.exception("Error procesando módulo %s: %s", mod.name, e)

        # Aplicar agregados: ajustar arousal y cortical bias
        # Normalizar influencia (ligera)
        self.arousal = float(np.clip(self.arousal + np.tanh(aggregated_influence["arousal_delta"]), 0.0, 1.0))
        self.cortical_feedback = float(np.clip(self.cortical_feedback + np.tanh(aggregated_influence["cortical_bias"]), -0.5, 0.5))

        # Aplicar boosts de saliencia: podemos incrementar salience en store (simple)
        if aggregated_influence["salience_boosts"]:
            total_boost = sum(aggregated_influence["salience_boosts"].values())
            # por simplicidad, aplicar multiplicador a salience existente
            for mod_name, boost in aggregated_influence["salience_boosts"].items():
                # ejemplo: si Insula sugiere boost, aumentar salience en somatosensory modalities
                for modality, arr in relayed_store.items():
                    if "somato" in modality or "touch" in modality:
                        for s in arr:
                            s["salience"] = float(np.clip(s["salience"] + boost * 0.2, 0.0, 1.0))

        metrics = self._metrics_snapshot()
        if self.logging_enabled:
            logger.info(f"Processed inputs={len(sensory_inputs)} -> relayed_total={metrics['total_relayed']}, arousal={self.arousal:.3f}")

        return {"relayed": relayed_store, "modules": {k: v.__dict__ for k, v in module_results.items()}, "metrics": metrics}

    def _metrics_snapshot(self) -> Dict[str, Any]:
        return {
            "arousal": self.arousal,
            "cortical_feedback": self.cortical_feedback,
            "global_max_relay": self.global_max_relay,
            "temporal_window_s": self.temporal_window_s,
            "total_inputs": self.total_inputs,
            "total_relayed": self.total_relayed,
            "avg_salience": float(np.mean(self.salience_history)) if self.salience_history else 0.0,
            "nuclei": {nid: {"efficiency": nuc.efficiency(), "last_relay_delta_s": time.time() - nuc.last_relay_time if nuc.last_relay_time > 0 else None,
                             "base_threshold": nuc.base_threshold} for nid, nuc in self.nuclei.items()}
        }


# ---------------------- Ejemplo de uso ----------------------

if __name__ == "__main__":
    # crear módulos
    rag = SimpleRAG()
    amyg = Amygdala(sensitivity=1.2)
    ins = Insula(sensitivity=0.9)
    hip = Hippocampus(novelty_threshold=0.65)
    pfc = PFC(top_down_focus={"vision": 0.08})   # ejemplo: priorizar visión
    acc = ACC()
    bg = BasalGanglia()

    modules = [amyg, ins, hip, pfc, acc, bg]
    t = ThalamusExtended(modules=modules, rag=rag, global_max_relay=6, temporal_window_s=0.03, logging_enabled=True)

    # registrar callback para acciones (p. ej. recuperar resultado RAG)
    def on_action(act):
        print("ON_ACTION CALLBACK ->", act)
    t.register_on_action(on_action)

    # batch de entrada
    now = time.time()
    batch = [
        {"modality": "vision", "signal": {"desc": "persona con cuchillo", "emotional_valence": -0.9}, "salience": {"intensity": 0.9, "novelty": 0.7, "urgency": 0.9}},
        {"modality": "auditory", "signal": {"desc": "pasos rapidos"}, "salience": {"intensity": 0.7, "novelty": 0.3}},
        {"modality": "somatosensory", "signal": {"desc": "dolor torax"}, "salience": {"intensity": 0.8, "urgency": 0.7, "emotional_valence": -0.6}},
        {"modality": "vision", "signal": {"desc": "luz intensa"}, "salience": {"intensity": 0.5, "novelty": 0.2}},
        {"modality": "cognitive", "signal": {"desc": "pensamiento intrusivo"}, "salience": {"novelty": 0.9, "emotional_valence": -0.7}},
    ]

    out = t.process_inputs(batch)
    print("\n=== RESULTADO PRINCIPAL ===")
    import json
    print(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n=== ESTADO EXPORTADO ===")
    print(t._metrics_snapshot())
