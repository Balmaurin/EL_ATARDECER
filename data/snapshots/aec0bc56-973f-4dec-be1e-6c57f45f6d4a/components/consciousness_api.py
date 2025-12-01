
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import asyncio

# Add consciousness paths
consciousness_paths = [
    os.path.join(os.getcwd(), "packages", "consciousness", "src"),
    os.path.join(os.getcwd(), "packages", "sheily_core", "src"),
    os.path.join(os.getcwd(), "tools"),
    os.getcwd()
]

for path in consciousness_paths:
    if path not in sys.path:
        sys.path.append(path)

app = FastAPI(title="EL-AMANECER Complete Consciousness API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global systems
consciousness_engine = None
training_system = None
evolution_engine = None
consciousness_modules = {}

class ConsciousInputModel(BaseModel):
    input: str
    context: dict = {}

class EvolutionRequestModel(BaseModel):
    component: str
    evolution_type: str = "mutation"

@app.on_event("startup")
async def startup():
    global consciousness_engine, training_system, evolution_engine, consciousness_modules
    print("Starting Complete Consciousness System...")
    
    # Load consciousness engine
    try:
        from conciencia.modulos.unified_consciousness_engine import UnifiedConsciousnessEngine
        consciousness_engine = UnifiedConsciousnessEngine()
        print("Unified Consciousness Engine started")
    except Exception as e:
        print(f"Consciousness engine: {e}")
        
        # Load individual modules
        try:
            from conciencia.modulos.digital_nervous_system import DigitalNervousSystem
            consciousness_modules['nervous_system'] = DigitalNervousSystem()
            print("Digital Nervous System loaded")
        except: pass
        
        try:
            from conciencia.modulos.human_emotional_system import HumanEmotionalSystem
            consciousness_modules['emotional_system'] = HumanEmotionalSystem(35)
            print("Emotional System loaded")
        except: pass
    
    # Load training system
    try:
        from tools.ai.auto_training_system import AutoTrainingSystem
        training_system = AutoTrainingSystem()
        await training_system.start_monitoring()
        print("Training system active")
    except Exception as e:
        print(f"Training system: {e}")
    
    # Load evolution engine
    try:
        from sheily_core.api.auto_evolution_engine import AutoEvolutionEngine
        evolution_engine = AutoEvolutionEngine()
        print("Evolution engine active")
    except Exception as e:
        print(f"Evolution engine: {e}")

@app.get("/")
async def root():
    return {
        "message": "EL-AMANECER Complete Consciousness System API", 
        "status": "active",
        "systems": {
            "consciousness": consciousness_engine is not None,
            "training": training_system is not None,
            "evolution": evolution_engine is not None,
            "modules": len(consciousness_modules)
        }
    }

@app.get("/api/consciousness/health")
async def consciousness_health():
    if consciousness_engine:
        return {
            "status": "conscious",
            "engine": "unified",
            "theories": ["IIT 4.0", "GWT", "FEP", "SMH", "Hebbian", "Circumplex"],
            "components": 37
        }
    return {
        "status": "partial",
        "engine": "modular",
        "active_modules": list(consciousness_modules.keys())
    }

@app.post("/api/consciousness/process")
async def process_conscious_experience(data: ConsciousInputModel):
    if consciousness_engine:
        try:
            # Try different method names
            method_to_use = None
            for method_name in ['process_experience', 'process_conscious_experience', 'process_input', 'process']:
                if hasattr(consciousness_engine, method_name):
                    method_to_use = getattr(consciousness_engine, method_name)
                    break
            
            if method_to_use:
                result = method_to_use({"input": data.input, "context": data.context})
                return {"status": "success", "result": result}
            else:
                return {"status": "error", "error": "No processing method available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    elif consciousness_modules:
        # Use available modules
        result = {"input": data.input, "modules_used": []}
        
        if 'nervous_system' in consciousness_modules:
            try:
                neural_response = consciousness_modules['nervous_system'].process_stimulus({
                    'text': data.input, 'intensity': 0.7
                }, data.context)
                result['neural_response'] = neural_response
                result['modules_used'].append('nervous_system')
            except: pass
        
        if 'emotional_system' in consciousness_modules:
            try:
                emotional_response = consciousness_modules['emotional_system'].process_emotion(
                    data.input, 0.7
                )
                result['emotional_response'] = emotional_response
                result['modules_used'].append('emotional_system')
            except: pass
        
        return {"status": "partial", "result": result}
    else:
        raise HTTPException(status_code=503, detail="No consciousness system available")

@app.get("/api/consciousness/status")
async def consciousness_status():
    status = {
        "consciousness_engine": consciousness_engine is not None,
        "training_system": training_system is not None,
        "evolution_engine": evolution_engine is not None,
        "active_modules": list(consciousness_modules.keys()),
        "system_health": "operational"
    }
    
    if consciousness_engine:
        try:
            if hasattr(consciousness_engine, 'get_consciousness_metrics'):
                metrics = consciousness_engine.get_consciousness_metrics()
                status["metrics"] = metrics
        except: pass
    
    return status

@app.post("/api/evolution/evolve")
async def evolve_component(data: EvolutionRequestModel):
    if not evolution_engine:
        raise HTTPException(status_code=503, detail="Evolution engine not available")
    
    try:
        result = await evolution_engine.evolve_system_component(
            data.component, data.evolution_type
        )
        return {"status": "success", "evolution_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evolution error: {str(e)}")

@app.get("/api/training/status")
async def training_status():
    if training_system:
        return {
            "status": "active",
            "monitoring": training_system.is_active if hasattr(training_system, 'is_active') else True
        }
    return {"status": "unavailable"}

@app.post("/api/training/feedback")
async def submit_training_feedback(feedback: dict):
    if not training_system:
        raise HTTPException(status_code=503, detail="Training system not available")
    
    try:
        result = await training_system.process_feedback(feedback)
        return {"status": "success", "training_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
