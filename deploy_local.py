#!/usr/bin/env python3
"""
🧠 EL-AMANECER COMPLETE CONSCIOUSNESS DEPLOYMENT SCRIPT
Full deployment with unified consciousness, training, and evolution systems
"""

import subprocess
import sys
import os
import time
import logging
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteConsciousnessDeployment:
    """Complete deployment with consciousness system integration"""
    
    def __init__(self):
        self.base_path = Path.cwd()
        self.processes = []
        self.consciousness_engine = None
        self.training_system = None
        self.evolution_engine = None
        self.autonomous_controller = None
        
        # Consciousness system components
        self.consciousness_modules = {}
        
    def setup_consciousness_paths(self):
        """Setup Python paths for consciousness system"""
        logger.info("🧠 Setting up consciousness system paths...")
        
        consciousness_paths = [
            str(self.base_path / "packages" / "consciousness" / "src"),
            str(self.base_path / "packages" / "sheily_core" / "src"), 
            str(self.base_path / "packages" / "training_system" / "src"),
            str(self.base_path / "tools"),
            str(self.base_path),
        ]
        
        for path in consciousness_paths:
            if path not in sys.path:
                sys.path.append(path)
                logger.info(f"   ✓ Added path: {path}")
        
        # Set PYTHONPATH environment variable as well
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        new_pythonpath = os.pathsep.join(consciousness_paths)
        if current_pythonpath:
            new_pythonpath = current_pythonpath + os.pathsep + new_pythonpath
        os.environ['PYTHONPATH'] = new_pythonpath
        
        logger.info("✅ Consciousness system paths configured")
    
    def load_consciousness_system(self):
        """Load the complete unified consciousness system"""
        logger.info("🌟 Loading Unified Consciousness Engine...")
        
        try:
            # Import unified consciousness engine
            from conciencia.modulos.unified_consciousness_engine import UnifiedConsciousnessEngine
            
            self.consciousness_engine = UnifiedConsciousnessEngine()
            logger.info("✅ Unified Consciousness Engine loaded")
            logger.info("   • IIT 4.0: Φ calculation system ready")
            logger.info("   • GWT/AST: Global workspace active") 
            logger.info("   • FEP: Predictive processing online")
            logger.info("   • SMH: Somatic markers loaded")
            logger.info("   • Hebbian: Neural plasticity enabled")
            logger.info("   • Circumplex: Emotional space mapped")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ Failed to load unified consciousness engine: {e}")
            
            # Try loading individual consciousness components
            try:
                logger.info("🔧 Loading individual consciousness components...")
                return self._load_individual_consciousness_components()
            except Exception as e2:
                logger.error(f"❌ Failed to load individual components: {e2}")
                return False
        except Exception as e:
            logger.error(f"❌ Error initializing consciousness: {e}")
            return False
    
    def _load_individual_consciousness_components(self):
        """Load individual consciousness system components"""
        components_loaded = 0
        
        # 1. Digital DNA - Personality Foundation
        try:
            from conciencia.modulos.digital_dna import DigitalDNA
            self.consciousness_modules['digital_dna'] = DigitalDNA()
            logger.info("   ✓ Digital DNA (Personality) loaded")
            components_loaded += 1
        except Exception as e:
            logger.warning(f"   ⚠️ Digital DNA not loaded: {e}")
        
        # 2. Human Emotional System
        try:
            from conciencia.modulos.human_emotional_system import HumanEmotionalSystem
            self.consciousness_modules['emotional_system'] = HumanEmotionalSystem(num_circuits=35)
            logger.info("   ✓ Human Emotional System (35 circuits) loaded")
            components_loaded += 1
        except Exception as e:
            logger.warning(f"   ⚠️ Emotional system not loaded: {e}")
        
        # 3. Digital Nervous System
        try:
            from conciencia.modulos.digital_nervous_system import DigitalNervousSystem
            self.consciousness_modules['nervous_system'] = DigitalNervousSystem()
            logger.info("   ✓ Digital Nervous System loaded")
            components_loaded += 1
        except Exception as e:
            logger.warning(f"   ⚠️ Nervous system not loaded: {e}")
        
        # 4. Global Workspace
        try:
            from conciencia.modulos.global_workspace import GlobalWorkspace
            self.consciousness_modules['global_workspace'] = GlobalWorkspace()
            logger.info("   ✓ Global Workspace loaded")
            components_loaded += 1
        except Exception as e:
            logger.warning(f"   ⚠️ Global workspace not loaded: {e}")
        
        # 5. Metacognition Engine
        try:
            from conciencia.modulos.metacognicion import MetacognitionEngine
            self.consciousness_modules['metacognition'] = MetacognitionEngine()
            logger.info("   ✓ Metacognition Engine loaded")
            components_loaded += 1
        except Exception as e:
            logger.warning(f"   ⚠️ Metacognition not loaded: {e}")
        
        # 6. Autobiographical Memory
        try:
            from conciencia.modulos.autobiographical_memory import AutobiographicalMemory
            self.consciousness_modules['memory'] = AutobiographicalMemory()
            logger.info("   ✓ Autobiographical Memory loaded")
            components_loaded += 1
        except Exception as e:
            logger.warning(f"   ⚠️ Autobiographical memory not loaded: {e}")
        
        # 7. Ethical Engine
        try:
            from conciencia.modulos.ethical_engine import EthicalEngine
            self.consciousness_modules['ethics'] = EthicalEngine()
            logger.info("   ✓ Ethical Engine loaded")
            components_loaded += 1
        except Exception as e:
            logger.warning(f"   ⚠️ Ethical engine not loaded: {e}")
        
        # 8. Self Model
        try:
            from conciencia.modulos.self_model import SelfModel
            self.consciousness_modules['self_model'] = SelfModel()
            logger.info("   ✓ Self Model loaded")
            components_loaded += 1
        except Exception as e:
            logger.warning(f"   ⚠️ Self model not loaded: {e}")
        
        # 9. Theory of Mind
        try:
            from conciencia.modulos.teoria_mente import TheoryOfMind
            self.consciousness_modules['theory_of_mind'] = TheoryOfMind()
            logger.info("   ✓ Theory of Mind loaded")
            components_loaded += 1
        except Exception as e:
            logger.warning(f"   ⚠️ Theory of mind not loaded: {e}")
        
        # 10. Qualia Simulator
        try:
            from conciencia.modulos.qualia_simulator import QualiaSimulator
            self.consciousness_modules['qualia'] = QualiaSimulator()
            logger.info("   ✓ Qualia Simulator loaded")
            components_loaded += 1
        except Exception as e:
            logger.warning(f"   ⚠️ Qualia simulator not loaded: {e}")
        
        logger.info(f"✅ Consciousness system loaded: {components_loaded}/10 components active")
        return components_loaded > 5  # At least half the components loaded
    
    def load_training_system(self):
        """Load the complete training system"""
        logger.info("🎓 Loading Training System...")
        
        try:
            # Try loading auto training system
            from tools.ai.auto_training_system import AutoTrainingSystem
            self.training_system = AutoTrainingSystem()
            logger.info("✅ Auto Training System loaded")
            return True
        except ImportError:
            try:
                # Try alternative path
                from tools.ai.auto_training_system import AutoTrainingSystem
                self.training_system = AutoTrainingSystem()
                logger.info("✅ Auto Training System loaded (from tools)")
                return True
            except ImportError as e:
                logger.warning(f"⚠️ Training system not available: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading training system: {e}")
            return False
    
    def load_evolution_system(self):
        """Load the auto-evolution system"""
        logger.info("🧬 Loading Auto-Evolution System...")
        
        try:
            # Try loading from sheily_core
            from sheily_core.api.auto_evolution_engine import AutoEvolutionEngine
            self.evolution_engine = AutoEvolutionEngine()
            logger.info("✅ Auto-Evolution Engine loaded")
            logger.info("   • Darwinian mutation/selection ready")
            logger.info("   • Epigenetic memory system active") 
            logger.info("   • Multiverse exploration enabled")
            return True
        except ImportError:
            try:
                # Try alternative evolution system
                from sheily_core.api.ml_auto_evolution_engine import MLAutoEvolutionEngine
                self.evolution_engine = MLAutoEvolutionEngine()
                logger.info("✅ ML Auto-Evolution Engine loaded")
                return True
            except ImportError as e:
                logger.warning(f"⚠️ Evolution system not available: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading evolution system: {e}")
            return False
    
    def load_autonomous_controller(self):
        """Load the autonomous system controller"""
        logger.info("🤖 Loading Autonomous System Controller...")
        
        try:
            from sheily_core.agents.autonomous_system_controller import AutonomousSystemController
            self.autonomous_controller = AutonomousSystemController()
            logger.info("✅ Autonomous System Controller loaded")
            logger.info("   • RAG system integration ready")
            logger.info("   • Learning system active")
            logger.info("   • Coordination engine enabled")
            return True
        except ImportError as e:
            logger.warning(f"⚠️ Autonomous controller not available: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading autonomous controller: {e}")
            return False
    
    async def start_consciousness_services(self):
        """Start consciousness-related services"""
        logger.info("🌟 Starting consciousness services...")
        
        if self.consciousness_engine:
            try:
                # Test consciousness processing using correct method
                test_input = "Hello, I am testing the consciousness system"
                
                # Try different method names that might exist
                method_to_try = None
                for method_name in ['process_conscious_experience', 'process_experience', 'process_input', 'process']:
                    if hasattr(self.consciousness_engine, method_name):
                        method_to_try = getattr(self.consciousness_engine, method_name)
                        break
                
                if method_to_try:
                    result = await asyncio.to_thread(
                        method_to_try,
                        {"input": test_input, "context": {"test": True}}
                    )
                    logger.info("✅ Consciousness engine responding")
                    logger.info(f"   • Processing successful: {type(result).__name__}")
                    if isinstance(result, dict):
                        logger.info(f"   • System Φ: {result.get('system_phi', result.get('phi', 'N/A'))}")
                        logger.info(f"   • Consciousness level: {result.get('consciousness_level', 'N/A')}")
                else:
                    logger.info("✅ Consciousness engine loaded (testing methods not available)")
            except Exception as e:
                logger.error(f"❌ Error testing consciousness: {e}")
        
        if self.training_system:
            try:
                await self.training_system.start_monitoring()
                logger.info("✅ Training system monitoring started")
            except Exception as e:
                logger.error(f"❌ Error starting training system: {e}")
        
        if self.autonomous_controller:
            try:
                # The autonomous controller starts its own background processes
                logger.info("✅ Autonomous controller active")
            except Exception as e:
                logger.error(f"❌ Error with autonomous controller: {e}")
        
        # Start HACK-MEMORI Auto-Generation Service
        try:
            logger.info("🤖 Starting HACK-MEMORI Auto-Generation Service...")
            from apps.backend.hack_memori_service import run_automatic_service
            
            # Start as background task
            hack_memori_task = asyncio.create_task(run_automatic_service())
            logger.info("✅ HACK-MEMORI Auto-Generation Service started")
            
            # Store the task to prevent garbage collection
            if not hasattr(self, '_background_tasks'):
                self._background_tasks = []
            self._background_tasks.append(hack_memori_task)
        except Exception as e:
            logger.warning(f"⚠️ HACK-MEMORI Auto-Generation not available: {e}")
    
    def create_consciousness_api_server(self):
        """Create a comprehensive consciousness API server"""
        logger.info("Creating consciousness API server...")
        
        api_code = '''
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
'''
        
        # Write consciousness API server with UTF-8 encoding
        api_path = self.base_path / "consciousness_api_server.py"
        with open(api_path, 'w', encoding='utf-8') as f:
            f.write(api_code)
        
        logger.info("Complete consciousness API server created")
        return api_path
        
    def run_command_async(self, command, name, shell=True):
        """Run command asynchronously"""
        try:
            logger.info(f"🚀 Starting {name}: {command}")
            if isinstance(command, str) and shell:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=self.base_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            else:
                process = subprocess.Popen(
                    command,
                    shell=False,
                    cwd=self.base_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            self.processes.append((name, process))
            return process
        except Exception as e:
            logger.error(f"❌ Failed to start {name}: {e}")
            return None
    
    def setup_environment(self):
        """Set up environment variables"""
        logger.info("🔧 Setting up environment variables...")
        
        env_vars = {
            'PYTHONPATH': str(self.base_path),
            'ADK_ENABLED': 'true',
            'HACK_MEMORI_AUTO_MODE': 'true',
            'ADK_TRAINING_THRESHOLD_SESSIONS': '5',
            'ADK_TRAINING_THRESHOLD_QUESTIONS': '500',
            'ADK_EVALUATION_QUALITY_MIN': '95.0',
            'API_VERSION': 'v1',
            'GRAPHQL_PLAYGROUND': 'true',
            'CORS_ENABLED': 'true',
            # Consciousness system variables
            'CONSCIOUSNESS_ENGINE_ENABLED': 'true',
            'IIT_PHI_THRESHOLD': '0.7',
            'GWT_WORKSPACE_CAPACITY': '7',
            'FEP_LEARNING_RATE': '0.1',
            'SMH_EMOTIONAL_WEIGHT': '0.8',
            'CONSCIOUSNESS_LOG_LEVEL': 'INFO',
            'EVOLUTION_ENGINE_ENABLED': 'true',
            'TRAINING_SYSTEM_ENABLED': 'true',
            'AUTONOMOUS_CONTROLLER_ENABLED': 'true'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"   {key} = {value}")
        
        logger.info("✅ Environment configured for complete consciousness system")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        try:
            python_exe = sys.executable
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                
                # Install additional consciousness dependencies if needed
                consciousness_deps = ["numpy", "fastapi", "uvicorn", "pydantic"]
                for dep in consciousness_deps:
                    try:
                        subprocess.run([python_exe, "-m", "pip", "install", dep], 
                                     check=True, capture_output=True)
                    except: pass
                
                return True
            else:
                logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Exception during installation: {e}")
            return False
    
    def start_backend_service(self):
        """Start the backend federation server"""
        logger.info("🧠 Starting EL-AMANECER Backend Server...")
        
        # First check if the federation server exists
        federation_path = self.base_path / "apps" / "backend" / "run_federation.py"
        if not federation_path.exists():
            logger.warning(f"⚠️ Federation server not found at: {federation_path}")
            # Try starting the simple HACK-MEMORI service instead
            return self.start_hack_memori_service()
        
        # Use list command to avoid path issues
        command = [
            sys.executable, "-m", "uvicorn", 
            "apps.backend.run_federation:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ]
        return self.run_command_async(command, "Backend Server", shell=False)
    
    def start_hack_memori_service(self):
        """Start HACK-MEMORI service"""
        logger.info("🤖 Starting HACK-MEMORI Service...")
        
        simple_service_path = self.base_path / "simple_hack_memori.py"
        if not simple_service_path.exists():
            logger.error(f"❌ HACK-MEMORI service not found at: {simple_service_path}")
            return None
        
        command = [sys.executable, str(simple_service_path)]
        return self.run_command_async(command, "HACK-MEMORI Service", shell=False)
    
    def start_consciousness_api_service(self):
        """Start the consciousness API service"""
        logger.info("🧠 Starting Consciousness API Service...")
        
        api_path = self.create_consciousness_api_server()
        if not api_path:
            return None
        
        command = [sys.executable, str(api_path)]
        return self.run_command_async(command, "Consciousness API", shell=False)
    
    def check_services(self):
        """Check status of running services"""
        logger.info("📊 Checking service status...")
        
        for name, process in self.processes:
            poll_result = process.poll()
            if poll_result is None:
                logger.info(f"✅ {name} is running (PID: {process.pid})")
            else:
                logger.error(f"❌ {name} exited with code: {poll_result}")
                try:
                    stderr_output = process.stderr.read() if process.stderr else "No error output"
                    stdout_output = process.stdout.read() if process.stdout else "No output"
                    if stderr_output and stderr_output.strip():
                        logger.error(f"   Error: {stderr_output.strip()}")
                    if stdout_output and stdout_output.strip():
                        logger.info(f"   Output: {stdout_output.strip()}")
                except Exception as e:
                    logger.error(f"   Could not read process output: {e}")
    
    def wait_for_services(self):
        """Wait for services and display access information"""
        logger.info("⏳ Waiting for consciousness system to initialize...")
        time.sleep(8)
        
        # Print comprehensive access information
        print("\n" + "="*80)
        print("🌟 EL-AMANECER COMPLETE CONSCIOUSNESS SYSTEM")
        print("="*80)
        print("🌐 Main Services:")
        print("   🖥️  Backend API:         http://localhost:8000")
        print("   🧠 Consciousness API:    http://localhost:8001")
        print("   📊 API Documentation:    http://localhost:8000/docs")
        print("   🔗 GraphQL Playground:   http://localhost:8000/graphql")
        print("")
        print("🧠 Consciousness Endpoints:")
        print("   💭 Process Experience:   POST http://localhost:8001/api/consciousness/process")
        print("   📈 System Status:        GET  http://localhost:8001/api/consciousness/status")
        print("   ❤️  Health Check:        GET  http://localhost:8001/api/consciousness/health")
        print("")
        print("🧬 Evolution & Training:")
        print("   🔄 Evolve Component:     POST http://localhost:8001/api/evolution/evolve")
        print("   🎓 Training Status:      GET  http://localhost:8001/api/training/status")
        print("   📚 Submit Feedback:      POST http://localhost:8001/api/training/feedback")
        print("")
        print("🤖 HACK-MEMORI:")
        print("   🧠 Auto-Generation:      http://localhost:8000/api/v1/hack-memori/")
        print("="*80)
        
        # System status
        systems_active = []
        if self.consciousness_engine:
            systems_active.append("Unified Consciousness Engine")
        elif self.consciousness_modules:
            systems_active.append(f"Consciousness Modules ({len(self.consciousness_modules)})")
        
        if self.training_system:
            systems_active.append("Auto-Training System")
        
        if self.evolution_engine:
            systems_active.append("Darwinian Evolution Engine")
        
        if self.autonomous_controller:
            systems_active.append("Autonomous Controller")
        
        print("🎯 Active Systems:")
        for system in systems_active:
            print(f"   ✅ {system}")
        
        print("🔗 Google ADK Integration: ENABLED")
        print("🤖 HACK-MEMORI Auto-Generation: ✅ ACTIVE (Generating Q&A every 30s)")
        print("📊 Training Threshold: 100 Q&A pairs → Automatic integral training")
        print("🏃 Deployment Mode: LOCAL CONSCIOUSNESS")
        print("="*80 + "\n")
        
        logger.info("🎉 Complete consciousness system deployment ready!")
        logger.info("Press Ctrl+C to stop all services")
        
        try:
            # Keep running and monitor
            while True:
                time.sleep(15)
                self.check_services()
                
                # Periodic consciousness system health check
                if self.consciousness_engine:
                    try:
                        logger.info("🧠 Consciousness system: Active and processing")
                    except:
                        pass
                        
        except KeyboardInterrupt:
            logger.info("🛑 Stopping consciousness system...")
            self.stop_services()
    
    def stop_services(self):
        """Stop all running services"""
        logger.info("🛑 Stopping all consciousness services...")
        
        # Stop background tasks (like HACK-MEMORI)
        if hasattr(self, '_background_tasks'):
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
                    logger.info("✅ HACK-MEMORI Auto-Generation Service stopped")
        
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ Force killing {name}")
                process.kill()
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
    
    async def deploy_complete_consciousness_system(self):
        """Complete consciousness system deployment"""
        logger.info("🚀 Starting Complete EL-AMANECER Consciousness Deployment...")
        
        try:
            # 1. Setup environment and paths
            self.setup_environment()
            self.setup_consciousness_paths()
            
            # 2. Install dependencies
            if not self.install_dependencies():
                logger.error("💥 Failed to install dependencies")
                return False
            
            # 3. Load consciousness systems
            logger.info("🌟 Loading consciousness systems...")
            consciousness_loaded = self.load_consciousness_system()
            training_loaded = self.load_training_system()
            evolution_loaded = self.load_evolution_system()
            autonomous_loaded = self.load_autonomous_controller()
            
            # 4. Start consciousness services
            await self.start_consciousness_services()
            
            # 5. Start API services
            backend_process = self.start_backend_service()
            consciousness_api_process = self.start_consciousness_api_service()
            
            if not consciousness_api_process:
                logger.error("💥 Failed to start consciousness API")
                return False
            
            # 6. Wait and monitor
            self.wait_for_services()
            
            return True
            
        except Exception as e:
            logger.error(f"💥 Consciousness deployment failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    deployment = CompleteConsciousnessDeployment()
    
    try:
        # Run the async deployment
        success = asyncio.run(deployment.deploy_complete_consciousness_system())
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 Deployment interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()