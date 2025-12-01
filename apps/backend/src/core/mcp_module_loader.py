"""
MCP Module Loader
Centralizes loading of optional modules for MCP Orchestrator
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MCPModuleLoader:
    """
    Loads and manages optional modules for MCP system
    """
    
    def __init__(self):
        self.loaded_modules = {}
        self.module_status = {}
        
    def load_theory_of_mind(self) -> Optional[Any]:
        """Load Theory of Mind module from consciousness package"""
        try:
            # Add consciousness package to path
            project_root = Path(__file__).parent.parent.parent.parent.parent
            consciousness_path = project_root / "packages" / "consciousness" / "src"

            # Debug: log the computed path
            print(f"[MCP-LOADER] Looking for consciousness at: {consciousness_path}")
            print(f"[MCP-LOADER] Absolute path: {consciousness_path.absolute()}")
            print(f"[MCP-LOADER] Current dir context: {Path.cwd()}")

            # Also try alternative path structure
            alt_consciousness_path = project_root / "packages" / "consciousness" / "src" / "conciencia"
            print(f"[MCP-LOADER] Alternative path: {alt_consciousness_path}")
            
            if not consciousness_path.exists():
                logger.warning(f"Consciousness package not found at {consciousness_path}")
                self.module_status["theory_of_mind"] = {
                    "loaded": False,
                    "error": "Package path not found"
                }
                return None
                
            if str(consciousness_path) not in sys.path:
                sys.path.insert(0, str(consciousness_path))
            
            from conciencia.modulos.teoria_mente import get_unified_tom
            
            # Initialize ToM
            unified_tom = get_unified_tom(enable_advanced=True)
            self.loaded_modules["theory_of_mind"] = unified_tom
            self.module_status["theory_of_mind"] = {
                "loaded": True,
                "has_advanced": unified_tom.has_advanced_capabilities,
                "version": "1.0.0",
                "location": str(consciousness_path)
            }
            
            logger.info("[OK] MCP loaded Theory of Mind from packages/consciousness")
            return unified_tom
            
        except ImportError as e:
            logger.warning(f"Could not load Theory of Mind: {e}")
            self.module_status["theory_of_mind"] = {
                "loaded": False,
                "error": str(e)
            }
            return None
        except Exception as e:
            logger.error(f"Error loading Theory of Mind: {e}")
            self.module_status["theory_of_mind"] = {
                "loaded": False,
                "error": str(e)
            }
            return None
    
    def load_training_system(self) -> Optional[Any]:
        """Load training system for uploads"""
        try:
            # Try to load from sheily_core or training-system package
            project_root = Path(__file__).parent.parent.parent.parent.parent
            training_paths = [
                project_root / "packages" / "training-system" / "src",
                project_root / "packages" / "sheily_core" / "src"
            ]
            
            for training_path in training_paths:
                if training_path.exists():
                    if str(training_path) not in sys.path:
                        sys.path.insert(0, str(training_path))
                    break
            
            # Try importing training system
            try:
                from sheily_core.training import QLoRAFinetuningPipeline
                training_system = QLoRAFinetuningPipeline()
                
                self.loaded_modules["training_system"] = training_system
                self.module_status["training_system"] = {
                    "loaded": True,
                    "version": "1.0.0"
                }
                logger.info("[OK] MCP loaded Training System")
                return training_system
            except ImportError:
                # Training system not available
                self.module_status["training_system"] = {
                    "loaded": False,
                    "error": "Optional training system not installed"
                }
                return None
                
        except Exception as e:
            logger.warning(f"Training system not available: {e}")
            self.module_status["training_system"] = {
                "loaded": False,
                "error": str(e)
            }
            return None
    
    def load_sheily_core(self) -> Optional[Any]:
        """Load sheily_core package for dashboard and other features"""
        try:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            sheily_path = project_root / "packages" / "sheily_core" / "src"
            
            if not sheily_path.exists():
                logger.info(f"Sheily core package not found at {sheily_path}")
                self.module_status["sheily_core"] = {
                    "loaded": False,
                    "error": "Package path not found"
                }
                return None
            
            if str(sheily_path) not in sys.path:
                sys.path.insert(0, str(sheily_path))
            
            # Try importing basic sheily_core module
            import sheily_core
            
            self.loaded_modules["sheily_core"] = sheily_core
            self.module_status["sheily_core"] = {
                "loaded": True,
                "version": getattr(sheily_core, '__version__', '1.0.0'),
                "location": str(sheily_path)
            }
            
            logger.info("[OK] MCP loaded Sheily Core package")
            return sheily_core
            
        except ImportError as e:
            logger.info(f"Sheily Core not available: {e}")
            self.module_status["sheily_core"] = {
                "loaded": False,
                "error": str(e)
            }
            return None
        except Exception as e:
            logger.warning(f"Error loading Sheily Core: {e}")
            self.module_status["sheily_core"] = {
                "loaded": False,
                "error": str(e)
            }
            return None
    
    def get_module_status(self) -> Dict[str, Any]:
        """Get status of all loaded modules"""
        return {
            "total_modules": len(self.module_status),
            "loaded_count": sum(1 for m in self.module_status.values() if m.get("loaded")),
            "modules": self.module_status
        }
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a loaded module by name"""
        return self.loaded_modules.get(module_name)


# Global module loader
mcp_module_loader = MCPModuleLoader()

# Auto-load modules on import
mcp_module_loader.load_theory_of_mind()
mcp_module_loader.load_sheily_core()
mcp_module_loader.load_training_system()

__all__ = ["mcp_module_loader", "MCPModuleLoader"]
