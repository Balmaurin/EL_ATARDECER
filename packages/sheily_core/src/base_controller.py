"""
Base controller or unified consciousness components
"""
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

class BaseController:
    """Base controller for unified consciousness components"""

    def __init__(self, name: str = "base_controller"):
        from datetime import datetime
        self.name = name
        self.logger = logging.getLogger(self.name)
        self._start_time = datetime.now()  # Track start time for uptime calculation
        self._file_handles = []  # Track open file handles for cleanup
        self._cache = {}  # Cache storage

    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from directory"""
        try:
            config_path = Path(__file__).parent / "config" / "system" / "base_controller.json"
            if config_path.exists():
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            self.logger.warning(f"Config file not found {config_path}")
            return {}

    def save_config(self, config: Dict[str, Any], config_file: Optional[str] = None) -> bool:
        """Save configuration to directory"""
        try:
            config_path = Path(__file__).parent / "config" / "system" / "base_controller.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"Config saved to {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving config {config_file} {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """REAL health check for controller with actual metrics"""
        from datetime import datetime
        import psutil
        import os
        
        try:
            # Get real system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            return {
                "controller": self.name,
                "status": "operational",
                "timestamp": datetime.now().isoformat(),  # REAL timestamp
                "system_metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_total_gb": round(memory.total / (1024**3), 2),
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "memory_percent": memory.percent,
                    "process_memory_mb": round(process_memory.rss / (1024**2), 2),
                },
                "uptime_seconds": (datetime.now() - getattr(self, '_start_time', datetime.now())).total_seconds(),
            }
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {
                "controller": self.name,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def cleanup(self):
        """REAL cleanup - release resources when shutting down"""
        self.logger.info(f"Cleaning up {self.name}")
        
        # Close any open file handles
        if hasattr(self, '_file_handles'):
            for handle in self._file_handles:
                try:
                    handle.close()
                except Exception as e:
                    self.logger.warning(f"Error closing file handle: {e}")
            self._file_handles.clear()
        
        # Clear caches
        if hasattr(self, '_cache'):
            self._cache.clear()
        
        # Close database connections if any
        if hasattr(self, '_db_connection'):
            try:
                self._db_connection.close()
            except Exception as e:
                self.logger.warning(f"Error closing database connection: {e}")
        
        self.logger.info(f"Cleanup completed for {self.name}")
