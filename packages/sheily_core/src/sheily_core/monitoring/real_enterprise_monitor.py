"""
Real Enterprise Monitoring - NO SIMULATIONS
Uses actual system metrics and logging
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class RealEnterpriseMonitor:
    """
    Real enterprise monitoring with actual system metrics
    NO MOCKS - Real CPU, memory, disk, network monitoring
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize monitoring
        
        Args:
            log_file: Path to application log file for error rate calculation
        """
        self.log_file_path = self._resolve_log_file(log_file)
        self.log_file = str(self.log_file_path)
        self.metrics_history: List[Dict] = []
        self.start_time = time.time()
        
        logger.info("📊 Real Enterprise Monitor initialized")
    
    def get_system_metrics(self) -> Dict:
        """Get real system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency_mhz": cpu_freq.current if cpu_freq else 0
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent
                },
                "swap": {
                    "total_gb": swap.total / (1024**3),
                    "used_gb": swap.used / (1024**3),
                    "percent": swap.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": disk.percent
                },
                "disk_io": {
                    "read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
                    "write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0
                },
                "network": {
                    "bytes_sent_mb": network.bytes_sent / (1024**2),
                    "bytes_recv_mb": network.bytes_recv / (1024**2),
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "process": {
                    "memory_mb": process_memory.rss / (1024**2),
                    "cpu_percent": process.cpu_percent()
                }
            }
            
            # Store in history
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 metrics
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to get system metrics: {e}")
            return {}
    
    def calculate_error_rate(self, time_window_seconds: int = 60) -> float:
        """
        Calculate error rate from log file
        
        Args:
            time_window_seconds: Time window to analyze
            
        Returns:
            Error rate (0.0 to 1.0)
        """
        try:
            log_path = self.log_file_path
            
            if not log_path.exists():
                logger.warning(f"⚠️ Log file not found: {self.log_file}")
                return 0.0
            
            cutoff_time = time.time() - time_window_seconds
            errors = 0
            total = 0
            
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        # Try to parse as JSON log
                        log_entry = json.loads(line)
                        
                        # Check timestamp
                        if 'timestamp' in log_entry:
                            log_time = datetime.fromisoformat(log_entry['timestamp']).timestamp()
                            if log_time < cutoff_time:
                                continue
                        
                        total += 1
                        
                        # Check for errors
                        if log_entry.get('level') in ['ERROR', 'CRITICAL']:
                            errors += 1
                            
                    except json.JSONDecodeError:
                        # Plain text log, check for ERROR
                        if 'ERROR' in line or 'CRITICAL' in line:
                            errors += 1
                        total += 1
            
            rate = (errors / total) if total > 0 else 0.0
            
            logger.info(f"📊 Error rate: {rate:.2%} ({errors}/{total} in last {time_window_seconds}s)")
            return rate
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate error rate: {e}")
            return 0.0
    
    def get_health_status(self) -> Dict:
        """Get overall system health status"""
        try:
            metrics = self.get_system_metrics()
            error_rate = self.calculate_error_rate()
            
            # Determine health status
            cpu_healthy = metrics.get("cpu", {}).get("percent", 0) < 80
            memory_healthy = metrics.get("memory", {}).get("percent", 0) < 85
            disk_healthy = metrics.get("disk", {}).get("percent", 0) < 90
            errors_healthy = error_rate < 0.05  # Less than 5% errors
            
            overall_healthy = all([cpu_healthy, memory_healthy, disk_healthy, errors_healthy])
            
            status = {
                "healthy": overall_healthy,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "components": {
                    "cpu": "healthy" if cpu_healthy else "warning",
                    "memory": "healthy" if memory_healthy else "warning",
                    "disk": "healthy" if disk_healthy else "warning",
                    "errors": "healthy" if errors_healthy else "critical"
                },
                "metrics": metrics,
                "error_rate": error_rate
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get health status: {e}")
            return {"healthy": False, "error": str(e)}
    
    def export_metrics(self, output_file: str):
        """Export metrics history to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            
            logger.info(f"💾 Metrics exported to: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export metrics: {e}")

    def _resolve_log_file(self, requested: Optional[str]) -> Path:
        """Resolve default log file ensuring the path exists."""
        candidates: List[Path] = []
        if requested:
            candidates.append(Path(requested))
        candidates.extend(
            [
                Path("logs/application.log"),
                Path("logs/sheily_ai.log"),
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        chosen = candidates[0]
        try:
            chosen.parent.mkdir(parents=True, exist_ok=True)
            chosen.touch(exist_ok=True)
        except Exception as exc:  # pragma: no cover - permisos
            logger.warning(f"⚠️ Could not prepare log file {chosen}: {exc}")
        return chosen


# Singleton
_real_enterprise_monitor: Optional[RealEnterpriseMonitor] = None


def get_real_enterprise_monitor(log_file: Optional[str] = None) -> RealEnterpriseMonitor:
    """Get singleton instance"""
    global _real_enterprise_monitor
    
    if _real_enterprise_monitor is None:
        _real_enterprise_monitor = RealEnterpriseMonitor(log_file)
    
    return _real_enterprise_monitor


# Demo
if __name__ == "__main__":
    print("📊 Real Enterprise Monitor Demo")
    print("=" * 50)
    
    monitor = get_real_enterprise_monitor()
    
    # Get metrics
    metrics = monitor.get_system_metrics()
    
    print(f"\n💻 System Metrics:")
    print(f"CPU: {metrics['cpu']['percent']:.1f}%")
    print(f"Memory: {metrics['memory']['percent']:.1f}%")
    print(f"Disk: {metrics['disk']['percent']:.1f}%")
    
    # Health status
    health = monitor.get_health_status()
    print(f"\n🏥 Health: {'✅ Healthy' if health['healthy'] else '⚠️ Warning'}")
