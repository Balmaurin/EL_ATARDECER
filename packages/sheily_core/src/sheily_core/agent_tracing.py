"""
Agent Tracing Module - REAL Implementation
Comprehensive tracing with actual event logging and metrics collection
"""
import contextlib
import logging
import time
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """Trace event data structure"""
    name: str
    timestamp: datetime
    data: Dict[str, Any]
    duration_ms: Optional[float] = None


@dataclass
class Trace:
    """Trace object with real event tracking"""
    agent_name: str
    operation: str
    trace_id: str
    start_time: datetime
    events: List[TraceEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_event(self, name: str, data: Dict[str, Any] = None):
        """REAL event logging"""
        event = TraceEvent(
            name=name,
            timestamp=datetime.now(),
            data=data or {}
        )
        self.events.append(event)
        logger.debug(f"Trace event [{self.trace_id}]: {name} - {data}")
    
    def add_timed_event(self, name: str, start_time: float, data: Dict[str, Any] = None):
        """Add event with duration"""
        duration_ms = (time.time() - start_time) * 1000
        event = TraceEvent(
            name=name,
            timestamp=datetime.now(),
            data=data or {},
            duration_ms=duration_ms
        )
        self.events.append(event)
        logger.debug(f"Trace event [{self.trace_id}]: {name} ({duration_ms:.2f}ms) - {data}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary"""
        return {
            "agent_name": self.agent_name,
            "operation": self.operation,
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_ms": (datetime.now() - self.start_time).total_seconds() * 1000,
            "events": [
                {
                    "name": event.name,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.data,
                    "duration_ms": event.duration_ms
                }
                for event in self.events
            ],
            "metadata": self.metadata
        }
    
    def save_to_file(self, trace_dir: Optional[Path] = None):
        """Save trace to file for analysis"""
        if trace_dir is None:
            trace_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "traces"
        
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / f"{self.trace_id}.json"
        
        try:
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            logger.debug(f"Trace saved to {trace_file}")
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")


class TraceManager:
    """Manager for agent traces"""
    
    def __init__(self, save_traces: bool = True):
        self.save_traces = save_traces
        self.active_traces: Dict[str, Trace] = {}
        logger.info("✅ TraceManager initialized")
    
    def create_trace(self, agent_name: str, operation: str, trace_id: Optional[str] = None) -> Trace:
        """Create a new trace"""
        import uuid
        if trace_id is None:
            trace_id = f"{agent_name}_{operation}_{uuid.uuid4().hex[:8]}"
        
        trace = Trace(
            agent_name=agent_name,
            operation=operation,
            trace_id=trace_id,
            start_time=datetime.now()
        )
        self.active_traces[trace_id] = trace
        return trace
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get active trace by ID"""
        return self.active_traces.get(trace_id)
    
    def finalize_trace(self, trace_id: str):
        """Finalize and save trace"""
        trace = self.active_traces.pop(trace_id, None)
        if trace and self.save_traces:
            trace.save_to_file()
        return trace


# Global trace manager
_trace_manager: Optional[TraceManager] = None


def get_trace_manager() -> TraceManager:
    """Get global trace manager instance"""
    global _trace_manager
    if _trace_manager is None:
        _trace_manager = TraceManager()
    return _trace_manager


@contextlib.contextmanager
def trace_agent_execution(agent_name: str, operation: str, trace_id: Optional[str] = None):
    """
    REAL tracing context manager for agent execution
    
    Args:
        agent_name: Name of the agent being traced
        operation: Name of the operation being performed
        trace_id: Optional trace ID (generated if not provided)
        
    Yields:
        Trace object with add_event method
    """
    manager = get_trace_manager()
    trace = manager.create_trace(agent_name, operation, trace_id)
    
    try:
        logger.info(f"🔍 Starting trace [{trace.trace_id}] for {agent_name}.{operation}")
        yield trace
    except Exception as e:
        trace.add_event("error", {"error": str(e), "error_type": type(e).__name__})
        logger.error(f"Error in trace [{trace.trace_id}]: {e}", exc_info=True)
        raise
    finally:
        manager.finalize_trace(trace.trace_id)
        total_duration = (datetime.now() - trace.start_time).total_seconds() * 1000
        logger.info(f"✅ Completed trace [{trace.trace_id}] in {total_duration:.2f}ms ({len(trace.events)} events)")


__all__ = ['trace_agent_execution', 'Trace', 'TraceEvent', 'TraceManager', 'get_trace_manager']
