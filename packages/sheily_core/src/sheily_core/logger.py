"""
Simple logger wrapper for sheily_core
"""
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class LogContext:
    """Context for structured logging"""
    component: str
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextLogger(logging.Logger):
    """Logger with REAL context support - tracks context metadata"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context_stack: list[Dict[str, Any]] = []
    
    @contextmanager
    def context(self, **kwargs):
        """REAL context manager for logging - tracks and applies context"""
        # Push context to stack
        context_data = kwargs.copy()
        context_data['_entered_at'] = __import__('datetime').datetime.now()
        self._context_stack.append(context_data)
        
        try:
            # Create adapter that includes context in all log records
            old_factory = logging.getLogRecordFactory()
            
            def record_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                # Add context to record
                if self._context_stack:
                    for ctx in self._context_stack:
                        for key, value in ctx.items():
                            if not key.startswith('_'):
                                setattr(record, f"ctx_{key}", value)
                return record
            
            logging.setLogRecordFactory(record_factory)
            
            yield self
            
        finally:
            # Pop context from stack
            if self._context_stack:
                self._context_stack.pop()
            
            # Restore original factory
            logging.setLogRecordFactory(old_factory)
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get current logging context - REAL implementation"""
        if not self._context_stack:
            return {}
        
        # Merge all contexts in stack
        merged = {}
        for ctx in self._context_stack:
            merged.update({k: v for k, v in ctx.items() if not k.startswith('_')})
        return merged

def get_logger(name: str) -> ContextLogger:
    """Get a logger with the given name"""
    # Get the logger and change its class
    logger = logging.getLogger(name)
    logger.__class__ = ContextLogger
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
