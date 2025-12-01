"""
REAL Security Module for sheily_core
Comprehensive security validation without fallbacks or mocks
"""
import re
import logging
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class SecurityMonitor:
    """
    Real security monitoring system with comprehensive threat detection
    """
    
    def __init__(self):
        # Dangerous command patterns (REAL comprehensive list)
        self.dangerous_commands = [
            # System commands
            r'\brm\s+-rf\b',
            r'\brm\s+-\w*rf\b',
            r'\bdel\s+/[sf]\b',  # Windows
            r'\bformat\b',
            r'\bshutdown\b',
            r'\breboot\b',
            r'\bhalt\b',
            r'\bpoweroff\b',
            
            # Database commands
            r'\bdrop\s+table\b',
            r'\bdrop\s+database\b',
            r'\btruncate\s+table\b',
            r'\bdelete\s+from\b.*\bwhere\s+1\s*=\s*1\b',
            r'\bupdate\b.*\bset\b.*\bwhere\s+1\s*=\s*1\b',
            
            # File system operations
            r'\bchmod\s+777\b',
            r'\bchown\b',
            r'\bmkdir\s+/',
            r'\brmdir\s+/',
            
            # Network operations
            r'\bcurl\b.*\bhttp',
            r'\bwget\b.*\bhttp',
            r'\bnc\s+-l\b',
            r'\bncat\s+-l\b',
            
            # Code execution
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\b__import__\s*\(',
            r'\bcompile\s*\(',
            r'\bexecfile\s*\(',
        ]
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"('|(\\')|(;)|(\\;)|(\|)|(\\|)|(\*)|(\\*)|(%)|(\\%)|(\+)|(\\+)|(\[)|(\\\[)|(\])|(\\\]))",
            r"(\bOR\b.*=.*)",
            r"(\bAND\b.*=.*)",
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bSELECT\b.*\bFROM\b)",
            r"(\bINSERT\b.*\bINTO\b)",
            r"(\bUPDATE\b.*\bSET\b)",
            r"(\bDELETE\b.*\bFROM\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"(--|\#|\/\*|\*\/)",
            r"(\bEXEC\b|\bEXECUTE\b)",
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<img[^>]*onerror\s*=',
            r'<svg[^>]*onload\s*=',
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'\.\.%2f',
            r'\.\.%5c',
            r'%2e%2e%2f',
            r'%2e%2e%5c',
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r'[;&|`]',
            r'\$\([^)]+\)',
            r'\$\{[^}]+\}',
            r'`[^`]+`',
        ]
        
        # Rate limiting per client
        self.client_requests: Dict[str, List[datetime]] = defaultdict(list)
        self.rate_limit_window = timedelta(minutes=1)
        self.max_requests_per_window = 60
        
        # Compile patterns for performance
        self.compiled_dangerous = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_commands]
        self.compiled_sql = [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_injection_patterns]
        self.compiled_xss = [re.compile(pattern, re.IGNORECASE) for pattern in self.xss_patterns]
        self.compiled_path = [re.compile(pattern, re.IGNORECASE) for pattern in self.path_traversal_patterns]
        self.compiled_cmd = [re.compile(pattern, re.IGNORECASE) for pattern in self.command_injection_patterns]
        
        logger.info("✅ SecurityMonitor initialized with comprehensive threat detection")
    
    def check_request(self, query: str, client_id: str) -> Tuple[bool, str]:
        """
        REAL security check - comprehensive validation
        
        Returns:
            Tuple[bool, str]: (is_safe, reason)
        """
        # Rate limiting check
        if not self._check_rate_limit(client_id):
            logger.warning(f"Rate limit exceeded for client {client_id}")
            return False, "Rate limit exceeded. Too many requests."
        
        # Sanitize input
        query_lower = query.lower()
        query_clean = query.strip()
        
        # Check dangerous commands
        for pattern in self.compiled_dangerous:
            if pattern.search(query):
                threat = pattern.pattern
                logger.warning(f"⚠️ Dangerous command detected: {threat} from client {client_id}")
                return False, f"Dangerous command pattern detected: {threat}"
        
        # Check SQL injection
        for pattern in self.compiled_sql:
            if pattern.search(query):
                logger.warning(f"⚠️ SQL injection attempt detected from client {client_id}")
                return False, "SQL injection pattern detected"
        
        # Check XSS
        for pattern in self.compiled_xss:
            if pattern.search(query):
                logger.warning(f"⚠️ XSS attempt detected from client {client_id}")
                return False, "XSS pattern detected"
        
        # Check path traversal
        for pattern in self.compiled_path:
            if pattern.search(query):
                logger.warning(f"⚠️ Path traversal attempt detected from client {client_id}")
                return False, "Path traversal pattern detected"
        
        # Check command injection
        for pattern in self.compiled_cmd:
            if pattern.search(query):
                logger.warning(f"⚠️ Command injection attempt detected from client {client_id}")
                return False, "Command injection pattern detected"
        
        # Check for suspicious length (potential buffer overflow)
        if len(query) > 100000:  # 100KB limit
            logger.warning(f"⚠️ Suspiciously long query from client {client_id}: {len(query)} chars")
            return False, "Query too long (potential buffer overflow attempt)"
        
        # Check for null bytes
        if '\x00' in query:
            logger.warning(f"⚠️ Null byte detected in query from client {client_id}")
            return False, "Null byte detected (potential injection attempt)"
        
        # All checks passed
        return True, "safe"
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = datetime.now()
        window_start = now - self.rate_limit_window
        
        # Clean old requests
        self.client_requests[client_id] = [
            req_time for req_time in self.client_requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.client_requests[client_id]) >= self.max_requests_per_window:
            return False
        
        # Record this request
        self.client_requests[client_id].append(now)
        return True
    
    def sanitize_input(self, input_str: str) -> str:
        """
        Sanitize input string - remove dangerous characters
        REAL implementation
        """
        # Remove null bytes
        sanitized = input_str.replace('\x00', '')
        
        # Remove control characters except newlines and tabs
        sanitized = ''.join(
            char for char in sanitized
            if ord(char) >= 32 or char in '\n\t'
        )
        
        # Limit length
        if len(sanitized) > 100000:
            sanitized = sanitized[:100000]
        
        return sanitized
    
    def get_security_stats(self) -> Dict[str, any]:
        """Get real security statistics"""
        total_clients = len(self.client_requests)
        active_clients = sum(
            1 for requests in self.client_requests.values()
            if requests and (datetime.now() - requests[-1]) < self.rate_limit_window
        )
        
        return {
            "total_clients_tracked": total_clients,
            "active_clients": active_clients,
            "rate_limit_window_seconds": self.rate_limit_window.total_seconds(),
            "max_requests_per_window": self.max_requests_per_window,
            "patterns_loaded": {
                "dangerous_commands": len(self.dangerous_commands),
                "sql_injection": len(self.sql_injection_patterns),
                "xss": len(self.xss_patterns),
                "path_traversal": len(self.path_traversal_patterns),
                "command_injection": len(self.command_injection_patterns),
            }
        }


# Global instance
_security_monitor: Optional[SecurityMonitor] = None


def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance - REAL implementation"""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor
