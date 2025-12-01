"""
MCP Coordinator - Coordinación del protocolo MCP (Model Context Protocol)
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """Types of MCP messages"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPStatus(Enum):
    """Status of MCP connection"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPMessage:
    """MCP protocol message"""
    message_id: str
    message_type: MCPMessageType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPConnection:
    """MCP connection information"""
    connection_id: str
    endpoint: str
    status: MCPStatus = MCPStatus.DISCONNECTED
    connected_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0


class MCPCoordinator:
    """
    Coordinator for Model Context Protocol
    Manages connections, message routing, and protocol compliance
    """
    
    def __init__(self, node_id: str = "sheily_node"):
        self.node_id = node_id
        self.connections: Dict[str, MCPConnection] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[str, List[callable]] = {}
        self.message_history: List[MCPMessage] = []
        self.is_running = False
        
        logger.info(f"MCPCoordinator initialized for node: {node_id}")
    
    # ==================== CONNECTION MANAGEMENT ====================
    
    async def connect(self, endpoint: str, connection_id: Optional[str] = None) -> str:
        """Establish MCP connection"""
        if connection_id is None:
            connection_id = f"conn_{len(self.connections) + 1}"
        
        if connection_id in self.connections:
            logger.warning(f"Connection {connection_id} already exists")
            return connection_id
        
        connection = MCPConnection(
            connection_id=connection_id,
            endpoint=endpoint,
            status=MCPStatus.CONNECTING
        )
        
        self.connections[connection_id] = connection
        
        try:
            # Simulate connection establishment
            await asyncio.sleep(0.1)
            
            connection.status = MCPStatus.CONNECTED
            connection.connected_at = datetime.now()
            connection.last_activity = datetime.now()
            
            logger.info(f"Connected to {endpoint} (connection: {connection_id})")
            
            return connection_id
            
        except Exception as e:
            connection.status = MCPStatus.ERROR
            logger.error(f"Failed to connect to {endpoint}: {e}")
            raise
    
    async def disconnect(self, connection_id: str) -> bool:
        """Close MCP connection"""
        if connection_id not in self.connections:
            logger.warning(f"Connection {connection_id} not found")
            return False
        
        connection = self.connections[connection_id]
        connection.status = MCPStatus.DISCONNECTED
        
        logger.info(f"Disconnected from {connection.endpoint}")
        
        return True
    
    def get_connection(self, connection_id: str) -> Optional[MCPConnection]:
        """Get connection by ID"""
        return self.connections.get(connection_id)
    
    def list_connections(self) -> List[MCPConnection]:
        """List all connections"""
        return list(self.connections.values())
    
    def get_active_connections(self) -> List[MCPConnection]:
        """Get all active connections"""
        return [
            conn for conn in self.connections.values()
            if conn.status == MCPStatus.CONNECTED
        ]
    
    # ==================== MESSAGE HANDLING ====================
    
    async def send_message(self, connection_id: str, message_type: MCPMessageType,
                          payload: Dict[str, Any], 
                          metadata: Optional[Dict] = None) -> str:
        """Send MCP message"""
        if connection_id not in self.connections:
            raise ValueError(f"Connection {connection_id} not found")
        
        connection = self.connections[connection_id]
        
        if connection.status != MCPStatus.CONNECTED:
            raise RuntimeError(f"Connection {connection_id} not connected")
        
        # Create message
        message_id = f"msg_{connection_id}_{connection.messages_sent + 1}"
        
        message = MCPMessage(
            message_id=message_id,
            message_type=message_type,
            payload=payload,
            metadata=metadata or {}
        )
        
        # Send message (simulated)
        await self._transmit_message(connection, message)
        
        # Update connection stats
        connection.messages_sent += 1
        connection.last_activity = datetime.now()
        
        # Record in history
        self.message_history.append(message)
        
        logger.debug(f"Sent {message_type.value} message: {message_id}")
        
        return message_id
    
    async def receive_message(self, connection_id: str, timeout: float = 5.0) -> Optional[MCPMessage]:
        """Receive MCP message"""
        if connection_id not in self.connections:
            raise ValueError(f"Connection {connection_id} not found")
        
        connection = self.connections[connection_id]
        
        try:
            # Simulate receiving message
            message = await asyncio.wait_for(
                self._receive_from_connection(connection),
                timeout=timeout
            )
            
            # Update connection stats
            connection.messages_received += 1
            connection.last_activity = datetime.now()
            
            # Record in history
            self.message_history.append(message)
            
            # Trigger handlers
            await self._trigger_handlers(message)
            
            logger.debug(f"Received {message.message_type.value} message: {message.message_id}")
            
            return message
            
        except asyncio.TimeoutError:
            logger.debug(f"No message received within {timeout}s")
            return None
    
    async def _transmit_message(self, connection: MCPConnection, message: MCPMessage):
        """Transmit message over connection (simulated)"""
        # In real implementation, would send over network
        await asyncio.sleep(0.01)  # Simulate network delay
    
    async def _receive_from_connection(self, connection: MCPConnection) -> MCPMessage:
        """Receive message from connection (simulated)"""
        # In real implementation, would receive from network
        # For now, create a mock response
        await asyncio.sleep(0.1)
        
        message_id = f"msg_{connection.connection_id}_recv_{connection.messages_received + 1}"
        
        return MCPMessage(
            message_id=message_id,
            message_type=MCPMessageType.RESPONSE,
            payload={'status': 'ok', 'data': 'mock_response'}
        )
    
    # ==================== MESSAGE ROUTING ====================
    
    def register_handler(self, message_type: str, handler: callable):
        """Register message handler"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        
        logger.debug(f"Registered handler for {message_type}")
    
    def unregister_handler(self, message_type: str, handler: callable):
        """Unregister message handler"""
        if message_type in self.message_handlers:
            try:
                self.message_handlers[message_type].remove(handler)
                logger.debug(f"Unregistered handler for {message_type}")
            except ValueError:
                pass
    
    async def _trigger_handlers(self, message: MCPMessage):
        """Trigger registered handlers for message"""
        message_type = message.message_type.value
        
        if message_type in self.message_handlers:
            for handler in self.message_handlers[message_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Handler error for {message_type}: {e}")
    
    # ==================== REQUEST/RESPONSE ====================
    
    async def request(self, connection_id: str, request_data: Dict[str, Any],
                     timeout: float = 10.0) -> Dict[str, Any]:
        """Send request and wait for response"""
        # Send request
        message_id = await self.send_message(
            connection_id,
            MCPMessageType.REQUEST,
            request_data
        )
        
        # Wait for response
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            response = await self.receive_message(connection_id, timeout=1.0)
            
            if response and response.message_type == MCPMessageType.RESPONSE:
                # Check if this is response to our request
                if response.metadata.get('request_id') == message_id:
                    return response.payload
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"No response received within {timeout}s")
    
    async def respond(self, connection_id: str, request_message: MCPMessage,
                     response_data: Dict[str, Any]):
        """Send response to a request"""
        metadata = {'request_id': request_message.message_id}
        
        await self.send_message(
            connection_id,
            MCPMessageType.RESPONSE,
            response_data,
            metadata=metadata
        )
    
    async def notify(self, connection_id: str, notification_data: Dict[str, Any]):
        """Send notification (no response expected)"""
        await self.send_message(
            connection_id,
            MCPMessageType.NOTIFICATION,
            notification_data
        )
    
    # ==================== BROADCAST ====================
    
    async def broadcast(self, message_type: MCPMessageType, payload: Dict[str, Any]):
        """Broadcast message to all active connections"""
        active_connections = self.get_active_connections()
        
        tasks = [
            self.send_message(conn.connection_id, message_type, payload)
            for conn in active_connections
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"Broadcast to {success_count}/{len(active_connections)} connections")
        
        return success_count
    
    # ==================== MONITORING ====================
    
    def get_connection_stats(self, connection_id: str) -> Optional[Dict]:
        """Get connection statistics"""
        if connection_id not in self.connections:
            return None
        
        connection = self.connections[connection_id]
        
        return {
            'connection_id': connection_id,
            'endpoint': connection.endpoint,
            'status': connection.status.value,
            'connected_at': connection.connected_at.isoformat() if connection.connected_at else None,
            'last_activity': connection.last_activity.isoformat() if connection.last_activity else None,
            'messages_sent': connection.messages_sent,
            'messages_received': connection.messages_received,
            'errors': connection.errors
        }
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        total_connections = len(self.connections)
        active_connections = len(self.get_active_connections())
        
        total_messages_sent = sum(c.messages_sent for c in self.connections.values())
        total_messages_received = sum(c.messages_received for c in self.connections.values())
        total_errors = sum(c.errors for c in self.connections.values())
        
        return {
            'node_id': self.node_id,
            'total_connections': total_connections,
            'active_connections': active_connections,
            'total_messages_sent': total_messages_sent,
            'total_messages_received': total_messages_received,
            'total_errors': total_errors,
            'message_history_size': len(self.message_history),
            'is_running': self.is_running
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        active = self.get_active_connections()
        
        # Check each connection
        connection_health = {}
        for conn in active:
            is_healthy = conn.status == MCPStatus.CONNECTED and conn.errors < 10
            connection_health[conn.connection_id] = is_healthy
        
        overall_healthy = all(connection_health.values()) if connection_health else False
        
        return {
            'overall_healthy': overall_healthy,
            'active_connections': len(active),
            'connection_health': connection_health,
            'timestamp': datetime.now().isoformat()
        }
    
    # ==================== LIFECYCLE ====================
    
    async def start(self):
        """Start coordinator"""
        self.is_running = True
        logger.info("MCP Coordinator started")
    
    async def stop(self):
        """Stop coordinator"""
        # Disconnect all connections
        for connection_id in list(self.connections.keys()):
            await self.disconnect(connection_id)
        
        self.is_running = False
        logger.info("MCP Coordinator stopped")
    
    def clear_history(self):
        """Clear message history"""
        self.message_history.clear()
        logger.info("Message history cleared")


# Global coordinator instance
_mcp_coordinator = None

def get_mcp_coordinator(node_id: str = "sheily_node") -> MCPCoordinator:
    """Get global MCP coordinator instance"""
    global _mcp_coordinator
    if _mcp_coordinator is None:
        _mcp_coordinator = MCPCoordinator(node_id)
    return _mcp_coordinator
