#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Implementation para Sheily AI
Implementa el protocolo completo de interoperabilidad entre agentes y herramientas
Basado en las especificaciones de Anthropic MCP
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp
import websockets

from sheily_core.agent_quality import evaluate_agent_quality
from sheily_core.agent_tracing import trace_agent_execution

logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS DE DATOS MCP
# =============================================================================


class MCPMessageType(Enum):
    """Tipos de mensajes en el protocolo MCP"""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPTransportType(Enum):
    """Tipos de transporte soportados"""

    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"


@dataclass
class MCPMessage:
    """Mensaje base del protocolo MCP"""

    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Convierte el mensaje a JSON"""
        data = {"jsonrpc": self.jsonrpc}
        if self.id:
            data["id"] = self.id
        if self.method:
            data["method"] = self.method
        if self.params:
            data["params"] = self.params
        if self.result is not None:
            data["result"] = self.result
        if self.error:
            data["error"] = self.error
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "MCPMessage":
        """Crea un mensaje desde JSON"""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class MCPTool:
    """Definición de herramienta MCP"""

    name: str
    title: Optional[str] = None
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la herramienta a diccionario"""
        result = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.title:
            result["title"] = self.title
        if self.output_schema:
            result["outputSchema"] = self.output_schema
        if self.annotations:
            result["annotations"] = self.annotations
        return result


@dataclass
class MCPResource:
    """Recurso MCP"""

    uri: str
    name: str
    description: str = ""
    mime_type: Optional[str] = None
    size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el recurso a diccionario"""
        result = {"uri": self.uri, "name": self.name, "description": self.description}
        if self.mime_type:
            result["mimeType"] = self.mime_type
        if self.size:
            result["size"] = self.size
        return result


@dataclass
class MCPPrompt:
    """Prompt MCP"""

    name: str
    description: str = ""
    arguments: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el prompt a diccionario"""
        result = {"name": self.name, "description": self.description}
        if self.arguments:
            result["arguments"] = self.arguments
        return result


# =============================================================================
# MCP SERVER BASE
# =============================================================================


class MCPServer:
    """Servidor base MCP"""

    def __init__(self, server_name: str, version: str = "1.0.0"):
        self.server_name = server_name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.capabilities = {
            "tools": {"listChanged": True},
            "resources": {},
            "prompts": {},
        }
        self.running = False

    def validate_message_quality(self, message: MCPMessage) -> float:
        """Validate message quality and structure"""
        score = 1.0
        
        # Structure check
        if not message.jsonrpc == "2.0": score -= 0.2
        if not message.id: score -= 0.1
        
        # Content check
        if message.method and len(message.method) > 50: score -= 0.1
        if message.params and len(str(message.params)) > 10000: score -= 0.1
        
        return max(0.0, score)

    async def start(self):
        """Inicia el servidor MCP"""
        self.running = True
        logger.info(f"MCP Server '{self.server_name}' started")

    async def stop(self):
        """Detiene el servidor MCP"""
        self.running = False
        logger.info(f"MCP Server '{self.server_name}' stopped")

    def register_tool(self, tool: MCPTool):
        """Registra una herramienta"""
        self.tools[tool.name] = tool
        logger.info(f"Tool registered: {tool.name}")

    def register_resource(self, resource: MCPResource):
        """Registra un recurso"""
        self.resources[resource.uri] = resource
        logger.info(f"Resource registered: {resource.uri}")

    def register_prompt(self, prompt: MCPPrompt):
        """Registra un prompt"""
        self.prompts[prompt.name] = prompt
        logger.info(f"Prompt registered: {prompt.name}")

    async def handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Maneja un mensaje entrante"""
        if message.method == "initialize":
            return await self._handle_initialize(message)
        elif message.method == "tools/list":
            return await self._handle_tools_list(message)
        elif message.method == "tools/call":
            return await self._handle_tools_call(message)
        elif message.method == "resources/list":
            return await self._handle_resources_list(message)
        elif message.method == "resources/read":
            return await self._handle_resources_read(message)
        elif message.method == "prompts/list":
            return await self._handle_prompts_list(message)
        elif message.method == "prompts/get":
            return await self._handle_prompts_get(message)
        else:
            return MCPMessage(
                id=message.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {message.method}",
                },
            )

    async def _handle_initialize(self, message: MCPMessage) -> MCPMessage:
        """Maneja inicialización"""
        return MCPMessage(
            id=message.id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": self.capabilities,
                "serverInfo": {"name": self.server_name, "version": self.version},
            },
        )

    async def _handle_tools_list(self, message: MCPMessage) -> MCPMessage:
        """Maneja lista de herramientas"""
        tools_list = [tool.to_dict() for tool in self.tools.values()]
        return MCPMessage(id=message.id, result={"tools": tools_list})

    async def _handle_tools_call(self, message: MCPMessage) -> MCPMessage:
        """Maneja llamada a herramienta"""
        try:
            params = message.params or {}
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            if tool_name not in self.tools:
                return MCPMessage(
                    id=message.id,
                    error={"code": -32602, "message": f"Tool not found: {tool_name}"},
                )

            # Ejecutar la herramienta
            result = await self._execute_tool(tool_name, tool_args)

            return MCPMessage(id=message.id, result={"content": result})

        except Exception as e:
            return MCPMessage(
                id=message.id,
                error={"code": -32000, "message": f"Tool execution error: {str(e)}"},
            )

    async def _execute_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Ejecuta una herramienta (debe ser implementado por subclases)"""
        raise NotImplementedError("Subclasses must implement _execute_tool")

    async def _handle_resources_list(self, message: MCPMessage) -> MCPMessage:
        """Maneja lista de recursos"""
        resources_list = [resource.to_dict() for resource in self.resources.values()]
        return MCPMessage(id=message.id, result={"resources": resources_list})

    async def _handle_resources_read(self, message: MCPMessage) -> MCPMessage:
        """Maneja lectura de recurso"""
        try:
            params = message.params or {}
            uri = params.get("uri")

            if uri not in self.resources:
                return MCPMessage(
                    id=message.id,
                    error={"code": -32602, "message": f"Resource not found: {uri}"},
                )

            # Leer el recurso
            content = await self._read_resource(uri)

            return MCPMessage(id=message.id, result={"contents": content})

        except Exception as e:
            return MCPMessage(
                id=message.id,
                error={"code": -32000, "message": f"Resource read error: {str(e)}"},
            )

    async def _read_resource(self, uri: str) -> List[Dict[str, Any]]:
        """Lee un recurso (debe ser implementado por subclases)"""
        raise NotImplementedError("Subclasses must implement _read_resource")

    async def _handle_prompts_list(self, message: MCPMessage) -> MCPMessage:
        """Maneja lista de prompts"""
        prompts_list = [prompt.to_dict() for prompt in self.prompts.values()]
        return MCPMessage(id=message.id, result={"prompts": prompts_list})

    async def _handle_prompts_get(self, message: MCPMessage) -> MCPMessage:
        """Maneja obtención de prompt"""
        try:
            params = message.params or {}
            prompt_name = params.get("name")
            prompt_args = params.get("arguments", {})

            if prompt_name not in self.prompts:
                return MCPMessage(
                    id=message.id,
                    error={
                        "code": -32602,
                        "message": f"Prompt not found: {prompt_name}",
                    },
                )

            # Obtener el prompt
            prompt_content = await self._get_prompt(prompt_name, prompt_args)

            return MCPMessage(
                id=message.id,
                result={
                    "description": self.prompts[prompt_name].description,
                    "messages": prompt_content,
                },
            )

        except Exception as e:
            return MCPMessage(
                id=message.id,
                error={"code": -32000, "message": f"Prompt get error: {str(e)}"},
            )

    async def _get_prompt(
        self, prompt_name: str, args: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Obtiene un prompt (debe ser implementado por subclases)"""
        raise NotImplementedError("Subclasses must implement _get_prompt")


# =============================================================================
# MCP CLIENT
# =============================================================================


class MCPClient:
    """Cliente MCP"""

    def __init__(self, transport_type: MCPTransportType = MCPTransportType.HTTP):
        self.transport_type = transport_type
        self.server_url: Optional[str] = None
        self.session_id: Optional[str] = None
        self.connected = False
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.pending_requests: Dict[str, asyncio.Future] = {}

    async def connect(self, server_url: str):
        """Conecta al servidor MCP"""
        self.server_url = server_url

        if self.transport_type == MCPTransportType.HTTP:
            self.connected = True
        elif self.transport_type == MCPTransportType.WEBSOCKET:
            self.websocket = await websockets.connect(server_url)
            self.connected = True
            # Iniciar listener de mensajes
            asyncio.create_task(self._listen_messages())
        elif self.transport_type == MCPTransportType.STDIO:
            # Para stdio, asumimos que ya está conectado
            self.connected = True

        logger.info(f"MCP Client connected to {server_url}")

    async def disconnect(self):
        """Desconecta del servidor MCP"""
        if self.websocket:
            await self.websocket.close()
        self.connected = False
        logger.info("MCP Client disconnected")

    async def initialize(self) -> Dict[str, Any]:
        """Inicializa la conexión MCP"""
        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "sheily-mcp-client", "version": "1.0.0"},
            },
        )
        return response

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Lista herramientas disponibles"""
        response = await self._send_request("tools/list")
        return response.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Llama a una herramienta"""
        response = await self._send_request(
            "tools/call", {"name": tool_name, "arguments": arguments}
        )
        return response.get("content", [])

    async def list_resources(self) -> List[Dict[str, Any]]:
        """Lista recursos disponibles"""
        response = await self._send_request("resources/list")
        return response.get("resources", [])

    async def read_resource(self, uri: str) -> List[Dict[str, Any]]:
        """Lee un recurso"""
        response = await self._send_request("resources/read", {"uri": uri})
        return response.get("contents", [])

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """Lista prompts disponibles"""
        response = await self._send_request("prompts/list")
        return response.get("prompts", [])

    async def get_prompt(
        self, prompt_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Obtiene un prompt"""
        params = {"name": prompt_name}
        if arguments:
            params["arguments"] = arguments
        response = await self._send_request("prompts/get", params)
        return response

    async def _send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Envía una solicitud al servidor"""
        message = MCPMessage(id=str(uuid.uuid4()), method=method, params=params)

        future = asyncio.Future()
        self.pending_requests[message.id] = future

        try:
            if self.transport_type == MCPTransportType.HTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.server_url,
                        json=json.loads(message.to_json()),
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        result = await response.json()
                        return result.get("result", {})

            elif self.transport_type == MCPTransportType.WEBSOCKET:
                await self.websocket.send(message.to_json())
                return await future

            elif self.transport_type == MCPTransportType.STDIO:
                # Para stdio, simular respuesta
                return {"result": "stdio_response"}

        except Exception as e:
            logger.error(f"Error sending MCP request: {e}")
            raise

    async def _listen_messages(self):
        """Escucha mensajes del servidor (para websockets)"""
        try:
            async for message in self.websocket:
                mcp_message = MCPMessage.from_json(message)

                if mcp_message.id in self.pending_requests:
                    future = self.pending_requests.pop(mcp_message.id)
                    if mcp_message.error:
                        future.set_exception(
                            Exception(mcp_message.error.get("message", "MCP Error"))
                        )
                    else:
                        future.set_result(mcp_message.result or {})
        except Exception as e:
            logger.error(f"Error listening to MCP messages: {e}")


# =============================================================================
# SERVIDORES MCP ESPECIALIZADOS
# =============================================================================


class CodeAnalysisMCPServer(MCPServer):
    """Servidor MCP para análisis de código"""

    def __init__(self):
        super().__init__("code-analysis-server", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        """Registra herramientas de análisis de código"""
        # Herramienta de análisis de código
        analyze_tool = MCPTool(
            name="analyze_code",
            title="Code Analysis Tool",
            description="Analyzes code for quality, bugs, and improvements",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code to analyze"},
                    "language": {
                        "type": "string",
                        "description": "Programming language",
                        "enum": ["python", "javascript", "java", "cpp"],
                    },
                },
                "required": ["code"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "message": {"type": "string"},
                                "line": {"type": "integer"},
                            },
                        },
                    },
                    "complexity_score": {"type": "number"},
                    "quality_score": {"type": "number"},
                },
            },
        )
        self.register_tool(analyze_tool)

    async def _execute_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Ejecuta herramientas de análisis de código"""
        if tool_name == "analyze_code":
            return await self._analyze_code(args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _analyze_code(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analiza código"""
        code = args.get("code", "")
        language = args.get("language", "python")

        issues = []
        issues = []
        
        # Calculate complexity score
        loc = len(code.split('\n'))
        complexity_score = min(1.0, loc / 500.0)  # Normalize
        
        # Calculate quality score based on heuristics
        quality_deductions = 0.0
        if "TODO" in code: quality_deductions += 0.1
        if "FIXME" in code: quality_deductions += 0.2
        if "except Exception:" in code: quality_deductions += 0.1
        if "print(" in code: quality_deductions += 0.05
        
        quality_score = max(0.0, 1.0 - quality_deductions)

        # Análisis simple
        if "TODO" in code:
            issues.append(
                {
                    "type": "info",
                    "message": "TODO comment found",
                    "line": code.find("TODO") // 50,  # Rough line estimate
                }
            )

        if len(code.split("\n")) > 50:
            issues.append(
                {
                    "type": "warning",
                    "message": "File is quite long, consider splitting",
                    "line": 1,
                }
            )

        if language == "python":
            if "def " in code and "return" not in code:
                issues.append(
                    {
                        "type": "warning",
                        "message": "Function without return statement",
                        "line": code.find("def ") // 50,
                    }
                )

        return [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "issues": issues,
                        "complexity_score": complexity_score,
                        "quality_score": quality_score,
                    }
                ),
            }
        ]


class DataProcessingMCPServer(MCPServer):
    """Servidor MCP para procesamiento de datos"""

    def __init__(self):
        super().__init__("data-processing-server", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        """Registra herramientas de procesamiento de datos"""
        process_tool = MCPTool(
            name="process_data",
            title="Data Processing Tool",
            description="Processes and analyzes datasets",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Array of numerical data",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["statistics", "normalize", "filter"],
                        "description": "Operation to perform",
                    },
                },
                "required": ["data", "operation"],
            },
        )
        self.register_tool(process_tool)

    async def _execute_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Ejecuta herramientas de procesamiento de datos"""
        if tool_name == "process_data":
            return await self._process_data(args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _process_data(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Procesa datos"""
        data = args.get("data", [])
        operation = args.get("operation", "statistics")

        if not isinstance(data, list):
            return [{"type": "text", "text": "Error: data must be an array"}]

        result = {}
        if operation == "statistics":
            result = {
                "count": len(data),
                "mean": sum(data) / len(data) if data else 0,
                "min": min(data) if data else 0,
                "max": max(data) if data else 0,
                "sum": sum(data),
            }
        elif operation == "normalize":
            if data:
                min_val = min(data)
                max_val = max(data)
                range_val = max_val - min_val
                if range_val > 0:
                    result["normalized"] = [(x - min_val) / range_val for x in data]
                else:
                    result["normalized"] = [0.5] * len(data)
            else:
                result["normalized"] = []
        elif operation == "filter":
            threshold = args.get("threshold", 0)
            result["filtered"] = [x for x in data if x > threshold]

        return [{"type": "text", "text": json.dumps(result)}]


# =============================================================================
# SISTEMA MCP UNIFICADO
# =============================================================================


class MCPSystem:
    """Sistema unificado MCP para Sheily AI"""

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.clients: Dict[str, MCPClient] = {}
        self.registry: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        """Inicia el sistema MCP"""
        # Crear servidores especializados
        code_server = CodeAnalysisMCPServer()
        data_server = DataProcessingMCPServer()

        await code_server.start()
        await data_server.start()

        self.servers["code-analysis"] = code_server
        self.servers["data-processing"] = data_server

        # Registrar en el registry
        self._register_servers()

        logger.info("MCP System started")

    async def stop(self):
        """Detiene el sistema MCP"""
        for server in self.servers.values():
            await server.stop()
        self.servers.clear()
        logger.info("MCP System stopped")

    def _register_servers(self):
        """Registra servidores en el registry local"""
        for server_name, server in self.servers.items():
            self.registry[server_name] = {
                "name": server.server_name,
                "version": server.version,
                "capabilities": server.capabilities,
                "tools": [tool.to_dict() for tool in server.tools.values()],
                "endpoint": f"mcp://{server_name}",
            }

    async def discover_tools(
        self, server_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Descubre herramientas disponibles"""
        all_tools = []

        if server_name and server_name in self.registry:
            # Herramientas de un servidor específico
            server_info = self.registry[server_name]
            all_tools.extend(server_info["tools"])
        else:
            # Herramientas de todos los servidores
            for server_info in self.registry.values():
                all_tools.extend(server_info["tools"])

        return all_tools

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Llama a una herramienta en un servidor específico"""
        if server_name not in self.servers:
            raise ValueError(f"Server not found: {server_name}")

        server = self.servers[server_name]

        # Crear mensaje de llamada a herramienta
        message = MCPMessage(
            id=str(uuid.uuid4()),
            method="tools/call",
            params={"name": tool_name, "arguments": arguments},
        )

        # Procesar el mensaje
        response = await server.handle_message(message)

        if response.error:
            raise Exception(
                f"MCP Error: {response.error.get('message', 'Unknown error')}"
            )

        return response.result

    async def get_server_info(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un servidor"""
        return self.registry.get(server_name)

    async def list_servers(self) -> List[Dict[str, Any]]:
        """Lista todos los servidores registrados"""
        return list(self.registry.values())


# =============================================================================
# INTEGRACIÓN CON SISTEMA DE AGENTES
# =============================================================================

# Instancia global del sistema MCP
mcp_system = MCPSystem()


async def initialize_mcp_system():
    """Inicializa el sistema MCP"""
    await mcp_system.start()


async def discover_mcp_tools() -> List[Dict[str, Any]]:
    """Descubre todas las herramientas MCP disponibles"""
    return await mcp_system.discover_tools()


async def call_mcp_tool(
    server_name: str, tool_name: str, arguments: Dict[str, Any]
) -> Any:
    """Llama a una herramienta MCP"""
    return await mcp_system.call_tool(server_name, tool_name, arguments)


# =============================================================================
# DEMO Y TESTING
# =============================================================================


async def demo_mcp_system():
    """Demostración del sistema MCP"""
    print("🔧 Inicializando sistema MCP...")
    await initialize_mcp_system()

    print("\n📋 Servidores registrados:")
    servers = await mcp_system.list_servers()
    for server in servers:
        print(f"  • {server['name']} ({server['endpoint']})")

    print("\n🔍 Descubriendo herramientas...")
    tools = await discover_mcp_tools()
    for tool in tools:
        print(f"  • {tool['name']}: {tool['description']}")

    print("\n🛠️ Probando herramienta de análisis de código...")
    try:
        result = await call_mcp_tool(
            "code-analysis",
            "analyze_code",
            {
                "code": "def hello():\n    print('Hello World')\n    # TODO: Add error handling",
                "language": "python",
            },
        )
        print(f"Resultado: {result}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n📊 Probando herramienta de procesamiento de datos...")
    try:
        result = await call_mcp_tool(
            "data-processing",
            "process_data",
            {"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "operation": "statistics"},
        )
        print(f"Resultado: {result}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n✅ Demo MCP completada!")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Clases principales
    "MCPMessage",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPServer",
    "MCPClient",
    "MCPSystem",
    # Servidores especializados
    "CodeAnalysisMCPServer",
    "DataProcessingMCPServer",
    # Sistema global
    "mcp_system",
    # Funciones de utilidad
    "initialize_mcp_system",
    "discover_mcp_tools",
    "call_mcp_tool",
    "demo_mcp_system",
]

# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - MCP Protocol Implementation"
