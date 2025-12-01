#!/usr/bin/env python3
"""
REAL CHAT INTERFACE - SHEILY OMEGA
==================================
Interfaz de chat en terminal que conecta directamente con el Controlador Autónomo.
Permite enviar comandos reales y recibir respuestas del sistema vivo.
"""

import asyncio
import logging
import sys
import os
import threading
import time
from typing import Dict, Any

# Configurar logging
logging.basicConfig(level=logging.ERROR)

# Añadir path para importaciones
sys.path.append(os.path.abspath("packages/sheily-core/src"))

try:
    from sheily_core.agents.autonomous_system_controller import (
        start_system_control, 
        stop_system_control, 
        get_system_status,
        autonomous_controller
    )
    from sheily_core.agents.coordination_system import functional_multi_agent_system
except ImportError as e:
    print(f"Error crítico de importación: {e}")
    sys.exit(1)

# Colores para terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

async def process_user_command(command: str):
    """Procesa un comando del usuario y lo convierte en tarea del sistema"""
    
    command = command.lower().strip()
    
    if command in ["exit", "quit", "salir"]:
        return "exit"
    
    if command in ["status", "estado"]:
        status = get_system_status()
        metrics = status.get('real_metrics', {})
        print(f"\n{Colors.HEADER}[CHART] ESTADO DEL SISTEMA:{Colors.ENDC}")
        print(f"   CPU: {metrics.get('cpu_load')}%")
        print(f"   RAM: {metrics.get('memory_usage')}%")
        print(f"   Agentes Activos: {metrics.get('active_agents')}/{metrics.get('total_agents')}")
        print(f"   Salud: {metrics.get('system_health_score'):.1f}%")
        return "continue"

    # Enrutamiento de comandos a tareas reales
    task = None
    
    if "analiza" in command and "archivo" in command:
        # Extracción simple de argumentos (mejorar con NLP en futuro)
        parts = command.split()
        path = parts[-1] if len(parts) > 1 else "README.md"
        task = {
            "type": "data_analysis",
            "function": "analyze_file",
            "parameters": {"path": path}
        }
        
    elif "limpia" in command or "clean" in command:
        task = {
            "type": "system_maintenance",
            "function": "cleanup_temp",
            "parameters": {"path": "./tmp"}
        }
        
    elif "diagnostico" in command or "diag" in command:
        task = {
            "type": "system_maintenance",
            "function": "system_diagnostics",
            "parameters": {}
        }
        
    elif "escribe" in command and "archivo" in command:
        # Ejemplo: escribe archivo test.txt hola mundo
        try:
            parts = command.split("archivo ")
            if len(parts) > 1:
                args = parts[1].split(" ", 1)
                filename = args[0]
                content = args[1] if len(args) > 1 else "Contenido vacío"
                
                task = {
                    "type": "code_engineering",
                    "function": "write_code",
                    "parameters": {"path": filename, "content": content}
                }
        except:
            print(f"{Colors.FAIL}[ERROR] Formato incorrecto. Uso: escribe archivo <nombre> <contenido>{Colors.ENDC}")

    if task:
        print(f"{Colors.BLUE}[LIGHTNING] Enviando tarea al núcleo: {task['type']} -> {task['function']}...{Colors.ENDC}")
        result = await functional_multi_agent_system.execute_distributed_task(task)
        
        if result.get("status") == "success":
            print(f"{Colors.GREEN}[OK] Tarea completada:{Colors.ENDC}")
            # Mostrar resultado formateado
            res_data = result.get("result", {})
            for k, v in res_data.items():
                print(f"   - {k}: {v}")
                
            # Registrar en memoria (si está disponible)
            if autonomous_controller.memory:
                autonomous_controller.memory.add_memory(
                    f"User command '{command}' executed successfully", 
                    "episodic", 
                    {"command": command}
                )
        else:
            print(f"{Colors.FAIL}[ERROR] Error en tarea: {result.get('message')}{Colors.ENDC}")
            
    else:
        print(f"{Colors.WARNING}[WARN] Comando no reconocido o no implementado aún.{Colors.ENDC}")
        print("Comandos disponibles: status, analiza archivo <path>, limpia, diagnostico, escribe archivo <path> <content>")

    return "continue"

async def chat_loop():
    print(f"{Colors.HEADER}[BOT] SHEILY OMEGA - INTERFAZ DE MANDO REAL{Colors.ENDC}")
    print("Inicializando sistemas autónomos...")
    
    # Iniciar cerebro
    start_system_control()
    await asyncio.sleep(2) # Esperar arranque
    
    print(f"{Colors.GREEN}[OK] Sistemas Online. Esperando órdenes.{Colors.ENDC}")
    print("Escribe 'exit' para salir.")
    
    while True:
        try:
            user_input = await asyncio.to_thread(input, f"\n{Colors.BOLD}Comandante > {Colors.ENDC}")
            action = await process_user_command(user_input)
            if action == "exit":
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"{Colors.FAIL}Error en loop de chat: {e}{Colors.ENDC}")

    print("\nApagando sistemas...")
    stop_system_control()
    print("Desconexión completada.")

if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        pass
