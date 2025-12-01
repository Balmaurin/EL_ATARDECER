"""
🎯 FEDERATED LEARNING CLI - SHEILY AI UNIFIED SYSTEM
============================================================
Interfaz de línea de comandos unificada para sistemas FL

CARACTERÍSTICAS:
- Comando único para todas las operaciones FL
- Soporte para todos los tipos de sistemas (healthcare, finance, iot, etc.)
- Modos demo, producción y experimentación
- Configuración interactiva
- Monitoreo en tiempo real
- Logging integrado

AUTORES: Sheily AI Team - Arquitectura Unificada v2.0
FECHA: 2025
"""

import asyncio
import click
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Import unified components
from federated_factory import FederatedFactory
from federated_core import (
    UseCase,
    BaseFederatedConfig,
    DependencyManager,
    common_logger,
)

# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================
def setup_cli_logging(verbose: bool = False):
    """Setup logging for CLI"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

def print_header(title: str, emoji: str = "🚀"):
    """Print formatted header"""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + len(emoji) + 2))

def print_success(message: str):
    """Print success message"""
    click.secho(f"✅ {message}", fg="green")

def print_error(message: str):
    """Print error message"""
    click.secho(f"❌ {message}", fg="red")

def print_warning(message: str):
    """Print warning message"""
    click.secho(f"⚠️  {message}", fg="yellow")

def print_info(message: str):
    """Print info message"""
    click.secho(f"ℹ️  {message}", fg="blue")

def print_metrics(metrics: Dict[str, Any]):
    """Print formatted metrics"""
    print("\n📊 Métricas del Sistema:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

def validate_system_requirements(system_type: str) -> bool:
    """Validate that required dependencies are available"""
    factory = FederatedFactory()
    requirements = factory.validate_system_requirements(system_type)

    missing_deps = [dep for dep, available in requirements.items() if not available]

    if missing_deps:
        print_warning(f"Dependencias faltantes para {system_type}: {', '.join(missing_deps)}")
        return False

    return True

async def run_demo_system(system_type: str, num_clients: int = 10, rounds: int = 2):
    """Run demo for specific system type"""
    try:
        # Validate requirements
        if not validate_system_requirements(system_type):
            return

        print_header(f"Demo de Sistema {system_type.title()}", "🎭")

        # Create system
        factory = FederatedFactory()
        system = factory.create_system(system_type, num_clients=num_clients)
        print_success(f"Sistema {system_type} creado exitosamente")

        # Register demo clients
        await register_demo_clients(system, system_type)

        # Run demo rounds
        for round_num in range(1, rounds + 1):
            print_info(f"Iniciando ronda {round_num}...")

            round_id = await system.start_federated_round(round_num)

            # Simulate some client updates
            await simulate_demo_updates(system, round_id)

            # Get metrics
            metrics = system.get_federated_metrics()
            print_metrics(metrics)

            # Wait a bit between rounds
            await asyncio.sleep(0.5)

        print_success(f"Demo de {system_type} completado exitosamente")

    except Exception as e:
        print_error(f"Error en demo de {system_type}: {str(e)}")

async def register_demo_clients(system, system_type: str):
    """Register demo clients for testing"""
    demo_clients = [
        {"id": f"{system_type}_client_1", "device": "server", "location": "Madrid"},
        {"id": f"{system_type}_client_2", "device": "mobile", "location": "Barcelona"},
        {"id": f"{system_type}_client_3", "device": "iot", "location": "Sevilla"},
    ]

    for client in demo_clients:
        success = await system.register_client(
            client_id=client["id"],
            public_key=f"demo_key_{client['id']}",
            device_type=client["device"],
            location=client["location"]
        )
        if success:
            print_success(f"Cliente registrado: {client['id']}")
        else:
            print_warning(f"Error registrando cliente: {client['id']}")

async def simulate_demo_updates(system, round_id: str):
    """Simulate client updates for demo"""
    # Simple simulation - in real scenario, this would be actual FL training
    for client_id in list(system.clients.keys())[:2]:  # Use first 2 clients
        try:
            # Import here to avoid circular imports
            from federated_core import BaseModelUpdate
            import torch

            # Create mock update
            mock_weights = {
                "layer1.weight": torch.randn(10, 5),
                "layer1.bias": torch.randn(10),
            }

            update = BaseModelUpdate(
                client_id=client_id,
                round_id=round_id,
                model_weights=mock_weights,
                local_loss=0.5 + torch.rand(1).item() * 0.3,
                local_accuracy=0.7 + torch.rand(1).item() * 0.3,
                num_samples=int(50 + torch.rand(1).item() * 100),
            )

            success = await system.receive_client_update(update)
            if success:
                print_info(f"Actualización procesada de {client_id}")
            else:
                print_warning(f"Actualización rechazada de {client_id}")

        except ImportError:
            print_warning("PyTorch no disponible - simulando actualización básica")
            # Basic simulation without torch
            import random
            performance_score = 0.8 + random.random() * 0.2
            await system.update_client_reputation(client_id, performance_score)

# ========================================================================
# CLICK GROUPS AND COMMANDS
# ========================================================================
@click.group()
@click.option('--verbose', '-v', is_flag=True, help="Verbose logging")
@click.pass_context
def cli(ctx, verbose):
    """Federated Learning Unified CLI - Sheily AI v2.0

    Sistema unificado para aprendizaje federado con especializaciones
    por caso de uso (healthcare, finance, iot, etc.)

    Ejemplos:
        federated_cli demo healthcare --clients 20
        federated_cli create healthcare --clients 10 --privacy maximum
        federated_cli metrics iot
        federated_cli validate finance
    """
    ctx.ensure_object(dict)
    setup_cli_logging(verbose)
    ctx.obj['verbose'] = verbose

    # Initialize factory
    try:
        factory = FederatedFactory()
        ctx.obj['factory'] = factory
        print_header("Federated Learning CLI v2.0", "🎯")
    except Exception as e:
        print_error(f"Error inicializando CLI: {str(e)}")
        sys.exit(1)

# ============================================================================
# DEMO COMMANDS
# ============================================================================
@cli.group()
def demo():
    """Comandos de demostración para sistemas FL"""
    pass

@demo.command()
@click.argument('system_type', type=click.Choice(['healthcare', 'finance', 'iot']))
@click.option('--clients', '-c', default=10, help="Número de clientes")
@click.option('--rounds', '-r', default=2, help="Número de rondas")
@click.pass_context
def system(ctx, system_type, clients, rounds):
    """Ejecutar demo completa de un sistema FL"""
    asyncio.run(run_demo_system(system_type, clients, rounds))

@demo.command()
@click.pass_context
def all(ctx):
    """Ejecutar demos de todos los sistemas disponibles"""
    factory = ctx.obj['factory']

    print_header("Demo de Todos los Sistemas FL", "🎪")

    available_systems = factory.get_available_systems()
    available_types = list(available_systems.keys())

    for system_type in available_types:
        if system_type in ['healthcare', 'finance', 'iot']:
            print_info(f"Ejecutando demo de {system_type}...")
            try:
                asyncio.run(run_demo_system(system_type, num_clients=5, rounds=1))
            except Exception as e:
                print_error(f"Error en demo de {system_type}: {str(e)}")

            print()  # Blank line between demos

    print_success("Demos completados")

# ============================================================================
# CREATE/CONFIG COMMANDS
# ============================================================================
@cli.group()
def create():
    """Crear sistemas FL con configuración personalizada"""
    pass

@create.command()
@click.argument('system_type', type=click.Choice(['healthcare', 'finance', 'iot']))
@click.option('--clients', '-c', default=10, help="Número de clientes")
@click.option('--privacy', default='high', type=click.Choice(['basic', 'high', 'maximum']))
@click.option('--device-type', default='mobile', type=click.Choice(['mobile', 'iot', 'server']))
@click.option('--interactive', '-i', is_flag=True, help="Configuración interactiva")
@click.pass_context
def system_config(ctx, system_type, clients, privacy, device_type, interactive):
    """Crear sistema FL con configuración personalizada"""

    factory = ctx.obj['factory']

    if interactive:
        print_header("Configuración Interactiva", "⚙️")
        clients = click.prompt("Número de clientes", default=clients, type=int)
        privacy = click.prompt("Nivel de privacidad", default=privacy,
                             type=click.Choice(['basic', 'high', 'maximum']))
        device_type = click.prompt("Tipo de dispositivo principal", default=device_type,
                                 type=click.Choice(['mobile', 'iot', 'server']))

    # Create config builder
    builder = factory.create_config_builder()

    # Configure based on options
    config = (builder
              .with_use_case({
                  'healthcare': UseCase.HEALTHCARE,
                  'finance': UseCase.FINANCE,
                  'iot': UseCase.IOT
              }[system_type])
              .with_privacy_level(privacy)
              .with_performance_optimization(device_type)
              .with_scale(clients)
              .build())

    # Create system
    try:
        system = factory.create_system(system_type, config)
        ctx.obj['active_system'] = system

        print_success(f"Sistema {system_type} creado exitosamente")
        print_info(f"Clientes configurados: {clients}")
        print_info(f"Nivel de privacidad: {privacy}")
        print_info(f"Optimizado para: {device_type}")

    except Exception as e:
        print_error(f"Error creando sistema: {str(e)}")

# ============================================================================
# MANAGEMENT COMMANDS
# ============================================================================
@cli.group()
def manage():
    """Gestión de sistemas FL activos"""
    pass

@manage.command()
@click.pass_context
def show_active(ctx):
    """Mostrar sistema activo actualmente"""
    active_system = ctx.obj.get('active_system')
    if active_system:
        print_info(f"Sistema activo: {type(active_system).__name__}")
        metrics = active_system.get_federated_metrics()
        print_metrics(metrics)
    else:
        print_warning("No hay sistema activo")

@manage.command()
@click.option('--rounds', '-r', default=1, help="Número de rondas a ejecutar")
@click.pass_context
def start_round(ctx, rounds):
    """Iniciar rondas de entrenamiento en sistema activo"""
    active_system = ctx.obj.get('active_system')
    if not active_system:
        print_error("No hay sistema activo. Usa 'federated_cli create' primero")
        return

    print_header("Iniciando Rondas de Entrenamiento", "🏃")

    async def _run_rounds():
        for round_num in range(1, rounds + 1):
            try:
                round_id = await active_system.start_federated_round(round_num)
                print_success(f"Ronda {round_num} iniciada: {round_id}")

                # Get updated metrics
                metrics = active_system.get_federated_metrics()
                print_metrics(metrics)

            except Exception as e:
                print_error(f"Error en ronda {round_num}: {str(e)}")

    asyncio.run(_run_rounds())

# ============================================================================
# MONITORING COMMANDS
# ============================================================================
@cli.group()
def monitor():
    """Comandos de monitoreo y métricas"""
    pass

@monitor.command()
@click.argument('system_type', required=False)
@click.option('--detailed', '-d', is_flag=True, help="Métricas detalladas")
@click.pass_context
def metrics(ctx, system_type, detailed):
    """Mostrar métricas de sistema activo o crear uno temporal"""
    active_system = ctx.obj.get('active_system')

    if active_system:
        print_header("Métricas del Sistema Activo", "📊")
        metrics = active_system.get_federated_metrics()
    elif system_type:
        print_header(f"Métricas de {system_type.title()}", "📊")
        factory = ctx.obj['factory']
        temp_system = factory.create_system(system_type)
        metrics = temp_system.get_federated_metrics()
    else:
        print_error("Especifica un tipo de sistema o crea uno primero con 'federated_cli create'")
        return

    print_metrics(metrics)

    if detailed:
        print("\n🔍 Detalles adicionales:")
        print(f"   Estado del sistema: {type(active_system).__name__ if active_system else 'temporal'}")
        print(f"   Número de rondas completadas: {len(active_system.active_rounds) if active_system else 0}")

@monitor.command()
@click.argument('system_type')
@click.pass_context
def validate(ctx, system_type):
    """Validar requerimientos y compatibilidad de un sistema"""
    print_header(f"Validación de {system_type.title()}", "🔍")

    factory = ctx.obj['factory']

    # Check availability
    available = factory.get_available_systems()
    if system_type not in available:
        print_error(f"Sistema '{system_type}' no disponible")
        print_info(f"Sistemas disponibles: {list(available.keys())}")
        return

    # Validate requirements
    requirements = factory.validate_system_requirements(system_type)

    print_info("Verificando requerimientos del sistema:")
    all_good = True
    for req, available in requirements.items():
        status = "✅" if available else "❌"
        color = "green" if available else "red"
        click.secho(f"   {status} {req}", fg=color)
        if not available:
            all_good = False

    if all_good:
        print_success(f"Sistema {system_type} está listo para usar")
    else:
        print_warning(f"Sistema {system_type} tiene dependencias faltantes")

# ============================================================================
# CONFIG COMMANDS
# ============================================================================
@cli.group()
def config():
    """Gestión de configuraciones FL"""
    pass

@config.command()
def templates():
    """Mostrar plantillas de configuración disponibles"""
    print_header("Plantillas de Configuración", "📋")

    templates = {
        "healthcare_basic": "Configuración básica para salud (10 clientes, privacidad estándar)",
        "healthcare_compliant": "Configuración GDPR/HIPAA compliant (20 clientes, máxima privacidad)",
        "finance_secure": "Configuración financiera segura (50 clientes, encriptación homomórfica)",
        "iot_edge": "Configuración IoT para edge computing (100 dispositivos, optimizada)",
        "research_minimal": "Configuración minimal para investigación (5 clientes, básica)",
    }

    for template, description in templates.items():
        print(f"   {template}: {description}")

@config.command()
@click.argument('template')
def show_template(template):
    """Mostrar configuración detallada de una plantilla"""
    print_header(f"Plantilla: {template}", "📄")

    # This would show actual config JSON for the template
    print_header(f"Validación de {system_type.title()}", "🔍")

    factory = ctx.obj['factory']

    # Check availability
    available = factory.get_available_systems()
    if system_type not in available:
        print_error(f"Sistema '{system_type}' no disponible")
        print_info(f"Sistemas disponibles: {list(available.keys())}")
        return

    # Validate requirements
    requirements = factory.validate_system_requirements(system_type)

    print_info("Verificando requerimientos del sistema:")
    all_good = True
    for req, available in requirements.items():
        status = "✅" if available else "❌"
        color = "green" if available else "red"
        click.secho(f"   {status} {req}", fg=color)
        if not available:
            all_good = False

    if all_good:
        print_success(f"Sistema {system_type} está listo para usar")
    else:
        print_warning(f"Sistema {system_type} tiene dependencias faltantes")

# ============================================================================
# CONFIG COMMANDS
# ============================================================================
@cli.group()
def config():
    """Gestión de configuraciones FL"""
    pass
