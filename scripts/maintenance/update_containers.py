#!/usr/bin/env python3
"""
EL-AMANECER V4 - Container Update Script
Actualiza contenedores con Google ADK integration y últimas mejoras
"""
import asyncio
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContainerUpdater:
    """Gestor de actualización de contenedores EL-AMANECER"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.compose_file = project_root / "docker-compose.yml"
        
    def run_command(self, command: str, description: str = "") -> bool:
        """Ejecutar comando y mostrar resultado"""
        logger.info(f"🔄 {description or command}")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} - Completado exitosamente")
                if result.stdout.strip():
                    logger.info(f"Output: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"❌ {description} - Error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {description} - Excepción: {e}")
            return False
    
    def update_containers(self) -> bool:
        """Proceso completo de actualización de contenedores"""
        logger.info("🚀 === INICIANDO ACTUALIZACIÓN DE CONTENEDORES EL-AMANECER V4 ===")
        logger.info(f"📁 Directorio del proyecto: {self.project_root}")
        logger.info(f"📄 Archivo compose: {self.compose_file}")
        
        # Verificar que existe docker-compose.yml
        if not self.compose_file.exists():
            logger.error("❌ docker-compose.yml no encontrado")
            return False
            
        steps = [
            # 1. Detener contenedores actuales
            ("docker-compose down", "Deteniendo contenedores existentes"),
            
            # 2. Limpiar imágenes locales (rebuild completo)
            ("docker system prune -f", "Limpiando sistema Docker"),
            
            # 3. Reconstruir con nuevas configuraciones
            ("docker-compose build --no-cache", "Reconstruyendo contenedores (puede tardar varios minutos)"),
            
            # 4. Iniciar servicios actualizados
            ("docker-compose up -d", "Iniciando servicios actualizados"),
            
            # 5. Verificar estado
            ("docker-compose ps", "Verificando estado de servicios"),
        ]
        
        for command, description in steps:
            if not self.run_command(command, description):
                logger.error(f"❌ Falló el paso: {description}")
                return False
            logger.info("---")
        
        # Esperar a que los servicios estén listos
        logger.info("⏳ Esperando que los servicios estén listos...")
        asyncio.sleep(10)
        
        # Verificación final
        logger.info("🔍 Verificación final de contenedores...")
        if self.run_command("docker-compose logs --tail=20", "Mostrando logs recientes"):
            logger.info("✨ === ACTUALIZACIÓN COMPLETADA EXITOSAMENTE ===")
            logger.info("🌐 GraphQL Federation Gateway: http://localhost:8080/graphql")
            logger.info("🎯 Google ADK: Habilitado con HACK-MEMORI automation")
            logger.info("📊 Frontend: http://localhost:3000")
            return True
        else:
            logger.error("❌ Problemas detectados en la verificación final")
            return False

def main():
    """Punto de entrada principal"""
    print("🎯 EL-AMANECER V4 - Container Update Script")
    print("═══════════════════════════════════════════")
    
    # Determinar directorio del proyecto
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    print(f"📂 Proyecto: {project_root}")
    print(f"🔧 Actualizando con Google ADK Integration...")
    print()
    
    # Ejecutar actualización
    updater = ContainerUpdater(project_root)
    success = updater.update_containers()
    
    if success:
        print("\n🎉 ¡ACTUALIZACIÓN COMPLETADA!")
        print("🚀 EL-AMANECER V4 está listo con Google ADK")
        print("📝 Nuevas características:")
        print("   • Google ADK Controller integrado")
        print("   • HACK-MEMORI automation mejorado") 
        print("   • REST + GraphQL APIs unificadas")
        print("   • Python 3.13 + healthchecks")
        sys.exit(0)
    else:
        print("\n❌ ACTUALIZACIÓN FALLÓ")
        print("🔍 Revisa los logs para más detalles")
        sys.exit(1)

if __name__ == "__main__":
    main()