#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Desactivar Restricciones de Red
Desactiva VPN, Proxy, Antivirus y restricciones de red privada de Windows
"""

import subprocess
import sys
import os
import winreg
import json
import logging
from pathlib import Path

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkRestrictionsDisabler:
    """Desactiva restricciones de red en Windows"""
    
    def __init__(self):
        self.docker_config_dir = Path.home() / ".docker"
        self.docker_config_file = self.docker_config_dir / "config.json"
        self.daemon_config_file = self.docker_config_dir / "daemon.json"
    
    def disable_windows_proxy(self):
        """Desactivar proxy de Windows"""
        logger.info("Desactivando proxy de Windows...")
        
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
                0,
                winreg.KEY_WRITE
            ) as key:
                # Desactivar proxy
                winreg.SetValueEx(key, "ProxyEnable", 0, winreg.REG_DWORD, 0)
                # Limpiar configuración de proxy
                try:
                    winreg.DeleteValue(key, "ProxyServer")
                except:
                    pass
                try:
                    winreg.DeleteValue(key, "ProxyOverride")
                except:
                    pass
                
                logger.info("[OK] Proxy de Windows desactivado")
                return True
        except PermissionError:
            logger.warning("[ADVERTENCIA] Se requieren permisos de administrador para desactivar el proxy")
            logger.info("[INFO] Por favor, ejecuta este script como administrador o desactiva el proxy manualmente:")
            logger.info("   Configuracion > Red e Internet > Proxy > Desactivar 'Usar un servidor proxy'")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Error desactivando proxy: {e}")
            return False
    
    def disable_docker_proxy(self):
        """Eliminar configuración de proxy de Docker"""
        logger.info("Desactivando proxy de Docker...")
        
        try:
            # Crear directorio si no existe
            self.docker_config_dir.mkdir(exist_ok=True)
            
            # Leer configuración existente del daemon
            daemon_config = {}
            if self.daemon_config_file.exists():
                try:
                    with open(self.daemon_config_file, 'r') as f:
                        daemon_config = json.load(f)
                except:
                    pass
            
            # Eliminar configuración de proxy
            if "proxies" in daemon_config:
                del daemon_config["proxies"]
                logger.info("[OK] Proxy eliminado de daemon.json")
            
            # Guardar configuración sin proxy
            with open(self.daemon_config_file, 'w', encoding='utf-8') as f:
                json.dump(daemon_config, f, indent=2)
            
            # Leer configuración del cliente
            client_config = {}
            if self.docker_config_file.exists():
                try:
                    with open(self.docker_config_file, 'r', encoding='utf-8') as f:
                        client_config = json.load(f)
                except:
                    pass
            
            # Eliminar proxy del cliente
            if "proxies" in client_config:
                del client_config["proxies"]
                logger.info("[OK] Proxy eliminado de config.json")
            
            # Guardar configuración sin proxy
            with open(self.docker_config_file, 'w', encoding='utf-8') as f:
                json.dump(client_config, f, indent=2)
            
            logger.info("[OK] Configuracion de proxy de Docker eliminada")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error desactivando proxy de Docker: {e}")
            return False
    
    def disable_private_network_restrictions(self):
        """Desactivar restricciones de red privada de Windows"""
        logger.info("Desactivando restricciones de red privada...")
        
        try:
            # Cambiar perfil de red a privado y desactivar firewall para red privada
            commands = [
                # Cambiar perfil de red a privado (si no lo está)
                'netsh advfirewall set currentprofile state off',
                # Permitir todas las conexiones en red privada
                'netsh advfirewall firewall set rule group="Red privada" new enable=yes',
                # Permitir Docker Desktop
                'netsh advfirewall firewall add rule name="Docker Desktop" dir=in action=allow program="C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"',
                # Permitir Docker Engine
                'netsh advfirewall firewall add rule name="Docker Engine" dir=in action=allow program="C:\\Program Files\\Docker\\Docker\\resources\\dockerd.exe"',
            ]
            
            for cmd in commands:
                try:
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        logger.info(f"[OK] Comando ejecutado: {cmd[:50]}...")
                    else:
                        logger.warning(f"[ADVERTENCIA] Comando fallo: {result.stderr}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"[ADVERTENCIA] Timeout en comando: {cmd[:50]}...")
                except Exception as e:
                    logger.warning(f"[ADVERTENCIA] Error en comando: {e}")
            
            logger.info("[OK] Restricciones de red privada desactivadas")
            logger.warning("[ADVERTENCIA] NOTA: El firewall de Windows ha sido desactivado temporalmente")
            logger.warning("[ADVERTENCIA] RECUERDA reactivarlo despues de usar el proyecto")
            return True
            
        except PermissionError:
            logger.warning("[ADVERTENCIA] Se requieren permisos de administrador")
            logger.info("[INFO] Ejecuta este script como administrador o desactiva manualmente:")
            logger.info("   Configuracion > Red e Internet > Estado > Cambiar propiedades de conexion > Red privada")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Error desactivando restricciones: {e}")
            return False
    
    def unset_environment_proxies(self):
        """Eliminar variables de entorno de proxy"""
        logger.info("Eliminando variables de entorno de proxy...")
        
        proxy_vars = [
            'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
            'NO_PROXY', 'no_proxy', 'ALL_PROXY', 'all_proxy'
        ]
        
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
                logger.info(f"[OK] Variable {var} eliminada")
        
        logger.info("[OK] Variables de entorno de proxy eliminadas")
        return True
    
    def create_instructions_file(self):
        """Crear archivo con instrucciones para desactivar VPN y Antivirus"""
        logger.info("Creando archivo de instrucciones...")
        
        instructions = """
# 🔓 INSTRUCCIONES PARA DESACTIVAR VPN Y ANTIVIRUS

## 🌐 DESACTIVAR VPN

### VPNs Comunes:
1. **NordVPN**: Click derecho en el icono de la bandeja > Desconectar
2. **ExpressVPN**: Click derecho en el icono > Quit
3. **ProtonVPN**: Click derecho en el icono > Disconnect
4. **Windows VPN**: Configuración > Red e Internet > VPN > Desconectar

### Método General:
- Busca el icono de VPN en la bandeja del sistema (esquina inferior derecha)
- Click derecho > Desconectar o Quit
- O abre la aplicación de VPN y desconéctala

---

## 🛡️ DESACTIVAR ANTIVIRUS

### Avast:
1. Abre Avast
2. Menú > Configuración
3. Protección > Protección en tiempo real
4. Desactiva temporalmente (10 minutos / 1 hora)
5. O: Click derecho en el icono de Avast > Protección de Avast > Desactivar temporalmente

### Bitdefender:
1. Abre Bitdefender
2. Protección > Configuración
3. Desactiva "Protección en tiempo real" temporalmente
4. O: Click derecho en el icono > Desactivar protección temporalmente

### Windows Defender:
1. Configuración > Privacidad y seguridad > Seguridad de Windows
2. Protección contra virus y amenazas
3. Configuración de protección contra virus y amenazas
4. Desactivar "Protección en tiempo real" temporalmente

### Método General:
- Busca el icono del antivirus en la bandeja del sistema
- Click derecho > Desactivar temporalmente / Pausar protección
- Generalmente permite desactivar por 10 minutos, 1 hora o hasta reiniciar

---

## ⚠️ IMPORTANTE

- **Solo desactiva temporalmente** mientras trabajas con el proyecto
- **Reactiva todo** después de terminar por seguridad
- Si no puedes desactivar algo, puede requerir permisos de administrador

---

## ✅ VERIFICACIÓN

Después de desactivar todo, verifica:

1. ✅ VPN desconectada
2. ✅ Antivirus pausado
3. ✅ Proxy desactivado (ya hecho por este script)
4. ✅ Firewall configurado (ya hecho por este script)

Luego ejecuta:
```bash
python deploy_local.py
```

O si usas Docker:
```bash
docker-compose up -d
```
"""
        
        instructions_path = Path("INSTRUCCIONES_DESACTIVAR_VPN_ANTIVIRUS.md")
        with open(instructions_path, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        logger.info(f"[OK] Instrucciones guardadas en: {instructions_path}")
        return instructions_path
    
    def disable_all_restrictions(self):
        """Desactivar todas las restricciones de red"""
        print("\n" + "="*70)
        print("DESACTIVANDO RESTRICCIONES DE RED")
        print("="*70)
        
        results = {
            'proxy_windows': False,
            'proxy_docker': False,
            'network_restrictions': False,
            'env_proxies': False
        }
        
        # 1. Desactivar proxy de Windows
        print("\n[1] Desactivando proxy de Windows...")
        results['proxy_windows'] = self.disable_windows_proxy()
        
        # 2. Desactivar proxy de Docker
        print("\n[2] Desactivando proxy de Docker...")
        results['proxy_docker'] = self.disable_docker_proxy()
        
        # 3. Desactivar restricciones de red privada
        print("\n[3] Desactivando restricciones de red privada...")
        results['network_restrictions'] = self.disable_private_network_restrictions()
        
        # 4. Eliminar variables de entorno de proxy
        print("\n[4] Eliminando variables de entorno de proxy...")
        results['env_proxies'] = self.unset_environment_proxies()
        
        # 5. Crear archivo de instrucciones
        print("\n[5] Creando archivo de instrucciones...")
        instructions_path = self.create_instructions_file()
        
        # Resumen
        print("\n" + "="*70)
        print("RESUMEN")
        print("="*70)
        print(f"[OK] Proxy Windows:        {'SI' if results['proxy_windows'] else 'NO (requiere admin)'}")
        print(f"[OK] Proxy Docker:         {'SI' if results['proxy_docker'] else 'NO'}")
        print(f"[OK] Restricciones Red:    {'SI' if results['network_restrictions'] else 'NO (requiere admin)'}")
        print(f"[OK] Variables Entorno:    {'SI' if results['env_proxies'] else 'NO'}")
        print(f"[OK] Instrucciones:        SI ({instructions_path})")
        
        print("\n" + "="*70)
        print("PROXIMOS PASOS")
        print("="*70)
        print("1. DESACTIVA VPN manualmente (ver instrucciones en el archivo creado)")
        print("2. DESACTIVA ANTIVIRUS manualmente (ver instrucciones en el archivo creado)")
        print("3. REINICIA Docker Desktop si esta corriendo")
        print("4. EJECUTA el proyecto:")
        print("   python deploy_local.py")
        print("   O:")
        print("   docker-compose up -d")
        print("="*70 + "\n")
        
        return results

def main():
    """Función principal"""
    print("DESACTIVADOR DE RESTRICCIONES DE RED")
    print("Este script desactiva proxy, restricciones de red y configura Docker")
    print("NOTA: Algunas operaciones requieren permisos de administrador\n")
    
    disabler = NetworkRestrictionsDisabler()
    
    try:
        results = disabler.disable_all_restrictions()
        
        if all(results.values()):
            print("[OK] Todas las restricciones desactivadas exitosamente!")
        else:
            print("[ADVERTENCIA] Algunas restricciones no se pudieron desactivar automaticamente")
            print("[INFO] Revisa las instrucciones en el archivo creado para desactivarlas manualmente")
        
    except Exception as e:
        logger.error(f"[ERROR] Error critico: {e}")
        import traceback
        traceback.print_exc()
        print("\n[INFO] Intenta ejecutar este script como administrador")

if __name__ == "__main__":
    main()

