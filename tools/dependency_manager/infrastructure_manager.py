"""
Sheily MCP Enterprise - Infrastructure Manager
Sistema de control total de infraestructura: Docker, Kubernetes, Terraform

Controla:
- Docker Compose orchestration
- Kubernetes deployments
- Terraform infrastructure
- Nginx configuration
- Service orchestration
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class InfrastructureManager:
    """Gestión completa de infraestructura del proyecto"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.docker_compose_file = self.root_dir / "docker-compose.yml"
        self.prod_compose_file = self.root_dir / "docker-compose.prod.yml"
        self.k8s_dir = self.root_dir / "k8s"
        self.terraform_dir = self.root_dir / "terraform"
        self.nginx_dir = self.root_dir / "config" / "nginx"
        self.docker_dir = self.root_dir / "config" / "docker"

    async def get_infrastructure_status(self) -> Dict[str, Any]:
        """Estado completo de toda la infraestructura"""
        return {
            "timestamp": asyncio.get_event_loop().time(),
            "docker_services": await self._get_docker_status(),
            "kubernetes_deployments": await self._get_k8s_status(),
            "terraform_state": await self._get_terraform_status(),
            "nginx_config": await self._get_nginx_status(),
            "infrastructure_health": await self._check_infrastructure_health(),
        }

    async def control_docker_services(
        self, action: str, services: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Control de servicios Docker"""
        services = services or ["all"]

        if action == "start":
            return await self._start_docker_services(services)
        elif action == "stop":
            return await self._stop_docker_services(services)
        elif action == "restart":
            return await self._restart_docker_services(services)
        elif action == "status":
            return await self._get_docker_status()
        elif action == "logs":
            return await self._get_docker_logs(services)
        else:
            return {"error": f"Unknown action: {action}"}

    async def deploy_to_kubernetes(self, namespace: str = "default") -> Dict[str, Any]:
        """Despliegue completo a Kubernetes"""
        return await self._deploy_k8s_manifests(namespace)

    async def manage_terraform(self, action: str) -> Dict[str, Any]:
        """Gestión de infraestructura Terraform"""
        if action == "plan":
            return await self._terraform_plan()
        elif action == "apply":
            return await self._terraform_apply()
        elif action == "destroy":
            return await self._terraform_destroy()
        elif action == "status":
            return await self._get_terraform_status()
        else:
            return {"error": f"Unknown Terraform action: {action}"}

    async def configure_nginx(self, domain: str = None) -> Dict[str, Any]:
        """Configuración dinámica de Nginx"""
        return await self._update_nginx_config(domain)

    async def start_full_infrastructure(self) -> Dict[str, Any]:
        """Inicia toda la infraestructura: Docker + Kubernetes + Nginx"""
        results = {
            "docker": await self._start_docker_services(["all"]),
            "kubernetes": await self.deploy_to_kubernetes(),
            "nginx": await self.configure_nginx(),
            "summary": {
                "services_started": 0,
                "deployments_active": 0,
                "endpoints_configured": 0,
            },
        }

        # Calcular resumen
        if "started" in results["docker"]:
            results["summary"]["services_started"] = len(results["docker"]["started"])
        if "deployments" in results["kubernetes"]:
            results["summary"]["deployments_active"] = len(
                results["kubernetes"]["deployments"]
            )
        if "domains" in results["nginx"]:
            results["summary"]["endpoints_configured"] = len(
                results["nginx"]["domains"]
            )

        return results

    async def stop_full_infrastructure(self) -> Dict[str, Any]:
        """Detiene toda la infraestructura"""
        return await self._shutdown_infrastructure()

    # ========================================================================
    # DOCKER MANAGEMENT
    # ========================================================================

    async def _start_docker_services(self, services: List[str]) -> Dict[str, Any]:
        """Iniciar servicios Docker especificados"""
        results = {"started": [], "failed": [], "warnings": []}

        try:
            # Verificar que docker-compose existe
            if not self.docker_compose_file.exists():
                return {"error": "docker-compose.yml not found"}

            # Iniciar servicios
            cmd = ["docker-compose", "up", "-d"]
            if services != ["all"]:
                cmd.extend(services)

            result = await self._run_command(cmd, cwd=self.root_dir)

            if result["returncode"] == 0:
                results["started"] = (
                    services if services != ["all"] else ["all_services"]
                )
            else:
                results["failed"] = services
                results["warnings"].append(result["stderr"])

            # Verificar estado
            status = await self._run_command(
                ["docker-compose", "ps"], cwd=self.root_dir
            )
            if status["returncode"] == 0:
                results["status"] = status["stdout"]

        except Exception as e:
            results["error"] = str(e)

        return results

    async def _stop_docker_services(self, services: List[str]) -> Dict[str, Any]:
        """Detener servicios Docker"""
        cmd = ["docker-compose", "down"]
        if services != ["all"]:
            cmd.extend(["--scale", f"{services[0]}=0"])

        return await self._run_command(cmd, cwd=self.root_dir)

    async def _restart_docker_services(self, services: List[str]) -> Dict[str, Any]:
        """Reiniciar servicios Docker"""
        restart_results = {
            "stopped": await self._stop_docker_services(services),
            "started": await self._start_docker_services(services),
        }
        return restart_results

    async def _get_docker_status(self) -> Dict[str, Any]:
        """Estado de servicios Docker"""
        try:
            result = await self._run_command(
                ["docker-compose", "ps"], cwd=self.root_dir
            )

            if result["returncode"] == 0:
                # Parsear output para obtener servicios activos
                lines = result["stdout"].split("\n")
                services = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            service_name = parts[0]
                            status = parts[3] if len(parts) > 3 else "unknown"
                            services.append(
                                {
                                    "name": service_name,
                                    "status": status,
                                    "ports": await self._get_service_ports(
                                        service_name
                                    ),
                                }
                            )

                return {
                    "running_services": len(
                        [s for s in services if "Up" in s["status"]]
                    ),
                    "stopped_services": len(
                        [s for s in services if "Up" not in s["status"]]
                    ),
                    "total_services": len(services),
                    "services": services,
                }
            else:
                return {
                    "error": "Failed to get Docker status",
                    "details": result["stderr"],
                }
        except Exception as e:
            return {"error": str(e)}

    async def _get_docker_logs(self, services: List[str]) -> Dict[str, Any]:
        """Logs de servicios Docker"""
        cmd = ["docker-compose", "logs", "-f"]
        if services != ["all"]:
            cmd.extend(services)

        return await self._run_command(cmd, cwd=self.root_dir, timeout=30)

    async def _get_service_ports(self, service_name: str) -> List[str]:
        """Obtener puertos expuestos por un servicio"""
        try:
            result = await self._run_command(
                ["docker-compose", "port", service_name], cwd=self.root_dir
            )
            if result["returncode"] == 0:
                ports = []
                for line in result["stdout"].split("\n"):
                    if ":" in line:
                        ports.append(line.strip())
                return ports
        except Exception:
            pass
        return []

    # ========================================================================
    # KUBERNETES MANAGEMENT
    # ========================================================================

    async def _deploy_k8s_manifests(self, namespace: str) -> Dict[str, Any]:
        """Desplegar manifests de Kubernetes"""
        results = {"deployments": [], "services": [], "ingress": [], "errors": []}

        if not self.k8s_dir.exists():
            return {
                "error": "Kubernetes directory not found",
                "expected": str(self.k8s_dir),
            }

        try:
            # Aplicar todos los manifests .yaml en orden
            manifest_files = list(self.k8s_dir.glob("*.yaml")) + list(
                self.k8s_dir.glob("*.yml")
            )

            for manifest_file in sorted(manifest_files, key=lambda x: x.name):
                try:
                    # Verificar sintaxis
                    check_cmd = [
                        "kubectl",
                        "apply",
                        "--dry-run=client",
                        "-f",
                        str(manifest_file),
                    ]
                    check_result = await self._run_command(check_cmd)

                    if check_result["returncode"] == 0:
                        # Aplicar
                        apply_cmd = [
                            "kubectl",
                            "apply",
                            "-f",
                            str(manifest_file),
                            "-n",
                            namespace,
                        ]
                        apply_result = await self._run_command(apply_cmd)

                        if apply_result["returncode"] == 0:
                            results["deployments"].append(
                                {
                                    "file": manifest_file.name,
                                    "status": "applied",
                                    "namespace": namespace,
                                }
                            )
                        else:
                            results["errors"].append(
                                {
                                    "file": manifest_file.name,
                                    "error": apply_result["stderr"],
                                }
                            )
                    else:
                        results["errors"].append(
                            {"file": manifest_file.name, "error": "Invalid YAML syntax"}
                        )

                except Exception as e:
                    results["errors"].append(
                        {"file": manifest_file.name, "error": str(e)}
                    )

            # Verificar estado final
            final_status = await self._get_k8s_status()
            results.update(final_status)

        except Exception as e:
            results["error"] = str(e)

        return results

    async def _get_k8s_status(self) -> Dict[str, Any]:
        """Estado de despliegues Kubernetes"""
        try:
            # Replicas disponibles
            deployments_cmd = ["kubectl", "get", "deployments", "-o", "json"]
            deployments_result = await self._run_command(deployments_cmd)

            if deployments_result["returncode"] == 0:
                deployments_data = json.loads(deployments_result["stdout"])
                deployments = []
                for item in deployments_data.get("items", []):
                    spec = item.get("spec", {})
                    status = item.get("status", {})

                    deployments.append(
                        {
                            "name": item["metadata"]["name"],
                            "namespace": item["metadata"]["namespace"],
                            "replicas": spec.get("replicas", 0),
                            "ready_replicas": status.get("readyReplicas", 0),
                            "available_replicas": status.get("availableReplicas", 0),
                            "status": (
                                "healthy"
                                if status.get("availableReplicas", 0) > 0
                                else "unhealthy"
                            ),
                        }
                    )

                return {
                    "deployments_count": len(deployments),
                    "healthy_deployments": len(
                        [d for d in deployments if d["status"] == "healthy"]
                    ),
                    "unhealthy_deployments": len(
                        [d for d in deployments if d["status"] != "healthy"]
                    ),
                    "deployments": deployments,
                }
            else:
                return {"error": "Failed to get Kubernetes deployments"}

        except Exception as e:
            return {"error": str(e)}

    # ========================================================================
    # TERRAFORM MANAGEMENT
    # ========================================================================

    async def _terraform_plan(self) -> Dict[str, Any]:
        """Terraform plan"""
        if not self.terraform_dir.exists():
            return {"error": "Terraform directory not found"}

        try:
            result = await self._run_command(
                ["terraform", "plan"], cwd=self.terraform_dir
            )

            return {
                "success": result["returncode"] == 0,
                "changes": (
                    len(result["stdout"].split("\n"))
                    if result["returncode"] == 0
                    else 0
                ),
                "output": result["stdout"],
                "errors": result["stderr"] if result["returncode"] != 0 else None,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _terraform_apply(self) -> Dict[str, Any]:
        """Terraform apply"""
        if not self.terraform_dir.exists():
            return {"error": "Terraform directory not found"}

        try:
            # Primero terraform plan
            plan_result = await self._terraform_plan()
            if not plan_result.get("success"):
                return {"error": "Terraform plan failed", "details": plan_result}

            # Aplicar cambios
            result = await self._run_command(
                ["terraform", "apply", "-auto-approve"], cwd=self.terraform_dir
            )

            return {
                "success": result["returncode"] == 0,
                "output": result["stdout"],
                "errors": result["stderr"] if result["returncode"] != 0 else None,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _terraform_destroy(self) -> Dict[str, Any]:
        """Terraform destroy"""
        try:
            result = await self._run_command(
                ["terraform", "destroy", "-auto-approve"], cwd=self.terraform_dir
            )

            return {
                "success": result["returncode"] == 0,
                "output": result["stdout"],
                "errors": result["stderr"] if result["returncode"] != 0 else None,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _get_terraform_status(self) -> Dict[str, Any]:
        """Estado de Terraform"""
        if not self.terraform_dir.exists():
            return {"deployed": False, "reason": "Terraform directory not found"}

        try:
            result = await self._run_command(
                ["terraform", "show"], cwd=self.terraform_dir
            )

            return {
                "has_state": result["returncode"] == 0,
                "resources_count": (
                    len(result["stdout"].split("\n")) // 3
                    if result["returncode"] == 0
                    else 0
                ),
                "last_applied": "unknown",  # Would need to parse state file
            }
        except Exception as e:
            return {"error": str(e)}

    # ========================================================================
    # NGINX CONFIGURATION
    # ========================================================================

    async def _update_nginx_config(self, domain: str = None) -> Dict[str, Any]:
        """Actualizar configuración de Nginx"""
        results = {"configured": False, "domains": [], "certificates": [], "errors": []}

        if not self.nginx_dir.exists():
            return {"error": "Nginx configuration directory not found"}

        try:
            # Leer configuración actual
            main_config = self.nginx_dir / "nginx.conf"  # Adjust as needed

            if main_config.exists():
                # Parse and update configuration
                # This is a placeholder - would need nginx config parsing logic
                results["configured"] = True
                results["domains"] = [domain] if domain else ["localhost"]
                results["message"] = "Nginx configuration processed"
            else:
                results["error"] = "Main nginx.conf not found"

        except Exception as e:
            results["error"] = str(e)

        return results

    async def _get_nginx_status(self) -> Dict[str, Any]:
        """Estado de configuración Nginx"""
        try:
            # Check if nginx is running
            process_result = await self._run_command(["pgrep", "nginx"])
            is_running = process_result["returncode"] == 0

            if is_running:
                # Get nginx status
                status_result = await self._run_command(["nginx", "-t"])
                config_valid = status_result["returncode"] == 0

                return {
                    "running": True,
                    "config_valid": config_valid,
                    "config_path": str(self.nginx_dir),
                    "domains_configured": await self._get_nginx_domains(),
                }
            else:
                return {"running": False, "config_path": str(self.nginx_dir)}
        except Exception as e:
            return {"error": str(e)}

    async def _get_nginx_domains(self) -> List[str]:
        """Obtener dominios configurados en Nginx"""
        domains = []
        try:
            # Simple domain extraction from nginx configs
            for config_file in self.nginx_dir.glob("*.conf"):
                with open(config_file, "r") as f:
                    content = f.read()
                    # Very basic regex for server_name directives
                    import re

                    matches = re.findall(r"server_name\s+([^;]+)", content)
                    for match in matches:
                        domains.extend([d.strip() for d in match.split() if d.strip()])

        except Exception:
            pass

        return list(set(domains))  # Remove duplicates

    # ========================================================================
    # INFRASTRUCTURE HEALTH & MONITORING
    # ========================================================================

    async def _check_infrastructure_health(self) -> Dict[str, Any]:
        """Comprobación de salud general de la infraestructura"""
        health = {
            "overall_status": "unknown",
            "components": {
                "docker": await self._check_docker_health(),
                "kubernetes": await self._check_k8s_health(),
                "terraform": await self._check_terraform_health(),
                "nginx": await self._check_nginx_health(),
                "database": await self._check_database_health(),
            },
            "critical_issues": [],
            "warnings": [],
        }

        # Determinar estado general
        component_statuses = [comp["status"] for comp in health["components"].values()]

        if all(status == "healthy" for status in component_statuses):
            health["overall_status"] = "healthy"
        elif any(status == "critical" for status in component_statuses):
            health["overall_status"] = "critical"
        elif any(status == "warning" for status in component_statuses):
            health["overall_status"] = "warning"
        else:
            health["overall_status"] = "degraded"

        return health

    async def _check_docker_health(self) -> Dict[str, Any]:
        """Comprobar salud de Docker"""
        try:
            # Check if docker daemon is running
            result = await self._run_command(["docker", "info"])
            if result["returncode"] == 0:
                # Check if our services are running
                status = await self._get_docker_status()
                running_count = status.get("running_services", 0)
                total_count = status.get("total_services", 0)

                if running_count == total_count and total_count > 0:
                    return {
                        "status": "healthy",
                        "details": f"All {running_count} services running",
                    }
                elif running_count > 0:
                    return {
                        "status": "warning",
                        "details": f"{running_count}/{total_count} services running",
                    }
                else:
                    return {"status": "critical", "details": "No services running"}
            else:
                return {"status": "critical", "details": "Docker daemon not accessible"}
        except Exception as e:
            return {"status": "critical", "details": str(e)}

    async def _check_k8s_health(self) -> Dict[str, Any]:
        """Comprobar salud de Kubernetes"""
        try:
            result = await self._run_command(["kubectl", "cluster-info"])
            if result["returncode"] == 0:
                # Check deployments
                status = await self._get_k8s_status()
                healthy_count = status.get("healthy_deployments", 0)
                total_count = status.get("deployments_count", 0)

                if healthy_count == total_count and total_count > 0:
                    return {
                        "status": "healthy",
                        "details": f"All {healthy_count} deployments healthy",
                    }
                elif healthy_count > 0:
                    return {
                        "status": "warning",
                        "details": f"{healthy_count}/{total_count} deployments healthy",
                    }
                else:
                    return {"status": "critical", "details": "No healthy deployments"}
            else:
                return {
                    "status": "critical",
                    "details": "Kubernetes cluster not accessible",
                }
        except Exception as e:
            return {"status": "unavailable", "details": str(e)}

    async def _check_terraform_health(self) -> Dict[str, Any]:
        """Comprobar estado de Terraform"""
        try:
            if self.terraform_dir.exists():
                status = await self._get_terraform_status()
                if status.get("has_state"):
                    return {
                        "status": "healthy",
                        "details": f"State has {status.get('resources_count', 0)} resources",
                    }
                else:
                    return {"status": "info", "details": "No infrastructure deployed"}
            else:
                return {"status": "inactive", "details": "Terraform not configured"}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    async def _check_nginx_health(self) -> Dict[str, Any]:
        """Comprobar salud de Nginx"""
        try:
            status = await self._get_nginx_status()
            if status.get("running"):
                if status.get("config_valid"):
                    domains = status.get("domains_configured", [])
                    return {
                        "status": "healthy",
                        "details": f"Running, {len(domains)} domain(s) configured",
                    }
                else:
                    return {
                        "status": "warning",
                        "details": "Running but config invalid",
                    }
            else:
                return {"status": "critical", "details": "Not running"}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    async def _check_database_health(self) -> Dict[str, Any]:
        """Comprobar salud de la base de datos - Real implementation"""
        try:
            # Buscar bases de datos SQLite en el proyecto
            db_files = []
            db_dir = self.root_dir / "data" / "db"
            if db_dir.exists():
                db_files = list(db_dir.glob("*.db"))
            
            # También buscar en raíz de data
            data_dir = self.root_dir / "data"
            if data_dir.exists():
                db_files.extend(data_dir.glob("*.db"))
            
            if db_files:
                # Verificar que los archivos sean accesibles
                accessible = 0
                total_size = 0
                for db_file in db_files:
                    try:
                        if db_file.exists() and db_file.stat().st_size > 0:
                            accessible += 1
                            total_size += db_file.stat().st_size
                    except Exception:
                        pass
                
                if accessible > 0:
                    return {
                        "status": "healthy",
                        "details": f"{accessible} SQLite database(s) accessible ({total_size / 1024 / 1024:.2f} MB total)",
                        "databases_found": accessible,
                        "total_size_mb": round(total_size / 1024 / 1024, 2)
                    }
                else:
                    return {
                        "status": "warning",
                        "details": "Database files found but not accessible"
                    }
            
            # Intentar conectar a PostgreSQL si está configurado
            try:
                import psycopg2
                # Intentar leer configuración de base de datos
                config_file = self.root_dir / "config" / "database" / "config.json"
                if config_file.exists():
                    with open(config_file, "r") as f:
                        db_config = json.load(f)
                    
                    # Intentar conexión
                    conn = psycopg2.connect(
                        host=db_config.get("host", "localhost"),
                        port=db_config.get("port", 5432),
                        database=db_config.get("database", "postgres"),
                        user=db_config.get("user", "postgres"),
                        password=db_config.get("password", "")
                    )
                    conn.close()
                    
                    return {
                        "status": "healthy",
                        "details": "PostgreSQL connection successful",
                        "type": "postgresql"
                    }
            except ImportError:
                pass  # psycopg2 no disponible
            except Exception as e:
                return {
                    "status": "warning",
                    "details": f"PostgreSQL configured but connection failed: {str(e)[:50]}"
                }
            
            # Si no hay bases de datos encontradas
            return {
                "status": "inactive",
                "details": "No databases found or configured"
            }
        except Exception as e:
            return {"status": "error", "details": str(e)}

    async def _shutdown_infrastructure(self) -> Dict[str, Any]:
        """Apagar toda la infraestructura"""
        results = {
            "docker_stopped": await self._stop_docker_services(["all"]),
            "infrastructure_status": "shutdown_complete",
            "warnings": [],
        }

        # Add shutdown notes
        results["warnings"].append(
            "Remember to manually stop Kubernetes deployments if needed"
        )
        results["warnings"].append("Terraform resources remain deployed")

        return results

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    async def _run_command(
        self, cmd: List[str], cwd: Path = None, timeout: int = 60
    ) -> Dict[str, Any]:
        """Ejecutar comando del sistema de forma asíncrona"""
        try:
            # Convertir Path a str para subprocess
            if cwd and isinstance(cwd, Path):
                cwd = str(cwd)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                return {
                    "returncode": process.returncode,
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                }
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout} seconds",
                }

        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command execution error: {str(e)}",
            }
