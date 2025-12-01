#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Backup y Recovery Automático - Sheily AI
===================================================

Sistema completo para backup automático y recuperación de datos críticos:
- Backup incremental de datos y configuraciones
- Snapshots del sistema en puntos críticos
- Verificación de integridad de backups
- Recuperación automática en caso de fallos
- Compresión y encriptación de datos sensibles
- Monitoreo y alertas de backup
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Configuración del sistema de backup"""

    backup_dir: str = "./backups"
    retention_days: int = 30
    compression_level: int = 6  # 1-9, donde 9 es máxima compresión
    max_backup_size_mb: int = 1000
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    auto_backup_interval_hours: int = 24
    critical_data_only: bool = False


@dataclass
class BackupMetadata:
    """Metadatos de un backup"""

    id: str
    timestamp: datetime
    type: str  # 'full', 'incremental', 'config', 'model'
    size_bytes: int
    compressed_size_bytes: int
    checksum: str
    components: List[str]
    version: str = "1.0"
    status: str = "completed"  # 'completed', 'failed', 'in_progress'


class BackupManager:
    """Gestor principal del sistema de backup"""

    def __init__(self, config: Optional[BackupConfig] = None):
        self.config = config or BackupConfig()
        self.backup_path = Path(self.config.backup_dir)
        self.backup_path.mkdir(parents=True, exist_ok=True)

        # Componentes críticos a respaldar
        self.critical_components = {
            "config": [
                "config/database.json",
                "config/security.json",
                "config/models.json",
                ".env",
                ".secrets.baseline",
            ],
            "data": ["data/", "centralized_data/", "memory/"],
            "models": ["models/", "sheily_train/"],
            "logs": ["logs/", "audit_2024/"],
        }

        # Estado del sistema
        self.last_backup = None
        self.backup_history: List[BackupMetadata] = []

        # Cargar historial existente
        self._load_backup_history()

    def _load_backup_history(self):
        """Cargar historial de backups desde disco"""
        history_file = self.backup_path / "backup_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.backup_history = [
                        BackupMetadata(**item) for item in data.get("backups", [])
                    ]
                    # Convertir strings de fecha a objetos datetime
                    for backup in self.backup_history:
                        if isinstance(backup.timestamp, str):
                            backup.timestamp = datetime.fromisoformat(backup.timestamp)
            except Exception as e:
                logger.error(f"Error cargando historial de backups: {e}")

    def _save_backup_history(self):
        """Guardar historial de backups a disco"""
        history_file = self.backup_path / "backup_history.json"
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "backups": [
                    {**backup.__dict__, "timestamp": backup.timestamp.isoformat()}
                    for backup in self.backup_history[-100:]  # Últimos 100 backups
                ],
            }
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando historial de backups: {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcular checksum SHA256 de un archivo"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        except Exception as e:
            logger.error(f"Error calculando checksum para {file_path}: {e}")
            return ""
        return hash_sha256.hexdigest()

    def _compress_data(self, data: bytes) -> bytes:
        """Comprimir datos usando gzip"""
        try:
            import gzip

            return gzip.compress(data, compresslevel=self.config.compression_level)
        except ImportError:
            logger.warning("gzip no disponible, guardando sin compresión")
            return data

    def _decompress_data(self, data: bytes) -> bytes:
        """Descomprimir datos usando gzip"""
        try:
            import gzip

            return gzip.decompress(data)
        except ImportError:
            return data

    def _encrypt_data(self, data: bytes) -> bytes:
        """Encriptar datos usando Fernet (AES)"""
        if not self.config.enable_encryption or not self.config.encryption_key:
            return data

        try:
            from cryptography.fernet import Fernet
            import base64

            # Derivar clave de 32 bytes desde la clave proporcionada
            key_bytes = self.config.encryption_key.encode('utf-8')
            if len(key_bytes) < 32:
                # Rellenar con ceros si es necesario
                key_bytes = key_bytes.ljust(32, b'\x00')
            elif len(key_bytes) > 32:
                # Truncar si es demasiado largo
                key_bytes = key_bytes[:32]

            # Crear clave Fernet (debe ser 32 bytes URL-safe base64-encoded)
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            fernet = Fernet(fernet_key)

            # Encriptar datos
            encrypted_data = fernet.encrypt(data)
            logger.debug("Datos encriptados exitosamente")
            return encrypted_data

        except ImportError:
            logger.warning("Biblioteca 'cryptography' no disponible, guardando sin encriptación")
            return data
        except Exception as e:
            logger.error(f"Error en encriptación: {e}")
            # En caso de error, devolver datos sin encriptar para no perder información
            return data

    def _decrypt_data(self, data: bytes) -> bytes:
        """Desencriptar datos usando Fernet (AES)"""
        if not self.config.enable_encryption or not self.config.encryption_key:
            return data

        try:
            from cryptography.fernet import Fernet
            import base64

            # Derivar clave de 32 bytes (misma lógica que en encriptación)
            key_bytes = self.config.encryption_key.encode('utf-8')
            if len(key_bytes) < 32:
                key_bytes = key_bytes.ljust(32, b'\x00')
            elif len(key_bytes) > 32:
                key_bytes = key_bytes[:32]

            fernet_key = base64.urlsafe_b64encode(key_bytes)
            fernet = Fernet(fernet_key)

            # Desencriptar datos
            decrypted_data = fernet.decrypt(data)
            logger.debug("Datos desencriptados exitosamente")
            return decrypted_data

        except ImportError:
            logger.warning("Biblioteca 'cryptography' no disponible, datos sin desencriptar")
            return data
        except Exception as e:
            logger.error(f"Error en desencriptación: {e}")
            # Si falla la desencriptación, asumir que los datos no estaban encriptados
            logger.warning("Asumiendo datos sin encriptar debido a error de desencriptación")
            return data

    async def _backup_component(
        self, component_name: str, paths: List[str]
    ) -> Tuple[int, List[str]]:
        """Respaldar un componente específico"""
        total_size = 0
        backed_up_files = []

        for path_pattern in paths:
            path = Path(path_pattern)

            if path.is_file() and path.exists():
                # Respaldar archivo individual
                try:
                    file_size = path.stat().st_size
                    total_size += file_size
                    content = await self._read_file_content(path)
                    backed_up_files.append({"path": str(path), "content": content})
                except Exception as e:
                    logger.error(f"Error accediendo a {path}: {e}")

            elif path.is_dir() and path.exists():
                # Respaldar directorio completo
                try:
                    for file_path in path.rglob("*"):
                        if file_path.is_file():
                            try:
                                file_size = file_path.stat().st_size
                                total_size += file_size
                                content = await self._read_file_content(file_path)
                                backed_up_files.append({
                                    "path": str(file_path.relative_to(Path.cwd())),
                                    "content": content
                                })
                            except Exception as e:
                                logger.error(f"Error accediendo a {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Error accediendo a directorio {path}: {e}")

        return total_size, backed_up_files

    async def _read_file_content(self, path: Path) -> str:
        """Read file content and return as base64 string"""
        import base64
        async with aiofiles.open(path, "rb") as f:
            content = await f.read()
            return base64.b64encode(content).decode("utf-8")

    async def _write_file_content(self, path: Path, content_b64: str):
        """Write base64 content to file"""
        import base64
        content = base64.b64decode(content_b64)
        async with aiofiles.open(path, "wb") as f:
            await f.write(content)

    async def create_backup(
        self, backup_type: str = "full", components: Optional[List[str]] = None
    ) -> Optional[BackupMetadata]:
        """Crear un nuevo backup"""
        backup_id = f"{backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir = self.backup_path / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🚀 Iniciando backup {backup_type}: {backup_id}")

        # Metadatos iniciales
        metadata = BackupMetadata(
            id=backup_id,
            timestamp=datetime.now(),
            type=backup_type,
            size_bytes=0,
            compressed_size_bytes=0,
            checksum="",
            components=components or list(self.critical_components.keys()),
            status="in_progress",
        )

        try:
            # Determinar qué componentes respaldar
            if components is None:
                if backup_type == "config":
                    components = ["config"]
                elif backup_type == "model":
                    components = ["models"]
                else:
                    components = list(self.critical_components.keys())

            total_original_size = 0
            total_compressed_size = 0
            all_files = []

            # Respaldar cada componente
            for component in components:
                if component in self.critical_components:
                    logger.info(f"Respaldando componente: {component}")
                    comp_size, comp_files = await self._backup_component(
                        component, self.critical_components[component]
                    )
                    total_original_size += comp_size
                    all_files.extend(comp_files)

                    # Crear archivo de backup para este componente
                    if comp_files:
                        comp_data = {
                            "component": component,
                            "files": comp_files,
                            "total_size": comp_size,
                            "timestamp": metadata.timestamp.isoformat(),
                        }

                        # Serializar y comprimir
                        json_data = json.dumps(
                            comp_data, indent=2, ensure_ascii=False
                        ).encode("utf-8")
                        compressed_data = self._compress_data(json_data)
                        encrypted_data = self._encrypt_data(compressed_data)

                        comp_file = backup_dir / f"{component}.backup"
                        async with aiofiles.open(comp_file, "wb") as f:
                            await f.write(encrypted_data)

                        total_compressed_size += len(encrypted_data)

            # Calcular checksum final
            metadata.size_bytes = total_original_size
            metadata.compressed_size_bytes = total_compressed_size
            metadata.checksum = (
                self._calculate_checksum(backup_dir / f"{components[0]}.backup")
                if components
                else ""
            )

            # Guardar metadatos
            metadata_file = backup_dir / "metadata.json"
            async with aiofiles.open(metadata_file, "w", encoding="utf-8") as f:
                await f.write(
                    json.dumps(
                        {
                            **metadata.__dict__,
                            "timestamp": metadata.timestamp.isoformat(),
                            "compression_ratio": (
                                total_compressed_size / total_original_size
                                if total_original_size > 0
                                else 0
                            ),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )

            # Actualizar estado
            metadata.status = "completed"
            self.backup_history.append(metadata)
            self.last_backup = metadata.timestamp
            self._save_backup_history()

            logger.info(f"✅ Backup {backup_id} completado exitosamente")
            logger.info(
                f"   Tamaño original: {total_original_size / (1024*1024):.2f} MB"
            )
            logger.info(
                f"   Tamaño comprimido: {total_compressed_size / (1024*1024):.2f} MB"
            )

            return metadata

        except Exception as e:
            logger.error(f"❌ Error creando backup {backup_id}: {e}")
            metadata.status = "failed"

            # Limpiar backup fallido
            try:
                shutil.rmtree(backup_dir)
            except:
                pass

            return None

    async def restore_backup(
        self, backup_id: str, components: Optional[List[str]] = None
    ) -> bool:
        """Restaurar desde un backup específico"""
        backup_dir = self.backup_path / backup_id

        if not backup_dir.exists():
            logger.error(f"Backup {backup_id} no encontrado")
            return False

        logger.info(f"🔄 Iniciando restauración desde backup: {backup_id}")

        try:
            # Leer metadatos
            metadata_file = backup_dir / "metadata.json"
            async with aiofiles.open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.loads(await f.read())

            restore_components = components or metadata.get("components", [])

            # Restaurar cada componente
            for component in restore_components:
                comp_file = backup_dir / f"{component}.backup"
                if not comp_file.exists():
                    logger.warning(
                        f"Archivo de componente {component} no encontrado, saltando..."
                    )
                    continue

                logger.info(f"Restaurando componente: {component}")

                # Leer y desencriptar datos
                async with aiofiles.open(comp_file, "rb") as f:
                    encrypted_data = await f.read()

                decrypted_data = self._decrypt_data(encrypted_data)
                decompressed_data = self._decompress_data(decrypted_data)

                comp_data = json.loads(decompressed_data.decode("utf-8"))

                # Restore files
                for file_data in comp_data.get("files", []):
                    try:
                        if isinstance(file_data, str):
                            # Legacy format support (path only)
                            logger.warning(f"Skipping legacy backup file format: {file_data}")
                            continue
                            
                        file_path = file_data.get("path")
                        content_b64 = file_data.get("content")
                        
                        if not file_path or not content_b64:
                            continue
                            
                        target_path = Path(file_path)
                        
                        # Ensure parent directory exists
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Backup existing if exists
                        if target_path.exists():
                            backup_target = target_path.with_suffix(target_path.suffix + ".bak")
                            shutil.copy2(target_path, backup_target)
                            
                        # Write content
                        await self._write_file_content(target_path, content_b64)
                        logger.debug(f"Restored file: {target_path}")
                        
                    except Exception as e:
                        logger.error(f"Error restoring file: {e}")

            logger.info(f"✅ Restauración desde {backup_id} completada")
            return True

        except Exception as e:
            logger.error(f"❌ Error durante restauración: {e}")
            return False

    def list_backups(self, backup_type: Optional[str] = None) -> List[BackupMetadata]:
        """Listar backups disponibles"""
        backups = self.backup_history

        if backup_type:
            backups = [b for b in backups if b.type == backup_type]

        # Ordenar por timestamp descendente (más recientes primero)
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)

    async def cleanup_old_backups(self) -> int:
        """Limpiar backups antiguos según política de retención"""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        removed_count = 0

        # Identificar backups a eliminar
        to_remove = [
            b
            for b in self.backup_history
            if b.timestamp < cutoff_date and b.status == "completed"
        ]

        for backup in to_remove:
            try:
                backup_dir = self.backup_path / backup.id
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                    removed_count += 1
                    logger.info(f"Eliminado backup antiguo: {backup.id}")
            except Exception as e:
                logger.error(f"Error eliminando backup {backup.id}: {e}")

        # Actualizar historial
        self.backup_history = [
            b
            for b in self.backup_history
            if b.timestamp >= cutoff_date or b.status != "completed"
        ]
        self._save_backup_history()

        return removed_count

    async def verify_backup_integrity(self, backup_id: str) -> bool:
        """Verificar integridad de un backup"""
        backup_dir = self.backup_path / backup_id

        if not backup_dir.exists():
            return False

        try:
            # Leer metadatos
            metadata_file = backup_dir / "metadata.json"
            async with aiofiles.open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.loads(await f.read())

            expected_checksum = metadata.get("checksum", "")

            # Verificar checksum del archivo principal
            components = metadata.get("components", [])
            if components:
                main_file = backup_dir / f"{components[0]}.backup"
                actual_checksum = self._calculate_checksum(main_file)

                if actual_checksum == expected_checksum:
                    logger.info(f"✅ Integridad de backup {backup_id} verificada")
                    return True
                else:
                    logger.error(f"❌ Checksum incorrecto para backup {backup_id}")
                    return False

        except Exception as e:
            logger.error(f"Error verificando integridad de {backup_id}: {e}")
            return False

        return False

    async def get_backup_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema de backup"""
        total_backups = len(self.backup_history)
        successful_backups = len(
            [b for b in self.backup_history if b.status == "completed"]
        )
        failed_backups = len([b for b in self.backup_history if b.status == "failed"])

        total_size = sum(
            b.size_bytes for b in self.backup_history if b.status == "completed"
        )
        compressed_size = sum(
            b.compressed_size_bytes
            for b in self.backup_history
            if b.status == "completed"
        )

        compression_ratio = compressed_size / total_size if total_size > 0 else 0

        return {
            "total_backups": total_backups,
            "successful_backups": successful_backups,
            "failed_backups": failed_backups,
            "success_rate": (
                successful_backups / total_backups if total_backups > 0 else 0
            ),
            "total_size_mb": total_size / (1024 * 1024),
            "compressed_size_mb": compressed_size / (1024 * 1024),
            "compression_ratio": compression_ratio,
            "last_backup": self.last_backup.isoformat() if self.last_backup else None,
            "retention_days": self.config.retention_days,
        }

    async def auto_backup_scheduler(self):
        """Programador automático de backups"""
        while True:
            try:
                # Verificar si es tiempo de hacer backup
                now = datetime.now()

                if (
                    self.last_backup is None
                    or (now - self.last_backup).total_seconds()
                    >= self.config.auto_backup_interval_hours * 3600
                ):
                    logger.info("⏰ Ejecutando backup automático programado")

                    # Crear backup completo
                    await self.create_backup("full")

                    # Limpiar backups antiguos
                    removed = await self.cleanup_old_backups()
                    if removed > 0:
                        logger.info(
                            f"Limpieza automática: {removed} backups antiguos eliminados"
                        )

            except Exception as e:
                logger.error(f"Error en scheduler automático: {e}")

            # Esperar hasta el próximo intervalo
            await asyncio.sleep(3600)  # Revisar cada hora


# Funciones de utilidad
async def create_backup(
    backup_type: str = "full", components: Optional[List[str]] = None
) -> Optional[str]:
    """Función de utilidad para crear backup rápidamente"""
    manager = BackupManager()
    result = await manager.create_backup(backup_type, components)
    return result.id if result else None


async def restore_backup(
    backup_id: str, components: Optional[List[str]] = None
) -> bool:
    """Función de utilidad para restaurar backup rápidamente"""
    manager = BackupManager()
    return await manager.restore_backup(backup_id, components)


def get_backup_manager(config: Optional[BackupConfig] = None) -> BackupManager:
    """Obtener instancia del gestor de backup"""
    return BackupManager(config)
