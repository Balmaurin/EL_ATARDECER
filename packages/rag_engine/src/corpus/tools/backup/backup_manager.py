#!/usr/bin/env python3
"""
Sistema de Backup y Restauración de Índices
============================================

Sistema para hacer backup automático de índices FAISS, ChromaDB, BM25
y otros componentes del sistema RAG, con soporte para backup offsite
en S3/MinIO.
"""

import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class IndexBackupManager:
    """Gestor de backups de índices RAG"""

    def __init__(
        self,
        index_base_path: Path,
        backup_base_path: Path = Path("corpus/_backups"),
        enable_s3: bool = False,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
    ):
        self.index_base_path = Path(index_base_path)
        self.backup_base_path = Path(backup_base_path)
        self.backup_base_path.mkdir(parents=True, exist_ok=True)
        
        self.enable_s3 = enable_s3
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix or "rag_indexes"

        # Componentes a respaldar
        self.components = {
            "faiss": ["*.faiss", "*.index"],
            "chromadb": ["chromadb/**"],
            "bm25": ["bm25/**", "*.toc"],
            "embeddings": ["*.parquet", "embeddings.parquet"],
            "metadata": ["*.json", "metadata*.json"],
        }

    def create_backup(
        self,
        snapshot_name: Optional[str] = None,
        include_embeddings: bool = True,
        compression: bool = True,
    ) -> Path:
        """Crear backup completo de todos los índices"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = snapshot_name or f"backup_{timestamp}"
        
        backup_dir = self.backup_base_path / snapshot_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔄 Iniciando backup: {snapshot_name}")

        # Crear metadata del backup
        metadata = {
            "snapshot_name": snapshot_name,
            "timestamp": timestamp,
            "index_base_path": str(self.index_base_path),
            "components": {},
            "version": "1.0",
        }

        # Respaldar cada componente
        for component, patterns in self.components.items():
            if component == "embeddings" and not include_embeddings:
                continue

            component_backup_dir = backup_dir / component
            component_backup_dir.mkdir(parents=True, exist_ok=True)

            files_backed_up = self._backup_component(
                component, patterns, component_backup_dir
            )

            metadata["components"][component] = {
                "files_count": len(files_backed_up),
                "files": files_backed_up,
            }

        # Guardar metadata
        metadata_path = backup_dir / "backup_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Backup completado: {backup_dir}")

        # Compresión si está habilitada
        if compression:
            compressed_backup = self._compress_backup(backup_dir)
            
            # Upload a S3 si está habilitado
            if self.enable_s3 and self.s3_bucket:
                self._upload_to_s3(compressed_backup)
            
            return compressed_backup
        
        return backup_dir

    def _backup_component(
        self, component: str, patterns: List[str], dest_dir: Path
    ) -> List[str]:
        """Respaldar un componente específico"""
        files_backed_up = []

        if component == "faiss":
            # Buscar archivos FAISS
            for pattern in patterns:
                for file_path in self.index_base_path.rglob(pattern):
                    if file_path.is_file():
                        dest_file = dest_dir / file_path.name
                        shutil.copy2(file_path, dest_file)
                        files_backed_up.append(str(file_path.relative_to(self.index_base_path)))
                        logger.debug(f"  📄 Backup: {file_path.name}")

        elif component == "chromadb":
            # Respaldar directorio completo de ChromaDB
            chroma_path = self.index_base_path / "chromadb"
            if chroma_path.exists() and chroma_path.is_dir():
                dest_chroma = dest_dir / "chromadb"
                shutil.copytree(chroma_path, dest_chroma, dirs_exist_ok=True)
                files_backed_up.append("chromadb/")

        elif component == "bm25":
            # Respaldar índices BM25 (Whoosh/Tantivy)
            bm25_path = self.index_base_path / "index" / "bm25"
            if bm25_path.exists():
                dest_bm25 = dest_dir / "bm25"
                shutil.copytree(bm25_path, dest_bm25, dirs_exist_ok=True)
                files_backed_up.append("index/bm25/")

        elif component == "embeddings":
            # Respaldar embeddings parquet
            for pattern in patterns:
                for file_path in self.index_base_path.rglob(pattern):
                    if file_path.is_file():
                        dest_file = dest_dir / file_path.name
                        shutil.copy2(file_path, dest_file)
                        files_backed_up.append(str(file_path.relative_to(self.index_base_path)))

        elif component == "metadata":
            # Respaldar archivos JSON de metadata
            for pattern in patterns:
                for file_path in self.index_base_path.rglob(pattern):
                    if file_path.is_file():
                        dest_file = dest_dir / file_path.name
                        shutil.copy2(file_path, dest_file)
                        files_backed_up.append(str(file_path.relative_to(self.index_base_path)))

        return files_backed_up

    def _compress_backup(self, backup_dir: Path) -> Path:
        """Comprimir backup en tar.gz"""
        archive_path = Path(f"{backup_dir}.tar.gz")
        
        logger.info(f"📦 Comprimiendo backup: {archive_path.name}")
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(backup_dir, arcname=backup_dir.name)
        
        # Eliminar directorio sin comprimir para ahorrar espacio
        shutil.rmtree(backup_dir)
        
        logger.info(f"✅ Backup comprimido: {archive_path}")
        return archive_path

    def _upload_to_s3(self, backup_path: Path) -> None:
        """Upload backup a S3/MinIO"""
        try:
            import boto3
            from botocore.exceptions import ClientError

            s3_client = boto3.client("s3")

            s3_key = f"{self.s3_prefix}/{backup_path.name}"

            logger.info(f"☁️  Uploading to S3: s3://{self.s3_bucket}/{s3_key}")

            s3_client.upload_file(str(backup_path), self.s3_bucket, s3_key)

            logger.info(f"✅ Upload completado: s3://{self.s3_bucket}/{s3_key}")

        except ImportError:
            logger.warning("⚠️  boto3 no disponible. Saltando upload a S3.")
        except ClientError as e:
            logger.error(f"❌ Error upload a S3: {e}")

    def restore_backup(
        self,
        backup_path: Path,
        restore_to: Optional[Path] = None,
    ) -> bool:
        """Restaurar backup desde archivo comprimido o directorio"""
        restore_to = Path(restore_to) if restore_to else self.index_base_path
        restore_to.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔄 Restaurando backup: {backup_path} -> {restore_to}")

        try:
            # Descomprimir si es necesario
            if backup_path.suffix == ".gz":
                backup_dir = self._decompress_backup(backup_path)
            else:
                backup_dir = backup_path

            # Leer metadata
            metadata_path = backup_dir / "backup_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.info(f"📋 Restaurando snapshot: {metadata.get('snapshot_name')}")

            # Restaurar cada componente
            for component in self.components.keys():
                component_backup = backup_dir / component
                if component_backup.exists():
                    self._restore_component(component, component_backup, restore_to)

            logger.info(f"✅ Restauración completada: {restore_to}")
            return True

        except Exception as e:
            logger.error(f"❌ Error restaurando backup: {e}")
            return False

    def _decompress_backup(self, archive_path: Path) -> Path:
        """Descomprimir backup"""
        extract_dir = archive_path.parent / archive_path.stem.replace(".tar", "")
        extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📦 Descomprimiendo: {archive_path.name}")

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir.parent)

        return extract_dir

    def _restore_component(
        self, component: str, component_backup: Path, restore_to: Path
    ) -> None:
        """Restaurar un componente específico"""
        if component == "chromadb":
            dest = restore_to / "chromadb"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(component_backup / "chromadb", dest)

        elif component == "bm25":
            dest = restore_to / "index" / "bm25"
            dest.mkdir(parents=True, exist_ok=True)
            if (component_backup / "bm25").exists():
                shutil.copytree(
                    component_backup / "bm25", dest, dirs_exist_ok=True
                )

        else:
            # Copiar archivos individuales
            for file_path in component_backup.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(component_backup)
                    dest_file = restore_to / relative_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_file)

        logger.debug(f"  ✅ Restaurado: {component}")

    def list_backups(self) -> List[Dict]:
        """Listar backups disponibles"""
        backups = []

        for backup_path in self.backup_base_path.iterdir():
            if backup_path.is_dir() or backup_path.suffix == ".gz":
                metadata_path = (
                    backup_path / "backup_metadata.json"
                    if backup_path.is_dir()
                    else backup_path.parent
                    / backup_path.stem.replace(".tar", "")
                    / "backup_metadata.json"
                )

                backup_info = {
                    "path": str(backup_path),
                    "name": backup_path.name,
                    "size_mb": (
                        sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file())
                        / (1024 * 1024)
                        if backup_path.is_dir()
                        else backup_path.stat().st_size / (1024 * 1024)
                    ),
                }

                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        backup_info.update(metadata)
                    except:
                        pass

                backups.append(backup_info)

        return sorted(backups, key=lambda x: x.get("timestamp", ""), reverse=True)

    def restore_from_s3(
        self, s3_key: str, restore_to: Optional[Path] = None
    ) -> bool:
        """Restaurar backup desde S3"""
        try:
            import boto3
            from botocore.exceptions import ClientError

            s3_client = boto3.client("s3")

            # Descargar a directorio temporal
            temp_backup = self.backup_base_path / "temp_restore.tar.gz"

            logger.info(f"☁️  Descargando desde S3: s3://{self.s3_bucket}/{s3_key}")

            s3_client.download_file(self.s3_bucket, s3_key, str(temp_backup))

            # Restaurar desde archivo descargado
            success = self.restore_backup(temp_backup, restore_to)

            # Limpiar archivo temporal
            temp_backup.unlink()

            return success

        except ImportError:
            logger.error("❌ boto3 no disponible")
            return False
        except ClientError as e:
            logger.error(f"❌ Error descargando desde S3: {e}")
            return False


if __name__ == "__main__":
    # Demo de uso
    manager = IndexBackupManager(
        index_base_path=Path("corpus/universal/latest"),
        backup_base_path=Path("corpus/_backups"),
    )

    # Crear backup
    backup_path = manager.create_backup()
    print(f"✅ Backup creado: {backup_path}")

    # Listar backups
    backups = manager.list_backups()
    print(f"\n📋 Backups disponibles: {len(backups)}")
    for backup in backups[:5]:
        print(f"  - {backup['name']}: {backup['size_mb']:.2f} MB")

