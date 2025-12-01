#!/usr/bin/env python3
"""
Script de Backup Automático
============================

Script para ejecutar backups automáticos después de cada indexación.
Puede ejecutarse manualmente o integrarse en el pipeline de indexación.
"""

import sys
from pathlib import Path

# Agregar paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from corpus.tools.backup.backup_manager import IndexBackupManager


def main():
    """Ejecutar backup automático"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Backup automático de índices RAG")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("corpus/universal/latest"),
        help="Path al directorio de índices",
    )
    parser.add_argument(
        "--backup-path",
        type=Path,
        default=Path("corpus/_backups"),
        help="Path para guardar backups",
    )
    parser.add_argument(
        "--snapshot-name",
        type=str,
        default=None,
        help="Nombre del snapshot (auto-generado si no se especifica)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="No incluir embeddings en el backup",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="No comprimir el backup",
    )
    parser.add_argument(
        "--enable-s3",
        action="store_true",
        help="Habilitar upload a S3",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.getenv("S3_BACKUP_BUCKET"),
        help="Bucket S3 para backups",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Listar backups existentes",
    )
    parser.add_argument(
        "--restore",
        type=Path,
        help="Path al backup para restaurar",
    )
    parser.add_argument(
        "--restore-to",
        type=Path,
        help="Path donde restaurar (default: index-path)",
    )
    
    args = parser.parse_args()
    
    # Crear manager
    manager = IndexBackupManager(
        index_base_path=args.index_path,
        backup_base_path=args.backup_path,
        enable_s3=args.enable_s3,
        s3_bucket=args.s3_bucket,
    )
    
    # Listar backups
    if args.list:
        backups = manager.list_backups()
        print(f"\n📋 Backups disponibles: {len(backups)}\n")
        for backup in backups[:20]:  # Mostrar últimos 20
            print(f"  {backup['name']}")
            print(f"    Size: {backup['size_mb']:.2f} MB")
            if 'timestamp' in backup:
                print(f"    Timestamp: {backup['timestamp']}")
            print()
        return
    
    # Restaurar backup
    if args.restore:
        restore_to = args.restore_to or args.index_path
        print(f"🔄 Restaurando backup: {args.restore} -> {restore_to}")
        success = manager.restore_backup(args.restore, restore_to)
        if success:
            print("✅ Restauración completada exitosamente")
            sys.exit(0)
        else:
            print("❌ Error en la restauración")
            sys.exit(1)
    
    # Crear backup
    print(f"🔄 Iniciando backup automático...")
    print(f"   Index path: {args.index_path}")
    print(f"   Backup path: {args.backup_path}")
    
    backup_path = manager.create_backup(
        snapshot_name=args.snapshot_name,
        include_embeddings=not args.skip_embeddings,
        compression=not args.no_compress,
    )
    
    print(f"\n✅ Backup completado exitosamente")
    print(f"   Location: {backup_path}")
    
    # Mostrar tamaño
    if backup_path.is_file():
        size_mb = backup_path.stat().st_size / (1024 * 1024)
    else:
        size_mb = sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file()) / (1024 * 1024)
    
    print(f"   Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()

