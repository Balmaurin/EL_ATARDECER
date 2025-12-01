#!/usr/bin/env python3
"""
SHEILYS Blockchain Wallet - Billetera para gestión de SHEILYS tokens y NFTs

Implementa funcionalidades completas de wallet para el ecosistema SHEILYS:
- Gestión de claves y direcciones
- Envío y recepción de tokens
- Staking y governance
- Gestión de NFTs
- Backup y recuperación
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .sheilys_blockchain import SHEILYSBlockchain, SHEILYSTransaction, TransactionType
from .sheilys_token import NFTCollection, SHEILYSTokenManager

# Configurar logging
logger = logging.getLogger(__name__)


class WalletKeys:
    """Gestión de claves para SHEILYS wallet"""

    def __init__(self):
        """Inicializar sistema de claves"""
        self.private_key: Optional[bytes] = None
        self.public_key: Optional[bytes] = None
        self.address: Optional[str] = None

    def generate_keys(self) -> str:
        """
        Generar nuevo par de claves usando Ed25519 (REAL)

        Returns:
            str: Address generado
        """
        # Generar par de claves Ed25519 REAL
        private_key_obj = ed25519.Ed25519PrivateKey.generate()
        public_key_obj = private_key_obj.public_key()

        # Serializar claves
        self.private_key = private_key_obj.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        self.public_key = public_key_obj.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

        # Generar address desde clave pública (hash SHA256 de la clave pública)
        public_hash = hashlib.sha256(self.public_key).digest()
        self.address = (
            base64.b64encode(public_hash[:20]).decode("utf-8").rstrip("=")
        )

        return self.address

    def import_private_key(self, private_key_hex: str) -> str:
        """
        Importar clave privada desde formato hexadecimal (Ed25519)

        Args:
            private_key_hex: Clave privada en formato hex (32 bytes = 64 hex chars)

        Returns:
            str: Address correspondiente
        """
        try:
            private_key_bytes = bytes.fromhex(private_key_hex)
            
            # Validar longitud de clave Ed25519 (32 bytes)
            if len(private_key_bytes) != 32:
                raise ValueError(f"Clave privada debe tener 32 bytes, tiene {len(private_key_bytes)}")
            
            # Crear objeto de clave privada Ed25519
            private_key_obj = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            public_key_obj = private_key_obj.public_key()
            
            self.private_key = private_key_bytes
            
            self.public_key = public_key_obj.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            
            # Generar address desde clave pública
            public_hash = hashlib.sha256(self.public_key).digest()
            self.address = (
                base64.b64encode(public_hash[:20]).decode("utf-8").rstrip("=")
            )
            return self.address
        except Exception as e:
            logger.error(f"Error importando clave privada: {e}")
            raise ValueError(f"Invalid private key format: {e}")

    def sign_transaction(self, transaction: SHEILYSTransaction) -> str:
        """
        Firmar transacción usando Ed25519 (REAL)

        Args:
            transaction: Transacción SHEILYS a firmar

        Returns:
            str: Firma en base64
        """
        if not self.private_key:
            raise ValueError("Wallet not initialized")

        # Crear objeto de clave privada Ed25519
        private_key_obj = ed25519.Ed25519PrivateKey.from_private_bytes(self.private_key)
        
        # Crear datos para firmar (hash de la transacción)
        tx_hash = transaction.calculate_hash()
        tx_hash_bytes = bytes.fromhex(tx_hash)
        
        # Firmar con Ed25519
        signature_bytes = private_key_obj.sign(tx_hash_bytes)
        
        # Retornar firma en base64
        return base64.b64encode(signature_bytes).decode("utf-8")
    
    def verify_signature(self, transaction: SHEILYSTransaction, signature_b64: str) -> bool:
        """
        Verificar firma de una transacción
        
        Args:
            transaction: Transacción a verificar
            signature_b64: Firma en base64
            
        Returns:
            bool: True si la firma es válida
        """
        try:
            if not self.public_key:
                return False
            
            # Crear objeto de clave pública Ed25519
            public_key_obj = ed25519.Ed25519PublicKey.from_public_bytes(self.public_key)
            
            # Decodificar firma
            signature_bytes = base64.b64decode(signature_b64)
            
            # Calcular hash de la transacción
            tx_hash = transaction.calculate_hash()
            tx_hash_bytes = bytes.fromhex(tx_hash)
            
            # Verificar firma
            public_key_obj.verify(signature_bytes, tx_hash_bytes)
            return True
        except Exception as e:
            logger.warning(f"Error verificando firma: {e}")
            return False

    def export_private_key(self) -> str:
        """Exportar clave privada en formato hexadecimal"""
        if not self.private_key:
            raise ValueError("No private key available")
        return self.private_key.hex()


class WalletEncryption:
    """Sistema de encriptación para wallet seguro"""

    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        """Derivar clave desde password usando PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        return kdf.derive(password.encode())

    @staticmethod
    def encrypt_data(data: bytes, key: bytes) -> Tuple[bytes, bytes]:
        """
        Encriptar datos

        Returns:
            Tuple[bytes, bytes]: (datos encriptados, IV)
        """
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Padding PKCS7
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padded_data = data + bytes([padding_length]) * padding_length

        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        return encrypted_data, iv

    @staticmethod
    def decrypt_data(encrypted_data: bytes, key: bytes, iv: bytes) -> bytes:
        """
        Desencriptar datos

        Returns:
            bytes: Datos desencriptados
        """
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

        # Remove PKCS7 padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]


class BlockchainWallet:
    """
    SHEILYS Blockchain Wallet - Wallet completa para el ecosistema

    Funcionalidades:
    - Gestión de tokens SHEILYS
    - Staking y rewards
    - Gestión de NFTs
    - Gobernanza del sistema
    - Backup y seguridad
    """

    def __init__(self):
        """Inicializar wallet SHEILYS"""
        self.keys = WalletKeys()
        self.address: Optional[str] = None
        self.encrypted = False

        # Conexión con blockchain y token manager
        self.blockchain: Optional[SHEILYSBlockchain] = None
        self.token_manager: Optional[SHEILYSTokenManager] = None

        # Cache de balances locales
        self.local_balances = {"sheilys": 0.0, "staked_sheilyns": 0.0, "nfts": []}

        # Historial de transacciones
        self.transaction_history: List[Dict[str, Any]] = []

        # Configuración de seguridad
        self.auto_backup = True
        self.backup_path = "./wallet_backups"
        self.security_level = "standard"
        
        # Archivo de wallet encriptada
        self.wallet_file: Optional[str] = None

    def create_wallet(self, password: Optional[str] = None) -> str:
        """
        Crear nueva wallet

        Args:
            password: Password para encriptación (opcional)

        Returns:
            str: Address de la wallet creada
        """
        try:
            # Generar claves
            self.address = self.keys.generate_keys()

            # Si hay password, encriptar
            if password:
                self._encrypt_wallet(password)
                self.encrypted = True

            # Crear directorio de backups
            if self.auto_backup and not os.path.exists(self.backup_path):
                os.makedirs(self.backup_path)

            # Backup inicial
            if self.auto_backup:
                self._create_backup()

            return self.address

        except Exception as e:
            raise Exception(f"Error creando wallet: {e}")

    def load_wallet(
        self, 
        private_key_hex: Optional[str] = None, 
        encrypted_data: Optional[Dict[str, str]] = None,
        password: Optional[str] = None
    ) -> str:
        """
        Cargar wallet desde clave privada o datos encriptados

        Args:
            private_key_hex: Clave privada en formato hex (si no está encriptada)
            encrypted_data: Datos encriptados con encrypted_key, iv, salt
            password: Password si está encriptada

        Returns:
            str: Address de la wallet cargada
        """
        try:
            # Si se proporcionan datos encriptados, desencriptar
            if encrypted_data and password:
                private_key_hex = self._decrypt_wallet_data(encrypted_data, password)
                self.encrypted = True
            
            if not private_key_hex:
                raise ValueError("Debe proporcionar private_key_hex o encrypted_data con password")

            # Importar clave
            self.address = self.keys.import_private_key(private_key_hex)

            logger.info(f"Wallet cargada exitosamente: {self.address}")
            return self.address

        except Exception as e:
            logger.error(f"Error cargando wallet: {e}", exc_info=True)
            raise Exception(f"Error cargando wallet: {e}")

    def _encrypt_wallet(self, password: str):
        """Encriptar wallet con password y PERSISTIR"""
        if not self.keys.private_key:
            logger.warning("No hay clave privada para encriptar")
            return

        # Generar salt
        salt = os.urandom(16)

        # Derivar clave
        key = WalletEncryption.derive_key(password, salt)

        # Encriptar clave privada
        encrypted_key, iv = WalletEncryption.encrypt_data(self.keys.private_key, key)

        # Guardar datos encriptados (AHORA SE PERSISTEN)
        self.encrypted_data = {
            "encrypted_key": base64.b64encode(encrypted_key).decode(),
            "iv": base64.b64encode(iv).decode(),
            "salt": base64.b64encode(salt).decode(),
            "address": self.address,
        }
        
        # PERSISTIR datos encriptados inmediatamente
        if self.auto_backup:
            self._save_encrypted_wallet()
        
        logger.info(f"Wallet encriptada y persistida: {self.address}")
    
    def _save_encrypted_wallet(self):
        """Guardar wallet encriptada en archivo"""
        if not hasattr(self, 'encrypted_data') or not self.encrypted_data:
            return
        
        try:
            if not os.path.exists(self.backup_path):
                os.makedirs(self.backup_path)
            
            wallet_filename = f"wallet_{self.address}.encrypted.json"
            wallet_path = os.path.join(self.backup_path, wallet_filename)
            
            wallet_data = {
                "address": self.address,
                "encrypted_data": self.encrypted_data,
                "backup_timestamp": datetime.now().isoformat(),
                "version": "1.0",
            }
            
            with open(wallet_path, "w") as f:
                json.dump(wallet_data, f, indent=2)
            
            self.wallet_file = wallet_path
            logger.info(f"Wallet encriptada guardada en: {wallet_path}")
            
        except Exception as e:
            logger.error(f"Error guardando wallet encriptada: {e}", exc_info=True)

    def _decrypt_wallet_data(self, encrypted_data_dict: Dict[str, str], password: str) -> str:
        """
        Desencriptar datos de wallet usando AES (IMPLEMENTACIÓN REAL)
        
        Args:
            encrypted_data_dict: Diccionario con encrypted_key, iv, salt
            password: Password para desencriptar
            
        Returns:
            str: Clave privada en formato hex
        """
        try:
            # Extraer datos encriptados
            encrypted_key_b64 = encrypted_data_dict.get("encrypted_key")
            iv_b64 = encrypted_data_dict.get("iv")
            salt_b64 = encrypted_data_dict.get("salt")
            
            if not all([encrypted_key_b64, iv_b64, salt_b64]):
                raise ValueError("Datos encriptados incompletos")
            
            # Decodificar desde base64
            encrypted_key = base64.b64decode(encrypted_key_b64)
            iv = base64.b64decode(iv_b64)
            salt = base64.b64decode(salt_b64)
            
            # Derivar clave desde password
            key = WalletEncryption.derive_key(password, salt)
            
            # Desencriptar
            decrypted_key_bytes = WalletEncryption.decrypt_data(encrypted_key, key, iv)
            
            # Convertir a hex string
            return decrypted_key_bytes.hex()
            
        except Exception as e:
            logger.error(f"Error desencriptando wallet: {e}", exc_info=True)
            raise ValueError(f"Error desencriptando wallet: {e}")

    def connect_to_blockchain(
        self, blockchain: SHEILYSBlockchain, token_manager: SHEILYSTokenManager
    ):
        """
        Conectar wallet con el sistema blockchain

        Args:
            blockchain: Instancia de blockchain
            token_manager: Instancia del token manager
        """
        self.blockchain = blockchain
        self.token_manager = token_manager

    def get_balance(self) -> Dict[str, Any]:
        """
        Obtener balances de la wallet

        Returns:
            dict: Balances completos
        """
        if not self.token_manager or not self.address:
            return self.local_balances.copy()

        try:
            # Actualizar balances desde blockchain
            self.local_balances["sheilys"] = self.token_manager.get_balance(
                self.address
            )
            self.local_balances["staked_sheilyns"] = (
                self.token_manager.get_staked_balance(self.address)
            )
            self.local_balances["nfts"] = self.token_manager.get_user_nfts(self.address)

            return self.local_balances.copy()

        except Exception as e:
            logger.error(f"Error obteniendo balance: {e}", exc_info=True)
            return self.local_balances.copy()

    def send_tokens(self, to_address: str, amount: float) -> bool:
        """
        Enviar tokens SHEILYS con FIRMA REAL

        Args:
            to_address: Dirección destinataria
            amount: Cantidad a enviar

        Returns:
            bool: True si la transacción fue exitosa
        """
        if not self.token_manager or not self.address or not self.blockchain:
            raise ValueError("Wallet not connected to blockchain")

        try:
            # Verificar balance suficiente
            current_balance = self.token_manager.get_balance(self.address)
            if current_balance < amount:
                raise ValueError(f"Insufficient SHEILYS balance. Tiene: {current_balance}, Necesita: {amount}")

            # Crear transacción con hash correcto
            tx_timestamp = time.time()
            tx = SHEILYSTransaction.create_with_hash(
                sender=self.address,
                receiver=to_address,
                amount=amount,
                transaction_type=TransactionType.TRANSFER,
                timestamp=tx_timestamp,
                signature="",  # Se firmará después
                metadata={
                    "transfer_type": "wallet_send",
                    "gas_used": 0.001,
                },
            )

            # FIRMAR transacción con clave privada REAL
            signature = self.keys.sign_transaction(tx)
            tx.signature = signature

            # Agregar transacción firmada a blockchain
            tx_added = self.blockchain.add_transaction(tx)
            if not tx_added:
                logger.error(f"Error: transacción no se pudo agregar a blockchain")
                return False

            # Ejecutar transferencia en token manager
            success = self.token_manager.transfer_tokens(
                self.address, to_address, amount
            )

            if success:
                # Actualizar historial
                self._add_transaction_to_history(
                    {
                        "type": "send",
                        "to": to_address,
                        "amount": amount,
                        "transaction_id": tx.transaction_id,
                        "signature": signature,
                        "timestamp": datetime.now(),
                        "status": "pending",  # Pendiente hasta confirmación en bloque
                    }
                )

                # Actualizar balances locales
                self.local_balances["sheilys"] -= amount
                
                logger.info(f"Tokens enviados: {amount} SHEILYS de {self.address} a {to_address}")

            return success

        except Exception as e:
            logger.error(f"Error enviando tokens: {e}", exc_info=True)
            return False

    def stake_tokens(self, amount: float, pool_name: str = "community_pool") -> bool:
        """
        Stake tokens para rewards

        Args:
            amount: Cantidad a stakear
            pool_name: Pool de staking

        Returns:
            bool: True si el stake fue exitoso
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            success = self.token_manager.stake_tokens(self.address, amount, pool_name)

            if success:
                # Actualizar historial
                self._add_transaction_to_history(
                    {
                        "type": "stake",
                        "amount": amount,
                        "pool": pool_name,
                        "timestamp": datetime.now(),
                        "status": "confirmed",
                    }
                )

                # Actualizar balances locales
                self.local_balances["sheilys"] -= amount
                self.local_balances["staked_sheilyns"] += amount

            return success

        except Exception as e:
            logger.error(f"Error staking tokens: {e}", exc_info=True)
            return False

    def claim_staking_rewards(self) -> float:
        """
        Claim staking rewards acumulados

        Returns:
            float: Cantidad de rewards claimed
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            claimed_amount = self.token_manager.claim_staking_rewards(self.address)

            if claimed_amount > 0:
                # Actualizar historial
                self._add_transaction_to_history(
                    {
                        "type": "claim_rewards",
                        "amount": claimed_amount,
                        "timestamp": datetime.now(),
                        "status": "confirmed",
                    }
                )

                # Actualizar balances locales
                self.local_balances["sheilys"] += claimed_amount

            return claimed_amount

        except Exception as e:
            logger.error(f"Error claiming staking rewards: {e}", exc_info=True)
            return 0.0

    def mint_nft(
        self, collection: NFTCollection, metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Mint nuevo NFT

        Args:
            collection: Colección del NFT
            metadata: Metadata del NFT

        Returns:
            str: Token ID del NFT minteado, o None si falló
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            token_id = self.token_manager.mint_nft(collection, self.address, metadata)

            if token_id:
                # Actualizar historial
                self._add_transaction_to_history(
                    {
                        "type": "mint_nft",
                        "token_id": token_id,
                        "collection": collection.value,
                        "timestamp": datetime.now(),
                        "status": "confirmed",
                    }
                )

                # Actualizar NFTs locales
                self.local_balances["nfts"].append(
                    {
                        "token_id": token_id,
                        "collection": collection.value,
                        "metadata": metadata,
                    }
                )

            return token_id

        except Exception as e:
            logger.error(f"Error minting NFT: {e}", exc_info=True)
            return None

    def transfer_nft(self, token_id: str, to_address: str) -> bool:
        """
        Transferir NFT a otra dirección

        Args:
            token_id: ID del token NFT
            to_address: Dirección destinataria

        Returns:
            bool: True si la transferencia fue exitosa
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            success = self.token_manager.transfer_nft(
                token_id, self.address, to_address
            )

            if success:
                # Actualizar historial
                self._add_transaction_to_history(
                    {
                        "type": "transfer_nft",
                        "token_id": token_id,
                        "to": to_address,
                        "timestamp": datetime.now(),
                        "status": "confirmed",
                    }
                )

                # Actualizar NFTs locales
                self.local_balances["nfts"] = [
                    nft
                    for nft in self.local_balances["nfts"]
                    if nft["token_id"] != token_id
                ]

            return success

        except Exception as e:
            logger.error(f"Error transfering NFT: {e}", exc_info=True)
            return False

    def create_governance_proposal(
        self, title: str, description: str, voting_period_days: int = 7
    ) -> str:
        """
        Crear propuesta de gobernanza

        Args:
            title: Título de la propuesta
            description: Descripción
            voting_period_days: Período de votación

        Returns:
            str: ID de la propuesta creada
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            proposal_id = self.token_manager.create_governance_proposal(
                self.address, title, description, voting_period_days
            )

            self._add_transaction_to_history(
                {
                    "type": "governance_proposal",
                    "proposal_id": proposal_id,
                    "title": title,
                    "timestamp": datetime.now(),
                    "status": "created",
                }
            )

            return proposal_id

        except Exception as e:
            logger.error(f"Error creating governance proposal: {e}", exc_info=True)
            raise

    def vote_on_proposal(self, proposal_id: str, vote_for: bool) -> bool:
        """
        Votar en una propuesta de gobernanza

        Args:
            proposal_id: ID de la propuesta
            vote_for: True para votar a favor

        Returns:
            bool: True si el voto fue registrado
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            success = self.token_manager.vote_on_proposal(
                proposal_id, self.address, vote_for
            )

            if success:
                self._add_transaction_to_history(
                    {
                        "type": "governance_vote",
                        "proposal_id": proposal_id,
                        "vote": "for" if vote_for else "against",
                        "timestamp": datetime.now(),
                        "status": "confirmed",
                    }
                )

            return success

        except Exception as e:
            logger.error(f"Error voting on proposal: {e}", exc_info=True)
            return False

    def _add_transaction_to_history(self, transaction: Dict[str, Any]):
        """Agregar transacción al historial local"""
        self.transaction_history.append(transaction)

        # Mantener historial limitado (últimas 1000 transacciones)
        if len(self.transaction_history) > 1000:
            self.transaction_history = self.transaction_history[-1000:]

    def get_transaction_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener historial de transacciones

        Args:
            limit: Máximo número de transacciones a retornar

        Returns:
            List[Dict]: Historial de transacciones
        """
        return self.transaction_history[-limit:]

    def _create_backup(self):
        """Crear backup de wallet"""
        if not self.auto_backup or not self.address:
            return

        try:
            backup_data = {
                "address": self.address,
                "public_key": (
                    base64.b64encode(self.keys.public_key).decode()
                    if self.keys.public_key
                    else None
                ),
                "balances": self.local_balances,
                "transaction_history": self.transaction_history[
                    -100:
                ],  # Últimas 100 tx
                "backup_timestamp": datetime.now().isoformat(),
                "version": "1.0",
            }

            # INCLUIR CLAVE PRIVADA ENCRIPTADA EN EL BACKUP (CRÍTICO)
            if self.encrypted and hasattr(self, "encrypted_data"):
                backup_data["encrypted_data"] = self.encrypted_data
            elif self.keys.private_key:
                # Si no está encriptada pero hay clave privada, incluirla encriptada en el backup
                # Para que el backup sea realmente recuperable
                if not hasattr(self, "encrypted_data") or not self.encrypted_data:
                    logger.warning("Backup sin encriptación: la clave privada no está encriptada en memoria")
                    # El backup incluirá solo la clave pública, pero advertir al usuario
                    backup_data["warning"] = "Private key not encrypted in this backup. Use encrypted wallet for security."

            backup_filename = (
                f"wallet_backup_{self.address}_{int(datetime.now().timestamp())}.json"
            )
            backup_path = os.path.join(self.backup_path, backup_filename)

            with open(backup_path, "w") as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Backup creado: {backup_path}")

        except Exception as e:
            logger.error(f"Error creando backup: {e}", exc_info=True)

    def export_wallet_data(self, include_encrypted: bool = True) -> Dict[str, Any]:
        """
        Exportar datos de wallet para backup (INCLUYE CLAVE PRIVADA ENCRIPTADA)

        Args:
            include_encrypted: Si incluir datos encriptados en el export

        Returns:
            dict: Datos de wallet completos
        """
        export_data = {
            "address": self.address,
            "balances": self.local_balances,
            "transaction_history": self.transaction_history,
            "encrypted": self.encrypted,
            "export_timestamp": datetime.now().isoformat(),
        }
        
        # INCLUIR CLAVE PRIVADA ENCRIPTADA PARA RECUPERACIÓN REAL
        if include_encrypted and self.encrypted and hasattr(self, "encrypted_data"):
            export_data["encrypted_data"] = self.encrypted_data
        elif include_encrypted and self.keys.private_key:
            # Si no está encriptada, advertir
            export_data["warning"] = "Private key not encrypted. Export is NOT secure."
        
        return export_data

    def import_wallet_data(self, wallet_data: Dict[str, Any]):
        """
        Importar datos de wallet desde backup

        Args:
            wallet_data: Datos de wallet exportados
        """
        self.address = wallet_data.get("address")
        self.local_balances = wallet_data.get("balances", {})
        self.transaction_history = wallet_data.get("transaction_history", [])
        self.encrypted = wallet_data.get("encrypted", False)

    def get_wallet_info(self) -> Dict[str, Any]:
        """Obtener información completa de la wallet"""
        return {
            "address": self.address,
            "balances": self.local_balances,
            "transaction_count": len(self.transaction_history),
            "connected_to_blockchain": self.blockchain is not None,
            "encrypted": self.encrypted,
            "security_level": self.security_level,
            "auto_backup": self.auto_backup,
        }
