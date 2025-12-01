#!/usr/bin/env python3
"""
SHEILYS Blockchain Core
Implementación del núcleo blockchain para el token SHEILYS de Sheily AI MCP Enterprise

Características:
- Proof-of-Stake (PoS) optimizado para enterprise
- Minting controlado por sistema IA
- Integración con gamificación y NFTs
- Zero-trust security validation
"""

import hashlib
import json
import logging
import secrets
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Configurar logging
logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Tipos de transacciones soportadas por SHEILYS blockchain"""

    TRANSFER = "transfer"
    STAKE = "stake"
    UNSTAKE = "unstake"
    REWARD = "reward"
    NFT_MINT = "nft_mint"
    NFT_TRANSFER = "nft_transfer"
    GAMIFICATION = "gamification"


class BlockStatus(Enum):
    """Estados de los bloques"""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"


@dataclass
class SHEILYSTransaction:
    """Transacción SHEILYS blockchain"""

    transaction_id: str
    sender: str
    receiver: str
    amount: float
    transaction_type: TransactionType
    timestamp: float
    signature: str
    metadata: Dict[str, Any]
    gas_used: float = 0.0
    block_height: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        data = asdict(self)
        data["transaction_type"] = self.transaction_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SHEILYSTransaction":
        """Crear desde diccionario"""
        data_copy = data.copy()
        data_copy["transaction_type"] = TransactionType(data["transaction_type"])
        return cls(**data_copy)

    def calculate_hash(self) -> str:
        """Calcular hash de la transacción (sin incluir transaction_id para evitar circularidad)"""
        tx_data = {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "transaction_type": self.transaction_type.value,
            "timestamp": self.timestamp,
            "metadata": json.dumps(self.metadata, sort_keys=True),
            "signature": self.signature,
        }
        tx_string = json.dumps(tx_data, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    @classmethod
    def create_with_hash(
        cls,
        sender: str,
        receiver: str,
        amount: float,
        transaction_type: TransactionType,
        timestamp: float,
        signature: str,
        metadata: Dict[str, Any],
        gas_used: float = 0.0,
        block_height: Optional[int] = None,
    ) -> "SHEILYSTransaction":
        """
        Crear transacción con hash calculado automáticamente
        
        Este método asegura que transaction_id sea el hash correcto
        """
        # Calcular hash ANTES de crear la transacción
        tx_data = {
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "transaction_type": transaction_type.value,
            "timestamp": timestamp,
            "metadata": json.dumps(metadata, sort_keys=True),
            "signature": signature,
        }
        tx_hash = hashlib.sha256(
            json.dumps(tx_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Crear transacción con transaction_id = hash
        return cls(
            transaction_id=tx_hash,
            sender=sender,
            receiver=receiver,
            amount=amount,
            transaction_type=transaction_type,
            timestamp=timestamp,
            signature=signature,
            metadata=metadata,
            gas_used=gas_used,
            block_height=block_height,
        )


@dataclass
class SHEILYSBlock:
    """Bloque SHEILYS blockchain"""

    block_height: int
    previous_hash: str
    timestamp: float
    transactions: List[SHEILYSTransaction]
    validator: str  # Address del validador (PoS)
    block_hash: str = ""
    status: BlockStatus = BlockStatus.PENDING
    difficulty: int = 1  # Para compatibilidad futura con PoW si es necesario
    nonce: str = ""

    def calculate_hash(self) -> str:
        """Calcular hash del bloque (Proof-of-Stake)"""
        block_data = {
            "block_height": self.block_height,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "transactions": [tx.calculate_hash() for tx in self.transactions],
            "validator": self.validator,
            "nonce": self.nonce,
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class SHEILYSBlockchain:
    """
    SHEILYS Blockchain Core - Sistema blockchain enterprise-grade

    Implementa Proof-of-Stake con características optimizadas para Sheily AI MCP:
    - Validación por agentes especializados
    - Minting controlado por IA
    - Integración con gamificación
    - Zero-trust security
    """

    def __init__(self, genesis_validator: str = "sheily_system_validator"):
        """Inicializar blockchain SHEILYS"""
        self.chain: List[SHEILYSBlock] = []
        self.pending_transactions: List[SHEILYSTransaction] = []
        self.stakes: Dict[str, float] = {}  # Address -> stake amount
        self.validators: List[str] = [genesis_validator]
        self.total_supply: float = 0.0
        self.circulating_supply: float = 0.0
        self.burned_supply: float = 0.0  # Tokens quemados
        self.block_time: int = 60  # 1 minuto entre bloques
        self.max_block_size: int = 1000  # Máximo 1000 transacciones por bloque
        self.min_stake: float = 1000.0  # Stake mínimo para validar

        # Sistema de balances REAL - trackea balances de todas las direcciones
        self.balances: Dict[str, float] = {}  # Address -> balance

        # Gamification integration
        self.gamification_contracts: Dict[str, Dict[str, Any]] = {}
        self.nft_contracts: Dict[str, Dict[str, Any]] = {}

        # Initialize with genesis block
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Crear bloque génesis"""
        genesis_amount = 1000000.0  # 1 millón SHEILYS iniciales
        genesis_timestamp = time.time()
        
        # Usar método create_with_hash para genesis transaction
        genesis_tx = SHEILYSTransaction.create_with_hash(
            sender="sheily_system",
            receiver="genesis_fund",
            amount=genesis_amount,
            transaction_type=TransactionType.REWARD,
            timestamp=genesis_timestamp,
            signature="genesis_signature_system",
            metadata={"genesis": True, "purpose": "initial_supply"},
        )

        genesis_block = SHEILYSBlock(
            block_height=0,
            previous_hash="0" * 64,
            timestamp=time.time(),
            transactions=[genesis_tx],
            validator="genesis_validator",
        )

        genesis_block.block_hash = genesis_block.calculate_hash()
        genesis_block.status = BlockStatus.FINALIZED
        genesis_block.nonce = secrets.token_hex(16)

        self.chain.append(genesis_block)
        self.total_supply = genesis_amount
        self.circulating_supply = genesis_amount
        self.burned_supply = 0.0
        
        # Inicializar balance del genesis_fund
        self.balances["genesis_fund"] = genesis_amount

    def add_transaction(self, transaction: SHEILYSTransaction) -> bool:
        """
        Agregar transacción al pool de transacciones pendientes

        Returns:
            bool: True si la transacción fue agregada exitosamente
        """
        try:
            # Validar transacción
            if not self._validate_transaction(transaction):
                return False

            # Verificar si ya existe
            if any(
                tx.transaction_id == transaction.transaction_id
                for tx in self.pending_transactions
            ):
                return False

            self.pending_transactions.append(transaction)
            return True

        except Exception as e:
            logger.error(f"Error agregando transacción: {e}", exc_info=True)
            return False

    def _validate_transaction(self, transaction: SHEILYSTransaction) -> bool:
        """Validar transacción básica"""
        try:
            # Validar campos requeridos
            if not all(
                [
                    transaction.transaction_id,
                    transaction.sender,
                    transaction.receiver,
                    transaction.amount > 0,
                    transaction.timestamp > 0,
                    transaction.signature,
                ]
            ):
                logger.warning(f"Transacción {transaction.transaction_id}: campos requeridos faltantes")
                return False

            # Validar timestamp (ventana más amplia para relojes desincronizados)
            current_time = time.time()
            if (
                transaction.timestamp > current_time + 600
            ):  # Máximo 10 minutos en el futuro
                logger.warning(f"Transacción {transaction.transaction_id}: timestamp futuro inválido")
                return False
            if (
                transaction.timestamp < current_time - 7200
            ):  # Máximo 2 horas en el pasado
                logger.warning(f"Transacción {transaction.transaction_id}: timestamp muy antiguo")
                return False

            # Validar hash - transaction_id DEBE ser igual al hash calculado
            calculated_hash = transaction.calculate_hash()
            if calculated_hash != transaction.transaction_id:
                logger.warning(
                    f"Transacción {transaction.transaction_id}: hash inválido. "
                    f"Esperado: {transaction.transaction_id}, Calculado: {calculated_hash}"
                )
                return False

            # Validar balance para transacciones que requieren fondos
            if transaction.transaction_type in [TransactionType.TRANSFER, TransactionType.STAKE]:
                sender_balance = self.balances.get(transaction.sender, 0.0)
                if sender_balance < transaction.amount:
                    logger.warning(
                        f"Transacción {transaction.transaction_id}: balance insuficiente. "
                        f"Tiene: {sender_balance}, Necesita: {transaction.amount}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validando transacción {transaction.transaction_id}: {e}")
            return False

    def create_block(self, validator_address: str) -> Optional[SHEILYSBlock]:
        """
        Crear nuevo bloque con transacciones pendientes

        Args:
            validator_address: Address del validador (stake holder)

        Returns:
            SHEILYSBlock: Nuevo bloque o None si no se puede crear
        """
        try:
            # Verificar que el validador tiene stake suficiente
            if self.stakes.get(validator_address, 0) < self.min_stake:
                return None

            # Limpiar transacciones expiradas
            self._clean_expired_transactions()

            # Obtener transacciones para el bloque
            transactions = self.pending_transactions[: self.max_block_size]

            if not transactions:
                return None

            # Crear bloque
            latest_block = self.chain[-1]
            new_block = SHEILYSBlock(
                block_height=latest_block.block_height + 1,
                previous_hash=latest_block.block_hash,
                timestamp=time.time(),
                transactions=transactions,
                validator=validator_address,
            )

            # Proof of Stake - nonce basado en stake del validador
            validator_stake = self.stakes.get(validator_address, 0.0)
            stake_hash = hashlib.sha256(
                f"{validator_address}{validator_stake}{time.time()}".encode()
            ).hexdigest()
            new_block.nonce = stake_hash[:16]
            new_block.block_hash = new_block.calculate_hash()

            # Remover transacciones del pool
            for tx in transactions:
                if tx in self.pending_transactions:
                    self.pending_transactions.remove(tx)
                    tx.block_height = new_block.block_height

            return new_block

        except Exception as e:
            logger.error(f"Error creando bloque: {e}", exc_info=True)
            return None

    def add_block(self, block: SHEILYSBlock) -> bool:
        """
        Agregar bloque a la cadena

        Args:
            block: Bloque a agregar

        Returns:
            bool: True si el bloque fue agregado exitosamente
        """
        try:
            # Validar bloque
            if not self._validate_block(block):
                return False

            # Agregar a la cadena
            block.status = BlockStatus.CONFIRMED
            self.chain.append(block)

            # Actualizar estado de la blockchain
            self._update_blockchain_state(block)

            return True

        except Exception as e:
            logger.error(f"Error agregando bloque: {e}", exc_info=True)
            return False

    def _validate_block(self, block: SHEILYSBlock) -> bool:
        """Validar bloque completo"""
        try:
            if not self.chain:
                return True  # Genesis block ya validado

            latest_block = self.chain[-1]

            # Validar altura del bloque
            if block.block_height != latest_block.block_height + 1:
                return False

            # Validar hash anterior
            if block.previous_hash != latest_block.block_hash:
                return False

            # Validar hash del bloque
            if block.calculate_hash() != block.block_hash:
                return False

            # Validar stake del validador
            if self.stakes.get(block.validator, 0) < self.min_stake:
                return False

            # Validar transacciones
            for tx in block.transactions:
                if not self._validate_transaction(tx):
                    return False

            return True

        except Exception:
            return False

    def _update_blockchain_state(self, block: SHEILYSBlock):
        """Actualizar estado global de la blockchain después de agregar bloque"""
        # ACTUALIZACIÓN REAL DE BALANCES Y ESTADO
        for tx in block.transactions:
            if tx.transaction_type == TransactionType.TRANSFER:
                # Transferir tokens entre direcciones
                sender_balance = self.balances.get(tx.sender, 0.0)
                receiver_balance = self.balances.get(tx.receiver, 0.0)
                
                if sender_balance >= tx.amount:
                    self.balances[tx.sender] = sender_balance - tx.amount
                    self.balances[tx.receiver] = receiver_balance + tx.amount
                    logger.debug(
                        f"Transferencia: {tx.sender} -> {tx.receiver}: {tx.amount} SHEILYS"
                    )
                    
            elif tx.transaction_type == TransactionType.REWARD:
                # Mint tokens (rewards, minting)
                receiver_balance = self.balances.get(tx.receiver, 0.0)
                self.balances[tx.receiver] = receiver_balance + tx.amount
                self.total_supply += tx.amount
                self.circulating_supply += tx.amount
                logger.debug(
                    f"Reward/Mint: {tx.receiver} recibió {tx.amount} SHEILYS"
                )
                
            elif tx.transaction_type == TransactionType.STAKE:
                # Staking ya se maneja en stake_tokens, pero actualizamos aquí también
                if tx.sender in self.balances:
                    # El balance ya se redujo en stake_tokens, solo verificamos
                    pass
                    
            elif tx.transaction_type == TransactionType.UNSTAKE:
                # Unstake tokens
                if tx.sender in self.stakes:
                    unstake_amount = min(tx.amount, self.stakes[tx.sender])
                    self.stakes[tx.sender] -= unstake_amount
                    self.balances[tx.sender] = self.balances.get(tx.sender, 0.0) + unstake_amount
                    logger.debug(
                        f"Unstake: {tx.sender} des-stakeó {unstake_amount} SHEILYS"
                    )
            
            # Manejar burn de tokens si está en metadata
            if "burn_amount" in tx.metadata:
                burn_amount = tx.metadata["burn_amount"]
                if tx.sender in self.balances:
                    self.balances[tx.sender] = max(0.0, self.balances[tx.sender] - burn_amount)
                    self.burned_supply += burn_amount
                    self.circulating_supply -= burn_amount
                    logger.debug(f"Burn: {burn_amount} SHEILYS quemados")

        # Marcar bloque como finalizado
        block.status = BlockStatus.FINALIZED

    def stake_tokens(self, address: str, amount: float) -> bool:
        """
        Stake tokens para convertirse en validador

        Args:
            address: Address que hace stake
            amount: Cantidad de SHEILYS a stakear

        Returns:
            bool: True si el stake fue exitoso
        """
        try:
            if amount < 0:
                logger.warning(f"Stake inválido: cantidad negativa {amount}")
                return False

            # VERIFICAR BALANCE REAL - CRÍTICO
            current_balance = self.balances.get(address, 0.0)
            if current_balance < amount:
                logger.warning(
                    f"Stake fallido: {address} no tiene suficientes tokens. "
                    f"Tiene: {current_balance}, Intenta stakear: {amount}"
                )
                return False

            current_stake = self.stakes.get(address, 0.0)

            # Reducir balance antes de aumentar stake
            self.balances[address] = current_balance - amount
            self.stakes[address] = current_stake + amount

            # Agregar como validador si supera el mínimo
            if (
                self.stakes[address] >= self.min_stake
                and address not in self.validators
            ):
                self.validators.append(address)
                logger.info(f"Validador agregado: {address} con stake de {self.stakes[address]}")

            # Crear transacción de stake con hash correcto usando create_with_hash
            stake_timestamp = time.time()
            stake_tx = SHEILYSTransaction.create_with_hash(
                sender=address,
                receiver="stake_contract",
                amount=amount,
                transaction_type=TransactionType.STAKE,
                timestamp=stake_timestamp,
                signature=f"stake_signature_{address}",
                metadata={"stake_type": "validator"},
            )

            self.add_transaction(stake_tx)
            logger.info(f"Stake exitoso: {address} stakeó {amount} SHEILYS")
            return True

        except Exception as e:
            logger.error(f"Error haciendo stake: {e}", exc_info=True)
            return False

    def _clean_expired_transactions(self):
        """Limpiar transacciones expiradas del pool"""
        current_time = time.time()
        max_age = 300  # 5 minutos máximo

        self.pending_transactions = [
            tx
            for tx in self.pending_transactions
            if current_time - tx.timestamp < max_age
        ]

    def get_blockchain_info(self) -> Dict[str, Any]:
        """Obtener información general de la blockchain"""
        return {
            "chain_length": len(self.chain),
            "pending_transactions": len(self.pending_transactions),
            "total_validators": len(self.validators),
            "total_staked": sum(self.stakes.values()),
            "total_supply": self.total_supply,
            "circulating_supply": self.circulating_supply,
            "burned_supply": self.burned_supply,
            "total_addresses": len(self.balances),
            "latest_block": self.chain[-1].block_height if self.chain else 0,
            "block_time": self.block_time,
        }
    
    def get_balance(self, address: str) -> float:
        """
        Obtener balance de una dirección
        
        Args:
            address: Dirección a consultar
            
        Returns:
            float: Balance de la dirección
        """
        return self.balances.get(address, 0.0)

    def get_block(self, height: int) -> Optional[SHEILYSBlock]:
        """Obtener bloque por altura"""
        if 0 <= height < len(self.chain):
            return self.chain[height]
        return None

    def get_transaction(self, tx_id: str) -> Optional[SHEILYSTransaction]:
        """Buscar transacción por ID en toda la cadena"""
        for block in self.chain:
            for tx in block.transactions:
                if tx.transaction_id == tx_id:
                    return tx
        return None

    # Gamification integration methods
    def register_gamification_contract(
        self, contract_name: str, contract_data: Dict[str, Any]
    ):
        """Registrar contrato de gamificación"""
        self.gamification_contracts[contract_name] = {
            "data": contract_data,
            "created_at": time.time(),
            "active": True,
        }

    def register_nft_contract(self, contract_name: str, contract_data: Dict[str, Any]):
        """Registrar contrato NFT"""
        self.nft_contracts[contract_name] = {
            "data": contract_data,
            "created_at": time.time(),
            "active": True,
            "total_supply": 0,
        }

    def mint_gamification_reward(
        self, player_address: str, reward_type: str, amount: float
    ) -> bool:
        """Mint reward tokens para gamificación"""
        try:
            reward_tx = SHEILYSTransaction.create_with_hash(
                sender="gamification_contract",
                receiver=player_address,
                amount=amount,
                transaction_type=TransactionType.REWARD,
                timestamp=time.time(),
                signature="gamification_system_signature",
                metadata={
                    "reward_type": reward_type,
                    "source": "gamification",
                    "player": player_address,
                },
            )

            return self.add_transaction(reward_tx)

        except Exception as e:
            logger.error(f"Error minting reward: {e}", exc_info=True)
            return False
