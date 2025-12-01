#!/usr/bin/env python3
"""
SHEILYS Token - Token nativo del ecosistema Sheily AI MCP Enterprise
Implementación completa compatible con Solana y Web3

El token SHEILYS facilita:
- Gamificación y recompensas automáticas desde blockchain
- Staking con yield rewards sobre blockchain
- NFT credentials verificables on-chain
- Gobernanza del ecosistema Sheily
- Pagos por servicios con contratos inteligentes
"""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .sheilys_blockchain import SHEILYSBlockchain, TransactionType, SHEILYSTransaction

# Configurar logging
logger = logging.getLogger(__name__)


class SHEILYSTokenStandard(Enum):
    """Estándares de token SHEILYS soportados"""

    SPL_TOKEN = "spl_token"  # Solana Program Library (SPL)
    ERC20_COMPATIBLE = "erc20_compatible"  # Compatible ERC-20


class NFTCollection(Enum):
    """Colecciones NFT disponibles en SHEILYS"""

    ACHIEVEMENT_BADGES = "achievement_badges"
    CREDENTIALS_CERTIFICATES = "credentials_certificates"
    LEARNING_TRACKS = "learning_tracks"
    GAMIFICATION_REWARDS = "gamification_rewards"
    GOVERNANCE_TOKENS = "governance_tokens"


@dataclass
class SHEILYSTokenMetadata:
    """Metadatos del token SHEILYS según estándar Solana Metaplex"""

    name: str
    symbol: str
    description: str
    image: str
    external_url: str
    attributes: List[Dict[str, Any]]
    properties: Dict[str, Any]

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convertir a formato Metaplex compatible"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "description": self.description,
            "image": self.image,
            "external_url": self.external_url,
            "attributes": self.attributes,
            "properties": self.properties,
        }


@dataclass
class SHEILYSNFT:
    """NFT SHEILYS con metadatos extendidos"""

    token_id: str
    collection: NFTCollection
    owner: str
    metadata: Dict[str, Any]
    minted_at: float
    last_transfer: float
    rarity_score: float
    utility_functions: List[str]

    def calculate_rarity(self) -> float:
        """Calcular rareza del NFT basado en sus atributos"""
        # Implementación simplificada - en producción usar algoritmo más sofisticado
        base_rarity = 1.0
        if self.collection == NFTCollection.ACHIEVEMENT_BADGES:
            base_rarity = 1.2
        elif self.collection == NFTCollection.CREDENTIALS_CERTIFICATES:
            base_rarity = 2.0
        elif self.collection == NFTCollection.GOVERNANCE_TOKENS:
            base_rarity = 3.0

        # Factor por edad (NFTs más antiguos son más raros)
        age_factor = min(1.5, (time.time() - self.minted_at) / (365 * 24 * 3600))
        return base_rarity * age_factor


class SHEILYSTokenManager:
    """
    SHEILYS Token Manager - Gestión completa del ecosistema token SHEILYS

    Funcionalidades:
    - Minting automático por IA
    - Rewards por gamificación
    - Staking con yield rewards
    - NFT minting y gestión
    - Gobierno del ecosistema
    """

    def __init__(self, blockchain: Optional[SHEILYSBlockchain] = None):
        """Inicializar token manager"""

        # Conectar con blockchain SHEILYS
        self.blockchain = blockchain or SHEILYSBlockchain()

        # Balances de tokens (SPL compatible)
        self.token_balances: Dict[str, float] = {}

        # Balances de staked tokens
        self.staked_balances: Dict[str, float] = {}
        
        # Tracking de staking para rewards reales
        self.staking_timestamps: Dict[str, float] = {}  # address -> timestamp del último claim
        self.staking_pool_tracking: Dict[str, Dict[str, Any]] = {}  # address -> pool info
        
        # Tracking de tokens quemados
        self.total_burned: float = 0.0
        
        # Tracking de votos para prevenir doble voto
        self.vote_tracking: Dict[str, Dict[str, bool]] = {}  # proposal_id -> {voter: True}

        # Colecciones NFT
        self.nft_collections: Dict[NFTCollection, List[SHEILYSNFT]] = {}
        self.nft_ownership: Dict[str, List[str]] = {}  # owner -> [token_ids]

        # Metadata del token principal
        self.token_metadata = SHEILYSTokenMetadata(
            name="SHEILYS AI",
            symbol="SHEILYS",
            description="Token nativo del ecosistema Sheily AI MCP Enterprise - Learn to Earn",
            image="https://sheily.ai/images/sheilys-token.png",
            external_url="https://sheily.ai",
            attributes=[
                {"trait_type": "Type", "value": "Educational Token"},
                {"trait_type": "Network", "value": "Solana"},
                {"trait_type": "Standard", "value": "SPL Token"},
                {"trait_type": "Features", "value": "Staking, NFTs, Governance"},
            ],
            properties={
                "educational_token": True,
                "learn_to_earn": True,
                "governance_enabled": True,
                "nft_credentials": True,
                "deflationary": True,
                "staking_enabled": True,
                "max_supply": 1000000000,
                "decimals": 9,
            },
        )

        # Sistema de rewards por gamificación
        self.reward_rates = {
            "exercise_completion": 3.0,  # SHEILYS por ejercicio correcto
            "dataset_generation": 10.0,  # SHEILYS por dataset generado
            "knowledge_sharing": 5.0,  # SHEILYS por compartir conocimiento
            "achievements": 25.0,  # SHEILYS por logros especiales
            "level_up": 50.0,  # SHEILYS por subir de nivel
        }

        # Sistema de staking
        self.staking_pools = {
            "validator_pool": {"apy": 12.0, "min_stake": 1000.0},
            "community_pool": {"apy": 8.0, "min_stake": 100.0},
            "education_pool": {"apy": 15.0, "min_stake": 500.0},
        }

        # Gobernanza
        self.proposals: Dict[str, Dict[str, Any]] = {}
        self.voting_power: Dict[str, float] = {}  # Basado en balance + stake

        # Inicializar colecciones NFT
        self._initialize_nft_collections()

    def _initialize_nft_collections(self):
        """Inicializar colecciones NFT por defecto"""
        for collection in NFTCollection:
            self.nft_collections[collection] = []

    def get_balance(self, address: str) -> float:
        """Obtener balance de SHEILYS de una dirección"""
        return self.token_balances.get(address, 0.0)

    def get_staked_balance(self, address: str) -> float:
        """Obtener balance staked de una dirección"""
        return self.staked_balances.get(address, 0.0)

    def mint_tokens(
        self, to_address: str, amount: float, reason: str = "minting"
    ) -> bool:
        """
        Mint tokens SHEILYS (controlado por sistema IA)

        Args:
            to_address: Dirección destinataria
            amount: Cantidad a mintear
            reason: Reason para transparencia

        Returns:
            bool: True si el mint fue exitoso
        """
        try:
            # Verificar límite de supply CONSIDERANDO BURNS
            current_supply = sum(self.token_balances.values()) + sum(
                self.staked_balances.values()
            )
            # El supply real es el inicial + minted - burned
            real_supply = current_supply - self.total_burned
            
            if real_supply + amount > self.token_metadata.properties["max_supply"]:
                logger.warning(
                    f"Mint fallido: excedería max supply. "
                    f"Supply actual: {real_supply}, Max: {self.token_metadata.properties['max_supply']}, "
                    f"Intenta mintear: {amount}"
                )
                return False

            # Mint tokens
            self.token_balances[to_address] = self.get_balance(to_address) + amount

            # Crear transacción con hash correcto
            tx_timestamp = time.time()
            tx = SHEILYSTransaction.create_with_hash(
                sender="sheily_system_minter",
                receiver=to_address,
                amount=amount,
                transaction_type=TransactionType.REWARD,
                timestamp=tx_timestamp,
                signature=f"system_mint_signature_{reason}",
                metadata={
                    "mint_reason": reason,
                    "type": "token_mint",
                    "supply_inflation": amount,
                },
            )
            tx_success = self.blockchain.add_transaction(tx)
            
            if tx_success:
                logger.info(f"Tokens minted: {amount} SHEILYS a {to_address} por {reason}")

            return tx_success

        except Exception as e:
            logger.error(f"Error minting tokens: {e}", exc_info=True)
            return False

    def transfer_tokens(
        self, from_address: str, to_address: str, amount: float
    ) -> bool:
        """
        Transferir tokens SHEILYS entre direcciones

        Returns:
            bool: True si la transferencia fue exitosa
        """
        try:
            available_balance = self.get_balance(from_address)
            if available_balance < amount:
                return False

            # Ejecutar transferencia
            self.token_balances[from_address] -= amount
            self.token_balances[to_address] = self.get_balance(to_address) + amount

            # Calcular burn ANTES de transferir (1% para deflación)
            burn_amount = amount * 0.01
            
            # Verificar que tenga suficiente para transferencia + burn
            if available_balance < (amount + burn_amount):
                logger.warning(
                    f"Transferencia fallida: balance insuficiente para transfer + burn. "
                    f"Tiene: {available_balance}, Necesita: {amount + burn_amount}"
                )
                return False

            # Ejecutar transferencia
            self.token_balances[from_address] -= amount
            self.token_balances[to_address] = self.get_balance(to_address) + amount

            # Aplicar burn (reducir del remitente y del total supply)
            if burn_amount > 0:
                self.token_balances[from_address] -= burn_amount
                self.total_burned += burn_amount
                
                # Actualizar balances en blockchain también
                if hasattr(self.blockchain, 'burned_supply'):
                    self.blockchain.burned_supply += burn_amount
                if hasattr(self.blockchain, 'circulating_supply'):
                    self.blockchain.circulating_supply -= burn_amount

            # Crear transacción con hash correcto usando método estático
            tx_timestamp = time.time()
            tx = SHEILYSTransaction.create_with_hash(
                sender=from_address,
                receiver=to_address,
                amount=amount,
                transaction_type=TransactionType.TRANSFER,
                timestamp=tx_timestamp,
                signature=f"{from_address}_transfer_sig",
                metadata={
                    "transfer_type": "direct_transfer",
                    "gas_used": 0.001,
                    "burn_amount": burn_amount,  # Incluir burn en metadata
                },
            )

            self.blockchain.add_transaction(tx)
            logger.info(f"Transferencia: {from_address} -> {to_address}: {amount} SHEILYS (burn: {burn_amount})")

            return True

        except Exception as e:
            logger.error(f"Error transferring tokens: {e}", exc_info=True)
            return False

    def stake_tokens(
        self, address: str, amount: float, pool_name: str = "community_pool"
    ) -> bool:
        """
        Stake tokens para ganar rewards

        Args:
            address: Dirección que hace stake
            amount: Cantidad a stakear
            pool_name: Pool de staking

        Returns:
            bool: True si el stake fue exitoso
        """
        try:
            if pool_name not in self.staking_pools:
                return False

            pool_config = self.staking_pools[pool_name]
            if amount < pool_config["min_stake"]:
                return False

            available_balance = self.get_balance(address)
            if available_balance < amount:
                return False

            # Mover a stake
            self.token_balances[address] -= amount
            self.staked_balances[address] = self.get_staked_balance(address) + amount
            
            # TRACKING REAL DE STAKING - guardar timestamp y pool
            self.staking_timestamps[address] = time.time()
            if address not in self.staking_pool_tracking:
                self.staking_pool_tracking[address] = {}
            self.staking_pool_tracking[address][pool_name] = {
                "amount": self.staked_balances[address],
                "start_time": time.time(),
                "last_claim": time.time(),
                "pool_config": pool_config,
            }

            # Registrar en blockchain
            self.blockchain.stake_tokens(address, amount)
            
            logger.info(f"Stake exitoso: {address} stakeó {amount} SHEILYS en {pool_name}")

            return True

        except Exception as e:
            logger.error(f"Error staking tokens: {e}", exc_info=True)
            return False

    def claim_staking_rewards(self, address: str) -> float:
        """
        Claim staking rewards acumulados basado en tiempo REAL

        Returns:
            float: Cantidad de rewards claimed
        """
        try:
            staked_amount = self.get_staked_balance(address)
            if staked_amount == 0:
                logger.warning(f"Claim fallido: {address} no tiene tokens staked")
                return 0.0

            # Obtener información de staking real
            if address not in self.staking_timestamps:
                logger.warning(f"Claim fallido: {address} no tiene timestamp de staking")
                return 0.0

            # Calcular tiempo REAL desde último claim o desde staking inicial
            last_claim_time = self.staking_timestamps.get(address, time.time())
            current_time = time.time()
            time_delta_seconds = current_time - last_claim_time
            
            # Calcular APY basado en pools del usuario
            if address in self.staking_pool_tracking and self.staking_pool_tracking[address]:
                # Obtener pool con mayor stake
                pools = self.staking_pool_tracking[address]
                max_pool = max(pools.items(), key=lambda x: x[1].get("amount", 0))
                pool_name, pool_info = max_pool
                pool_config = pool_info.get("pool_config", {})
                apy = pool_config.get("apy", 10.0)  # APY del pool
            else:
                # Fallback a APY promedio
                apy = 10.0

            # Calcular rewards REALES basados en tiempo transcurrido
            daily_reward_rate = (apy / 100) / 365
            time_delta_days = time_delta_seconds / (24 * 3600)
            claimed_amount = staked_amount * daily_reward_rate * time_delta_days

            # Limitar a máximo razonable (evitar explotación)
            max_daily_claim = staked_amount * (apy / 100) / 365 * 2  # Máximo 2 días por claim
            claimed_amount = min(claimed_amount, max_daily_claim)

            if claimed_amount > 0:
                # Actualizar timestamp de último claim
                self.staking_timestamps[address] = current_time
                if address in self.staking_pool_tracking:
                    for pool_info in self.staking_pool_tracking[address].values():
                        pool_info["last_claim"] = current_time

                # Mint rewards
                success = self.mint_tokens(address, claimed_amount, f"staking_rewards_{address}")
                if success:
                    logger.info(
                        f"Rewards claimed: {address} reclamó {claimed_amount:.6f} SHEILYS "
                        f"por {time_delta_days:.2f} días de staking"
                    )
                    return claimed_amount

            return 0.0

        except Exception as e:
            logger.error(f"Error claiming staking rewards: {e}", exc_info=True)
            return 0.0

    def reward_gamification_action(self, user_address: str, action_type: str) -> float:
        """
        Reward tokens por acciones de gamificación

        Args:
            user_address: Usuario que recibe reward
            action_type: Tipo de acción ('exercise_completion', etc.)

        Returns:
            float: Cantidad de tokens rewarded
        """
        if action_type not in self.reward_rates:
            return 0.0

        reward_amount = self.reward_rates[action_type]

        # Mint reward tokens
        success = self.mint_tokens(
            user_address, reward_amount, f"gamification_{action_type}"
        )

        return reward_amount if success else 0.0

    def mint_nft(
        self, collection: NFTCollection, owner: str, metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Mint un nuevo NFT SHEILYS

        Args:
            collection: Colección del NFT
            owner: Propietario del NFT
            metadata: Metadata del NFT

        Returns:
            str: Token ID del NFT minteado, o None si falló
        """
        try:
            # Generar token ID único
            token_id = f"{collection.value}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

            # Crear NFT
            nft = SHEILYSNFT(
                token_id=token_id,
                collection=collection,
                owner=owner,
                metadata=metadata,
                minted_at=time.time(),
                last_transfer=time.time(),
                rarity_score=0.0,  # Se calcula después
                utility_functions=self._get_nft_utility_functions(collection),
            )

            # Calcular rareza
            nft.rarity_score = nft.calculate_rarity()

            # Agregar a colección
            self.nft_collections[collection].append(nft)

            # Actualizar ownership
            if owner not in self.nft_ownership:
                self.nft_ownership[owner] = []
            self.nft_ownership[owner].append(token_id)

            # Crear transacción REAL con objeto SHEILYSTransaction
            tx_timestamp = time.time()
            nft_tx = SHEILYSTransaction.create_with_hash(
                sender="sheily_nft_minter",
                receiver=owner,
                amount=1,  # NFTs son únicos
                transaction_type=TransactionType.NFT_MINT,
                timestamp=tx_timestamp,
                signature=f"nft_mint_signature_{token_id}",
                metadata={
                    "nft_token_id": token_id,
                    "collection": collection.value,
                    "rarity": nft.rarity_score,
                },
            )

            # Registrar en blockchain
            success = self.blockchain.add_transaction(nft_tx)
            if success:
                logger.info(f"NFT minted: {token_id} en colección {collection.value} para {owner}")
                return token_id
            else:
                logger.error(f"Error minting NFT: transacción no se pudo agregar a blockchain")
                return None

        except Exception as e:
            logger.error(f"Error minting NFT: {e}", exc_info=True)
            return None

    def _get_nft_utility_functions(self, collection: NFTCollection) -> List[str]:
        """Obtener funciones de utilidad de un NFT basado en su colección"""
        utilities = {
            NFTCollection.ACHIEVEMENT_BADGES: [
                "access_premium_features",
                "vote_governance",
                "exclusive_community",
            ],
            NFTCollection.CREDENTIALS_CERTIFICATES: [
                "verify_credentials",
                "access_certified_content",
                "professional_networking",
            ],
            NFTCollection.LEARNING_TRACKS: [
                "unlock_advance_content",
                "certificate_verification",
                "career_credits",
            ],
            NFTCollection.GAMIFICATION_REWARDS: [
                "boosted_rewards",
                "exclusive_events",
                "leaderboard_bonus",
            ],
            NFTCollection.GOVERNANCE_TOKENS: [
                "voting_power_boost",
                "proposal_creation",
                "delegate_voting",
            ],
        }
        return utilities.get(collection, [])

    def transfer_nft(self, token_id: str, from_address: str, to_address: str) -> bool:
        """
        Transferir NFT entre direcciones

        Returns:
            bool: True si la transferencia fue exitosa
        """
        try:
            # Encontrar NFT
            nft = None
            collection = None
            for coll_name, nfts in self.nft_collections.items():
                for n in nfts:
                    if n.token_id == token_id:
                        nft = n
                        collection = coll_name
                        break
                if nft:
                    break

            if not nft or nft.owner != from_address:
                return False

            # Actualizar propietario
            nft.owner = to_address
            nft.last_transfer = time.time()

            # Actualizar ownership tracking
            if from_address in self.nft_ownership:
                self.nft_ownership[from_address].remove(token_id)
            if to_address not in self.nft_ownership:
                self.nft_ownership[to_address] = []
            self.nft_ownership[to_address].append(token_id)

            # Crear transacción REAL con objeto SHEILYSTransaction
            collection_str = collection.value if collection else "unknown"
            tx_timestamp = time.time()
            nft_transfer_tx = SHEILYSTransaction.create_with_hash(
                sender=from_address,
                receiver=to_address,
                amount=1,
                transaction_type=TransactionType.NFT_TRANSFER,
                timestamp=tx_timestamp,
                signature=f"nft_transfer_signature_{token_id}",
                metadata={
                    "nft_token_id": token_id,
                    "collection": collection_str,
                    "rarity": nft.rarity_score,
                },
            )

            # Registrar en blockchain
            success = self.blockchain.add_transaction(nft_transfer_tx)
            if success:
                logger.info(f"NFT transferido: {token_id} de {from_address} a {to_address}")
                return True
            else:
                logger.error(f"Error en transferencia NFT: transacción no se pudo agregar")
                return False

        except Exception as e:
            logger.error(f"Error transferring NFT: {e}", exc_info=True)
            return False

    def get_user_nfts(self, address: str) -> List[Dict[str, Any]]:
        """Obtener NFTs de un usuario"""
        user_nfts = []
        for collection, nfts in self.nft_collections.items():
            for nft in nfts:
                if nft.owner == address:
                    user_nfts.append(
                        {
                            "token_id": nft.token_id,
                            "collection": collection.value,
                            "metadata": nft.metadata,
                            "rarity_score": nft.rarity_score,
                            "minted_at": nft.minted_at,
                            "utility_functions": nft.utility_functions,
                        }
                    )
        return user_nfts

    def create_governance_proposal(
        self, proposer: str, title: str, description: str, voting_period_days: int = 7
    ) -> str:
        """
        Crear propuesta de gobernanza (REQUIERE MÍNIMO DE TOKENS)

        Returns:
            str: ID de la propuesta
        """
        # REQUERIR MÍNIMO DE TOKENS PARA CREAR PROPUESTA
        min_proposal_tokens = 1000.0  # Mínimo 1000 SHEILYS para crear propuesta
        voting_power = self.get_balance(proposer) + self.get_staked_balance(proposer)
        
        if voting_power < min_proposal_tokens:
            logger.warning(
                f"Propuesta fallida: {proposer} no tiene suficientes tokens. "
                f"Tiene: {voting_power}, Requiere: {min_proposal_tokens}"
            )
            raise ValueError(
                f"Se requieren al menos {min_proposal_tokens} SHEILYS para crear una propuesta. "
                f"Tienes: {voting_power}"
            )

        proposal_id = (
            f"prop_{int(time.time())}_{hashlib.sha256(title.encode()).hexdigest()[:8]}"
        )

        self.proposals[proposal_id] = {
            "id": proposal_id,
            "proposer": proposer,
            "title": title,
            "description": description,
            "created_at": time.time(),
            "voting_ends": time.time() + (voting_period_days * 24 * 3600),
            "total_votes": 0,
            "votes_for": 0,
            "votes_against": 0,
            "status": "active",
        }
        
        # Inicializar tracking de votos para esta propuesta
        if proposal_id not in self.vote_tracking:
            self.vote_tracking[proposal_id] = {}
        
        logger.info(f"Propuesta creada: {proposal_id} por {proposer}")

        return proposal_id

    def vote_on_proposal(self, proposal_id: str, voter: str, votes_for: bool) -> bool:
        """
        Votar en una propuesta de gobernanza (PREVIENE DOBLE VOTO)

        Args:
            proposal_id: ID de la propuesta
            voter: Dirección del votante
            votes_for: True para votar a favor, False para votar en contra

        Returns:
            bool: True si el voto fue registrado
        """
        if proposal_id not in self.proposals:
            logger.warning(f"Voto fallido: propuesta {proposal_id} no existe")
            return False

        proposal = self.proposals[proposal_id]

        if time.time() > proposal["voting_ends"]:
            logger.warning(f"Voto fallido: período de votación terminado para {proposal_id}")
            return False

        # PREVENIR DOBLE VOTO - verificar si ya votó
        if proposal_id not in self.vote_tracking:
            self.vote_tracking[proposal_id] = {}
        
        if voter in self.vote_tracking[proposal_id]:
            logger.warning(f"Voto fallido: {voter} ya votó en propuesta {proposal_id}")
            return False

        # Calcular poder de voto
        voting_power = self.get_balance(voter) + self.get_staked_balance(voter)
        
        if voting_power <= 0:
            logger.warning(f"Voto fallido: {voter} no tiene poder de voto (balance: {self.get_balance(voter)}, staked: {self.get_staked_balance(voter)})")
            return False

        # Registrar voto
        if votes_for:
            proposal["votes_for"] += voting_power
        else:
            proposal["votes_against"] += voting_power

        proposal["total_votes"] += voting_power
        
        # Marcar que este voter ya votó
        self.vote_tracking[proposal_id][voter] = True
        
        logger.info(
            f"Voto registrado: {voter} votó {'A FAVOR' if votes_for else 'EN CONTRA'} "
            f"en {proposal_id} con poder de voto {voting_power}"
        )

        return True

    def get_token_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del token SHEILYS"""
        total_supply = sum(self.token_balances.values()) + sum(
            self.staked_balances.values()
        )
        total_staked = sum(self.staked_balances.values())
        total_holders = len([b for b in self.token_balances.values() if b > 0])
        total_stakers = len([s for s in self.staked_balances.values() if s > 0])

        # Calcular métricas NFT
        total_nfts = sum(len(nfts) for nfts in self.nft_collections.values())

        return {
            "token_metadata": self.token_metadata.to_metadata_dict(),
            "total_supply": total_supply,
            "circulating_supply": total_supply - self.total_burned,
            "burned_supply": self.total_burned,
            "staked_supply": total_staked,
            "holders_count": total_holders,
            "stakers_count": total_stakers,
            "staking_ratio": total_staked / total_supply if total_supply > 0 else 0,
            "burn_ratio": self.total_burned / total_supply if total_supply > 0 else 0,
            "total_nfts": total_nfts,
            "nft_collections": {
                k.value: len(v) for k, v in self.nft_collections.items()
            },
            "active_proposals": len(
                [p for p in self.proposals.values() if p["status"] == "active"]
            ),
            "reward_rates": self.reward_rates,
        }
