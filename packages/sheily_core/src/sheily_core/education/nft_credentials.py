"""
Sistema de NFTs para Credenciales Educativas en Sheily AI
Implementa certificaciones verificables basadas en blockchain
Basado en investigación: Hyperledger Besu e-learning system, Web3 attitudes

Características:
- NFTs ERC-721 para diplomas y certificados
- Metadata educativa completa
- Verificación inmutable de logros
- Integración con Solana/Metaplex
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CredentialType(Enum):
    """Tipos de credenciales educativas"""

    COURSE_CERTIFICATE = "course_certificate"
    SKILL_BADGE = "skill_badge"
    ACHIEVEMENT_AWARD = "achievement_award"
    PARTICIPATION_CERTIFICATE = "participation_certificate"
    ASSESSMENT_COMPLETION = "assessment_completion"
    TUTORING_ACHIEVEMENT = "tutoring_achievement"


class VerificationStatus(Enum):
    """Estados de verificación de credenciales"""

    PENDING = "pending"
    VERIFIED = "verified"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class EducationalMetadata:
    """Metadata completa para credenciales educativas"""

    institution_name: str
    institution_id: str
    course_name: str
    course_id: str
    instructor_name: str
    instructor_id: str
    learner_name: str
    learner_id: str
    issue_date: datetime
    expiry_date: Optional[datetime] = None
    grade_achieved: Optional[str] = None
    credit_hours: Optional[float] = None
    competencies: List[str] = field(default_factory=list)
    learning_outcomes: List[str] = field(default_factory=list)
    verification_url: Optional[str] = None
    blockchain_tx_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertir metadata a diccionario para NFT"""
        return {
            "institution": {"name": self.institution_name, "id": self.institution_id},
            "course": {"name": self.course_name, "id": self.course_id},
            "instructor": {"name": self.instructor_name, "id": self.instructor_id},
            "learner": {"name": self.learner_name, "id": self.learner_id},
            "dates": {
                "issued": self.issue_date.isoformat(),
                "expires": self.expiry_date.isoformat() if self.expiry_date else None,
            },
            "academic_details": {
                "grade": self.grade_achieved,
                "credit_hours": self.credit_hours,
                "competencies": self.competencies,
                "learning_outcomes": self.learning_outcomes,
            },
            "verification": {
                "url": self.verification_url,
                "blockchain_tx": self.blockchain_tx_hash,
            },
        }


@dataclass
class NFTEducationalCredential:
    """Estructura de NFT para credenciales educativas"""

    token_id: str
    credential_type: CredentialType
    metadata: EducationalMetadata
    status: VerificationStatus = VerificationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Verificar si la credencial es válida"""
        if self.status != VerificationStatus.VERIFIED:
            return False
        if self.metadata.expiry_date and datetime.now() > self.metadata.expiry_date:
            return False
        return True

    @property
    def nft_metadata(self) -> Dict[str, Any]:
        """Metadata completa para el NFT"""
        # Generate actual image URL or use default
        image_url = self._generate_credential_image_url()
        
        return {
            "name": f"Sheily AI {self.credential_type.value.replace('_', ' ').title()}",
            "description": f"Verifiable educational credential issued by Sheily AI: {self.metadata.course_name}",
            "image": image_url,
            "attributes": [
                {"trait_type": "Credential Type", "value": self.credential_type.value},
                {"trait_type": "Institution", "value": self.metadata.institution_name},
                {"trait_type": "Course", "value": self.metadata.course_name},
                {"trait_type": "Grade", "value": self.metadata.grade_achieved or "N/A"},
                {
                    "trait_type": "Issue Date",
                    "value": self.metadata.issue_date.isoformat(),
                },
                {"trait_type": "Status", "value": self.status.value},
            ],
            "educational_data": self.metadata.to_dict(),
        }
    
    def _generate_credential_image_url(self) -> str:
        """Generate credential image URL"""
        # In production, this would:
        # 1. Generate a visual certificate using PIL/Pillow
        # 2. Upload to IPFS or Arweave
        # 3. Return the decentralized storage URL
        
        # For now, return a placeholder that indicates the credential type
        return f"https://sheily.ai/credentials/images/{self.credential_type.value}/{self.token_id}.png"


class NFTEducationCredentials:
    """
    Sistema de NFTs para credenciales educativas
    Gestiona creación, verificación y consulta de certificados blockchain
    """

    def __init__(self):
        self.credentials: Dict[str, NFTEducationalCredential] = {}
        self.user_credentials: Dict[str, List[str]] = {}  # user_id -> [token_ids]
        self.logger = logging.getLogger(__name__)

        # Simulación de conexión blockchain (en producción usar Metaplex/Solana)
        self.blockchain_connected = False

        self.logger.info("🎓 NFT Education Credentials system initialized")

    async def create_credential(
        self,
        learner_id: str,
        credential_type: CredentialType,
        metadata: EducationalMetadata,
    ) -> Dict[str, Any]:
        """
        Crear nueva credencial educativa como NFT
        """
        try:
            # Generar token ID único
            token_id = f"sheily_edu_{credential_type.value}_{learner_id}_{int(datetime.now().timestamp())}"

            # Crear credencial
            credential = NFTEducationalCredential(
                token_id=token_id,
                credential_type=credential_type,
                metadata=metadata,
                status=VerificationStatus.PENDING,
            )

            # Almacenar credencial
            self.credentials[token_id] = credential

            # Agregar a lista de usuario
            if learner_id not in self.user_credentials:
                self.user_credentials[learner_id] = []
            self.user_credentials[learner_id].append(token_id)

            # Simular minteado en blockchain (en producción usar Metaplex)
            mint_result = await self._mint_nft_credential(credential)

            if mint_result["success"]:
                credential.status = VerificationStatus.VERIFIED
                credential.metadata.blockchain_tx_hash = mint_result.get("tx_hash")
                credential.updated_at = datetime.now()

            self.logger.info(f"✅ Educational credential created: {token_id}")

            return {
                "success": True,
                "token_id": token_id,
                "credential_type": credential_type.value,
                "status": credential.status.value,
                "blockchain_tx": mint_result.get("tx_hash"),
                "nft_metadata": credential.nft_metadata,
            }

        except Exception as e:
            self.logger.error(f"Error creating educational credential: {e}")
            return {"success": False, "error": str(e)}

    async def _mint_nft_credential(
        self, credential: NFTEducationalCredential
    ) -> Dict[str, Any]:
        """
        Mintear NFT en blockchain (simulado para desarrollo)
        En producción: usar Metaplex SDK para Solana
        """
        try:
            if not self.blockchain_connected:
                # Simulación para desarrollo
                return {
                    "success": True,
                    "tx_hash": f"simulated_tx_{credential.token_id}_{int(datetime.now().timestamp())}",
                    "blockchain": "solana_devnet",
                    "token_address": f"simulated_address_{credential.token_id}",
                }

            # TODO: Implementar minteado real con Metaplex
            # - Crear metadata JSON en IPFS/Arweave
            # - Mintear NFT usando Metaplex
            # - Retornar transaction hash y token address

            return {
                "success": True,
                "tx_hash": "real_blockchain_tx_hash",
                "blockchain": "solana_mainnet",
                "token_address": "real_solana_token_address",
            }

        except Exception as e:
            self.logger.error(f"Error minting NFT credential: {e}")
            return {"success": False, "error": str(e)}

    async def verify_credential(self, token_id: str) -> Dict[str, Any]:
        """
        Verificar autenticidad de credencial educativa
        """
        try:
            if token_id not in self.credentials:
                return {
                    "valid": False,
                    "error": "Credential not found",
                    "token_id": token_id,
                }

            credential = self.credentials[token_id]

            # Verificar estado
            if credential.status != VerificationStatus.VERIFIED:
                return {
                    "valid": False,
                    "error": f"Credential status: {credential.status.value}",
                    "token_id": token_id,
                    "status": credential.status.value,
                }

            # Verificar expiración
            if (
                credential.metadata.expiry_date
                and datetime.now() > credential.metadata.expiry_date
            ):
                credential.status = VerificationStatus.EXPIRED
                return {
                    "valid": False,
                    "error": "Credential expired",
                    "token_id": token_id,
                    "expiry_date": credential.metadata.expiry_date.isoformat(),
                }

            # Verificar en blockchain (simulado)
            blockchain_verification = await self._verify_on_blockchain(token_id)

            return {
                "valid": blockchain_verification["valid"],
                "token_id": token_id,
                "credential_type": credential.credential_type.value,
                "learner_id": credential.metadata.learner_id,
                "institution": credential.metadata.institution_name,
                "course": credential.metadata.course_name,
                "issue_date": credential.metadata.issue_date.isoformat(),
                "grade": credential.metadata.grade_achieved,
                "blockchain_verified": blockchain_verification["verified"],
                "metadata": credential.nft_metadata,
            }

        except Exception as e:
            self.logger.error(f"Error verifying credential: {e}")
            return {"valid": False, "error": str(e), "token_id": token_id}

    async def _verify_on_blockchain(self, token_id: str) -> Dict[str, Any]:
        """
        Verificar credencial en blockchain (simulado para desarrollo)
        En producción: consultar Solana blockchain
        """
        try:
            if not self.blockchain_connected:
                # Simulación para desarrollo
                return {
                    "valid": True,
                    "verified": True,
                    "blockchain": "solana_devnet",
                    "last_verified": datetime.now().isoformat(),
                }

            # TODO: Implementar verificación real
            # - Consultar token en Solana
            # - Verificar ownership y metadata
            # - Confirmar no ha sido revocado

            return {
                "valid": True,
                "verified": True,
                "blockchain": "solana_mainnet",
                "last_verified": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"valid": False, "verified": False, "error": str(e)}

    async def get_user_credentials(
        self, user_id: str, include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Obtener todas las credenciales de un usuario
        """
        try:
            if user_id not in self.user_credentials:
                return []

            user_creds = []
            for token_id in self.user_credentials[user_id]:
                if token_id in self.credentials:
                    credential = self.credentials[token_id]

                    # Filtrar expiradas si no se solicitan
                    if not include_expired and not credential.is_valid:
                        continue

                    user_creds.append(
                        {
                            "token_id": token_id,
                            "credential_type": credential.credential_type.value,
                            "status": credential.status.value,
                            "course_name": credential.metadata.course_name,
                            "institution": credential.metadata.institution_name,
                            "issue_date": credential.metadata.issue_date.isoformat(),
                            "grade": credential.metadata.grade_achieved,
                            "is_valid": credential.is_valid,
                            "nft_metadata": credential.nft_metadata,
                        }
                    )

            return user_creds

        except Exception as e:
            self.logger.error(f"Error getting user credentials: {e}")
            return []

    async def revoke_credential(self, token_id: str, reason: str) -> Dict[str, Any]:
        """
        Revocar una credencial (solo administradores)
        """
        try:
            if token_id not in self.credentials:
                return {"success": False, "error": "Credential not found"}

            credential = self.credentials[token_id]
            credential.status = VerificationStatus.REVOKED
            credential.updated_at = datetime.now()

            # TODO: Implementar revocación en blockchain (burn token)

            self.logger.info(f"🚫 Credential revoked: {token_id}, Reason: {reason}")

            return {
                "success": True,
                "token_id": token_id,
                "status": "revoked",
                "reason": reason,
                "revoked_at": credential.updated_at.isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error revoking credential: {e}")
            return {"success": False, "error": str(e)}

    async def get_credential_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema de credenciales
        """
        try:
            total_credentials = len(self.credentials)
            verified_credentials = sum(
                1
                for c in self.credentials.values()
                if c.status == VerificationStatus.VERIFIED
            )
            revoked_credentials = sum(
                1
                for c in self.credentials.values()
                if c.status == VerificationStatus.REVOKED
            )
            expired_credentials = sum(
                1
                for c in self.credentials.values()
                if c.status == VerificationStatus.EXPIRED
            )

            # Estadísticas por tipo
            type_distribution = {}
            for credential in self.credentials.values():
                cred_type = credential.credential_type.value
                if cred_type not in type_distribution:
                    type_distribution[cred_type] = 0
                type_distribution[cred_type] += 1

            # Usuarios únicos
            unique_users = len(self.user_credentials)

            return {
                "total_credentials": total_credentials,
                "verified_credentials": verified_credentials,
                "revoked_credentials": revoked_credentials,
                "expired_credentials": expired_credentials,
                "unique_users": unique_users,
                "type_distribution": type_distribution,
                "verification_rate": (verified_credentials / max(total_credentials, 1))
                * 100,
            }

        except Exception as e:
            self.logger.error(f"Error getting credential stats: {e}")
            return {"error": str(e), "total_credentials": 0}

    async def search_credentials(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Buscar credenciales por criterios
        """
        try:
            results = []

            for token_id, credential in self.credentials.items():
                match = True

                # Filtrar por criterios
                if (
                    "learner_id" in query
                    and credential.metadata.learner_id != query["learner_id"]
                ):
                    match = False
                if (
                    "course_id" in query
                    and credential.metadata.course_id != query["course_id"]
                ):
                    match = False
                if (
                    "institution_id" in query
                    and credential.metadata.institution_id != query["institution_id"]
                ):
                    match = False
                if (
                    "credential_type" in query
                    and credential.credential_type.value != query["credential_type"]
                ):
                    match = False
                if "status" in query and credential.status.value != query["status"]:
                    match = False

                if match:
                    results.append(
                        {
                            "token_id": token_id,
                            "credential_type": credential.credential_type.value,
                            "status": credential.status.value,
                            "learner_id": credential.metadata.learner_id,
                            "course_name": credential.metadata.course_name,
                            "institution": credential.metadata.institution_name,
                            "issue_date": credential.metadata.issue_date.isoformat(),
                            "is_valid": credential.is_valid,
                        }
                    )

            return results

        except Exception as e:
            self.logger.error(f"Error searching credentials: {e}")
            return []


# Instancia global (singleton)
_nft_credentials: Optional[NFTEducationCredentials] = None


def get_nft_credentials() -> NFTEducationCredentials:
    """Obtener instancia singleton del sistema NFT de credenciales"""
    global _nft_credentials
    if _nft_credentials is None:
        _nft_credentials = NFTEducationCredentials()
    return _nft_credentials
