"""
Token Economy Educativa para Sheily AI
Implementa sistema de recompensas educativas usando tokens SHEILYS
Basado en investigación: REAL8 Learn-to-Earn, Token Economy pedagógica, Modelos económicos

Características:
- Learn-to-Earn con SHEILYS tokens
- Sistema de staking educativo
- Recompensas por engagement y aprendizaje
- Integración con sistema de recompensas existente
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Integración completa con sistemas reales de Sheily AI
from packages.sheily_core.src.sheily_core.blockchain.sheily_spl_manager import get_sheily_spl_manager
from packages.sheily_core.src.sheily_core.rewards.reward_system import SheilyRewardSystem

logger = logging.getLogger(__name__)


class EducationActivity(Enum):
    """Tipos de actividades educativas que generan recompensas"""

    COURSE_COMPLETION = "course_completion"
    QUIZ_SUCCESS = "quiz_success"
    PEER_FEEDBACK = "peer_feedback"
    DISCUSSION_PARTICIPATION = "discussion_participation"
    TUTORING_SESSION = "tutoring_session"
    STUDY_GROUP_ATTENDANCE = "study_group_attendance"
    RESEARCH_CONTRIBUTION = "research_contribution"
    LEARNING_MILESTONE = "learning_milestone"


@dataclass
class EducationalReward:
    """Estructura de recompensa educativa"""

    activity_type: EducationActivity
    base_sheilys: float
    multiplier: float = 1.0
    bonus_criteria: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def total_sheilys(self) -> float:
        """Calcular recompensa total con multiplicador"""
        return round(self.base_sheilys * self.multiplier, 2)


@dataclass
class LearningSession:
    """Sesión de aprendizaje con métricas"""

    user_id: str
    activity_type: EducationActivity
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    engagement_level: str = "low"  # low, medium, high
    completion_status: str = "in_progress"  # in_progress, completed, failed


class EducationalTokenEconomy:
    """
    Sistema de economía de tokens educativos
    Integra SHEILYS tokens con actividades de aprendizaje
    """

    def __init__(self):
        # Integración completa con sistemas reales de Sheily AI
        self.spl_manager = get_sheily_spl_manager()
        self.reward_system = SheilyRewardSystem(vault_path="./rewards/vault/education")
        self.active_sessions: Dict[str, LearningSession] = {}
        self.reward_history: List[Dict[str, Any]] = []

        # Configuración de recompensas base
        self.reward_config = {
            EducationActivity.COURSE_COMPLETION: EducationalReward(
                EducationActivity.COURSE_COMPLETION,
                50.0,
                description="Completar un curso completo",
            ),
            EducationActivity.QUIZ_SUCCESS: EducationalReward(
                EducationActivity.QUIZ_SUCCESS,
                10.0,
                description="Aprobar un quiz con calificación >= 80%",
            ),
            EducationActivity.PEER_FEEDBACK: EducationalReward(
                EducationActivity.PEER_FEEDBACK,
                5.0,
                description="Proporcionar feedback constructivo a un compañero",
            ),
            EducationActivity.DISCUSSION_PARTICIPATION: EducationalReward(
                EducationActivity.DISCUSSION_PARTICIPATION,
                3.0,
                description="Participar activamente en discusiones",
            ),
            EducationActivity.TUTORING_SESSION: EducationalReward(
                EducationActivity.TUTORING_SESSION,
                25.0,
                description="Realizar una sesión de tutoría",
            ),
            EducationActivity.STUDY_GROUP_ATTENDANCE: EducationalReward(
                EducationActivity.STUDY_GROUP_ATTENDANCE,
                8.0,
                description="Asistir a grupo de estudio",
            ),
            EducationActivity.RESEARCH_CONTRIBUTION: EducationalReward(
                EducationActivity.RESEARCH_CONTRIBUTION,
                30.0,
                description="Contribuir a investigación académica",
            ),
            EducationActivity.LEARNING_MILESTONE: EducationalReward(
                EducationActivity.LEARNING_MILESTONE,
                15.0,
                description="Alcanzar hito de aprendizaje importante",
            ),
        }

        # Multiplicadores por engagement
        self.engagement_multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5,
            "exceptional": 2.0,
        }

        logger.info("🎓 Educational Token Economy inicializado")

    async def start_learning_session(
        self,
        user_id: str,
        activity_type: EducationActivity,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Iniciar una sesión de aprendizaje
        Retorna session_id único
        """
        session_id = f"{user_id}_{activity_type.value}_{datetime.now().timestamp()}"

        session = LearningSession(
            user_id=user_id,
            activity_type=activity_type,
            start_time=datetime.now(),
            metrics=metadata or {},
        )

        self.active_sessions[session_id] = session
        logger.info(f"📚 Sesión de aprendizaje iniciada: {session_id}")
        return session_id

    async def complete_learning_session(
        self,
        session_id: str,
        quality_score: float = 0.0,
        engagement_level: str = "medium",
        additional_metrics: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Completar sesión de aprendizaje y calcular recompensa
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Sesión no encontrada: {session_id}")

        session = self.active_sessions[session_id]
        session.end_time = datetime.now()
        session.quality_score = quality_score
        session.engagement_level = engagement_level
        session.completion_status = "completed"

        if additional_metrics:
            session.metrics.update(additional_metrics)

        # Calcular duración
        duration = (
            session.end_time - session.start_time
        ).total_seconds() / 60  # minutos
        session.metrics["duration_minutes"] = duration

        # Calcular recompensa
        reward = await self._calculate_educational_reward(session)

        # Registrar en sistema de recompensas
        reward_record = await self._register_reward(session, reward)

        # Limpiar sesión activa
        del self.active_sessions[session_id]

        logger.info(
            f"✅ Sesión completada: {session_id}, Recompensa: {reward.total_sheilys} SHEILYS"
        )
        return reward_record

    async def _calculate_educational_reward(
        self, session: LearningSession
    ) -> EducationalReward:
        """
        Calcular recompensa educativa basada en sesión
        Implementa lógica de REAL8 Learn-to-Earn + investigación token economy
        """
        base_reward = self.reward_config[session.activity_type]

        # Multiplicador por engagement
        engagement_multiplier = self.engagement_multipliers.get(
            session.engagement_level, 1.0
        )

        # Multiplicador por calidad
        quality_multiplier = 0.5 + (session.quality_score * 0.5)  # 0.5 a 1.0

        # Multiplicador por duración (para actividades largas)
        duration_multiplier = 1.0
        if session.metrics.get("duration_minutes", 0) > 60:  # Más de 1 hora
            duration_multiplier = 1.2
        elif session.metrics.get("duration_minutes", 0) > 120:  # Más de 2 horas
            duration_multiplier = 1.5

        # Multiplicador por dificultad/complejidad
        complexity_multiplier = session.metrics.get("complexity_multiplier", 1.0)

        # Calcular multiplicador total
        total_multiplier = (
            engagement_multiplier
            * quality_multiplier
            * duration_multiplier
            * complexity_multiplier
        )

        # Crear recompensa final
        final_reward = EducationalReward(
            activity_type=session.activity_type,
            base_sheilys=base_reward.base_sheilys,
            multiplier=round(total_multiplier, 2),
            bonus_criteria={
                "engagement_level": session.engagement_level,
                "quality_score": session.quality_score,
                "duration_minutes": session.metrics.get("duration_minutes", 0),
                "complexity_multiplier": complexity_multiplier,
            },
            description=f"{base_reward.description} - {session.engagement_level} engagement",
        )

        return final_reward

    async def _register_reward(
        self, session: LearningSession, reward: EducationalReward
    ) -> Dict[str, Any]:
        """
        Registrar recompensa en sistema de SHEILYS usando SPL manager real
        Integración completa con blockchain y sistema de recompensas
        """
        try:
            # 1. Mintear tokens SHEILYS usando SPL manager real
            spl_transaction = self.spl_manager.mint_tokens(
                user_id=session.user_id,
                amount=int(reward.total_sheilys),  # Convertir a int para SPL
                reason=f"Educational activity: {session.activity_type.value}",
            )

            # 2. Registrar en sistema de recompensas Sheilys
            reward_data = {
                "session_id": f"{session.user_id}_{session.activity_type.value}_{session.start_time.timestamp()}",
                "domain": "education",
                "quality_score": session.quality_score,
                "tokens_used": session.metrics.get(
                    "duration_minutes", 0
                ),  # Usar duración como proxy
                "query": f"Educational activity: {session.activity_type.value}",
                "response": f"Completed with {session.engagement_level} engagement and {session.quality_score} quality",
                "user_id": session.user_id,
                "activity_type": session.activity_type.value,
                "engagement_level": session.engagement_level,
                "duration_minutes": session.metrics.get("duration_minutes", 0),
                "blockchain_tx": (
                    spl_transaction.transaction_id
                    if hasattr(spl_transaction, "transaction_id")
                    else spl_transaction.get("transaction_id", "unknown")
                ),
            }

            sheilys_record = self.reward_system.record_reward(reward_data)

            # 3. Guardar en historial educativo
            history_record = {
                "session_id": sheilys_record["reward_id"],
                "user_id": session.user_id,
                "activity_type": session.activity_type.value,
                "reward_details": {
                    "base_sheilys": reward.base_sheilys,
                    "multiplier": reward.multiplier,
                    "total_sheilys": reward.total_sheilys,
                    "bonus_criteria": reward.bonus_criteria,
                },
                "session_metrics": {
                    "quality_score": session.quality_score,
                    "engagement_level": session.engagement_level,
                    "duration_minutes": session.metrics.get("duration_minutes", 0),
                    "completion_status": session.completion_status,
                },
                "blockchain_tx": (
                    spl_transaction.transaction_id
                    if hasattr(spl_transaction, "transaction_id")
                    else spl_transaction.get("transaction_id", "unknown")
                ),
                "sheilys_system_id": sheilys_record["reward_id"],
                "timestamp": session.end_time.isoformat(),
            }

            self.reward_history.append(history_record)

            logger.info(
                f"💎 Recompensa registrada: {reward.total_sheilys} SHEILYS minteados para {session.user_id}"
            )
            logger.info(
                f"🔗 TX Blockchain: {spl_transaction.transaction_id if hasattr(spl_transaction, 'transaction_id') else spl_transaction.get('transaction_id', 'unknown')}"
            )
            logger.info(f"📊 Sistema Sheilys ID: {sheilys_record['reward_id']}")

            return {
                "success": True,
                "reward_id": sheilys_record["reward_id"],
                "total_sheilys": reward.total_sheilys,
                "activity_type": session.activity_type.value,
                "blockchain_tx": (
                    spl_transaction.transaction_id
                    if hasattr(spl_transaction, "transaction_id")
                    else spl_transaction.get("transaction_id", "unknown")
                ),
                "sheilys_record": sheilys_record,
                "multipliers_applied": {
                    "engagement": self.engagement_multipliers.get(
                        session.engagement_level, 1.0
                    ),
                    "quality": 0.5 + (session.quality_score * 0.5),
                    "duration": (
                        1.2 if session.metrics.get("duration_minutes", 0) > 60 else 1.0
                    ),
                    "complexity": session.metrics.get("complexity_multiplier", 1.0),
                },
            }

        except Exception as e:
            logger.error(f"Error registrando recompensa educativa: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_sheilys": reward.total_sheilys,
            }

    async def get_user_educational_balance(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener balance educativo del usuario usando SPL manager real
        """
        try:
            # Obtener balance real de SHEILYS desde blockchain
            user_balance = self.spl_manager.get_user_balance(user_id)

            # Filtrar recompensas educativas
            educational_rewards = [
                r for r in self.reward_history if r["user_id"] == user_id
            ]

            total_educational_sheilys = sum(
                r["reward_details"]["total_sheilys"] for r in educational_rewards
            )

            # Estadísticas por tipo de actividad
            activity_stats = {}
            for reward in educational_rewards:
                activity = reward["activity_type"]
                if activity not in activity_stats:
                    activity_stats[activity] = {
                        "count": 0,
                        "total_sheilys": 0.0,
                        "avg_quality": 0.0,
                        "qualities": [],
                    }

                activity_stats[activity]["count"] += 1
                activity_stats[activity]["total_sheilys"] += reward["reward_details"][
                    "total_sheilys"
                ]
                if reward["session_metrics"]["quality_score"] > 0:
                    activity_stats[activity]["qualities"].append(
                        reward["session_metrics"]["quality_score"]
                    )

            # Calcular promedios
            for activity, stats in activity_stats.items():
                if stats["qualities"]:
                    stats["avg_quality"] = sum(stats["qualities"]) / len(
                        stats["qualities"]
                    )
                del stats["qualities"]  # Limpiar datos temporales

            return {
                "user_id": user_id,
                "total_sheilys_balance": user_balance.get("token_balance", 0),
                "educational_sheilys": total_educational_sheilys,
                "educational_percentage": (
                    total_educational_sheilys
                    / max(user_balance.get("token_balance", 1), 1)
                )
                * 100,
                "activity_stats": activity_stats,
                "total_sessions": len(educational_rewards),
                "last_activity": (
                    educational_rewards[-1]["timestamp"]
                    if educational_rewards
                    else None
                ),
                "blockchain_info": {
                    "account_address": user_balance.get("token_account"),
                    "mint_address": user_balance.get("mint_address"),
                    "last_updated": user_balance.get("last_updated"),
                },
            }

        except Exception as e:
            logger.error(f"Error obteniendo balance educativo: {e}")
            return {
                "user_id": user_id,
                "error": str(e),
                "total_sheilys_balance": 0,
                "educational_sheilys": 0,
            }

    async def get_educational_leaderboard(
        self, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Obtener leaderboard de usuarios por actividad educativa
        """
        try:
            # Agrupar por usuario
            user_stats = {}
            for reward in self.reward_history:
                user_id = reward["user_id"]
                if user_id not in user_stats:
                    user_stats[user_id] = {
                        "user_id": user_id,
                        "total_sheilys": 0,
                        "total_sessions": 0,
                        "avg_quality": 0,
                        "qualities": [],
                        "activities": set(),
                    }

                user_stats[user_id]["total_sheilys"] += reward["reward_details"][
                    "total_sheilys"
                ]
                user_stats[user_id]["total_sessions"] += 1
                user_stats[user_id]["activities"].add(reward["activity_type"])

                if reward["session_metrics"]["quality_score"] > 0:
                    user_stats[user_id]["qualities"].append(
                        reward["session_metrics"]["quality_score"]
                    )

            # Calcular promedios y preparar resultado
            leaderboard = []
            for user_id, stats in user_stats.items():
                if stats["qualities"]:
                    stats["avg_quality"] = sum(stats["qualities"]) / len(
                        stats["qualities"]
                    )
                stats["unique_activities"] = len(stats["activities"])
                del stats["qualities"]
                del stats["activities"]
                leaderboard.append(stats)

            # Ordenar por SHEILYS totales
            leaderboard.sort(key=lambda x: x["total_sheilys"], reverse=True)

            return leaderboard[:limit]

        except Exception as e:
            logger.error(f"Error obteniendo leaderboard educativo: {e}")
            return []

    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas generales del sistema educativo
        """
        try:
            total_sessions = len(self.reward_history)
            total_sheilys_distributed = sum(
                r["reward_details"]["total_sheilys"] for r in self.reward_history
            )

            # Estadísticas por actividad
            activity_distribution = {}
            for reward in self.reward_history:
                activity = reward["activity_type"]
                if activity not in activity_distribution:
                    activity_distribution[activity] = {
                        "count": 0,
                        "total_sheilys": 0,
                        "avg_sheilys": 0,
                    }
                activity_distribution[activity]["count"] += 1
                activity_distribution[activity]["total_sheilys"] += reward[
                    "reward_details"
                ]["total_sheilys"]

            # Calcular promedios
            for activity, stats in activity_distribution.items():
                stats["avg_sheilys"] = stats["total_sheilys"] / stats["count"]

            # Usuarios únicos
            unique_users = len(set(r["user_id"] for r in self.reward_history))

            return {
                "total_sessions": total_sessions,
                "total_sheilys_distributed": total_sheilys_distributed,
                "unique_users": unique_users,
                "activity_distribution": activity_distribution,
                "avg_sheilys_per_session": total_sheilys_distributed
                / max(total_sessions, 1),
                "avg_sessions_per_user": total_sessions / max(unique_users, 1),
                "active_sessions": len(self.active_sessions),
            }

        except Exception as e:
            logger.error(f"Error obteniendo estadísticas del sistema: {e}")
            return {
                "error": str(e),
                "total_sessions": 0,
                "total_sheilys_distributed": 0,
            }


# Instancia global (singleton)
_educational_token_economy: Optional[EducationalTokenEconomy] = None


def get_educational_token_economy() -> EducationalTokenEconomy:
    """Obtener instancia singleton del sistema de token economy educativa"""
    global _educational_token_economy
    if _educational_token_economy is None:
        _educational_token_economy = EducationalTokenEconomy()
    return _educational_token_economy
