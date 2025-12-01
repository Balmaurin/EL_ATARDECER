"""
MODELS - Modelos de datos comunes para EL-AMANECER API
======================================================

Modelos de base de datos compartidos por todos los routers.
Elimina dependencias circulares y warnings de importación.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

# ==========================================
# ENUMERATIONS COMUNES
# ==========================================

class UserRole(str, Enum):
    """Roles de usuario en el sistema"""
    ADMIN = "admin"
    PREMIUM = "premium"
    BASIC = "basic"
    ALPHA_TESTER = "alpha_tester"

class ConversationStatus(str, Enum):
    """Estados de conversación"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class MessageType(str, Enum):
    """Tipos de mensaje"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class EmotionType(str, Enum):
    """Tipos de emoción reconocidos"""
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    LOVE = "love"
    TRUST = "trust"
    JOY = "joy"
    ANXIETY = "anxiety"
    CALM = "calm"
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"
    EXCITEMENT = "excitement"
    NEUTRAL = "neutral"

# ==========================================
# MODELOS PYDANTIC PARA API
# ==========================================

class BaseResponseModel(BaseModel):
    """Modelo base para respuestas de API"""
    success: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"

class ErrorResponseModel(BaseResponseModel):
    """Modelo para respuestas de error"""
    success: bool = False
    error_code: str = ""
    error_details: Optional[Dict[str, Any]] = None

class UserProfile(BaseModel):
    """Perfil de usuario básico"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    role: UserRole = UserRole.BASIC
    tokens_balance: int = 0
    level: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConsciousnessMetrics(BaseModel):
    """Métricas de consciencia y emoción"""
    phi_value: float = 0.0
    emotion: EmotionType = EmotionType.NEUTRAL
    arousal: float = 0.5
    complexity: float = 0.0
    user_recognized: bool = False
    conversation_count: int = 0
    awareness_level: float = 0.0
    cognitive_load: float = 0.0
    thinking_steps: List[str] = Field(default_factory=list)

class ConversationMessage(BaseModel):
    """Mensaje en una conversación"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    type: MessageType = MessageType.USER
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tokens_used: Optional[int] = None
    consciousness_data: Optional[ConsciousnessMetrics] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Conversation(BaseModel):
    """Conversación completa"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str = ""
    status: ConversationStatus = ConversationStatus.ACTIVE
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    total_tokens: int = 0
    consciousness_level: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TokenTransaction(BaseModel):
    """Transacción de tokens SHEILY"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    amount: int
    description: str
    type: str = "interaction"  # conversation, premium, reward, etc.
    timestamp: datetime = Field(default_factory=datetime.now)
    balance_before: int = 0
    balance_after: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConsciousnessState(BaseModel):
    """Estado completo de consciencia del sistema"""
    level: str = "0"
    score: float = 0.0
    emotion: str = "neutral"
    load: float = 0.0
    last_thought: str = ""
    total_memories: int = 0
    learning_experiences: int = 0
    average_quality: float = 0.0

class SystemHealth(BaseModel):
    """Estado de salud del sistema"""
    status: str = "healthy"
    version: str = "1.0.0"
    uptime: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    active_connections: int = 0

# ==========================================
# MODELOS PARA FUNCIONALIDADES AVANZADAS
# ==========================================

class CommunityPost(BaseModel):
    """Post de la comunidad"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    content: str
    tags: List[str] = Field(default_factory=list)
    upvotes: int = 0
    downvotes: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: str = "active"

class VaultMessage(BaseModel):
    """Mensaje seguro en vault"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    encrypted_content: str
    salt: str
    iv: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ==========================================
# FUNCTIONS DE AYUDA
# ==========================================

def create_user_profile(username: str, email: str, role: UserRole = UserRole.BASIC) -> UserProfile:
    """Crear un perfil de usuario nuevo"""
    return UserProfile(
        username=username,
        email=email,
        role=role,
        tokens_balance=500,  # Bonus inicial
        preferences={
            "language": "es",
            "theme": "dark",
            "consciousness_level": "medium"
        }
    )

def create_conversation(user_id: str, title: str = "") -> Conversation:
    """Crear una nueva conversación"""
    if not title:
        title = f"Conversación {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    return Conversation(
        user_id=user_id,
        title=title,
        metadata={
            "model": "mental_health_counseling_gemma_7b_merged",
            "consciousness_enabled": True,
            "rag_enabled": True
        }
    )

def calculate_consciousness_metrics(messages: List[ConversationMessage]) -> ConsciousnessMetrics:
    """Calcular métricas de consciencia basadas en mensajes"""
    if not messages:
        return ConsciousnessMetrics()

    # Cálculo simplificado basado en patrones
    total_emotions = sum(1 for msg in messages if msg.consciousness_data)
    avg_emotions = total_emotions / len(messages) if messages else 0

    return ConsciousnessMetrics(
        phi_value=min(1.0, avg_emotions * 0.3),  # Φ escala con emociones detectadas
        emotion=EmotionType.NEUTRAL,  # Podría calcular el promedio
        arousal=0.5,  # Nivel base
        complexity=min(1.0, len(messages) * 0.1),  # Más mensajes = más complejidad
        conversation_count=len(messages),
        awareness_level=avg_emotions,
        thinking_steps=["Percepción", "Procesamiento", "Respuesta"]
    )

# ==========================================
# CONFIGURACIÓN DE MODELOS
# ==========================================

# Paths comunes que las rutas pueden usar
DATABASE_PATH = Path("data/db/gamified_database.db")
VAULT_PATH = Path("data/vault.db")
USER_PROFILES_PATH = Path("data/user_profiles")

# Asegurar que los directorios existan
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
VAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
USER_PROFILES_PATH.mkdir(parents=True, exist_ok=True)

__all__ = [
    # Enums
    "UserRole", "ConversationStatus", "MessageType", "EmotionType",

    # Base models
    "BaseResponseModel", "ErrorResponseModel",

    # User models
    "UserProfile",

    # Conversation models
    "ConversationMessage", "Conversation",

    # Token models
    "TokenTransaction",

    # Consciousness models
    "ConsciousnessMetrics", "ConsciousnessState",

    # System models
    "SystemHealth",

    # Advanced models
    "CommunityPost", "VaultMessage",

    # Helper functions
    "create_user_profile", "create_conversation", "calculate_consciousness_metrics",

    # Constants
    "DATABASE_PATH", "VAULT_PATH", "USER_PROFILES_PATH"
]
