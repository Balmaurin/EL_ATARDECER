"""
Sistema de autenticación JWT para Sheily AI
Gestión de usuarios, tokens y seguridad
"""

import logging
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from ..config.settings import settings
from ...models.base import APIToken, User
from ...models.database import get_db_session

logger = logging.getLogger(__name__)

# Configuración de seguridad
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)


class AuthService:
    """Servicio de autenticación y gestión de usuarios"""

    def __init__(self):
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.jwt_expiration_hours * 60
        self.refresh_token_expire_days = settings.refresh_token_expire_days

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verificar contraseña con manejo de errores seguro"""
        try:
            if not plain_password or not hashed_password:
                logger.warning("Intento de verificar contraseña con campos vacíos")
                return False
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Error verificando contraseña: {e}")
            return False

    def get_password_hash(self, password: str) -> str:
        """Generar hash de contraseña"""
        return pwd_context.hash(password)

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Crear token de acceso JWT"""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )

        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Crear token de refresh JWT"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verificar y decodificar token JWT"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None

    def authenticate_user(
        self, db: Session, email: str, password: str
    ) -> Optional[User]:
        """Autenticar usuario con email y contraseña"""
        try:
            if not email or not password:
                logger.warning("Intento de autenticación con campos vacíos")
                return None

            user = db.query(User).filter(User.email == email).first()
            if not user:
                logger.info(f"Usuario no encontrado: {email}")
                return None

            if not user.is_active:
                logger.warning(f"Intento de login con usuario inactivo: {email}")
                return None

            if not self.verify_password(password, user.hashed_password):
                logger.warning(f"Contraseña incorrecta para usuario: {email}")
                return None

            logger.info(f"Usuario autenticado exitosamente: {email}")
            return user

        except Exception as e:
            logger.error(f"Error durante autenticación para {email}: {e}")
            return None

    def get_current_user(self, db: Session, token: str) -> Optional[User]:
        """Obtener usuario actual desde token con mejor validación"""
        try:
            if not token:
                logger.warning("Intento de obtener usuario con token vacío")
                return None

            payload = self.verify_token(token)
            if not payload or payload.get("type") != "access":
                logger.warning("Token inválido o no es de acceso")
                return None

            user_id_str: str = payload.get("sub")
            if user_id_str is None:
                logger.warning("Token sin user_id (sub)")
                return None

            try:
                user_id = int(user_id_str)
            except (ValueError, TypeError):
                logger.error(f"User ID inválido en token: {user_id_str}")
                return None

            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                logger.warning(f"Usuario no encontrado para ID: {user_id}")
                return None

            if not user.is_active:
                logger.warning(f"Usuario inactivo intentó acceder: {user.email}")
                return None

            return user

        except Exception as e:
            logger.error(f"Error obteniendo usuario actual: {e}")
            return None

    def create_user(
        self, db: Session, email: str, username: str, password: str, **kwargs
    ) -> User:
        """Crear nuevo usuario"""
        # Verificar que no exista
        if db.query(User).filter(User.email == email).first():
            raise HTTPException(status_code=400, detail="Email already registered")

        if db.query(User).filter(User.username == username).first():
            raise HTTPException(status_code=400, detail="Username already taken")

        # Crear usuario
        hashed_password = self.get_password_hash(password)
        db_user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            full_name=kwargs.get("full_name"),
            role=kwargs.get("role", "user"),
        )

        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        logger.info(f"User created: {username} ({email})")
        return db_user

    def update_user(self, db: Session, user_id: int, **updates) -> User:
        """Actualizar usuario"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Actualizar campos permitidos
        allowed_fields = ["full_name", "is_active", "preferences"]
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(user, field, value)

        db.commit()
        db.refresh(user)
        return user

    def change_password(
        self, db: Session, user_id: int, old_password: str, new_password: str
    ) -> bool:
        """Cambiar contraseña de usuario"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False

        if not self.verify_password(old_password, user.hashed_password):
            return False

        user.hashed_password = self.get_password_hash(new_password)
        db.commit()

        logger.info(f"Password changed for user: {user.username}")
        return True

    def create_api_token(
        self,
        db: Session,
        user_id: int,
        name: str,
        permissions: Optional[Dict] = None,
    ) -> APIToken:
        """Crear token de API para usuario"""
        import secrets
        from hashlib import sha256

        # Generar token aleatorio
        token_plain = secrets.token_urlsafe(32)
        token_hash = sha256(token_plain.encode()).hexdigest()

        # Crear registro en BD
        api_token = APIToken(
            user_id=user_id,
            name=name,
            token_hash=token_hash,
            permissions=permissions or {},
            expires_at=None,  # Sin expiración por defecto
        )

        db.add(api_token)
        db.commit()
        db.refresh(api_token)

        # Devolver token (solo se muestra una vez)
        api_token._plain_token = token_plain
        logger.info(f"API token created for user {user_id}: {name}")
        return api_token

    def verify_api_token(self, db: Session, token: str) -> Optional[User]:
        """Verificar token de API"""
        token_hash = sha256(token.encode()).hexdigest()
        api_token = (
            db.query(APIToken)
            .filter(APIToken.token_hash == token_hash, APIToken.is_active.is_(True))
            .first()
        )

        if not api_token:
            return None

        # Actualizar last_used_at
        api_token.last_used_at = datetime.utcnow()
        db.commit()

        return api_token.user

    def revoke_api_token(self, db: Session, token_id: int, user_id: int) -> bool:
        """Revocar token de API"""
        api_token = (
            db.query(APIToken)
            .filter(APIToken.id == token_id, APIToken.user_id == user_id)
            .first()
        )

        if not api_token:
            return False

        api_token.is_active = False
        db.commit()

        logger.info(f"API token revoked: {token_id}")
        return True


# Instancia global del servicio de autenticación
auth_service = AuthService()


# Dependencias FastAPI
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db_session),
) -> User:
    """Dependencia para obtener usuario actual desde JWT"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = auth_service.get_current_user(db, credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Dependencia para usuario activo"""
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Dependencia para usuario administrador"""
    if current_user.role not in ["admin", "super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


# Funciones de utilidad
def create_user_response(user: User) -> Dict[str, Any]:
    """Crear respuesta de usuario (sin datos sensibles)"""
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "preferences": user.preferences,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


def create_token_response(
    access_token: str, refresh_token: str, user: User
) -> Dict[str, Any]:
    """Crear respuesta de login con tokens"""
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.jwt_expiration_hours * 3600,
        "user": create_user_response(user),
    }


# Exportar
__all__ = [
    "AuthService",
    "auth_service",
    "get_current_user",
    "get_current_active_user",
    "get_current_admin_user",
    "create_user_response",
    "create_token_response",
]
