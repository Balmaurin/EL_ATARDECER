"""
API Dependencies Module
Common dependencies for FastAPI endpoints with REAL authentication
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from jose import JWTError, jwt

from apps.backend.src.models.user import User
from apps.backend.src.models.database import get_db_session, User as DBUser
from apps.backend.src.core.config.settings import settings
from sqlalchemy.orm import Session

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db_session)
) -> User:
    """
    Get current authenticated user from JWT token
    REAL IMPLEMENTATION - No mocks or stubs

    Validates JWT token and loads user from database
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode JWT token
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )

        # Extract user_id from token
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    # Load user from database
    db_user = db.query(DBUser).filter(DBUser.id == int(user_id)).first()
    if db_user is None:
        raise credentials_exception

    # Check if user is active
    if not db_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    # Convert DB user to API User model
    return User(
        id=str(db_user.id),
        username=db_user.username,
        email=db_user.email,
        created_at=db_user.created_at.isoformat(),
        updated_at=db_user.updated_at.isoformat(),
        is_active=db_user.is_active,
        is_verified=db_user.is_verified
    )


async def get_admin_user(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
) -> User:
    """
    Get current user and verify admin permissions
    REAL IMPLEMENTATION - Checks actual role from database
    """
    # Load full user from database to check role
    db_user = db.query(DBUser).filter(DBUser.id == int(current_user.id)).first()

    if not db_user or db_user.role not in ["admin", "superadmin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise
    Useful for endpoints that work for both authenticated and anonymous users
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except:
        return None


__all__ = [
    "get_current_user",
    "get_admin_user",
    "get_optional_user",
    "security"
]
