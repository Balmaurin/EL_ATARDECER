'''
Enterprise Security Manager
Real authentication implementation using AuthService
'''

from typing import Any, Dict, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..auth.service import auth_service
from ...models.database import get_db_session
from ...models.database import User as DBUser

class SecurityManager:
    ''' Enterprise security management system '''
    
    def __init__(self):
        self._bearer_scheme = HTTPBearer(auto_error=False)
        self._rate_limits = {}

security_manager = SecurityManager()

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db_session)
) -> DBUser:
    ''' FastAPI dependency for authenticated users - Real implementation '''
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
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user

def require_admin(current_user: DBUser = Depends(get_current_user)) -> DBUser:
    ''' Require admin role '''
    if current_user.role not in ["admin", "super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

def require_enterprise(current_user: DBUser = Depends(get_current_user)) -> DBUser:
    ''' Require enterprise role or higher '''
    if current_user.role not in ["enterprise", "admin", "super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Enterprise subscription required"
        )
    return current_user

__all__ = [
    'security_manager',
    'SecurityManager',
    'get_current_user',
    'require_admin',
    'require_enterprise'
]
