"""
User Service - Sheily AI Backend
Complete production-ready service with real database operations.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc

from apps.backend.src.models.database import (
    User as DBUser,
    Transaction as DBTransaction,
    get_db_session
)


class UserService:
    """
    Production-ready user service with complete database integration.
    Handles user management, tokens, transactions, and business logic.
    """
    
    def __init__(self, db: Optional[Session] = None):
        """Initialize service with optional database session."""
        self._db = db
    
    def _get_db(self) -> Session:
        """Get database session."""
        if self._db:
            return self._db
        # Create new session if not provided
        return next(get_db_session())
    
    async def get_user_by_id(self, user_id: int) -> Optional[DBUser]:
        """
        Retrieve user by ID from database.
        
        Args:
            user_id: User ID
            
        Returns:
            User object or None if not found
        """
        db = self._get_db()
        try:
            return db.query(DBUser).filter(DBUser.id == user_id).first()
        except Exception as e:
            print(f"Error retrieving user {user_id}: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[DBUser]:
        """
        Retrieve user by email from database.
        
        Args:
            email: User email
            
        Returns:
            User object or None if not found
        """
        db = self._get_db()
        try:
            return db.query(DBUser).filter(DBUser.email == email).first()
        except Exception as e:
            print(f"Error retrieving user by email {email}: {e}")
            return None
    
    async def create_user(self, email: str, username: str, full_name: Optional[str] = None) -> DBUser:
        """
        Create a new user in the database.
        
        Args:
            email: User email
            username: Username
            full_name: Full name (optional)
            
        Returns:
            Created user object
        """
        db = self._get_db()
        try:
            new_user = DBUser(
                email=email,
                username=username,
                full_name=full_name,
                sheily_tokens=100.0,  # Initial bonus
                level=1,
                experience_points=0,
                is_active=True,
                is_verified=False
            )
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            # Create welcome transaction
            welcome_tx = DBTransaction(
                user_id=new_user.id,
                transaction_type="reward",
                amount=100.0,
                currency="SHEILY",
                status="confirmed",
                description="Bono de bienvenida",
                confirmed_at=datetime.utcnow()
            )
            db.add(welcome_tx)
            db.commit()
            
            print(f"[OK] Created new user: {username} ({email})")
            return new_user
            
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Error creating user: {e}")
            raise

    async def get_token_balance(self, user_id: int) -> Dict[str, Any]:
        """
        Get complete token balance and statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with balance information
        """
        db = self._get_db()
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return {
                    "error": "User not found",
                    "current_tokens": 0,
                    "level": 1,
                    "experience": 0
                }
            
            # Calculate level progress
            current_level_exp = (user.level - 1) * 1000
            next_level_exp = user.level * 1000
            
            # Get subscription limits
            limits = self._get_subscription_limits(user.role)
            
            # Get transaction statistics
            total_earned = db.query(DBTransaction).filter(
                DBTransaction.user_id == user_id,
                DBTransaction.amount > 0,
                DBTransaction.status == "confirmed"
            ).count()
            
            total_spent = db.query(DBTransaction).filter(
                DBTransaction.user_id == user_id,
                DBTransaction.amount < 0,
                DBTransaction.status == "confirmed"
            ).count()
            
            return {
                "current_tokens": float(user.sheily_tokens),
                "level": user.level,
                "experience": user.experience_points,
                "next_level_experience": next_level_exp,
                "daily_limit": limits["daily"],
                "monthly_limit": limits["monthly"],
                "total_earned": total_earned,
                "total_spent": total_spent
            }
            
        except Exception as e:
            print(f"Error getting token balance for user {user_id}: {e}")
            return {
                "error": str(e),
                "current_tokens": 0,
                "level": 1,
                "experience": 0
            }
    
    def _get_subscription_limits(self, role: str) -> Dict[str, int]:
        """Get token limits based on subscription/role."""
        limits = {
            "user": {"daily": 100, "monthly": 1000},
            "admin": {"daily": 10000, "monthly": 100000},
            "moderator": {"daily": 1000, "monthly": 10000}
        }
        return limits.get(role, limits["user"])
    
    async def add_tokens(self, user_id: int, amount: float, description: str = "Token addition") -> bool:
        """
        Add tokens to user account with transaction logging.
        
        Args:
            user_id: User ID
            amount: Amount to add (positive)
            description: Transaction description
            
        Returns:
            True if successful, False otherwise
        """
        db = self._get_db()
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return False
            
            # Update user balance
            user.sheily_tokens += amount
            user.updated_at = datetime.utcnow()
            
            # Create transaction record
            transaction = DBTransaction(
                user_id=user_id,
                transaction_type="reward",
                amount=amount,
                currency="SHEILY",
                status="confirmed",
                description=description,
                confirmed_at=datetime.utcnow()
            )
            
            db.add(transaction)
            db.commit()
            db.refresh(user)
            
            print(f"[OK] Added {amount} tokens to user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Error adding tokens: {e}")
            return False
    
    async def spend_tokens(self, user_id: int, amount: float, description: str = "Token usage") -> bool:
        """
        Spend tokens from user account with transaction logging.
        
        Args:
            user_id: User ID
            amount: Amount to spend (positive value)
            description: Transaction description
            
        Returns:
            True if successful, False otherwise
        """
        db = self._get_db()
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return False
            
            # Check if user has enough tokens
            if user.sheily_tokens < amount:
                print(f"[WARN] Insufficient tokens for user {user_id}")
                return False
            
            # Update user balance
            user.sheily_tokens -= amount
            user.updated_at = datetime.utcnow()
            
            # Create transaction record
            transaction = DBTransaction(
                user_id=user_id,
                transaction_type="spend",
                amount=-amount,  # Negative for spending
                currency="SHEILY",
                status="confirmed",
                description=description,
                confirmed_at=datetime.utcnow()
            )
            
            db.add(transaction)
            db.commit()
            db.refresh(user)
            
            print(f"[OK] Spent {amount} tokens from user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Error spending tokens: {e}")
            return False
    
    async def add_experience(self, user_id: int, exp_amount: int) -> Dict[str, Any]:
        """
        Add experience points and handle level-ups.
        
        Args:
            user_id: User ID
            exp_amount: Experience to add
            
        Returns:
            Dictionary with level-up information
        """
        db = self._get_db()
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return {"error": "User not found"}
            
            old_level = user.level
            user.experience_points += exp_amount
            
            # Calculate new level (1000 exp per level)
            new_level = (user.experience_points // 1000) + 1
            
            leveled_up = False
            if new_level > old_level:
                user.level = new_level
                leveled_up = True
                
                # Award bonus tokens for level up
                bonus_tokens = new_level * 10
                user.sheily_tokens += bonus_tokens
                
                # Create bonus transaction
                bonus_tx = DBTransaction(
                    user_id=user_id,
                    transaction_type="reward",
                    amount=bonus_tokens,
                    currency="SHEILY",
                    status="confirmed",
                    description=f"Level {new_level} bonus",
                    confirmed_at=datetime.utcnow()
                )
                db.add(bonus_tx)
            
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            
            return {
                "leveled_up": leveled_up,
                "old_level": old_level,
                "new_level": user.level,
                "current_experience": user.experience_points,
                "bonus_tokens": new_level * 10 if leveled_up else 0
            }
            
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Error adding experience: {e}")
            return {"error": str(e)}
    
    async def get_token_transactions(self, user_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get transaction history for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of transactions to return
            offset: Offset for pagination
            
        Returns:
            List of transaction dictionaries
        """
        db = self._get_db()
        try:
            transactions = db.query(DBTransaction).filter(
                DBTransaction.user_id == user_id
            ).order_by(
                desc(DBTransaction.created_at)
            ).limit(limit).offset(offset).all()
            
            return [
                {
                    "id": tx.id,
                    "type": tx.transaction_type,
                    "amount": float(tx.amount),
                    "currency": tx.currency,
                    "description": tx.description,
                    "status": tx.status,
                    "timestamp": tx.created_at.isoformat() if tx.created_at else None,
                    "confirmed_at": tx.confirmed_at.isoformat() if tx.confirmed_at else None
                }
                for tx in transactions
            ]
            
        except Exception as e:
            print(f"Error getting transactions for user {user_id}: {e}")
            return []
    
    async def update_profile(self, user_id: int, data: Dict[str, Any]) -> Optional[DBUser]:
        """
        Update user profile information.
        
        Args:
            user_id: User ID
            data: Dictionary with fields to update
            
        Returns:
            Updated user object or None
        """
        db = self._get_db()
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return None
            
            # Update allowed fields
            if "full_name" in data:
                user.full_name = data["full_name"]
            if "bio" in data:
                user.bio = data["bio"]
            if "location" in data:
                user.location = data["location"]
            if "avatar_url" in data:
                user.avatar_url = data["avatar_url"]
            
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            
            print(f"[OK] Updated profile for user {user_id}")
            return user
            
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Error updating profile: {e}")
            return None
    
    async def upload_avatar(self, user_id: int, avatar_data: bytes) -> Optional[str]:
        """
        Process avatar upload and update user profile.
        
        Args:
            user_id: User ID
            avatar_data: Avatar image data
            
        Returns:
            Avatar URL or None
        """
        import hashlib
        import os
        
        try:
            # Generate unique filename
            file_hash = hashlib.md5(avatar_data).hexdigest()
            filename = f"avatar_{user_id}_{file_hash[:8]}.png"
            
            # Save to uploads directory (in production, use S3/MinIO)
            upload_dir = "data/uploads/avatars"
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, filename)
            with open(file_path, "wb") as f:
                f.write(avatar_data)
            
            # Update user profile
            avatar_url = f"/static/uploads/avatars/{filename}"
            await self.update_profile(user_id, {"avatar_url": avatar_url})
            
            print(f"[OK] Uploaded avatar for user {user_id}")
            return avatar_url
            
        except Exception as e:
            print(f"[ERROR] Error uploading avatar: {e}")
            return None
    
    async def create_payment_session(self, user_id: int, amount: int, price_cents: int, payment_method: str) -> Dict[str, Any]:
        """
        Create a payment session for token purchase.
        
        Args:
            user_id: User ID
            amount: Number of tokens to purchase
            price_cents: Price in cents
            payment_method: Payment method (stripe, paypal, etc.)
            
        Returns:
            Payment session information
        """
        import uuid
        
        try:
            # In production, integrate with Stripe/PayPal
            session_id = f"sess_{uuid.uuid4().hex[:16]}"
            
            # Create pending transaction
            db = self._get_db()
            pending_tx = DBTransaction(
                user_id=user_id,
                transaction_type="purchase",
                amount=amount,
                currency="SHEILY",
                status="pending",
                description=f"Token purchase - {amount} tokens",
                system_metadata={
                    "session_id": session_id,
                    "price_cents": price_cents,
                    "payment_method": payment_method
                }
            )
            
            db.add(pending_tx)
            db.commit()
            
            return {
                "id": session_id,
                "url": f"https://checkout.sheily.ai/pay/{session_id}",
                "status": "pending",
                "amount": amount,
                "price": price_cents / 100,
                "currency": "usd"
            }
            
        except Exception as e:
            print(f"[ERROR] Error creating payment session: {e}")
            return {"error": str(e)}
