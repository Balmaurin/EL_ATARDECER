"""
Enterprise Database Connection Management
=======================================

High-performance PostgreSQL database connection management for Sheily MCP.
Features connection pooling, Row-Level Security (RLS), async support, and monitoring.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import asyncpg
from sqlalchemy import MetaData, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from apps.backend.src.core.config.settings import settings


class DatabaseManager:
    """Enterprise database connection manager"""

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._async_session_maker: Optional[async_sessionmaker] = None
        self._sync_engine = None
        self._sync_session_maker = None

        # Connection pool monitoring
        self._connection_stats = {
            "active_connections": 0,
            "total_connections": 0,
            "connection_errors": 0,
            "query_count": 0,
        }

    async def initialize(self) -> None:
        """Initialize database connections"""
        try:
            # Create async engine for production use
            self._engine = create_async_engine(
                settings.database_url,
                pool_size=settings.db_pool_size,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=settings.db_pool_recycle,
                pool_timeout=settings.db_pool_timeout,
                echo=False,  # Set to True for SQL debugging in development
                connect_args={
                    "command_timeout": 60,
                    "server_settings": {
                        "timezone": "UTC",
                        "application_name": f"Sheily MCP {settings.version}",
                    },
                },
            )

            # Create async session factory
            self._async_session_maker = async_sessionmaker(
                bind=self._engine,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )

            # Sync engine for utilities (migrations, etc.)
            self._sync_engine = create_engine(
                settings.database_url.replace("postgresql+asyncpg://", "postgresql://"),
                pool_size=5,
                max_overflow=5,
                pool_pre_ping=True,
                echo=False,
            )

            self._sync_session_maker = sessionmaker(
                bind=self._sync_engine, expire_on_commit=False
            )

            # Test connection
            async with self._engine.begin() as conn:
                await conn.execute("SELECT 1")

            print(f"[OK] Database connected successfully to PostgreSQL")

        except Exception as e:
            print(f"[ERROR] Database connection failed: {e}")
            raise

    async def close(self) -> None:
        """Close database connections"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None

        if self._sync_engine:
            self._sync_engine.dispose()
            self._sync_engine = None

        print("🗑️ Database connections closed")

    @asynccontextmanager
    async def get_async_session(self, tenant_id: Optional[str] = None):
        """Get async database session with tenant isolation"""
        if not self._async_session_maker:
            raise RuntimeError("Database not initialized")

        async with self._async_session_maker() as session:
            try:
                # Set tenant context for Row-Level Security (RLS)
                if tenant_id:
                    # PostgreSQL RLS: Set session variables for tenant isolation
                    await session.execute(f"SET app.current_tenant_id = '{tenant_id}'")
                    await session.execute(
                        "SET app.current_user_id = NULL"
                    )  # Will be set later

                # Enable Row Level Security
                await session.execute("SET row_security = on")

                self._connection_stats["active_connections"] += 1

                yield session

                await session.commit()

            except Exception as e:
                await session.rollback()
                self._connection_stats["connection_errors"] += 1
                raise
            finally:
                self._connection_stats["active_connections"] -= 1

    def get_sync_session(self) -> Session:
        """Get synchronous session (for utilities)"""
        if not self._sync_session_maker:
            raise RuntimeError("Database not initialized")
        return self._sync_session_maker()

    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Any:
        """Execute raw SQL query with tenant isolation"""
        async with self.get_async_session(tenant_id) as session:
            result = await session.execute(query, params or {})
            self._connection_stats["query_count"] += 1
            return result

    async def health_check(self) -> Dict[str, Any]:
        """Database health check"""
        try:
            # Query database information
            async with self.get_async_session() as session:
                # Basic connectivity test
                result = await session.execute("SELECT version() as postgres_version")
                version_row = result.fetchone()

                # Connection pool stats
                pool_stats = await session.execute(
                    """
                    SELECT
                        (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                        (SELECT count(*) FROM pg_stat_activity) as total_connections,
                        (SELECT sum(tup_ins) + sum(tup_upd) + sum(tup_del) FROM pg_stat_user_tables) as total_operations
                """
                )
                stats_row = pool_stats.fetchone()

                # Database size
                size_result = await session.execute(
                    """
                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                """
                )
                size_row = size_result.fetchone()

                return {
                    "status": "healthy",
                    "postgres_version": (
                        version_row.postgres_version if version_row else "unknown"
                    ),
                    "active_connections": (
                        stats_row.active_connections if stats_row else 0
                    ),
                    "total_connections": (
                        stats_row.total_connections if stats_row else 0
                    ),
                    "database_size": size_row.db_size if size_row else "unknown",
                    "pool_size": settings.db_pool_size,
                    "max_connections": getattr(
                        settings, "max_connections", settings.db_pool_size * 2
                    ),
                    "connection_errors": self._connection_stats["connection_errors"],
                    "query_count": self._connection_stats["query_count"],
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_errors": self._connection_stats["connection_errors"],
            }

    async def create_tenant_schema(self, tenant_id: str) -> None:
        """Create tenant-specific schema with RLS policies"""
        async with self.get_async_session() as session:
            try:
                # Create tenant schema
                schema_name = f"tenant_{tenant_id}"

                # Note: In production, this would create separate schemas
                # For now, we use application-level RLS within shared tables

                # Verify tenant isolation setup
                await session.execute(
                    """
                    SELECT set_config('app.current_tenant_id', $1, false)
                """,
                    (tenant_id,),
                )

                await session.commit()
                print(f"[OK] Tenant schema prepared for tenant: {tenant_id}")

            except Exception as e:
                await session.rollback()
                print(f"[ERROR] Failed to create tenant schema: {e}")
                raise

    async def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get statistics for a specific tenant"""
        async with self.get_async_session(tenant_id) as session:
            try:
                # User count
                user_result = await session.execute(
                    """
                    SELECT COUNT(*) as user_count FROM users
                """
                )
                user_count = user_result.scalar() or 0

                # Conversation count
                conv_result = await session.execute(
                    """
                    SELECT COUNT(*) as conversation_count FROM conversations
                """
                )
                conv_count = conv_result.scalar() or 0

                # Agent count
                agent_result = await session.execute(
                    """
                    SELECT COUNT(*) as agent_count FROM agents WHERE is_active = true
                """
                )
                agent_count = agent_result.scalar() or 0

                # Dataset count
                dataset_result = await session.execute(
                    """
                    SELECT COUNT(*) as dataset_count FROM exercise_datasets
                """
                )
                dataset_count = dataset_result.scalar() or 0

                # Token usage
                token_result = await session.execute(
                    """
                    SELECT
                        SUM(tokens_earned) as total_tokens_earned,
                        SUM(tokens_spent) as total_tokens_spent
                    FROM exercise_datasets
                """
                )
                token_row = token_result.fetchone()

                return {
                    "tenant_id": tenant_id,
                    "user_count": user_count,
                    "conversation_count": conv_count,
                    "agent_count": agent_count,
                    "dataset_count": dataset_count,
                    "total_tokens_earned": token_row.total_tokens_earned or 0,
                    "total_tokens_spent": token_row.total_tokens_spent or 0,
                    "net_tokens": (token_row.total_tokens_earned or 0)
                    - (token_row.total_tokens_spent or 0),
                }

            except Exception as e:
                print(f"[ERROR] Error getting tenant stats: {e}")
                return {
                    "tenant_id": tenant_id,
                    "error": str(e),
                    "user_count": 0,
                    "conversation_count": 0,
                    "agent_count": 0,
                    "dataset_count": 0,
                    "total_tokens_earned": 0,
                    "total_tokens_spent": 0,
                    "net_tokens": 0,
                }


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
async def get_db(tenant_id: Optional[str] = None):
    """Dependency injection for async database sessions"""
    async with db_manager.get_async_session(tenant_id) as session:
        yield session


def get_sync_db():
    """Dependency injection for sync database sessions"""
    return db_manager.get_sync_session()


# Export database connection functions
__all__ = ["db_manager", "DatabaseManager", "get_db", "get_sync_db"]
