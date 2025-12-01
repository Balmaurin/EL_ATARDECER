"""
Database connection and utilities for EL-AMANECER
Async database implementation using aiosqlite
"""

import logging
import aiosqlite
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class AsyncDatabase:
    """Async database connection manager"""

    def __init__(self, url: str):
        self.url = url
        self.db_path = url.replace("sqlite:///", "")
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self):
        """Establish database connection"""
        try:
            # Ensure directory exists
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            
            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row
            logger.info(f"Connected to database: {self.db_path}")
            return self
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    async def disconnect(self):
        """Close database connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database disconnected")

    async def execute(self, query: str, parameters: tuple = ()) -> aiosqlite.Cursor:
        """Execute a query"""
        if not self._connection:
            await self.connect()
        return await self._connection.execute(query, parameters)

    async def fetch_one(self, query: str, parameters: tuple = ()) -> Optional[Dict[str, Any]]:
        """Fetch a single row"""
        if not self._connection:
            await self.connect()
        async with self._connection.execute(query, parameters) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def fetch_all(self, query: str, parameters: tuple = ()) -> List[Dict[str, Any]]:
        """Fetch all rows"""
        if not self._connection:
            await self.connect()
        async with self._connection.execute(query, parameters) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def commit(self):
        """Commit transaction"""
        if self._connection:
            await self._connection.commit()

# Database instance - use settings
from .config.settings import settings
db = AsyncDatabase(settings.database_url)

async def get_db():
    """Dependency injection for database connection"""
    if not db._connection:
        await db.connect()
    return db
