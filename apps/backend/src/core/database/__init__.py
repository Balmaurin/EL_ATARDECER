# Import session management from session.py
from .session import (
    db_manager,
    DatabaseManager,
    get_db,
    get_sync_db
)

# Import models from models.database
from ...models.database import (
    Base,
    engine,
    SessionLocal,
    User,
    Conversation,
    Message,
    Exercise,
    Dataset,
    Document,
    Embedding,
    Transaction,
    SystemMetric,
    CacheEntry,
    get_db_session
)

# Re-export for backward compatibility
__all__ = [
    # Session management
    "db_manager",
    "DatabaseManager",
    "get_db",
    "get_sync_db",
    # Models and engine
    "Base",
    "engine",
    "SessionLocal",
    "User",
    "Conversation",
    "Message",
    "Exercise",
    "Dataset",
    "Document",
    "Embedding",
    "Transaction",
    "SystemMetric",
    "CacheEntry",
    "get_db_session"
]
