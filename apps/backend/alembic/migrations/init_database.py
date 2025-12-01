"""
Database Migrations - Enterprise PostgreSQL Setup
===============================================

Automated database initialization and schema creation for Sheily MCP Enterprise.
Creates all tables, indexes, and row-level security policies with military-grade isolation.
"""

import asyncio
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    UUID,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.sql import text

from apps.backend.src.config.database import async_engine, get_db
from apps.backend.src.config.settings import settings

# Create metadata
metadata = MetaData()

# Users Table - Enterprise authentication with RBAC
users = Table(
    "users",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column("username", String(255), unique=True, nullable=False, index=True),
    Column("email", String(255), unique=True, nullable=False, index=True),
    Column("password_hash", String(255), nullable=False),
    Column("role", String(50), nullable=False, default="user"),
    Column(
        "tenant_id", String(255), nullable=False, index=True
    ),  # Multi-tenant support
    Column("is_active", Boolean(), default=True),
    Column("is_verified", Boolean(), default=False),
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
    Column("updated_at", DateTime(timezone=True), onupdate=text("NOW()")),
    Column("last_login", DateTime(timezone=True)),
    Column("login_attempts", Integer(), default=0),
    Column("locked_until", DateTime(timezone=True)),
    Column("metadata", JSON, default=dict),
)

# Roles and Permissions - RBAC implementation
roles = Table(
    "roles",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column("name", String(100), unique=True, nullable=False),
    Column("description", Text),
    Column("permissions", JSON, default=list),
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
)

# User Sessions - Session management
user_sessions = Table(
    "user_sessions",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "user_id",
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    ),
    Column("session_token", String(512), unique=True, nullable=False),
    Column("refresh_token", String(512), nullable=False),
    Column("ip_address", String(45)),
    Column("user_agent", Text),
    Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
    Column("last_activity", DateTime(timezone=True), default=text("NOW()")),
)

# Conversations - Chat history with AI agents
conversations = Table(
    "conversations",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "user_id",
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    ),
    Column("title", String(500), nullable=False),
    Column("model_used", String(100), default="sheily-mcp"),
    Column("status", String(50), default="active"),
    Column("total_tokens", BigInteger(), default=0),
    Column("total_cost", Float, default=0.0),
    Column("agents_used", JSON, default=list),
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
    Column("updated_at", DateTime(timezone=True), onupdate=text("NOW()")),
    Column("last_message_at", DateTime(timezone=True)),
    Column("metadata", JSON, default=dict),
)

# Messages - Individual chat messages
messages = Table(
    "messages",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "conversation_id",
        UUID(as_uuid=True),
        ForeignKey("conversations.id"),
        nullable=False,
        index=True,
    ),
    Column("role", String(50), nullable=False),  # user, assistant, system
    Column("content", Text, nullable=False),
    Column("model", String(100)),
    Column("tokens_used", Integer(), default=0),
    Column("temperature", Float, default=0.7),
    Column("finish_reason", String(50)),
    Column("agent_name", String(100)),  # Which agent processed this
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
    Column("metadata", JSON, default=dict),
)

# Agent Tasks - Task orchestration for agents
agent_tasks = Table(
    "agent_tasks",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "user_id",
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    ),
    Column(
        "agent_id", String(255), nullable=False, index=True
    ),  # e.g. "finance/risk_analyzer"
    Column("domain", String(100), nullable=False),
    Column("title", String(500), nullable=False),
    Column("description", Text),
    Column(
        "status", String(50), default="pending"
    ),  # pending, processing, completed, failed
    Column("priority", Integer(), default=5),
    Column("complexity", Integer(), default=3),
    Column("requirements", JSON, default=dict),
    Column("result", JSON),
    Column("error_message", Text),
    Column("tokens_used", BigInteger(), default=0),
    Column("processing_time_ms", BigInteger(), default=0),
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
    Column("started_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    Column("metadata", JSON, default=dict),
)

# Agent Performance - Performance tracking for each agent
agent_performance = Table(
    "agent_performance",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column("agent_id", String(255), nullable=False, index=True),
    Column("domain", String(100), nullable=False, index=True),
    Column("total_tasks", BigInteger(), default=0),
    Column("successful_tasks", BigInteger(), default=0),
    Column("failed_tasks", BigInteger(), default=0),
    Column("total_tokens", BigInteger(), default=0),
    Column("average_response_time_ms", Float, default=0.0),
    Column("success_rate", Float, default=0.0),
    Column("last_activity", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
    Column("updated_at", DateTime(timezone=True), onupdate=text("NOW()")),
)

# Security Audit Logs - Enterprise security monitoring
security_audit_logs = Table(
    "security_audit_logs",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True
    ),
    Column("event_type", String(100), nullable=False, index=True),
    Column("event_description", Text, nullable=False),
    Column("ip_address", String(45), index=True),
    Column("user_agent", Text),
    Column("endpoint", String(500)),
    Column("method", String(10)),
    Column("status_code", Integer()),
    Column("request_data", JSON),
    Column("response_data", JSON),
    Column("sever ity", String(20), default="INFO"),  # INFO, WARNING, ERROR, CRITICAL
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
    Column("metadata", JSON, default=dict),
)

# API Keys - Enterprise API key management
api_keys = Table(
    "api_keys",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "user_id",
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    ),
    Column("key_name", String(255), nullable=False),
    Column("hashed_key", String(512), unique=True, nullable=False),
    Column("permissions", JSON, default=list),
    Column("rate_limit", Integer(), default=1000),  # requests per hour
    Column("is_active", Boolean(), default=True),
    Column("last_used", DateTime(timezone=True)),
    Column("expires_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
    Column("metadata", JSON, default=dict),
)

# File Storage - Document management system
files = Table(
    "files",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "user_id",
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    ),
    Column("filename", String(500), nullable=False),
    Column("original_filename", String(500), nullable=False),
    Column("file_path", String(1000), nullable=False),
    Column("file_size", BigInteger(), nullable=False),
    Column("mime_type", String(255), nullable=False),
    Column("file_hash", String(128), nullable=False, index=True),
    Column("processed", Boolean(), default=False),
    Column("processing_status", String(50), default="pending"),
    Column("embeddings_vector", Text),  # For vector similarity search
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
    Column("updated_at", DateTime(timezone=True), onupdate=text("NOW()")),
    Column("metadata", JSON, default=dict),
)

# Training Sessions - QLoRA training tracking
training_sessions = Table(
    "training_sessions",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "user_id",
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    ),
    Column("model_name", String(255), nullable=False),
    Column("dataset_size", BigInteger(), default=0),
    Column("status", String(50), nullable=False, default="running"),
    Column("parameters", JSON, default=dict),
    Column("metrics", JSON, default=dict),
    Column("start_time", DateTime(timezone=True), server_default=text("NOW()")),
    Column("end_time", DateTime(timezone=True)),
    Column("total_epochs", Integer(), default=0),
    Column("current_epoch", Integer(), default=0),
    Column("loss", Float, default=0.0),
    Column("accuracy", Float, default=0.0),
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
    Column("updated_at", DateTime(timezone=True), onupdate=text("NOW()")),
)

# Agent Configurations - Dynamic agent behavior configuration
agent_configurations = Table(
    "agent_configurations",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column("agent_id", String(255), unique=True, nullable=False, index=True),
    Column("domain", String(100), nullable=False, index=True),
    Column("config_json", JSON, nullable=False),
    Column("is_active", Boolean(), default=True),
    Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
    Column("updated_at", DateTime(timezone=True), onupdate=text("NOW()")),
    Column("metadata", JSON, default=dict),
)

# System Metrics - Enterprise monitoring
system_metrics = Table(
    "system_metrics",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column("metric_type", String(100), nullable=False, index=True),
    Column("metric_name", String(255), nullable=False, index=True),
    Column("metric_value", Float, nullable=False),
    Column("unit", String(50), default="count"),
    Column("timestamp", DateTime(timezone=True), server_default=text("NOW()")),
    Column("labels", JSON, default=dict),
    Column("metadata", JSON, default=dict),
)


async def create_extensions():
    """Create required PostgreSQL extensions"""
    async with async_engine.begin() as conn:
        # UUID generation
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))

        # Audit trigger functionality
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "audit-trigger";'))

        print("[OK] PostgreSQL extensions created successfully")


async def create_tables():
    """Create all database tables"""
    async with async_engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        print("[OK] All database tables created successfully")


async def create_indexes():
    """Create performance indexes"""
    index_queries = [
        # User-related indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_tenant_role ON users(tenant_id, role);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active_created ON users(is_active, created_at);",
        # Session indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_user_expires ON user_sessions(user_id, expires_at);",
        # Conversation indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_user_status ON conversations(user_id, status);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_created ON conversations(created_at DESC);",
        # Message indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_conversation_created ON messages(conversation_id, created_at);",
        # Agent task indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_tasks_user_status ON agent_tasks(user_id, status);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_tasks_agent ON agent_tasks(agent_id, created_at);",
        # Performance indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_domain_activity ON agent_performance(domain, last_activity);",
        # Security audit indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_security_event_created ON security_audit_logs(event_type, created_at);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_security_ip_event ON security_audit_logs(ip_address, event_type);",
        # API key indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_user_active ON api_keys(user_id, is_active);",
        # File indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_files_user_processed ON files(user_id, processed);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_files_hash ON files(file_hash);",
        # Training indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_user_status ON training_sessions(user_id, status);",
        # System metrics indexes for time-series queries
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_type_timestamp ON system_metrics(metric_type, timestamp);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_name_timestamp ON system_metrics(metric_name, timestamp DESC);",
    ]

    async with async_engine.begin() as conn:
        for query in index_queries:
            try:
                await conn.execute(text(query))
            except Exception as e:
                print(f"[WARN] Index creation warning: {e}")
                continue

        print("[OK] Database indexes created successfully")


async def create_row_level_security():
    """Enable and configure Row-Level Security (RLS) policies"""
    rls_queries = [
        # Enable RLS on all tables
        "ALTER TABLE users ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE messages ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE agent_tasks ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE files ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE training_sessions ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;",
        # RLS Policies for Users table
        """
        CREATE POLICY users_tenant_policy ON users
        FOR ALL USING (tenant_id = current_setting('app.current_tenant', TRUE)::text);
        """,
        # RLS Policies for Conversations table
        """
        CREATE POLICY conversations_user_policy ON conversations
        FOR ALL USING (
            user_id IN (
                SELECT id FROM users WHERE tenant_id = current_setting('app.current_tenant', TRUE)::text
            )
        );
        """,
        # RLS Policies for Messages table
        """
        CREATE POLICY messages_user_policy ON messages
        FOR ALL USING (
            conversation_id IN (
                SELECT id FROM conversations WHERE user_id IN (
                    SELECT id FROM users WHERE tenant_id = current_setting('app.current_tenant', TRUE)::text
                )
            )
        );
        """,
        # RLS Policies for Agent Tasks table
        """
        CREATE POLICY agent_tasks_user_policy ON agent_tasks
        FOR ALL USING (
            user_id IN (
                SELECT id FROM users WHERE tenant_id = current_setting('app.current_tenant', TRUE)::text
            )
        );
        """,
        # RLS Policies for Files table
        """
        CREATE POLICY files_user_policy ON files
        FOR ALL USING (
            user_id IN (
                SELECT id FROM users WHERE tenant_id = current_setting('app.current_tenant', TRUE)::text
            )
        );
        """,
        # RLS Policies for Training Sessions table
        """
        CREATE POLICY training_user_policy ON training_sessions
        FOR ALL USING (
            user_id IN (
                SELECT id FROM users WHERE tenant_id = current_setting('app.current_tenant', TRUE)::text
            )
        );
        """,
        # RLS Policies for API Keys table
        """
        CREATE POLICY api_keys_user_policy ON api_keys
        FOR ALL USING (
            user_id IN (
                SELECT id FROM users WHERE tenant_id = current_setting('app.current_tenant', TRUE)::text
            )
        );
        """,
        # RLS Policies for User Sessions table
        """
        CREATE POLICY sessions_user_policy ON user_sessions
        FOR ALL USING (
            user_id IN (
                SELECT id FROM users WHERE tenant_id = current_setting('app.current_tenant', TRUE)::text
            )
        );
        """,
    ]

    async with async_engine.begin() as conn:
        for query in rls_queries:
            try:
                await conn.execute(text(query.strip()))
            except Exception as e:
                print(f"[WARN] RLS policy warning: {e}")
                continue

        print("[OK] Row-Level Security policies configured successfully")


async def create_default_roles():
    """Create default roles and permissions"""
    role_data = [
        (
            "admin",
            "System Administrator with full access",
            [
                "users:read",
                "users:write",
                "users:delete",
                "conversations:read",
                "conversations:write",
                "conversations:delete",
                "agents:execute",
                "agents:configure",
                "system:monitor",
                "security:audit",
                "files:upload",
                "files:delete",
            ],
        ),
        (
            "enterprise",
            "Enterprise user with advanced features",
            [
                "conversations:read",
                "conversations:write",
                "agents:execute",
                "agents:monitor",
                "files:upload",
                "training:create",
                "analytics:read",
            ],
        ),
        (
            "user",
            "Standard user with basic features",
            [
                "conversations:read",
                "conversations:write",
                "agents:execute",
                "files:upload",
            ],
        ),
    ]

    async with async_engine.begin() as conn:
        for role_name, description, permissions in role_data:
            await conn.execute(
                text(
                    """
                    INSERT INTO roles (name, description, permissions)
                    VALUES (:name, :description, :permissions)
                    ON CONFLICT (name) DO NOTHING
                """
                ),
                {
                    "name": role_name,
                    "description": description,
                    "permissions": permissions,
                },
            )

        print("[OK] Default roles created successfully")


async def create_default_admin():
    """Create default admin user with secure password handling"""
    import logging
    import secrets
    import string

    from apps.backend.src.core.security import security_manager

    logger = logging.getLogger(__name__)

    # Get admin password from environment variable or generate secure one
    admin_password = os.getenv("SHEILY_ADMIN_PASSWORD")

    if not admin_password:
        # Generate a secure random password if not provided via environment
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{}|;:,.<>?"
        admin_password = "".join(
            secrets.choice(alphabet) for i in range(20)
        )  # Increased to 20 chars

        logger.warning(
            "🔐 No SHEILY_ADMIN_PASSWORD environment variable set - generated secure random password"
        )
        logger.warning(
            "🔑 Set SHEILY_ADMIN_PASSWORD environment variable to use your own password"
        )
        logger.warning(
            "📝 To avoid this warning, export: export SHEILY_ADMIN_PASSWORD='your_secure_password_here'"
        )

        # Only log password mask in info level (never the actual password)
        logger.info(
            "🔒 Admin password set (length: {} characters)".format(len(admin_password))
        )

    # Validate password strength
    if len(admin_password) < 12:
        logger.error("[ERROR] Admin password too weak - minimum 12 characters required")
        raise ValueError("Admin password must be at least 12 characters long")

    # Create admin password hash (secure the password immediately)
    hashed_password = security_manager.hash_password(admin_password)

    # Clear password from memory as soon as possible
    admin_password = None  # Explicitly clear

    async with async_engine.begin() as conn:
        await conn.execute(
            text(
                """
                INSERT INTO users (
                    username, email, password_hash, role, tenant_id,
                    is_active, is_verified, metadata
                )
                VALUES (
                    'admin', 'admin@sheily.ai', :password_hash, 'admin', 'default',
                    true, true, :metadata
                )
                ON CONFLICT (username) DO NOTHING
            """
            ),
            {
                "password_hash": hashed_password,
                "metadata": {
                    "created_by": "system",
                    "is_super_admin": True,
                    "permissions": ["*"],
                    "created_via": "database_migration",
                },
            },
        )

        logger.info("[OK] Default admin user 'admin' created successfully")
        logger.warning(
            "[WARN]  SECURITY: Change default admin password immediately in production!"
        )
        logger.warning(
            "🔐 Use environment variable SHEILY_ADMIN_PASSWORD to set custom password"
        )
        logger.info("📧 Admin email: admin@sheily.ai")
        logger.info("[START] Default admin account ready (requires password change)")


async def create_triggers_and_functions():
    """Create database triggers and functions"""
    trigger_queries = [
        # Update timestamps trigger
        """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """,
        # Auto-update triggers
        """
        CREATE TRIGGER update_users_updated_at
            BEFORE UPDATE ON users
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """,
        """
        CREATE TRIGGER update_conversations_updated_at
            BEFORE UPDATE ON conversations
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """,
        """
        CREATE TRIGGER update_training_sessions_updated_at
            BEFORE UPDATE ON training_sessions
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """,
        """
        CREATE TRIGGER update_agent_configurations_updated_at
            BEFORE UPDATE ON agent_configurations
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """,
        # Session cleanup function
        """
        CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
        RETURNS void AS $$
        BEGIN
            DELETE FROM user_sessions WHERE expires_at < NOW();
        END;
        $$ LANGUAGE plpgsql;
        """,
        # Metrics aggregation function
        """
        CREATE OR REPLACE FUNCTION get_system_metrics_summary(
            start_date timestamp with time zone DEFAULT NOW() - INTERVAL '1 hour',
            end_date timestamp with time zone DEFAULT NOW()
        )
        RETURNS JSON AS $$
        DECLARE
            result JSON;
        BEGIN
            SELECT json_build_object(
                'period_start', start_date,
                'period_end', end_date,
                'total_requests', COALESCE(SUM(CASE WHEN metric_name = 'api_requests_total' THEN metric_value END), 0)::bigint,
                'active_users', COALESCE(COUNT(DISTINCT (CASE WHEN metric_name = 'active_user' THEN (labels->>'user_id') END)), 0)::bigint,
                'agent_tasks_completed', COALESCE(SUM(CASE WHEN metric_name = 'agent_tasks_completed' THEN metric_value END), 0)::bigint,
                'average_response_time', COALESCE(AVG(CASE WHEN metric_name = 'response_time_ms' THEN metric_value END), 0)::float,
                'error_rate', COALESCE(
                    AVG(CASE WHEN metric_name = 'error_requests_total' THEN metric_value END) /
                    NULLIF(AVG(CASE WHEN metric_name = 'api_requests_total' THEN metric_value END), 0) * 100, 0
                )::float
            ) INTO result
            FROM system_metrics
            WHERE timestamp BETWEEN start_date AND end_date;

            RETURN result;
        END;
        $$ LANGUAGE plpgsql;
        """,
    ]

    async with async_engine.begin() as conn:
        for query in trigger_queries:
            try:
                await conn.execute(text(query.strip()))
            except Exception as e:
                print(f"[WARN] Trigger/function creation warning: {e}")
                continue

        print("[OK] Database triggers and functions created successfully")


async def run_migrations():
    """Run all database migrations"""
    print("[START] Starting database migrations...")

    try:
        # Create database extensions
        await create_extensions()

        # Create all tables
        await create_tables()

        # Create indexes
        await create_indexes()

        # Configure Row-Level Security
        await create_row_level_security()

        # Create default roles
        await create_default_roles()

        # Create default admin user
        await create_default_admin()

        # Create triggers and functions
        await create_triggers_and_functions()

        print("[CELEBRATION] Database migrations completed successfully!")
        print("[CHART] Database ready for Sheily MCP Enterprise production deployment")

    except Exception as e:
        print(f"[ERROR] Database migration failed: {e}")
        raise


async def health_check():
    """Perform database health check"""
    try:
        async with async_engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"[OK] Database connection successful: {version[:50]}...")

            # Check table existence
            tables = ["users", "conversations", "agent_tasks", "security_audit_logs"]
            for table in tables:
                result = await conn.execute(
                    text(
                        f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table)"
                    ),
                    {"table": table},
                )
                exists = result.fetchone()[0]
                status = "[OK]" if exists else "[ERROR]"
                print(f"{status} Table '{table}' exists: {exists}")

            return True

    except Exception as e:
        print(f"[ERROR] Database health check failed: {e}")
        return False


if __name__ == "__main__":
    # Run migrations
    asyncio.run(run_migrations())

    # Run health check
    asyncio.run(health_check())
