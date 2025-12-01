"""
Database models and connection for Sheily AI Backend
Real database implementation with SQLAlchemy
"""

import os
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

# Database URL - use SQLite for development, can be changed to PostgreSQL/MySQL for production
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sheily_ai.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args=(
        {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    ),
    echo=False,  # Set to True for SQL query logging
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

get_db_session = get_db


# Base class for all models
class Base(DeclarativeBase):
    pass


class User(Base):
    """Real user model with all necessary fields"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)  # For future auth
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    role = Column(String(50), default="user")  # user, admin, moderator

    # Token system
    sheily_tokens = Column(Float, default=100.0)
    level = Column(Integer, default=1)
    experience_points = Column(Integer, default=0)

    # Profile
    avatar_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)

    # Blockchain
    wallet_address = Column(String(255), unique=True, nullable=True)
    blockchain_balance = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    exercises = relationship("Exercise", back_populates="user")
    datasets = relationship("Dataset", back_populates="user")
    transactions = relationship("Transaction", back_populates="user")


class Conversation(Base):
    """Real conversation model"""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(500), nullable=True)
    model_used = Column(String(100), default="default-model")
    total_messages = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)

    # Status
    is_active = Column(Boolean, default=True)
    status = Column(String(50), default="active")  # active, archived, deleted

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_message_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    """Real message model"""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    content = Column(Text, nullable=False)
    role = Column(String(50), nullable=False)  # user, assistant, system
    model = Column(String(100), nullable=True)
    tokens_used = Column(Integer, default=0)

    # Metadata
    message_metadata = Column(
        JSON, nullable=True
    )  # Store additional data like embeddings, etc.

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class Exercise(Base):
    """Real exercise model"""

    __tablename__ = "exercises"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exercise_type = Column(String(50), nullable=False)  # yesno, truefalse, multiple
    question_count = Column(Integer, default=20)
    correct_answers = Column(Integer, default=0)
    incorrect_answers = Column(Integer, default=0)
    accuracy_percentage = Column(Float, default=0.0)
    tokens_earned = Column(Float, default=0.0)
    time_taken_seconds = Column(Integer, default=0)

    # Answers data
    answers_data = Column(JSON, nullable=True)  # Store detailed answers

    # Status
    completed = Column(Boolean, default=False)
    status = Column(
        String(50), default="in_progress"
    )  # in_progress, completed, abandoned

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="exercises")


class Dataset(Base):
    """Real dataset model"""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    exercise_type = Column(String(50), nullable=False)
    total_questions = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    tokens_earned = Column(Float, default=0.0)

    # Dataset metadata
    dataset_metadata = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)  # List of tags

    # Status
    is_public = Column(Boolean, default=False)
    status = Column(String(50), default="active")  # active, archived, deleted

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="datasets")


class Document(Base):
    """Real document model for RAG"""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, default=0)
    mime_type = Column(String(100), nullable=False)
    content_hash = Column(String(128), unique=True, nullable=False)

    # Processing status
    processing_status = Column(
        String(50), default="pending"
    )  # pending, processing, completed, failed
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)

    # Content
    content_text = Column(Text, nullable=True)
    content_length = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)

    # Metadata
    document_metadata = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)

    # Status
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])


class Embedding(Base):
    """Real embedding model for vector search"""

    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, default=0)
    content_chunk = Column(Text, nullable=False)

    # Vector data (stored as JSON for simplicity, can be optimized later)
    embedding_vector = Column(JSON, nullable=False)
    vector_dimension = Column(Integer, default=384)  # Default dimension

    # Metadata
    embedding_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


class Tenant(Base):
    """Multi-tenant model for enterprise features"""

    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    tenant_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    domain = Column(String(255), unique=True, index=True, nullable=False)
    status = Column(String(50), default="active")  # active, suspended, inactive, pending
    admin_email = Column(String(255), nullable=False)
    contact_email = Column(String(255), nullable=True)

    # Resource usage
    current_users = Column(Integer, default=0)
    current_storage_gb = Column(Float, default=0.0)
    api_calls_today = Column(Integer, default=0)

    # Limits
    max_users = Column(Integer, default=100)
    max_agents = Column(Integer, default=10)
    max_api_calls_per_hour = Column(Integer, default=1000)
    max_storage_gb = Column(Float, default=10.0)

    # Features JSON
    enabled_features = Column(JSON, default=lambda: ["basic"])

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Transaction(Base):
    """Real blockchain transaction model"""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    transaction_type = Column(
        String(50), nullable=False
    )  # send, receive, stake, reward, exercise
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="SHEILY")

    # Transaction details
    transaction_hash = Column(String(255), unique=True, nullable=True)
    from_address = Column(String(255), nullable=True)
    to_address = Column(String(255), nullable=True)
    blockchain_tx_id = Column(String(255), nullable=True)

    # Status
    status = Column(String(50), default="pending")  # pending, confirmed, failed
    confirmations = Column(Integer, default=0)

    # Metadata
    system_metadata = Column(JSON, nullable=True)
    description = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    confirmed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="transactions")


class SystemMetric(Base):
    """Real system metrics model"""

    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)

    # Context
    component = Column(String(100), nullable=True)  # api, database, ai_model, etc.
    server_id = Column(String(100), nullable=True)

    # Metadata
    metric_metadata = Column(JSON, nullable=True)

    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow)


class CacheEntry(Base):
    """Real cache entry model"""

    __tablename__ = "cache_entries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cache_key = Column(String(500), nullable=False, unique=True)
    cache_value = Column(JSON, nullable=False)
    cache_type = Column(String(50), default="response")  # response, embedding, etc.

    # Expiration
    expires_at = Column(DateTime, nullable=True)
    ttl_seconds = Column(Integer, nullable=True)

    # Metadata
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


# Create all tables
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


# Drop all tables (for testing/reset)
def drop_tables():
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)


# Get database session
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize database with sample data
def init_sample_data():
    """Initialize database with sample data"""
    db = SessionLocal()

    try:
        # Check if we already have data
        if db.query(User).first():
            print("[OK] Database already initialized")
            return

        # Create sample user
        sample_user = User(
            email="demo@sheily.ai",
            username="sheily_demo",
            full_name="Usuario Demo Sheily",
            sheily_tokens=1000.0,
            level=5,
            experience_points=2500,
            bio="Usuario de demostración del sistema Sheily AI",
            is_active=True,
            is_verified=True,
        )

        db.add(sample_user)
        db.commit()
        db.refresh(sample_user)

        # Create sample conversation
        sample_conversation = Conversation(
            user_id=sample_user.id,
            title="Conversación de ejemplo",
            model_used="gemma-2-9b",
            total_messages=4,
            total_tokens_used=150,
        )

        db.add(sample_conversation)
        db.commit()
        db.refresh(sample_conversation)

        # Create sample messages
        messages_data = [
            {
                "content": "¡Hola! Soy Sheily AI. ¿En qué puedo ayudarte?",
                "role": "assistant",
            },
            {"content": "¿Qué es el aprendizaje automático?", "role": "user"},
            {
                "content": "El aprendizaje automático es una rama de la inteligencia artificial que permite a los sistemas aprender y mejorar automáticamente a partir de la experiencia, sin ser programados explícitamente para cada tarea específica.",
                "role": "assistant",
            },
            {"content": "¡Gracias! Eso fue muy útil.", "role": "user"},
        ]

        for i, msg_data in enumerate(messages_data):
            message = Message(
                conversation_id=sample_conversation.id,
                content=msg_data["content"],
                role=msg_data["role"],
                tokens_used=len(msg_data["content"].split()) * 2,  # Rough estimate
            )
            db.add(message)

        # Create sample exercise
        sample_exercise = Exercise(
            user_id=sample_user.id,
            exercise_type="yesno",
            question_count=20,
            correct_answers=15,
            incorrect_answers=5,
            accuracy_percentage=75.0,
            tokens_earned=25.0,
            completed=True,
            status="completed",
            completed_at=datetime.utcnow(),
        )

        db.add(sample_exercise)
        db.commit()

        # Create sample dataset
        sample_dataset = Dataset(
            user_id=sample_user.id,
            name="Dataset de ejemplo - Preguntas Sí/No",
            description="Dataset generado automáticamente desde ejercicios de usuario",
            exercise_type="yesno",
            total_questions=20,
            accuracy=75.0,
            tokens_earned=25.0,
            is_public=False,
        )

        db.add(sample_dataset)
        db.commit()

        # Create sample transactions
        transactions_data = [
            {"type": "reward", "amount": 100.0, "description": "Bono de bienvenida"},
            {
                "type": "exercise",
                "amount": 25.0,
                "description": "Recompensa por ejercicio completado",
            },
            {
                "type": "send",
                "amount": -10.0,
                "description": "Envío de tokens a otro usuario",
            },
        ]

        for tx_data in transactions_data:
            transaction = Transaction(
                user_id=sample_user.id,
                transaction_type=tx_data["type"],
                amount=tx_data["amount"],
                description=tx_data["description"],
                status="confirmed",
                confirmed_at=datetime.utcnow(),
            )
            db.add(transaction)

        db.commit()

        print("[OK] Database initialized with sample data")
        print(f"   👤 Created user: {sample_user.username}")
        print(
            f"   💬 Created conversation with {sample_conversation.total_messages} messages"
        )
        print(
            f"   [TARGET] Created exercise: {sample_exercise.accuracy_percentage}% accuracy"
        )
        print(f"   [CHART] Created dataset: {sample_dataset.name}")
        print(f"   ⛓️ Created {len(transactions_data)} transactions")

    except Exception as e:
        print(f"[ERROR] Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("[REFRESH] Creating database tables...")
    create_tables()
    print("[OK] Database tables created")

    print("[REFRESH] Initializing sample data...")
    init_sample_data()
    print("[OK] Database initialization complete")
