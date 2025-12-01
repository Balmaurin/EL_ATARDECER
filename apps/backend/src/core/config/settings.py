"""
Settings configuration for EL-AMANECER Backend
Centralized configuration management
"""

import os
from pathlib import Path

# Version information
version = "1.0.0"

# Cache settings
cache_enabled = True
cache_ttl = 3600  # 1 hour

# Database settings
database_url = "sqlite:///data/sheily_dashboard.db"

# Security settings
secret_key = "sheily-secret-key-change-in-production"
algorithm = "HS256"
access_token_expire_minutes = 30
refresh_token_expire_days = 7

# AI/API settings
gemini_api_key = None  # Set via environment variable

# File paths
base_dir = Path(__file__).parent.parent.parent.parent.parent  # apps/backend/src/core/config/ -> EL-AMANECER root
data_dir = base_dir / "data"
models_dir = base_dir / "models"
logs_dir = base_dir / "logs"

# Ensure directories exist
data_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)

# Training settings
max_training_jobs = 5
default_epochs = 3
default_batch_size = 4

# LLM Service settings
llm_service_url = "http://localhost:8003/v1/completions"
llm_model_id = "gemma-2b"

# Temporal settings
temporal_address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
temporal_task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "hack-memori-training")

# Server settings
server_host = os.getenv("SERVER_HOST", "0.0.0.0")
server_port = int(os.getenv("SERVER_PORT", "8000"))
api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
websocket_url = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/ws")

# External service URLs
training_service_url = os.getenv("TRAINING_SERVICE_URL", "http://localhost:9001/train")
rag_service_url = os.getenv("RAG_SERVICE_URL", "http://localhost:9100/retrieve")
memory_service_url = os.getenv("MEMORY_SERVICE_URL", "http://localhost:9200/memory")


class LLMSettings:
    """LLM configuration settings"""
    
    def __init__(self):
        # Path to the local GGUF model
        self.model_path = str(base_dir / "modelsLLM" / "mental_health_counseling_gemma_7b_merged.Q4_K_M.gguf")
        # Context window size
        self.n_ctx = 4096
        # Number of threads for inference
        self.n_threads = 4
        # Chat format (llama-2, chatml, etc.)
        self.chat_format = "chatml"  # Gemma 2 uses ChatML format
        # Verbose logging
        self.verbose = False


class Settings:
    """Settings class for dependency injection"""

    def __init__(self):
        self.version = version
        self.cache_enabled = cache_enabled
        self.database_url = database_url
        self.secret_key = secret_key
        self.jwt_secret_key = secret_key  # Alias for compatibility
        self.algorithm = algorithm
        self.jwt_algorithm = algorithm  # Alias for compatibility
        self.access_token_expire_minutes = access_token_expire_minutes
        self.jwt_expiration_hours = access_token_expire_minutes / 60  # Alias for compatibility
        self.refresh_token_expire_days = refresh_token_expire_days
        # Add LLM settings
        self.llm = LLMSettings()
        # Add service URLs
        self.llm_service_url = llm_service_url
        self.llm_model_id = llm_model_id
        self.temporal_address = temporal_address
        self.temporal_task_queue = temporal_task_queue
        # Server settings
        self.server_host = server_host
        self.server_port = server_port
        self.api_base_url = api_base_url
        self.websocket_url = websocket_url
        # External service URLs
        self.training_service_url = training_service_url
        self.rag_service_url = rag_service_url
        self.memory_service_url = memory_service_url


# Create a settings instance for import compatibility
settings = Settings()
