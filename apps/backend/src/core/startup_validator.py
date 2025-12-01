"""
Startup Validator - Fail-Fast System Validation
NO FALLBACKS - System must fail immediately if critical dependencies are missing
"""

import sys
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class StartupValidationError(Exception):
    """Critical startup validation failure - system cannot start"""
    pass


class StartupValidator:
    """
    Validates all critical system dependencies on startup.
    NO FALLBACKS - If validation fails, system MUST NOT start.
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate all critical systems.
        Returns: (success, errors, warnings)
        """
        logger.info("🔍 Starting system validation...")

        # Critical validations - MUST pass
        self._validate_database()
        self._validate_redis()
        self._validate_jwt_config()
        self._validate_embedding_service()
        self._validate_llm_config()

        # Report results
        if self.errors:
            logger.error(f"❌ CRITICAL ERRORS FOUND ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"   • {error}")
            return False, self.errors, self.warnings

        if self.warnings:
            logger.warning(f"⚠️  WARNINGS FOUND ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"   • {warning}")

        logger.info("✅ All critical validations passed")
        return True, [], self.warnings

    def _validate_database(self):
        """Validate database connection and models"""
        try:
            from src.models.database import engine, Base
            from sqlalchemy import inspect, text

            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            # Verify all tables exist
            inspector = inspect(engine)
            existing_tables = inspector.get_table_names()

            required_tables = [
                'users', 'conversations', 'messages', 'exercises', 'datasets',
                'documents', 'embeddings', 'tenants', 'transactions',
                'system_metrics', 'cache_entries'
            ]

            missing_tables = [t for t in required_tables if t not in existing_tables]

            if missing_tables:
                self.errors.append(
                    f"Database missing required tables: {', '.join(missing_tables)}. "
                    "Run: python -m src.models.database to initialize."
                )
            else:
                logger.info("✓ Database connection and schema validated")

        except Exception as e:
            self.errors.append(f"Database validation failed: {str(e)}")

    def _validate_redis(self):
        """Validate Redis connection - NO FALLBACK"""
        try:
            import redis
            from src.core.config.settings import settings

            # Get Redis URL from settings
            redis_url = getattr(settings, 'redis_url', None)

            if not redis_url:
                self.errors.append(
                    "Redis URL not configured. Set REDIS_URL environment variable. "
                    "NO FALLBACK - Redis is required for production."
                )
                return

            # Test connection
            client = redis.from_url(redis_url)
            client.ping()
            logger.info("✓ Redis connection validated")

        except ImportError:
            self.errors.append(
                "Redis library not installed. Run: pip install redis. "
                "NO FALLBACK - Redis is required."
            )
        except Exception as e:
            self.errors.append(
                f"Redis connection failed: {str(e)}. "
                "NO FALLBACK - Redis is required for rate limiting and caching."
            )

    def _validate_jwt_config(self):
        """Validate JWT configuration"""
        try:
            from src.core.config.settings import settings

            # Check secret key
            secret_key = getattr(settings, 'secret_key', None)
            if not secret_key or secret_key == 'your-secret-key-here':
                self.errors.append(
                    "JWT secret_key not configured or using default value. "
                    "Set SECRET_KEY environment variable with a strong random key."
                )
                return

            if len(secret_key) < 32:
                self.warnings.append(
                    "JWT secret_key is too short. Use at least 32 characters."
                )

            # Check algorithm
            algorithm = getattr(settings, 'algorithm', None)
            if algorithm not in ['HS256', 'HS384', 'HS512']:
                self.warnings.append(
                    f"JWT algorithm '{algorithm}' may not be secure. Use HS256, HS384, or HS512."
                )

            logger.info("✓ JWT configuration validated")

        except Exception as e:
            self.errors.append(f"JWT config validation failed: {str(e)}")

    def _validate_embedding_service(self):
        """Validate embedding service - NO FALLBACK"""
        try:
            from src.core.embeddings.embedding_service import EmbeddingService

            service = EmbeddingService()

            # Check if model is loaded
            if not hasattr(service, 'model') or service.model is None:
                self.errors.append(
                    "Embedding model not loaded. Check model installation. "
                    "NO FALLBACK - Embeddings are required for RAG."
                )
                return

            # Test embedding generation
            test_text = "System validation test"
            embedding = service.generate_embedding(test_text)

            if embedding is None or len(embedding) == 0:
                self.errors.append(
                    "Embedding generation failed. Model not working correctly."
                )
                return

            logger.info(f"✓ Embedding service validated (dimension: {len(embedding)})")

        except ImportError as e:
            self.errors.append(
                f"Embedding service import failed: {str(e)}. "
                "Check sentence-transformers installation."
            )
        except Exception as e:
            self.errors.append(f"Embedding service validation failed: {str(e)}")

    def _validate_llm_config(self):
        """Validate LLM configuration"""
        try:
            from src.core.llm.llm_factory import LLMFactory

            factory = LLMFactory()

            # Check if any LLM providers are configured
            has_openai = self._check_openai_config()
            has_local = self._check_local_llm_config()

            if not has_openai and not has_local:
                self.errors.append(
                    "No LLM provider configured. Configure either OpenAI API key "
                    "or local LLM model path."
                )
                return

            logger.info("✓ LLM configuration validated")

        except Exception as e:
            self.errors.append(f"LLM config validation failed: {str(e)}")

    def _check_openai_config(self) -> bool:
        """Check if OpenAI is configured"""
        try:
            from src.core.config.settings import settings
            import os

            openai_key = os.getenv('OPENAI_API_KEY') or getattr(settings, 'openai_api_key', None)
            return openai_key is not None and openai_key != ''
        except:
            return False

    def _check_local_llm_config(self) -> bool:
        """Check if local LLM is configured"""
        try:
            from src.core.config.settings import settings
            import os

            # Check for local model path
            model_path = getattr(settings, 'local_llm_model_path', None)
            if model_path and os.path.exists(model_path):
                return True

            # Check config files
            config_paths = [
                'config/ai/llm/llm_config.json',
                'config/ai/llm/llama_config.json'
            ]

            for path in config_paths:
                if os.path.exists(path):
                    return True

            return False
        except:
            return False


def validate_startup() -> None:
    """
    Main startup validation function.
    Call this at application startup - will raise exception if validation fails.
    """
    validator = StartupValidator()
    success, errors, warnings = validator.validate_all()

    if not success:
        error_msg = "\n".join([
            "=" * 80,
            "❌ CRITICAL STARTUP VALIDATION FAILED",
            "=" * 80,
            "",
            "The following critical errors prevent system startup:",
            "",
            *[f"  • {error}" for error in errors],
            "",
            "System cannot start with these errors. Fix them and restart.",
            "=" * 80
        ])

        raise StartupValidationError(error_msg)

    if warnings:
        logger.warning("\n" + "\n".join([
            "=" * 80,
            "⚠️  STARTUP WARNINGS",
            "=" * 80,
            "",
            *[f"  • {warning}" for warning in warnings],
            "",
            "=" * 80
        ]))


if __name__ == "__main__":
    # Allow running as standalone validation script
    logging.basicConfig(level=logging.INFO)
    try:
        validate_startup()
        print("\n✅ All validations passed - system ready to start")
        sys.exit(0)
    except StartupValidationError as e:
        print(f"\n{e}")
        sys.exit(1)
