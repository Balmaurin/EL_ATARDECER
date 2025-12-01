"""
Setup configuration for sheily_core
REAL IMPLEMENTATION - No duplicates, properly organized
"""
from setuptools import find_packages, setup

# Production dependencies - Core functionality
PRODUCTION_DEPS = [
    # Web framework
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "starlette>=0.27.0",
    "websockets>=12.0",
    
    # Data validation and settings
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    
    # LLM and ML
    "llama-cpp-python>=0.2.0",
    "transformers>=4.35.0",
    "torch>=2.1.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "sentence-transformers>=2.2.0",
    
    # Vector databases and search
    "faiss-cpu>=1.7.4",
    "chromadb>=0.4.0",
    
    # Database
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "asyncpg>=0.29.0",
    "alembic>=1.12.0",
    
    # Caching and messaging
    "redis>=5.0.0",
    
    # Security
    "bcrypt>=4.0.0",
    "cryptography>=41.0.0",
    "pyjwt[crypto]>=2.8.0",
    
    # HTTP clients
    "aiohttp>=3.9.0",
    "httpx>=0.25.0",
    
    # Utilities
    "python-dotenv>=1.0.0",
    "rich>=13.6.0",
    "click>=8.1.0",
    "tqdm>=4.66.0",
    "psutil>=5.9.0",
    
    # Data processing
    "pandas>=2.1.0",
    "scikit-learn>=1.3.0",
    
    # Logging
    "python-json-logger>=2.0.0",
    "structlog>=23.2.0",
    
    # Monitoring
    "prometheus-client>=0.19.0",
]

# Development dependencies - Testing and code quality
DEV_DEPS = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.11.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "mypy>=1.7.0",
    "bandit>=1.7.0",
]

# Documentation dependencies
DOCS_DEPS = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.4.0",
]

# Optional infrastructure dependencies (commented out - install separately if needed)
# INFRA_DEPS = [
#     "docker>=6.1.0",
#     "kubernetes>=28.1.0",
# ]

setup(
    name="sheily_core",
    version="1.0.0",
    packages=find_packages(),
    install_requires=PRODUCTION_DEPS,
    extras_require={
        "dev": DEV_DEPS,
        "docs": DOCS_DEPS,
        "all": DEV_DEPS + DOCS_DEPS,
    },
    python_requires=">=3.10",
)
