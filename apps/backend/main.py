#!/usr/bin/env python3
"""
EL-AMANECER Backend - Main Application Entry Point
==================================================

Consolidated GraphQL Federation Gateway for all backend services:
- GraphQL API (primary interface)
- Consciousness Neural System
- TODO Management System
- HACK-MEMORI Training System
- Authentication & User Management
"""

import sys
import logging
from pathlib import Path

# Setup Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

from federation_server import FederationGateway

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point"""
    import uvicorn

    logger.info("=" * 80)
    logger.info("EL-AMANECER Backend - Starting Federation Gateway")
    logger.info("=" * 80)

    # Initialize gateway and create app
    gateway = FederationGateway()
    app = gateway.create_app()

    # Run with uvicorn
    host = "0.0.0.0"
    port = 8000

    logger.info(f"Starting server at http://{host}:{port}")
    logger.info(f"GraphQL endpoint: http://{host}:{port}/graphql")
    logger.info(f"GraphQL Playground: http://{host}:{port}/graphql")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
