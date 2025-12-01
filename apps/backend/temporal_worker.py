"""
Temporal Worker para ejecutar workflows de Hack-Memori
"""

import asyncio
import logging
from temporalio.client import Client
from temporalio.worker import Worker
from apps.backend.temporal_workflows import TrainingSessionWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Inicia el worker de Temporal"""
    # Conectar a Temporal usando settings
    from apps.backend.src.core.config.settings import settings
    client = await Client.connect(settings.temporal_address)
    
    logger.info("✅ Connected to Temporal")
    
    # Crear worker usando settings (ya importado arriba)
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[TrainingSessionWorkflow],
        activities=[
            # Las actividades se importan desde temporal_workflows
        ],
    )
    
    logger.info("🚀 Starting Temporal worker for Hack-Memori...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

