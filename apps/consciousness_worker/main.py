#!/usr/bin/env python3
"""
🧠 Consciousness Worker Service
Procesa eventos de consciencia de forma asíncrona usando Dapr Pub/Sub
Desacoplado del backend principal para garantizar que la API nunca se caiga
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI
from dapr.clients import DaprClient
from dapr.ext.fastapi import DaprApp
import redis
import uvicorn

# Setup paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
consciousness_path = project_root / "packages" / "consciousness" / "src"
if str(consciousness_path) not in sys.path:
    sys.path.append(str(consciousness_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Consciousness Worker Service")
dapr_app = DaprApp(app)

# Redis client for storing results
redis_client = None
consciousness_system = None


def init_redis():
    """Initialize Redis connection"""
    global redis_client
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        redis_client = None


def init_consciousness_system():
    """Initialize consciousness system"""
    global consciousness_system
    try:
        from conciencia.meta_cognition_system import MetaCognitionSystem
        
        consciousness_dir = project_root / "data" / "consciousness"
        consciousness_dir.mkdir(parents=True, exist_ok=True)
        
        consciousness_system = MetaCognitionSystem(
            consciousness_dir=str(consciousness_dir),
            emergence_threshold=0.85
        )
        logger.info("✅ Consciousness system initialized")
    except ImportError as e:
        logger.warning(f"⚠️ MetaCognitionSystem not available: {e}")
        try:
            from conciencia.modulos import IITEngine
            consciousness_system = IITEngine()
            logger.info("✅ Using IIT Engine as fallback")
        except ImportError:
            logger.error("❌ No consciousness system available")
            consciousness_system = None
    except Exception as e:
        logger.error(f"❌ Error initializing consciousness: {e}")
        consciousness_system = None


@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    logger.info("🧠 Starting Consciousness Worker Service...")
    init_redis()
    init_consciousness_system()
    logger.info("✅ Consciousness Worker ready")


@dapr_app.subscribe(pubsub="consciousness-pubsub", topic="consciousness.stimulus")
async def process_consciousness_stimulus(event_data: Dict[str, Any]):
    """
    Procesa un estímulo de consciencia recibido vía Dapr Pub/Sub
    
    Event payload:
    {
        "input": "texto del estímulo",
        "user_id": "user-123",
        "context_type": "chat" (opcional)
    }
    """
    try:
        # Parse event data
        if isinstance(event_data, dict) and "data" in event_data:
            data = json.loads(event_data["data"]) if isinstance(event_data["data"], str) else event_data["data"]
        else:
            data = event_data
        
        stimulus = data.get("input", "")
        user_id = data.get("user_id", "anonymous")
        context = data.get("context", {})
        context_type = data.get("context_type", "general")
        
        logger.info(f"📥 Processing consciousness stimulus for user {user_id}: {stimulus[:50]}...")
        
        if not consciousness_system:
            logger.error("❌ Consciousness system not available")
            return {"status": "error", "message": "Consciousness system not initialized"}
        
        # Process with consciousness system
        result = await asyncio.to_thread(
            consciousness_system.process_meta_cognitive_loop,
            stimulus,
            {**context, "context_type": context_type},
            max_recursion_depth=2
        )
        
        # Get metrics
        metrics = consciousness_system.get_consciousness_metrics()
        
        # Prepare result
        processed_result = {
            "user_id": user_id,
            "stimulus": stimulus,
            "result": result,
            "metrics": metrics,
            "phi_value": metrics.get('current_meta_awareness', 0.67),
            "emotional_depth": result.get('meta_awareness_updated', 0.7),
            "mindfulness_level": metrics.get('consciousness_stability', 0.8),
            "processed_at": asyncio.get_event_loop().time()
        }
        
        # Store result in Redis for API to read
        if redis_client:
            redis_key = f"consciousness:result:{user_id}"
            redis_client.setex(
                redis_key,
                3600,  # 1 hour TTL
                json.dumps(processed_result)
            )
            logger.info(f"✅ Result stored in Redis: {redis_key}")
        
        # Also store in ChromaDB if available
        try:
            # Store in consciousness data directory
            result_file = project_root / "data" / "consciousness" / f"{user_id}_latest.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(processed_result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save to file: {e}")
        
        logger.info(f"✅ Consciousness processing complete for {user_id}")
        return {"status": "success", "user_id": user_id}
        
    except Exception as e:
        logger.error(f"❌ Error processing consciousness stimulus: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "consciousness-worker",
        "consciousness_system": consciousness_system is not None,
        "redis": redis_client is not None
    }


@app.get("/metrics")
async def metrics():
    """Get consciousness system metrics"""
    if not consciousness_system:
        return {"error": "Consciousness system not available"}
    
    try:
        metrics_data = consciousness_system.get_consciousness_metrics()
        return {
            "status": "success",
            "metrics": metrics_data
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    logger.info(f"🧠 Starting Consciousness Worker on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

