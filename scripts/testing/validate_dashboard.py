"""
EL-AMANECER V4 - Dashboard Validation Script
============================================

Validates the Consciousness Dashboard functionality by checking:
1. Backend API health
2. Consciousness metrics endpoint
3. Real-time data format
4. System status reporting
"""

import asyncio
import aiohttp
import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DashboardValidator")

BASE_URL = "http://localhost:8000/api"

class DashboardValidator:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()

    async def check_health(self, session: aiohttp.ClientSession) -> bool:
        """Check API health endpoint"""
        try:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Health Check: PASS ({data.get('status')})")
                    return True
                else:
                    logger.error(f"❌ Health Check: FAIL (Status {response.status})")
                    return False
        except Exception as e:
            logger.error(f"❌ Health Check: FAIL (Connection error: {e})")
            return False

    async def check_consciousness_metrics(self, session: aiohttp.ClientSession) -> bool:
        """Check consciousness metrics endpoint"""
        try:
            # Correct endpoint: /api/dashboard/consciousness
            async with session.get(f"{BASE_URL}/dashboard/consciousness") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate required fields for dashboard
                    required_fields = ["consciousness", "analytics", "status"]
                    missing = [f for f in required_fields if f not in data]
                    
                    if not missing:
                        cons_data = data.get("consciousness", {})
                        logger.info(f"✅ Consciousness Metrics: PASS (Phi: {cons_data.get('phi_value', 0):.2f})")
                        return True
                    else:
                        logger.warning(f"⚠️ Consciousness Metrics: PARTIAL (Missing: {missing})")
                        return False 
                else:
                    logger.error(f"❌ Consciousness Metrics: FAIL (Status {response.status})")
                    return False
        except Exception as e:
            logger.error(f"❌ Consciousness Metrics: FAIL ({e})")
            return False

    async def check_system_status(self, session: aiohttp.ClientSession) -> bool:
        """Check overall system status for dashboard"""
        try:
            # Correct endpoint: /api/dashboard/status
            async with session.get(f"{BASE_URL}/dashboard/status") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ System Status: PASS")
                    return True
                else:
                    logger.error(f"❌ System Status: FAIL (Status {response.status})")
                    return False
        except Exception as e:
            logger.error(f"❌ System Status: FAIL ({e})")
            return False

    async def run_validation(self):
        """Run full validation suite"""
        logger.info("🚀 Starting Dashboard Validation...")
        
        async with aiohttp.ClientSession() as session:
            # 1. Health Check
            health_ok = await self.check_health(session)
            self.results["health"] = health_ok
            
            if not health_ok:
                logger.critical("⛔ Backend appears down. Cannot proceed with dashboard validation.")
                return False

            # 2. Consciousness Metrics (Core Dashboard Data)
            metrics_ok = await self.check_consciousness_metrics(session)
            self.results["metrics"] = metrics_ok
            
            # 3. System Status
            status_ok = await self.check_system_status(session)
            self.results["status"] = status_ok
            
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("="*50)
        
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        logger.info(f"Total Checks: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        
        if passed == total:
            logger.info("✅ DASHBOARD BACKEND READY")
            return True
        else:
            logger.warning("⚠️ DASHBOARD BACKEND ISSUES DETECTED")
            return False

if __name__ == "__main__":
    validator = DashboardValidator()
    try:
        # Check if backend is likely running (simple socket check could be added here)
        # For now, just run the async loop
        success = asyncio.run(validator.run_validation())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Validation stopped by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Validation failed with error: {e}")
        sys.exit(1)
