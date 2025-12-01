"""
Real System Verification Script
Tests all real implementations to ensure they work correctly
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def verify_nft_system():
    """Verify NFT credentials system"""
    try:
        logger.info("🔍 Verifying NFT System...")
        from packages.sheily_core.src.sheily_core.education.real_nft_credentials import (
            get_real_nft_credentials,
            BlockchainConfig
        )
        
        nft_system = await get_real_nft_credentials(BlockchainConfig(network="devnet"))
        balance = await nft_system.get_balance()
        
        logger.info(f"✅ NFT System OK (Balance: {balance} SOL)")
        return True
    except Exception as e:
        logger.error(f"❌ NFT System FAILED: {e}")
        return False


def verify_multimodal():
    """Verify multimodal processor"""
    try:
        logger.info("🔍 Verifying Multimodal Processor...")
        from packages.sheily_core.src.sheily_core.utils.real_multimodal_processor import (
            get_real_multimodal_processor
        )
        
        processor = get_real_multimodal_processor()
        
        logger.info(f"✅ Multimodal Processor OK (Device: {processor.device})")
        return True
    except Exception as e:
        logger.error(f"❌ Multimodal Processor FAILED: {e}")
        return False


def verify_training_system():
    """Verify training system"""
    try:
        logger.info("🔍 Verifying Training System...")
        from packages.sheily_core.src.sheily_core.training.real_training_system import (
            get_real_training_system
        )
        
        trainer = get_real_training_system()
        
        logger.info(f"✅ Training System OK (Device: {trainer.device})")
        return True
    except Exception as e:
        logger.error(f"❌ Training System FAILED: {e}")
        return False


def verify_semantic_search():
    """Verify semantic search"""
    try:
        logger.info("🔍 Verifying Semantic Search...")
        from packages.sheily_core.src.sheily_core.search.real_semantic_search import (
            get_real_semantic_search
        )
        
        search = get_real_semantic_search()
        
        # Test with sample documents
        docs = ["Machine learning is AI", "Python is a programming language"]
        search.add_documents(docs)
        
        results = search.search("What is AI?", k=1)
        
        logger.info(f"✅ Semantic Search OK (Found {len(results)} results)")
        return True
    except Exception as e:
        logger.error(f"❌ Semantic Search FAILED: {e}")
        return False


def verify_llm_inference():
    """Verify LLM inference"""
    try:
        logger.info("🔍 Verifying LLM Inference...")
        from packages.sheily_core.src.sheily_core.inference.real_llm_inference import (
            get_real_llm_inference
        )
        
        llm = get_real_llm_inference()
        
        logger.info(f"✅ LLM Inference OK (Device: {llm.device})")
        return True
    except Exception as e:
        logger.error(f"❌ LLM Inference FAILED: {e}")
        return False


def verify_enterprise_monitor():
    """Verify enterprise monitor"""
    try:
        logger.info("🔍 Verifying Enterprise Monitor...")
        from packages.sheily_core.src.sheily_core.monitoring.real_enterprise_monitor import (
            get_real_enterprise_monitor
        )
        
        monitor = get_real_enterprise_monitor()
        metrics = monitor.get_system_metrics()
        
        logger.info(f"✅ Enterprise Monitor OK (CPU: {metrics['cpu']['percent']:.1f}%)")
        return True
    except Exception as e:
        logger.error(f"❌ Enterprise Monitor FAILED: {e}")
        return False


async def main():
    """Run all verifications"""
    print("=" * 60)
    print("🚀 REAL SYSTEM VERIFICATION")
    print("=" * 60)
    print()
    
    results = {}
    
    # Run verifications
    results['NFT System'] = await verify_nft_system()
    results['Multimodal Processor'] = verify_multimodal()
    results['Training System'] = verify_training_system()
    results['Semantic Search'] = verify_semantic_search()
    results['LLM Inference'] = verify_llm_inference()
    results['Enterprise Monitor'] = verify_enterprise_monitor()
    
    # Summary
    print()
    print("=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for system, status in results.items():
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"{system:.<40} {status_str}")
    
    print()
    print(f"Total: {passed}/{total} systems verified")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        return 0
    else:
        print("⚠️ SOME SYSTEMS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
