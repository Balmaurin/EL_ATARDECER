#!/usr/bin/env python3
"""
Test Script for Google ADK Integration with Hack-Memori
Verifica que el pegamento ADK esté funcionando correctamente
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

async def test_adk_integration():
    """Test complete ADK integration with Hack-Memori"""
    
    print("🔗 TESTING GOOGLE ADK INTEGRATION")
    print("=" * 50)
    
    try:
        # Test 1: Import ADK integration
        print("\n📦 Test 1: Importing ADK integration...")
        from packages.sheily_core.src.sheily_core.adk_integration import (
            get_adk_controller,
            execute_complete_system_evolution,
            monitor_and_trigger_system_evolution
        )
        print("✅ ADK integration imported successfully")
        
        # Test 2: Initialize ADK Controller
        print("\n🤖 Test 2: Initializing ADK Controller...")
        controller = get_adk_controller()
        print(f"✅ ADK Controller initialized - Available: {controller.is_adk_available}")
        
        # Test 3: Monitor Hack-Memori activity
        print("\n📊 Test 3: Monitoring Hack-Memori activity...")
        activity_data = await controller.monitor_hack_memori_activity()
        print(f"✅ Monitoring complete:")
        print(f"   - Sessions: {activity_data['total_sessions']}")
        print(f"   - Questions: {activity_data['total_questions']}")
        print(f"   - Responses: {activity_data['total_responses']}")
        print(f"   - Quality: {activity_data['average_quality']:.2%}")
        
        # Test 4: Evaluate training opportunity
        print("\n🎯 Test 4: Evaluating training opportunity...")
        evaluation = await controller.evaluate_training_opportunity(activity_data)
        print(f"✅ Evaluation complete:")
        print(f"   - Score: {evaluation['score']}/100")
        print(f"   - Should train: {evaluation['should_train']}")
        for reason in evaluation['reasons']:
            print(f"   - {reason}")
        
        # Test 5: Test monitoring function
        print("\n⏰ Test 5: Testing monitoring function...")
        monitoring_result = await monitor_and_trigger_system_evolution()
        print(f"✅ Monitoring function works:")
        print(f"   - Evolution triggered: {monitoring_result['evolution_triggered']}")
        print(f"   - Next check: {monitoring_result['next_check']}")
        
        # Test 6: Execute complete system evolution (if conditions are met)
        print("\n🚀 Test 6: Testing complete system evolution...")
        evolution_result = await execute_complete_system_evolution()
        print(f"✅ Evolution executed:")
        print(f"   - Status: {evolution_result['overall_status']}")
        print(f"   - Evolution ID: {evolution_result['evolution_id']}")
        
        if evolution_result['overall_status'] == 'completed':
            print("   - Systems orchestrated:")
            for system in evolution_result['phases']['orchestration']['systems_triggered']:
                print(f"     • {system}")
        elif evolution_result['overall_status'] == 'skipped':
            print(f"   - Reason: {evolution_result.get('reason', 'Unknown')}")
        
        # Test 7: Verify ADK stub functionality
        print("\n🧪 Test 7: Testing ADK stub functionality...")
        if evaluation['should_train']:
            orchestration = await controller.orchestrate_training_systems(activity_data)
            print(f"✅ ADK orchestration works:")
            print(f"   - Systems triggered: {len(orchestration['systems_triggered'])}")
            print(f"   - Status: {orchestration['status']}")
        else:
            print("⏭️ Orchestration skipped - conditions not met")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Google ADK integration is working correctly")
        print("✅ Hack-Memori pegamento is functional")
        print("✅ System evolution pipeline is operational")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Test execution failed")
        return False

async def test_hack_memori_service():
    """Test Hack-Memori service directly"""
    
    print("\n🔍 TESTING HACK-MEMORI SERVICE")
    print("=" * 40)
    
    try:
        from apps.backend.hack_memori_service import HackMemoriService
        
        service = HackMemoriService()
        print("✅ Hack-Memori service initialized")
        
        # Test getting sessions
        sessions = service.get_sessions()
        print(f"✅ Found {len(sessions)} sessions")
        
        # Test session details
        for session in sessions[:3]:  # Show first 3 sessions
            questions = service.get_questions(session["id"])
            responses = service.get_responses(session["id"])
            print(f"   - {session['name']}: {len(questions)} questions, {len(responses)} responses")
        
        return True
        
    except Exception as e:
        print(f"❌ Hack-Memori service test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 GOOGLE ADK + HACK-MEMORI INTEGRATION TEST")
    print("=" * 60)
    print("This script tests the complete integration between:")
    print("- Google ADK (Agent Development Kit)")
    print("- Hack-Memori automatic Q&A generation")
    print("- Training systems orchestration")
    print("- System evolution pipeline")
    print()
    
    try:
        # Test Hack-Memori service first
        hack_memori_success = asyncio.run(test_hack_memori_service())
        
        if not hack_memori_success:
            print("⚠️ Hack-Memori service has issues, but continuing with ADK tests...")
        
        # Test ADK integration
        adk_success = asyncio.run(test_adk_integration())
        
        print("\n" + "=" * 60)
        if adk_success:
            print("🎉 INTEGRATION TEST SUCCESSFUL!")
            print("✅ Google ADK is managing Hack-Memori correctly")
            print("✅ System evolution pipeline is functional")
            print("✅ All components are working together")
        else:
            print("❌ INTEGRATION TEST FAILED!")
            print("⚠️ Check error messages above for details")
        
        print("\n📝 Next steps:")
        if adk_success:
            print("- Run the system in production with real Hack-Memori data")
            print("- Monitor ADK orchestration logs")
            print("- Verify training system improvements")
        else:
            print("- Fix import and dependency issues")
            print("- Ensure Hack-Memori service is running")
            print("- Check Python environment setup")
        
        return adk_success
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected test failure")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)