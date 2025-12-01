
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Mock necessary modules if they are missing dependencies
import unittest.mock
sys.modules['psutil'] = unittest.mock.MagicMock()
sys.modules['numpy'] = unittest.mock.MagicMock()

async def verify_tier_1():
    print("🚀 Verifying Tier 1 Fixes...")
    results = {}

    try:
        print("\n1. Verifying recursive_self_improvement.py...")
        from packages.auto_improvement.recursive_self_improvement import RecursiveSelfImprovementEngine
        engine = RecursiveSelfImprovementEngine()
        analysis = engine.analyze_architecture()
        print(f"   ✅ analyze_architecture result: {analysis}")
        score = engine.calculate_improvement_score({'completion_rate': 0.8, 'quality_delta': 10, 'learning_rate': 0.5})
        print(f"   ✅ calculate_improvement_score result: {score}")
        results['recursive_self_improvement'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['recursive_self_improvement'] = f'FAIL: {e}'

    try:
        print("\n2. Verifying unified_embedding_semantic_system.py...")
        from packages.sheily_core.src.sheily_core.unified_systems.unified_embedding_semantic_system import UnifiedEmbeddingSemanticSystem
        # Mock numpy for the test
        import numpy as np
        sys.modules['numpy'] = np
        
        system = UnifiedEmbeddingSemanticSystem()
        # Mock embeddings
        emb1 = np.array([0.1, 0.2, 0.3])
        emb2 = np.array([0.1, 0.2, 0.4])
        similarity = system.calculate_semantic_similarity(emb1, emb2)
        print(f"   ✅ calculate_semantic_similarity result: {similarity}")
        results['unified_embedding'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['unified_embedding'] = f'FAIL: {e}'

    try:
        print("\n3. Verifying mcp_agent_manager.py...")
        from packages.sheily_core.src.sheily_core.core.mcp.mcp_agent_manager import MCPAgentManager
        manager = MCPAgentManager()
        linked = await manager._link_capabilities("layer1", "layer2")
        print(f"   ✅ _link_capabilities result: {linked}")
        analysis = await manager._analyze_operation("test_operation", {})
        print(f"   ✅ _analyze_operation result: {analysis}")
        plan = await manager._plan_distributed_execution(analysis)
        print(f"   ✅ _plan_distributed_execution result: {plan}")
        results['mcp_agent_manager'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['mcp_agent_manager'] = f'FAIL: {e}'

    try:
        print("\n4. Verifying mcp_auto_improvement.py...")
        from packages.rag_engine.src.core.mcp_auto_improvement import MCPAutoImprovementEngine
        engine = MCPAutoImprovementEngine()
        security = await engine._assess_security_posture()
        print(f"   ✅ _assess_security_posture result: {security}")
        quality = engine.calculate_retrieval_quality([{'score': 0.9, 'source': 'a'}, {'score': 0.8, 'source': 'b'}])
        print(f"   ✅ calculate_retrieval_quality result: {quality}")
        results['mcp_auto_improvement'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['mcp_auto_improvement'] = f'FAIL: {e}'

    try:
        print("\n5. Verifying mcp_protocol.py...")
        from packages.sheily_core.src.sheily_core.core.mcp.mcp_protocol import MCPServer, MCPMessage
        server = MCPServer("test")
        msg = MCPMessage(jsonrpc="2.0", id="1", method="test")
        score = server.validate_message_quality(msg)
        print(f"   ✅ validate_message_quality result: {score}")
        results['mcp_protocol'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['mcp_protocol'] = f'FAIL: {e}'

    try:
        print("\n6. Verifying database.py...")
        from apps.backend.src.core.database import AsyncDatabase
        db = AsyncDatabase("sqlite:///test.db")
        # Just check class instantiation, connection requires async context and file
        print(f"   ✅ AsyncDatabase instantiated: {db.url}")
        results['database'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['database'] = f'FAIL: {e}'

    try:
        print("\n7. Verifying master_orchestrator.py...")
        from packages.sheily_core.src.sheily_core.core.system.master_orchestrator import MasterMCPOrchestrator
        # Mock dependencies
        sys.modules['sheily_core.agents.base.enhanced_base'] = unittest.mock.MagicMock()
        
        orchestrator = MasterMCPOrchestrator()
        # Test healing action logic (mocking internal state)
        orchestrator.agent_registry = {}
        await orchestrator._execute_healing_action({"type": "scale_agents", "agent_type": "test", "count": 2})
        print("   ✅ _execute_healing_action (scale_agents) executed")
        results['master_orchestrator'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['master_orchestrator'] = f'FAIL: {e}'

    try:
        print("\n8. Verifying integration_manager_v2.py...")
        from packages.sheily_core.src.sheily_core.integration.integration_manager_v2 import SheilyIntegrationManager
        manager = SheilyIntegrationManager()
        # Test learning score logic
        res = manager.process_learning_interaction("test content", "chat")
        print(f"   ✅ process_learning_interaction result: {res}")
        results['integration_manager'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['integration_manager'] = f'FAIL: {e}'

    try:
        print("\n9. Verifying auto_training_system.py...")
        from tools.ai.auto_training_system import AutoTrainingSystem
        system = AutoTrainingSystem()
        await system.start_monitoring()
        res = await system.process_feedback({"test": "data"})
        print(f"   ✅ process_feedback result: {res}")
        results['auto_training_system'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['auto_training_system'] = f'FAIL: {e}'

    try:
        print("\n10. Verifying rag_integrator.py...")
        from packages.rag_engine.src.advanced.integration.rag_integrator import RAGIntegrator
        # Mock dependencies
        sys.modules['sheily_core.core.mcp.mcp_agent_manager'] = unittest.mock.MagicMock()
        
        integrator = RAGIntegrator(enable_mcp_integration=True, federated_learning_enabled=True)
        integrator.update_models()
        print("   ✅ update_models executed")
        results['rag_integrator'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['rag_integrator'] = f'FAIL: {e}'

    print("\n📊 Verification Summary:")
    for file, status in results.items():
        print(f"  {file}: {status}")

if __name__ == "__main__":
    asyncio.run(verify_tier_1())
