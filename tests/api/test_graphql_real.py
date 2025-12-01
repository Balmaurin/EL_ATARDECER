"""
GraphQL API Test Suite - Real Implementation Testing
Tests all queries and mutations against the running GraphQL server
"""

import asyncio
import json
import requests
from typing import Dict, Any

# GraphQL endpoint
GRAPHQL_URL = "http://localhost:8000/graphql"


def execute_query(query: str, variables: Dict[str, Any] = None) -> Dict:
    """Execute a GraphQL query"""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    response = requests.post(GRAPHQL_URL, json=payload)
    return response.json()


def test_system_metrics():
    """Test system metrics query"""
    print("\n" + "="*60)
    print("🔍 TEST 1: System Metrics")
    print("="*60)
    
    query = """
    query {
      systemMetrics {
        cpuPercent
        memoryPercent
        diskPercent
        timestamp
      }
    }
    """
    
    result = execute_query(query)
    
    if "data" in result and result["data"]["systemMetrics"]:
        metrics = result["data"]["systemMetrics"]
        print(f"✅ SUCCESS")
        print(f"   CPU: {metrics['cpuPercent']:.1f}%")
        print(f"   Memory: {metrics['memoryPercent']:.1f}%")
        print(f"   Disk: {metrics['diskPercent']:.1f}%")
        print(f"   Timestamp: {metrics['timestamp']}")
        return True
    else:
        print(f"❌ FAILED: {result}")
        return False


def test_add_documents():
    """Test adding documents to search"""
    print("\n" + "="*60)
    print("📝 TEST 2: Add Search Documents")
    print("="*60)
    
    query = """
    mutation {
      addSearchDocuments(
        documents: [
          "Machine learning is a subset of artificial intelligence that enables systems to learn from data",
          "Python is a high-level programming language widely used in data science and AI",
          "Neural networks are computational models inspired by biological neural networks",
          "Deep learning uses multiple layers of neural networks for complex pattern recognition",
          "Natural language processing helps computers understand and generate human language"
        ]
      )
    }
    """
    
    result = execute_query(query)
    
    if "data" in result and result["data"]["addSearchDocuments"]:
        print("✅ SUCCESS - Documents added to search index")
        return True
    else:
        print(f"❌ FAILED: {result}")
        return False


def test_semantic_search():
    """Test semantic search"""
    print("\n" + "="*60)
    print("🔍 TEST 3: Semantic Search")
    print("="*60)
    
    query = """
    query {
      semanticSearch(query: "What is artificial intelligence?", k: 3) {
        document
        score
        index
      }
    }
    """
    
    result = execute_query(query)
    
    if "data" in result and result["data"]["semanticSearch"]:
        results = result["data"]["semanticSearch"]
        print(f"✅ SUCCESS - Found {len(results)} results")
        for i, r in enumerate(results, 1):
            print(f"\n   Result {i} (Score: {r['score']:.3f}):")
            print(f"   {r['document'][:80]}...")
        return True
    else:
        print(f"❌ FAILED: {result}")
        return False


def test_nft_balance():
    """Test NFT balance query"""
    print("\n" + "="*60)
    print("💰 TEST 4: NFT Balance")
    print("="*60)
    
    query = """
    query {
      nftBalance
    }
    """
    
    result = execute_query(query)
    
    if "data" in result:
        balance = result["data"]["nftBalance"]
        print(f"✅ SUCCESS")
        print(f"   Wallet Balance: {balance} SOL")
        return True
    else:
        print(f"❌ FAILED: {result}")
        return False


def test_generate_text():
    """Test LLM text generation"""
    print("\n" + "="*60)
    print("🤖 TEST 5: LLM Text Generation")
    print("="*60)
    
    query = """
    mutation {
      generateText(
        prompt: "Explain machine learning in one sentence:",
        maxTokens: 50
      ) {
        text
        model
      }
    }
    """
    
    result = execute_query(query)
    
    if "data" in result and result["data"]["generateText"]:
        gen = result["data"]["generateText"]
        print(f"✅ SUCCESS")
        print(f"   Model: {gen['model']}")
        print(f"   Generated: {gen['text'][:200]}...")
        return True
    else:
        print(f"❌ FAILED: {result}")
        return False


def test_health_endpoint():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("🏥 TEST 6: Health Check Endpoint")
    print("="*60)
    
    response = requests.get("http://localhost:8000/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"✅ SUCCESS")
        print(f"   Healthy: {health.get('healthy', False)}")
        print(f"   Uptime: {health.get('uptime_seconds', 0):.1f}s")
        return True
    else:
        print(f"❌ FAILED: {response.status_code}")
        return False


def test_root_endpoint():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("📡 TEST 7: Root API Endpoint")
    print("="*60)
    
    response = requests.get("http://localhost:8000/")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SUCCESS")
        print(f"   API: {data.get('message')}")
        print(f"   Version: {data.get('version')}")
        print(f"\n   Systems:")
        for system, status in data.get('systems', {}).items():
            print(f"     {system}: {status}")
        return True
    else:
        print(f"❌ FAILED: {response.status_code}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("🚀 GRAPHQL API REAL IMPLEMENTATION TEST SUITE")
    print("="*60)
    print(f"Testing endpoint: {GRAPHQL_URL}")
    
    results = {}
    
    try:
        # Run tests
        results["Root Endpoint"] = test_root_endpoint()
        results["Health Check"] = test_health_endpoint()
        results["System Metrics"] = test_system_metrics()
        results["Add Documents"] = test_add_documents()
        results["Semantic Search"] = test_semantic_search()
        results["NFT Balance"] = test_nft_balance()
        results["LLM Generation"] = test_generate_text()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to GraphQL server")
        print("   Make sure the server is running:")
        print("   python apps/backend/src/api/real_graphql_api.py")
        return
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, status in results.items():
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"{test:.<40} {status_str}")
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - GRAPHQL API FULLY FUNCTIONAL!")
    else:
        print("⚠️ SOME TESTS FAILED")
    
    print("="*60)


if __name__ == "__main__":
    main()
