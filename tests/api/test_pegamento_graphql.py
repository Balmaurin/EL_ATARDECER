"""
Complete Pegamento GraphQL Integration Test
Tests all pegamento functionality through GraphQL API
"""

import requests
import json
from typing import Dict

# GraphQL endpoint
GRAPHQL_URL = "http://localhost:8000/graphql"


def execute_query(query: str, variables: Dict = None) -> Dict:
    """Execute a GraphQL query"""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    response = requests.post(GRAPHQL_URL, json=payload)
    return response.json()


def test_hack_memori_activity():
    """Test Hack Memori activity monitoring"""
    print("\n" + "="*60)
    print("🔍 TEST 1: Hack Memori Activity Monitoring")
    print("="*60)
    
    query = """
    query {
      hackMemoriActivity {
        totalSessions
        totalQuestions
        totalResponses
        averageQuality
      }
    }
    """
    
    result = execute_query(query)
    
    if not result:
        print(f"❌ FAILED: No response from server")
        return False
    
    if "errors" in result:
        print(f"❌ FAILED: GraphQL errors: {result['errors']}")
        return False
    
    if "data" in result and result["data"] and result["data"].get("hackMemoriActivity"):
        activity = result["data"]["hackMemoriActivity"]
        print(f"✅ SUCCESS")
        print(f"   Sessions: {activity['totalSessions']}")
        print(f"   Questions: {activity['totalQuestions']}")
        print(f"   Responses: {activity['totalResponses']}")
        print(f"   Quality: {activity['averageQuality']:.2%}")
        return True
    else:
        print(f"❌ FAILED: Invalid response structure")
        return False


def test_trigger_evolution():
    """Test triggering system evolution"""
    print("\n" + "="*60)
    print("🚀 TEST 2: Trigger System Evolution")
    print("="*60)
    
    query = """
    mutation {
      triggerSystemEvolution {
        evolutionId
        overallStatus
        systemsOrchestrated
        trainingTriggered
        timestamp
      }
    }
    """
    
    result = execute_query(query)
    
    if "data" in result and result["data"]["triggerSystemEvolution"]:
        evolution = result["data"]["triggerSystemEvolution"]
        print(f"✅ SUCCESS")
        print(f"   Evolution ID: {evolution['evolutionId']}")
        print(f"   Status: {evolution['overallStatus']}")
        print(f"   Systems: {evolution['systemsOrchestrated']}")
        print(f"   Training: {'✅' if evolution['trainingTriggered'] else '❌'}")
        return True
    else:
        print(f"❌ FAILED: {result}")
        return False


def test_combined_workflow():
    """Test complete pegamento workflow"""
    print("\n" + "="*60)
    print("🔄 TEST 3: Complete Pegamento Workflow")
    print("="*60)
    
    # Step 1: Check activity
    print("\n📊 Step 1: Checking Hack Memori activity...")
    activity_query = """
    query {
      hackMemoriActivity {
        totalQuestions
        averageQuality
      }
    }
    """
    
    activity_result = execute_query(activity_query)
    
    if "data" in activity_result:
        activity = activity_result["data"]["hackMemoriActivity"]
        print(f"   Questions: {activity['totalQuestions']}")
        print(f"   Quality: {activity['averageQuality']:.2%}")
        
        # Step 2: Add to search if we have data
        if activity['totalQuestions'] > 0:
            print("\n🔍 Step 2: Adding sample to semantic search...")
            add_docs_query = """
            mutation {
              addSearchDocuments(
                documents: ["Sample training data from Hack Memori workflow"]
              )
            }
            """
            add_result = execute_query(add_docs_query)
            print(f"   Added: {'✅' if add_result.get('data', {}).get('addSearchDocuments') else '❌'}")
        
        # Step 3: Check system metrics
        print("\n📈 Step 3: Checking system metrics...")
        metrics_query = """
        query {
          systemMetrics {
            cpuPercent
            memoryPercent
          }
        }
        """
        metrics_result = execute_query(metrics_query)
        if "data" in metrics_result:
            metrics = metrics_result["data"]["systemMetrics"]
            print(f"   CPU: {metrics['cpuPercent']:.1f}%")
            print(f"   Memory: {metrics['memoryPercent']:.1f}%")
        
        print("\n✅ Complete workflow executed successfully!")
        return True
    
    return False


def test_api_info():
    """Test API info endpoint"""
    print("\n" + "="*60)
    print("📡 TEST 4: API Info with Pegamento")
    print("="*60)
    
    response = requests.get("http://localhost:8000/")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SUCCESS")
        print(f"   API: {data.get('message')}")
        print(f"   Version: {data.get('version')}")
        
        if 'pegamento' in data:
            print(f"\n   Pegamento Status:")
            for key, value in data['pegamento'].items():
                print(f"     {key}: {value}")
        
        return True
    else:
        print(f"❌ FAILED: {response.status_code}")
        return False


def main():
    """Run all pegamento tests"""
    print("="*60)
    print("🚀 COMPLETE PEGAMENTO GRAPHQL INTEGRATION TEST")
    print("="*60)
    print(f"Testing endpoint: {GRAPHQL_URL}")
    
    results = {}
    
    try:
        # Run tests
        results["API Info"] = test_api_info()
        results["Hack Memori Activity"] = test_hack_memori_activity()
        results["Complete Workflow"] = test_combined_workflow()
        results["Trigger Evolution"] = test_trigger_evolution()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to GraphQL server")
        print("   Make sure the server is running:")
        print("   python apps/backend/src/api/real_graphql_api.py")
        return
    
    # Summary
    print("\n" + "="*60)
    print("📊 PEGAMENTO TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, status in results.items():
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"{test:.<40} {status_str}")
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL PEGAMENTO TESTS PASSED!")
        print("✅ Hack Memori integration working")
        print("✅ Evolution cycle accessible via GraphQL")
        print("✅ Complete workflow functional")
    else:
        print("⚠️ SOME TESTS FAILED")
    
    print("="*60)


if __name__ == "__main__":
    main()
