import requests
import json
import sys

URL = "http://localhost:8000/graphql"
QUERY = """
query { 
    consciousness(consciousnessId: "main") { 
        phiValue 
        emotionalDepth 
        mindfulnessLevel 
        currentEmotion 
    } 
}
"""

def validate_connection():
    print(f"🚀 Testing GraphQL Connection to {URL}...")
    try:
        response = requests.post(
            URL, 
            json={'query': QUERY}, 
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "consciousness" in data["data"]:
                cons = data["data"]["consciousness"]
                print("✅ Connection Successful!")
                print(f"🧠 Consciousness Data Received:")
                print(f"   - Phi Value: {cons.get('phiValue')}")
                print(f"   - Emotion: {cons.get('currentEmotion')}")
                print(f"   - Depth: {cons.get('emotionalDepth')}")
                return True
            else:
                print(f"❌ Invalid Response Structure: {data}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_connection()
    sys.exit(0 if success else 1)
