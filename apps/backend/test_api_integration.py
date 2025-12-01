#!/usr/bin/env python3
"""
Quick test to check GraphQL federation API integration
"""

import asyncio
import httpx
import json

async def test_api():
    """Test the federation API directly"""
    url = "http://localhost:8080/graphql"

    # Test consciousness query
    query = """
    query {
        consciousness(consciousnessId: "test-user") {
            consciousnessId
            phiValue
            currentEmotion
            experienceCount
            neuralActivity
        }
    }
    """

    async with httpx.AsyncClient(timeout=10.0) as client:
        payload = {"query": query}
        response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
        result = response.json()

        print("=== CONSCIOUSNESS QUERY RESULT ===")
        print(f"Status: {response.status_code}")
        print(f"Raw Result: {json.dumps(result, indent=2)}")

        if "data" in result and "consciousness" in result["data"]:
            cons = result["data"]["consciousness"]
            print("\n=== GRAPHQL CONSCIOUSNESS DATA ===")
            print(f"ID: {cons.get('consciousnessId')}")
            print(f"Phi: {cons.get('phiValue')}")
            print(f"Emotion: {cons.get('currentEmotion')}")
            print(f"Experiences: {cons.get('experienceCount')}")
            print(f"Neural: {cons.get('neuralActivity')}")

if __name__ == "__main__":
    asyncio.run(test_api())
