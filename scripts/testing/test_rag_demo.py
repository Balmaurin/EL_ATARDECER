import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from packages.rag_engine.src.advanced.parametric_rag import ParametricRAG

def ingest_and_query():
    print("🧠 Initializing RAG Engine...")
    rag = ParametricRAG()
    
    # 1. Ingest Documents
    print("\n📥 Ingesting Documents...")
    documents = [
        ("doc1", "The theory of Integrated Information (IIT) defines consciousness as intrinsic information."),
        ("doc2", "El Amanecer is an advanced AI system designed to simulate artificial consciousness."),
        ("doc3", "Parametric RAG injects knowledge directly into model parameters instead of context.")
    ]
    
    for doc_id, text in documents:
        rag.add_document(doc_id, text)
        print(f"   - Added {doc_id}")
        
    # 2. Query
    print("\n🔍 Testing Query...")
    query = "What is El Amanecer?"
    print(f"   Query: {query}")
    
    result = rag.generate_with_parametric_rag(query)
    
    print("\n📝 Result:")
    print(f"   Answer: {result['generated_answer']}")
    print(f"   Retrieved: {len(result['retrieved_docs'])} docs")
    
    if len(result['retrieved_docs']) > 0:
        print("✅ RAG Test Passed!")
    else:
        print("❌ RAG Test Failed (No docs retrieved)")

if __name__ == "__main__":
    ingest_and_query()
