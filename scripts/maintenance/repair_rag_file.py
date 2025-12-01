import sys

FILE_PATH = "packages/rag_engine/src/advanced/parametric_rag.py"

def repair_file():
    print(f"🔧 Repairing {FILE_PATH}...")
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # The corruption seems to be ']' characters inserted.
        # Let's try removing them.
        repaired_content = content.replace(']', '')
        
        # Verify if it looks like python
        if "class ParametricRAG" in repaired_content or "def __init__" in repaired_content:
            print("✅ Content looks valid after stripping ']'.")
            
            with open(FILE_PATH, 'w', encoding='utf-8') as f:
                f.write(repaired_content)
            print("✅ File saved.")
        else:
            print("❌ Repair failed: Content does not look like valid Python.")
            print(repaired_content[:200])
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    repair_file()
