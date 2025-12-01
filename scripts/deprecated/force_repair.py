import os

def force_repair(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        print(f"Checking {filepath}...")
        print(f"Start: {content[:20]}")
        
        if content.startswith(']'):
            print(f"Repairing {filepath}...")
            new_content = content[1::2]
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Repaired {filepath}.")
        else:
            print(f"Skipping {filepath} (does not start with ])")

    except Exception as e:
        print(f"Error: {e}")

files = [
    r"packages/sheily_core/src/sheily_core/tools/neuro_training_v2.py",
    r"packages/rag_engine/src/advanced/qr_lora.py",
    r"packages/consciousness/__init__.py",
    r"packages/sheily_core/src/sheily_core/__init__.py",
    r"packages/__init__.py"
]

for f in files:
    force_repair(f)
