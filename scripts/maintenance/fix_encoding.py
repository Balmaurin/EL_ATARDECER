import os

def fix_encoding(filepath):
    try:
        # Try reading as UTF-16LE (PowerShell default)
        with open(filepath, 'r', encoding='utf-16le') as f:
            content = f.read()
        
        # Write back as UTF-8
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed encoding for {filepath}")
    except Exception as e:
        print(f"Error fixing encoding for {filepath}: {e}")

fix_encoding(r"packages/sheily_core/src/sheily_core/tools/neuro_training_v2.py")
