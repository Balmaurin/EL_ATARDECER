import os

def repair_file_v2(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content:
            return False

        # Check if file follows the pattern ]c]h]a]r
        # We check the first few characters
        if content.startswith(']') and len(content) > 1:
            # Check a sample to be sure
            is_pattern = True
            limit = min(len(content), 100)
            for i in range(0, limit, 2):
                if content[i] != ']':
                    is_pattern = False
                    break
            
            if is_pattern:
                print(f"Repairing v2 {filepath}...")
                original_len = len(content)
                new_content = content[1::2]
                new_len = len(new_content)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Repaired v2 {filepath}. Size changed from {original_len} to {new_len}")
                return True
    except Exception as e:
        print(f"Error reading/repairing {filepath}: {e}")
    return False

def main():
    paths_to_check = [
        'scripts',
        'packages',
        'apps'
    ]
    
    count = 0
    for path in paths_to_check:
        if os.path.isfile(path):
             if repair_file_v2(path):
                 count += 1
        else:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.py') or file.endswith('.md') or file.endswith('.json'):
                        filepath = os.path.join(root, file)
                        if repair_file_v2(filepath):
                            count += 1
    
    print(f"Total files repaired v2: {count}")

if __name__ == "__main__":
    main()
