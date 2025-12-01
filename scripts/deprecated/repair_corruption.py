import os

def repair_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if '' in content or '' in content:
            print(f"Repairing {filepath}...")
            original_len = len(content)
            content = content.replace('', '')
            content = content.replace('', '')
            new_len = len(content)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Repaired {filepath}. Size changed from {original_len} to {new_len}")
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
             if repair_file(path):
                 count += 1
        else:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.py') or file.endswith('.md') or file.endswith('.json'):
                        filepath = os.path.join(root, file)
                        if repair_file(filepath):
                            count += 1
    
    print(f"Total files repaired: {count}")

if __name__ == "__main__":
    main()
