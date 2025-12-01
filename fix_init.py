import os

content = '''"""

Package initialization.

"""

__version__ = "1.0.0"

__author__ = "Sheily AI Team"

'''

for root, dirs, files in os.walk('.'):
    for file in files:
        if file == '__init__.py':
            path = os.path.join(root, file)
            if os.path.getsize(path) == 0:
                with open(path, 'w') as f:
                    f.write(content)
                print(f"Fixed {path}")
