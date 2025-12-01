import os
import py_compile

def check_syntax(start_path):
    output_file = "syntax_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Checking syntax in {start_path}...\n")
        errors = []
        for root, dirs, files in os.walk(start_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    try:
                        py_compile.compile(full_path, doraise=True)
                    except py_compile.PyCompileError as e:
                        f.write(f"Syntax error in {full_path}: {e}\n")
                        errors.append(full_path)
                    except Exception as e:
                        f.write(f"Error checking {full_path}: {e}\n")
                        errors.append(full_path)
        
        if not errors:
            f.write("No syntax errors found.\n")
        else:
            f.write(f"Found {len(errors)} files with syntax errors.\n")

if __name__ == "__main__":
    check_syntax(r"c:\Users\YO\Desktop\EL-AMANECERV3-main - copia\packages")
