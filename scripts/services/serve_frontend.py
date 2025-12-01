import http.server
import socketserver
import os
import sys

PORT = 8080
PUBLIC_DIR = os.path.abspath("apps/frontend/public")
SRC_DIR = os.path.abspath("apps/frontend/src")

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # If the path starts with /src/, serve from the src directory
        if path.startswith("/src/"):
            # Remove '/src/' from the start and join with SRC_DIR
            relative_path = path[5:] 
            return os.path.join(SRC_DIR, relative_path)
        
        # Otherwise, serve from the public directory
        return os.path.join(PUBLIC_DIR, path.lstrip("/"))

    def log_message(self, format, *args):
        # Suppress logging to keep console clean, or redirect to a file if needed
        pass

if __name__ == "__main__":
    # Change to the project root directory to ensure relative paths work
    # Assuming this script is run from project root or scripts/
    # We'll force the CWD to be the project root based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Re-evaluate paths after chdir
    PUBLIC_DIR = os.path.abspath("apps/frontend/public")
    SRC_DIR = os.path.abspath("apps/frontend/src")

    print(f"Starting Frontend Server on http://localhost:{PORT}")
    print(f"Serving Root: {PUBLIC_DIR}")
    print(f"Mapping /src to: {SRC_DIR}")

    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
