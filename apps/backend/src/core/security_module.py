class SecurityError(Exception):
    pass

def validate_command_args(args):
    pass

def validate_timeout(timeout):
    pass

def sanitize_filename(filename):
    return filename.replace("..", "").replace("/", "").replace("\\", "")

def sanitize_path(path):
    """Sanitiza un path para evitar directory traversal básico"""
    if not path:
        return ""
    return path.replace("..", "")
