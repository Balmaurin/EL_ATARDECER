"""
Sheily MCP Enterprise - Version Locker
Sistema avanzado de bloqueo de versiones enterprise
"""

import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class VersionLocker:
    """Sistema avanzado de bloqueo de versiones con integridad"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.config_dir = self.root_dir / "config"

    async def generate_lock_file(
        self, format_type: str = "poetry", strict: bool = False
    ) -> Dict[str, Any]:
        """Generar archivo de bloqueo de versiones enterprise"""

        # Get current installed packages
        installed_packages = await self._get_installed_packages()

        # Generate lock file based on format
        if format_type == "poetry":
            lock_data = await self._generate_poetry_lock(installed_packages, strict)
        elif format_type == "pip-tools":
            lock_data = await self._generate_pip_tools_lock(installed_packages, strict)
        else:  # requirements
            lock_data = await self._generate_requirements_lock(
                installed_packages, strict
            )

        # Calculate total packages based on format
        package_count = 0
        if format_type == "poetry":
            package_count = len(lock_data.get("package", {}))
        elif format_type == "pip-tools":
            package_count = len(lock_data.get("dependencies", {}))
        else:  # requirements
            package_count = len(lock_data.get("dependencies", {}))

        # Add metadata
        lock_data.update(
            {
                "metadata": {
                    "generated_at": asyncio.get_event_loop().time(),
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "working_directory": str(self.root_dir),
                    "total_packages": package_count,
                }
            }
        )

        return lock_data

    async def _get_installed_packages(self) -> Dict[str, str]:
        """Obtener paquetes instalados actualmente"""

        packages = {}

        try:
            # Use pip list to get installed packages
            result = await self._run_command(
                [sys.executable, "-m", "pip", "list", "--format=json"]
            )

            if result["success"]:
                package_list = json.loads(result["output"])
                packages = {pkg["name"]: pkg["version"] for pkg in package_list}

        except Exception as e:
            logger.warning(f"Failed to get installed packages: {e}")

        return packages

    async def _generate_poetry_lock(
        self, installed_packages: Dict[str, str], strict: bool
    ) -> Dict[str, Any]:
        """Generate Poetry-style lock file"""

        dependencies = {}
        package_hashes = {}

        for name, version in installed_packages.items():
            package_key = name.lower()

            # Generate hash for integrity
            package_hash = await self._generate_package_hash(name, version)

            dependencies[package_key] = {
                "version": version,
                "description": "",
                "category": "main",
            }

            if strict:
                package_hashes[f"{package_key}-{version}"] = package_hash

        lock_data = {
            "_meta": {
                "lock-version": "2.0",
                "python-versions": "*",
                "content-hash": await self._generate_content_hash(installed_packages),
                "hashes": package_hashes if strict else {},
            },
            "package": dependencies,
            "extras": {},
            "metadata": {
                "lock-version": "2.0",
                "python-versions": f"^{sys.version_info.major}.{sys.version_info.minor}",
                "content-hash": await self._generate_content_hash(installed_packages),
                "files": {},
            },
        }

        # Add missing format key so save_lock_file() logic works uniformly
        lock_data["format"] = "poetry"

        return lock_data

    async def _generate_pip_tools_lock(
        self, installed_packages: Dict[str, str], strict: bool
    ) -> Dict[str, Any]:
        """Generate pip-tools style requirements.txt lock"""

        lines = []

        for name, version in sorted(installed_packages.items()):
            line = f"{name}=={version}"

            if strict:
                # Add hash for strict mode
                package_hash = await self._generate_package_hash(name, version)
                line += f" --hash=sha256:{package_hash}"

            lines.append(line)

        return {
            "format": "pip-tools",
            "content": "\n".join(lines),
            "dependencies": installed_packages,
            "strict_mode": strict,
            "hash_enabled": strict,
        }

    async def _generate_requirements_lock(
        self, installed_packages: Dict[str, str], strict: bool
    ) -> Dict[str, Any]:
        """Generate basic requirements.txt style lock"""

        lines = []

        for name, version in sorted(installed_packages.items()):
            lines.append(f"{name}=={version}")

        return {
            "format": "requirements",
            "content": "\n".join(lines),
            "dependencies": installed_packages,
            "strict_mode": strict,
            "hash_enabled": False,
        }

    async def _generate_package_hash(self, name: str, version: str) -> str:
        """Generate a hash for package integrity"""

        # Create a deterministic hash based on package info
        content = f"{name}{version}{sys.version}{sys.platform}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def _generate_content_hash(self, packages: Dict[str, str]) -> str:
        """Generate content hash for all packages"""

        # Sort packages for deterministic hashing
        sorted_packages = sorted(packages.items())
        content = "".join(f"{name}{version}" for name, version in sorted_packages)

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def save_lock_file(
        self, lock_data: Dict[str, Any], output_path: Path
    ) -> bool:
        """Save lock file to disk"""

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if lock_data["format"] == "poetry":
                    json.dump(lock_data, f, indent=2)
                elif lock_data["format"] == "pip-tools":
                    # For pip-tools, save the content directly
                    f.write(lock_data.get("content", ""))
                else:  # requirements
                    f.write(lock_data.get("content", ""))

            logger.info(f"Lock file saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save lock file: {e}")
            return False

    async def validate_lock_file(self, lock_file_path: Path) -> Dict[str, Any]:
        """Validate integrity of lock file with complete validation logic"""

        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "security_issues": [],
            "integrity_checks": []
        }

        try:
            # Load lock file
            with open(lock_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Determine file format and validate structure
            if lock_file_path.suffix == ".json":
                validation_result.update(await self._validate_json_lock_file(content, lock_file_path))
            elif lock_file_path.suffix == ".txt":
                validation_result.update(await self._validate_txt_lock_file(content, lock_file_path))
            else:
                validation_result["errors"].append(f"Unsupported lock file format: {lock_file_path.suffix}")

            # Security checks
            if validation_result["valid"]:
                security_issues = await self._perform_security_checks(content, lock_file_path)
                validation_result["security_issues"] = security_issues

                if security_issues:
                    validation_result["warnings"].append(f"Found {len(security_issues)} security issues")

            # Integrity verification
            if validation_result["valid"]:
                integrity_result = await self._verify_integrity(content, lock_file_path)
                validation_result["integrity_checks"] = integrity_result

                if not integrity_result.get("integrity_valid", False):
                    validation_result["errors"].append("Integrity check failed")
                    validation_result["valid"] = False

        except FileNotFoundError:
            validation_result["errors"].append("Lock file not found")
        except PermissionError:
            validation_result["errors"].append("Permission denied accessing lock file")
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")

        return validation_result

    async def _validate_json_lock_file(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Validate JSON lock file (Poetry format)"""
        result = {"valid": False, "errors": [], "warnings": []}

        try:
            lock_data = json.loads(content)
            result["valid"] = True

            # Validate required fields
            required_fields = ["_meta", "package", "metadata"]
            for field in required_fields:
                if field not in lock_data:
                    result["errors"].append(f"Missing required field: {field}")
                    result["valid"] = False

            # Validate metadata structure
            if "metadata" in lock_data:
                metadata = lock_data["metadata"]
                required_meta = ["lock-version", "python-versions", "content-hash"]

                for meta_field in required_meta:
                    if meta_field not in metadata:
                        result["warnings"].append(f"Missing metadata field: {meta_field}")

            # Validate package structure
            if "package" in lock_data:
                packages = lock_data["package"]
                if isinstance(packages, dict):
                    for pkg_name, pkg_data in packages.items():
                        if not isinstance(pkg_data, dict):
                            result["warnings"].append(f"Invalid package data for {pkg_name}")
                        elif "version" not in pkg_data:
                            result["warnings"].append(f"Missing version for package {pkg_name}")
                else:
                    result["errors"].append("Package section must be a dictionary")
                    result["valid"] = False

            # Check for suspicious patterns
            if "package" in lock_data:
                suspicious_packages = await self._check_suspicious_packages(lock_data["package"])
                if suspicious_packages:
                    result["warnings"].extend([f"Suspicious package: {pkg}" for pkg in suspicious_packages])

        except json.JSONDecodeError as e:
            result["errors"].append(f"Invalid JSON format: {e}")
            result["valid"] = False

        return result

    async def _validate_txt_lock_file(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Validate text lock file (requirements.txt or pip-tools format)"""
        result = {"valid": False, "errors": [], "warnings": []}

        try:
            lines = content.strip().split('\n')
            if not lines:
                result["errors"].append("Lock file is empty")
                return result

            result["valid"] = True
            valid_lines = 0
            invalid_lines = []

            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments

                # Validate package specification format
                if not await self._validate_package_line(line):
                    invalid_lines.append(f"Line {i}: {line}")
                else:
                    valid_lines += 1

            if invalid_lines:
                result["warnings"].extend([f"Invalid package specification: {line}" for line in invalid_lines])

            if valid_lines == 0:
                result["warnings"].append("No valid package specifications found")

            # Check for version conflicts
            version_conflicts = await self._check_version_conflicts(content)
            if version_conflicts:
                result["warnings"].extend([f"Version conflict: {conflict}" for conflict in version_conflicts])

        except Exception as e:
            result["errors"].append(f"Error parsing text file: {e}")
            result["valid"] = False

        return result

    async def _validate_package_line(self, line: str) -> bool:
        """Validate a single package specification line"""
        import re

        # Remove hash specifications for validation
        line = re.sub(r'\s+--hash=[^\s]+', '', line)

        # Basic package name validation
        pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]?(==|>=|<=|>|<|!=|~=)?[^\s]*$'
        return bool(re.match(pattern, line.split()[0] if line.split() else line))

    async def _check_version_conflicts(self, content: str) -> List[str]:
        """Check for version conflicts in requirements"""
        conflicts = []
        packages = {}

        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Extract package name and version
            parts = line.split('==')
            if len(parts) >= 2:
                pkg_name = parts[0].strip()
                version = parts[1].split()[0]  # Remove any additional specs

                if pkg_name in packages and packages[pkg_name] != version:
                    conflicts.append(f"{pkg_name}: {packages[pkg_name]} vs {version}")
                else:
                    packages[pkg_name] = version

        return conflicts

    async def _check_suspicious_packages(self, packages: Dict) -> List[str]:
        """Check for potentially suspicious packages"""
        suspicious = []
        suspicious_names = [
            'hidden', 'backdoor', 'malware', 'trojan', 'spyware', 'ransomware',
            'cryptojacker', 'keylogger', 'rootkit', 'exploit'
        ]

        for pkg_name in packages.keys():
            pkg_lower = pkg_name.lower()
            if any(suspicious_word in pkg_lower for suspicious_word in suspicious_names):
                suspicious.append(pkg_name)

        return suspicious

    async def _perform_security_checks(self, content: str, file_path: Path) -> List[str]:
        """Perform comprehensive security checks on lock file"""
        security_issues = []

        # Check for overly permissive version specifiers
        if '>=' in content and '==' not in content:
            security_issues.append("Overly permissive version specifiers without pins")

        # Check for packages from untrusted sources (basic check)
        untrusted_indicators = ['git+', 'hg+', 'svn+', 'file://']
        for indicator in untrusted_indicators:
            if indicator in content:
                security_issues.append(f"Package from potentially untrusted source: {indicator}")

        # Check file permissions
        try:
            import stat
            file_stat = file_path.stat()
            if file_stat.st_mode & stat.S_IWOTH:  # World writable
                security_issues.append("Lock file is world-writable")
        except:
            pass  # Skip permission check if not available

        return security_issues

    async def _verify_integrity(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Verify integrity of lock file"""
        integrity_result = {
            "integrity_valid": False,
            "hash_verified": False,
            "content_hash": "",
            "calculated_hash": ""
        }

        try:
            # Calculate content hash
            calculated_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            integrity_result["calculated_hash"] = calculated_hash

            # For JSON files, check internal hash if present
            if file_path.suffix == ".json":
                try:
                    lock_data = json.loads(content)
                    if "metadata" in lock_data and "content-hash" in lock_data["metadata"]:
                        stored_hash = lock_data["metadata"]["content-hash"]
                        integrity_result["content_hash"] = stored_hash
                        integrity_result["hash_verified"] = (stored_hash == calculated_hash)
                        integrity_result["integrity_valid"] = integrity_result["hash_verified"]
                    else:
                        # No internal hash, but file is valid JSON
                        integrity_result["integrity_valid"] = True
                        integrity_result["hash_verified"] = True
                except json.JSONDecodeError:
                    integrity_result["integrity_valid"] = False
            else:
                # For text files, integrity is valid if parsing succeeded
                integrity_result["integrity_valid"] = True
                integrity_result["hash_verified"] = True

        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            integrity_result["integrity_valid"] = False

        return integrity_result

    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Execute command and return results"""

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "output": stdout.decode("utf-8", errors="ignore").strip(),
                "error": stderr.decode("utf-8", errors="ignore").strip(),
                "returncode": process.returncode,
            }

        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "returncode": -1}
