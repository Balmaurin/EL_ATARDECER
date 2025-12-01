#!/usr/bin/env python3
"""
SHEILY AI - INPUT SANITIZATION & VALIDATION SYSTEM
===============================================

Sistema avanzado de sanitización y validación de inputs para prevenir
vulnerabilidades de seguridad y garantizar integridad de datos.

Incluye:
- Sanitización XSS
- Validación SQL Injection
- Filtrado de comandos peligrosos
- Validación de tipos de datos
- Rate limiting integrado
"""

import hashlib
import html
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import bleach

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Niveles de severidad para validaciones"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InputType(Enum):
    """Tipos de input soportados"""

    TEXT = "text"
    HTML = "html"
    SQL = "sql"
    JSON = "json"
    EMAIL = "email"
    URL = "url"
    FILE_PATH = "file_path"
    COMMAND = "command"
    CODE = "code"


@dataclass
class ValidationRule:
    """Regla de validación individual"""

    name: str
    pattern: str
    severity: ValidationSeverity
    description: str
    enabled: bool = True


@dataclass
class ValidationResult:
    """Resultado de validación"""

    is_valid: bool
    sanitized_value: Any
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0


@dataclass
class SanitizationConfig:
    """Configuración de sanitización"""

    allow_html_tags: List[str] = field(
        default_factory=lambda: ["p", "br", "strong", "em", "u"]
    )
    allow_html_attrs: Dict[str, List[str]] = field(default_factory=dict)
    max_length: int = 10000
    allow_scripts: bool = False
    allow_styles: bool = False
    strip_comments: bool = True


class InputSanitizer:
    """
    Sanitizador avanzado de inputs para Sheily AI
    Previene XSS, SQL injection, command injection y otros ataques
    """

    def __init__(self, config: Optional[SanitizationConfig] = None):
        self.config = config or SanitizationConfig()

        # Patrones de validación predefinidos
        self.validation_rules = self._load_default_rules()

        # Cache para validaciones repetidas
        self._validation_cache: Dict[str, ValidationResult] = {}
        self._cache_max_size = 1000

        logger.info("🛡️ Input Sanitizer initialized with security rules")

    def _load_default_rules(self) -> Dict[str, List[ValidationRule]]:
        """Cargar reglas de validación por defecto con manejo de excepciones"""
        try:
            rules = {
                InputType.TEXT.value: [
                    ValidationRule(
                        name="xss_prevention",
                        pattern=r"<script[^>]*>.*?</script>",
                        severity=ValidationSeverity.CRITICAL,
                        description="Prevenir inyección de scripts XSS",
                    ),
                    ValidationRule(
                        name="sql_injection",
                        pattern=r"(\b(union|select|insert|update|delete|drop|create|alter)\b.*\b(from|into|table|database)\b)",
                        severity=ValidationSeverity.CRITICAL,
                        description="Detectar posibles inyecciones SQL",
                    ),
                    ValidationRule(
                        name="command_injection",
                        pattern=r"[;&|`$()<>]",
                        severity=ValidationSeverity.HIGH,
                        description="Prevenir inyección de comandos del sistema",
                    ),
                    ValidationRule(
                        name="path_traversal",
                        pattern=r"\.\./|\.\.\\",
                        severity=ValidationSeverity.HIGH,
                        description="Prevenir traversal de directorios",
                    ),
                ],
                InputType.HTML.value: [
                    ValidationRule(
                        name="dangerous_tags",
                        pattern=r"<(script|object|embed|form|input|meta|link|iframe)[^>]*>",
                        severity=ValidationSeverity.CRITICAL,
                        description="Etiquetas HTML peligrosas",
                    ),
                    ValidationRule(
                        name="javascript_urls",
                        pattern=r"javascript:",
                        severity=ValidationSeverity.HIGH,
                        description="URLs JavaScript peligrosas",
                    ),
                ],
                InputType.SQL.value: [
                    ValidationRule(
                        name="sql_keywords",
                        pattern=r"\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b",
                        severity=ValidationSeverity.CRITICAL,
                        description="Palabras clave SQL peligrosas",
                    ),
                    ValidationRule(
                        name="sql_comments",
                        pattern=r"--|#",
                        severity=ValidationSeverity.MEDIUM,
                        description="Comentarios SQL potencialmente peligrosos",
                    ),
                ],
                InputType.EMAIL.value: [
                    ValidationRule(
                        name="email_format",
                        pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                        severity=ValidationSeverity.MEDIUM,
                        description="Formato de email válido",
                    )
                ],
                InputType.URL.value: [
                    ValidationRule(
                        name="url_format",
                        pattern=r"^https?://[^\s/$.?#].[^\s]*$",
                        severity=ValidationSeverity.MEDIUM,
                        description="Formato de URL válido",
                    ),
                    ValidationRule(
                        name="localhost_prevention",
                        pattern=r"localhost|127\.0\.0\.1|0\.0\.0\.0",
                        severity=ValidationSeverity.HIGH,
                        description="Prevenir acceso a localhost",
                    ),
                ],
                InputType.FILE_PATH.value: [
                    ValidationRule(
                        name="path_safety",
                        pattern=r"^[a-zA-Z0-9._/-]+$",
                        severity=ValidationSeverity.MEDIUM,
                        description="Caracteres seguros en rutas de archivo",
                    ),
                    ValidationRule(
                        name="absolute_paths",
                        pattern=r"^/",
                        severity=ValidationSeverity.HIGH,
                        description="Prevenir rutas absolutas",
                    ),
                ],
                InputType.COMMAND.value: [
                    ValidationRule(
                        name="shell_metacharacters",
                        pattern=r"[;&|`$()<>]",
                        severity=ValidationSeverity.CRITICAL,
                        description="Metacaracteres de shell peligrosos",
                    ),
                    ValidationRule(
                        name="command_chaining",
                        pattern=r"&&|\|\||;",
                        severity=ValidationSeverity.CRITICAL,
                        description="Encadenamiento de comandos",
                    ),
                ],
                InputType.CODE.value: [
                    ValidationRule(
                        name="code_injection",
                        pattern=r"(eval|exec|compile)\s*\(",
                        severity=ValidationSeverity.CRITICAL,
                        description="Funciones de ejecución de código peligrosas",
                    ),
                    ValidationRule(
                        name="import_injection",
                        pattern=r"__import__|importlib",
                        severity=ValidationSeverity.HIGH,
                        description="Importaciones dinámicas peligrosas",
                    ),
                ],
            }
            return rules
        except Exception as e:
            logger.error(f"Error loading default validation rules: {e}")
            # Return minimal ruleset as fallback
            return {
                InputType.TEXT.value: [
                    ValidationRule(
                        name="basic_validation",
                        pattern=r".*",
                        severity=ValidationSeverity.LOW,
                        description="Basic input validation",
                    )
                ]
            }

    def sanitize_and_validate(
        self, input_value: Any, input_type: InputType, strict_mode: bool = True
    ) -> ValidationResult:
        """
        Sanitizar y validar input completo

        Args:
            input_value: Valor a sanitizar y validar
            input_type: Tipo de input
            strict_mode: Modo estricto (rechaza inputs sospechosos)

        Returns:
            ValidationResult: Resultado completo de sanitización y validación
        """
        start_time = time.time()

        # Convertir a string si es necesario
        if not isinstance(input_value, str):
            input_value = str(input_value)

        # Crear hash para cache
        cache_key = hashlib.md5(
            f"{input_value}:{input_type.value}:{strict_mode}".encode()
        ).hexdigest()

        # Verificar cache
        if cache_key in self._validation_cache:
            cached_result = self._validation_cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result

        result = ValidationResult(
            is_valid=True, sanitized_value=input_value, violations=[], warnings=[]
        )

        try:
            # Paso 1: Sanitización básica
            sanitized = self._basic_sanitize(input_value, input_type)

            # Paso 2: Validación según tipo
            violations = self._validate_by_type(sanitized, input_type)

            # Paso 3: Filtrado adicional si es necesario
            if input_type == InputType.HTML:
                sanitized = self._sanitize_html(sanitized)
            elif input_type == InputType.SQL:
                sanitized = self._sanitize_sql(sanitized)

            # Paso 4: Verificar violaciones críticas
            critical_violations = [
                v for v in violations if v["severity"] == ValidationSeverity.CRITICAL
            ]

            if critical_violations and strict_mode:
                result.is_valid = False
                result.violations = critical_violations
            else:
                result.sanitized_value = sanitized
                result.violations = violations
                result.warnings = [
                    v["description"]
                    for v in violations
                    if v["severity"] != ValidationSeverity.CRITICAL
                ]

            # Paso 5: Verificar longitud máxima
            if len(str(sanitized)) > self.config.max_length:
                result.is_valid = False
                result.violations.append(
                    {
                        "rule": "max_length",
                        "severity": ValidationSeverity.HIGH.value,
                        "description": f"Input exceeds maximum length of {self.config.max_length} characters",
                    }
                )

        except Exception as e:
            logger.error(f"Error during sanitization/validation: {e}")
            result.is_valid = False
            result.violations.append(
                {
                    "rule": "processing_error",
                    "severity": ValidationSeverity.CRITICAL.value,
                    "description": f"Processing error: {str(e)}",
                }
            )

        result.processing_time = time.time() - start_time

        # Cachear resultado
        if len(self._validation_cache) < self._cache_max_size:
            self._validation_cache[cache_key] = result

        return result

    def _basic_sanitize(self, value: str, input_type: InputType) -> str:
        """Sanitización básica de inputs"""
        if not value:
            return value

        # Eliminar caracteres de control
        value = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value)

        # Normalizar espacios
        value = " ".join(value.split())

        # Sanitización específica por tipo
        if input_type in [InputType.TEXT, InputType.HTML]:
            # Escapar HTML entities
            value = html.escape(value, quote=True)
        elif input_type == InputType.SQL:
            # Escapar comillas simples para SQL
            value = value.replace("'", "''")
        elif input_type == InputType.COMMAND:
            # Remover metacaracteres peligrosos
            value = re.sub(r"[;&|`$()<>]", "", value)

        return value

    def _validate_by_type(
        self, value: str, input_type: InputType
    ) -> List[Dict[str, Any]]:
        """Validar input según su tipo"""
        violations = []

        if input_type.value not in self.validation_rules:
            return violations

        for rule in self.validation_rules[input_type.value]:
            if not rule.enabled:
                continue

            # Verificar si la regla coincide (para reglas de detección de patrones peligrosos)
            if rule.name in [
                "xss_prevention",
                "sql_injection",
                "command_injection",
                "path_traversal",
                "dangerous_tags",
                "javascript_urls",
                "sql_keywords",
                "shell_metacharacters",
                "command_chaining",
                "code_injection",
                "import_injection",
            ]:
                if re.search(rule.pattern, value, re.IGNORECASE):
                    violations.append(
                        {
                            "rule": rule.name,
                            "severity": rule.severity.value,
                            "description": rule.description,
                            "pattern": rule.pattern,
                        }
                    )
            # Verificar formato válido (para reglas de formato)
            elif rule.name in ["email_format", "url_format", "path_safety"]:
                if not re.match(rule.pattern, value):
                    violations.append(
                        {
                            "rule": rule.name,
                            "severity": rule.severity.value,
                            "description": rule.description,
                            "pattern": rule.pattern,
                        }
                    )

        return violations

    def _sanitize_html(self, value: str) -> str:
        """Sanitización específica para HTML"""
        # Usar bleach para sanitización HTML avanzada
        try:
            return bleach.clean(
                value,
                tags=self.config.allow_html_tags,
                attributes=self.config.allow_html_attrs,
                strip=self.config.strip_comments,
                strip_comments=self.config.strip_comments,
            )
        except ImportError:
            # Fallback si bleach no está disponible
            logger.warning("Bleach not available, using basic HTML sanitization")
            # Remover tags peligrosos
            dangerous_tags = [
                "script",
                "object",
                "embed",
                "form",
                "input",
                "meta",
                "link",
                "iframe",
            ]
            for tag in dangerous_tags:
                value = re.sub(
                    rf"<{tag}[^>]*>.*?</{tag}>",
                    "",
                    value,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                value = re.sub(rf"<{tag}[^>]*>", "", value, flags=re.IGNORECASE)
            return value

    def _sanitize_sql(self, value: str) -> str:
        """Sanitización específica para SQL"""
        # Remover comentarios
        value = re.sub(r"--.*$", "", value, flags=re.MULTILINE)
        value = re.sub(r"#.*$", "", value, flags=re.MULTILINE)
        value = re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL)

        # Escapar caracteres peligrosos
        value = value.replace("'", "''")
        value = value.replace("\\", "\\\\")

        return value

    def validate_and_sanitize_input(
        self,
        input_data: Dict[str, Any],
        validation_schema: Optional[Dict[str, InputType]] = None,
    ) -> Dict[str, Any]:
        """
        Validar y sanitizar un diccionario completo de inputs

        Args:
            input_data: Datos a validar
            validation_schema: Esquema de validación por campo

        Returns:
            Dict con datos sanitizados y resultados de validación
        """
        if validation_schema is None:
            # Esquema por defecto - asumir texto para campos desconocidos
            validation_schema = {key: InputType.TEXT for key in input_data.keys()}

        sanitized_data = {}
        validation_results = {}

        for field, value in input_data.items():
            input_type = validation_schema.get(field, InputType.TEXT)

            result = self.sanitize_and_validate(value, input_type)

            if result.is_valid:
                sanitized_data[field] = result.sanitized_value
            else:
                logger.warning(
                    f"Validation failed for field '{field}': {result.violations}"
                )
                # En caso de fallo, usar valor sanitizado pero marcar como inválido
                sanitized_data[field] = result.sanitized_value

            validation_results[field] = {
                "is_valid": result.is_valid,
                "violations": result.violations,
                "warnings": result.warnings,
            }

        return {
            "sanitized_data": sanitized_data,
            "validation_results": validation_results,
            "overall_valid": all(r["is_valid"] for r in validation_results.values()),
        }


# Instancia global del sanitizador
_global_sanitizer = None


def get_input_sanitizer() -> InputSanitizer:
    """Obtener instancia global del sanitizador"""
    global _global_sanitizer
    if _global_sanitizer is None:
        _global_sanitizer = InputSanitizer()
    return _global_sanitizer


def validate_and_sanitize_input(
    input_data: Dict[str, Any], validation_schema: Optional[Dict[str, InputType]] = None
) -> Dict[str, Any]:
    """
    Función de conveniencia para validar y sanitizar inputs

    Args:
        input_data: Datos a procesar
        validation_schema: Esquema de validación

    Returns:
        Datos sanitizados con resultados de validación
    """
    sanitizer = get_input_sanitizer()
    return sanitizer.validate_and_sanitize_input(input_data, validation_schema)


def sanitize_text_input(text: str) -> str:
    """Sanitizar input de texto simple con manejo de excepciones"""
    try:
        sanitizer = get_input_sanitizer()
        result = sanitizer.sanitize_and_validate(text, InputType.TEXT)
        return result.sanitized_value if result.is_valid else ""
    except Exception as e:
        logger.error(f"Error sanitizing text input: {e}")
        return ""


def sanitize_html_input(html_content: str) -> str:
    """Sanitizar input HTML"""
    sanitizer = get_input_sanitizer()
    result = sanitizer.sanitize_and_validate(html_content, InputType.HTML)
    return result.sanitized_value if result.is_valid else ""


def sanitize_sql_input(sql_query: str) -> str:
    """Sanitizar input SQL"""
    sanitizer = get_input_sanitizer()
    result = sanitizer.sanitize_and_validate(sql_query, InputType.SQL)
    return result.sanitized_value if result.is_valid else ""


# Decorador para endpoints con validación automática
def validate_input(validation_schema: Dict[str, InputType]):
    """
    Decorador para validar inputs automáticamente en endpoints

    Args:
        validation_schema: Esquema de validación por campo
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extraer datos del request (asumiendo FastAPI)
            request_data = {}
            for arg_name, arg_value in kwargs.items():
                if isinstance(arg_value, dict):
                    request_data.update(arg_value)
                elif hasattr(arg_value, "__dict__"):
                    # Para modelos Pydantic
                    request_data.update(arg_value.__dict__)

            # Validar y sanitizar
            validation_result = validate_and_sanitize_input(
                request_data, validation_schema
            )

            if not validation_result["overall_valid"]:
                # Log de seguridad
                logger.warning(
                    f"Input validation failed: {validation_result['validation_results']}"
                )

                # En modo estricto, rechazar la request
                from fastapi import HTTPException

                raise HTTPException(status_code=400, detail="Input validation failed")

            # Reemplazar datos originales con sanitizados
            kwargs["validated_data"] = validation_result["sanitized_data"]

            return await func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    # Demo del sistema de sanitización
    print("🛡️ SHEILY AI - INPUT SANITIZATION SYSTEM DEMO")
    print("=" * 60)

    sanitizer = InputSanitizer()

    # Test cases
    test_cases = [
        ("<script>alert('XSS')</script>Hello World", InputType.HTML, "XSS Prevention"),
        ("'; DROP TABLE users; --", InputType.SQL, "SQL Injection Prevention"),
        ("$(rm -rf /)", InputType.COMMAND, "Command Injection Prevention"),
        ("../../../etc/passwd", InputType.FILE_PATH, "Path Traversal Prevention"),
        ("user@example.com", InputType.EMAIL, "Email Validation"),
        ("https://example.com", InputType.URL, "URL Validation"),
    ]

    for test_input, input_type, description in test_cases:
        print(f"\n🧪 Testing: {description}")
        print(f"Input: {test_input}")

        result = sanitizer.sanitize_and_validate(test_input, input_type)

        print(f"Valid: {result.is_valid}")
        print(f"Sanitized: {result.sanitized_value}")

        if result.violations:
            print(f"Violations: {len(result.violations)}")
            for violation in result.violations[:2]:  # Mostrar máximo 2
                print(
                    f"  - {violation['severity'].upper()}: {violation['description']}"
                )

        if result.warnings:
            print(f"Warnings: {result.warnings[:2]}")  # Mostrar máximo 2

    print("\n[OK] Input Sanitization System Demo Complete!")
    print("🛡️ Sistema listo para prevenir vulnerabilidades de seguridad")
