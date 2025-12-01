#!/usr/bin/env python3
"""
RECURSIVE SELF-IMPROVEMENT ENGINE - Sistema de Auto-Mejora Real
===============================================================

Sistema de auto-mejora recursiva para análisis y mejora de código:
- Análisis real de código Python usando AST
- Detección de patrones problemáticos
- Aplicación de mejoras con validación y reversión
- Tracking de mejoras aplicadas
- Métricas reales de progreso

Implementación completamente funcional sin simulaciones.
"""

import ast
import asyncio
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Intentar importar dependencias opcionales
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil no disponible, análisis de recursos limitado")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy no disponible, usando listas nativas")


@dataclass
class CodeImprovement:
    """Mejora de código detectada y aplicable"""
    improvement_id: str
    file_path: str
    improvement_type: str  # 'code_quality', 'performance', 'security', 'structure'
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    line_number: Optional[int] = None
    original_code: Optional[str] = None
    improved_code: Optional[str] = None
    confidence_score: float = 0.0  # 0.0 - 1.0
    applied: bool = False
    applied_at: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class ImprovementLoop:
    """Bucle de auto-mejora activo"""

    loop_id: str
    loop_type: str  # 'code_analysis', 'performance_optimization', 'security_hardening'
    target_directory: str
    current_iteration: int
    improvements_found: int
    improvements_applied: int
    improvements_successful: int
    active: bool = True
    started_at: datetime = field(default_factory=datetime.now)
    last_iteration_at: Optional[datetime] = None


@dataclass
class SelfModification:
    """Modificación de código aplicada"""

    mod_id: str
    target_file: str
    modification_type: str
    description: str
    confidence_score: float
    risk_level: str  # 'low', 'medium', 'high'
    backup_path: Optional[str] = None
    reversible: bool = True
    applied_at: Optional[datetime] = None
    success: bool = False
    validation_passed: bool = False
    rollback_available: bool = False


class RecursiveSelfImprovementEngine:
    """Motor de auto-mejora recursiva REAL para código"""

    def __init__(self,
                 project_root: Optional[str] = None,
                 improvement_dir: str = "auto_improvement/state",
                 safety_boundaries: Optional[Dict[str, Any]] = None,
                 singularity_dir: Optional[str] = None):

        # Determinar directorio del proyecto
        if project_root:
            self.project_root = Path(project_root)
        else:
            # Intentar encontrar raíz del proyecto
            current = Path(__file__).resolve()
            while current != current.parent:
                if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
                    self.project_root = current
                    break
                current = current.parent
            else:
                self.project_root = Path.cwd()
        
        self.improvement_dir = Path(improvement_dir)
        self.improvement_dir.mkdir(parents=True, exist_ok=True)

        if singularity_dir is not None:
            self.singularity_dir = Path(singularity_dir)
            self.singularity_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.singularity_dir = None

        # Estado de auto-mejora
        self.improvement_loops: Dict[str, ImprovementLoop] = {}
        self.improvements_found: List[CodeImprovement] = []
        self.self_modifications: List[SelfModification] = []
        self.applied_improvements_history: List[Dict[str, Any]] = []

        # Métricas reales
        self.metrics: Dict[str, Any] = {
            'total_files_analyzed': 0,
            'total_improvements_found': 0,
            'total_improvements_applied': 0,
            'total_improvements_successful': 0,
            'total_improvements_failed': 0,
            'files_modified': 0,
            'code_complexity_reduction': 0.0,
            'security_issues_fixed': 0,
            'performance_improvements': 0
        }

        # Límites de seguridad
        self.safety_boundaries = safety_boundaries or {
            'max_loop_iterations': 50,
            'max_files_per_iteration': 100,
            'require_human_approval': False,  # Para auto-mejora completamente automática
            'backup_before_modify': True,
            'validate_after_modify': True,
            'max_risk_level': 'medium',  # 'low', 'medium', 'high'
            'allow_irreversible_changes': False
        }

        logger.info("🚀 Recursive Self-Improvement Engine inicializado")
        logger.info(f"   Directorio del proyecto: {self.project_root}")
        logger.info(f"   Límites de seguridad activos: {len(self.safety_boundaries)}")

    async def start_improvement_loop(self, 
                                     target_directory: Optional[str] = None,
                                     loop_type: str = "code_analysis",
                                     max_iterations: int = 10) -> ImprovementLoop:
        """
        Inicia un bucle de auto-mejora REAL
        
        Args:
            target_directory: Directorio a analizar (default: proyecto completo)
            loop_type: Tipo de bucle ('code_analysis', 'performance_optimization', 'security_hardening')
            max_iterations: Máximo de iteraciones
            
        Returns:
            ImprovementLoop iniciado
        """
        logger.info(f"🔄 Iniciando bucle de mejora: {loop_type}")

        target_dir = Path(target_directory) if target_directory else self.project_root
        if not target_dir.exists():
            raise ValueError(f"Directorio no encontrado: {target_dir}")

        loop_id = f"{loop_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        loop = ImprovementLoop(
            loop_id=loop_id,
            loop_type=loop_type,
            target_directory=str(target_dir),
            current_iteration=0,
            improvements_found=0,
            improvements_applied=0,
            improvements_successful=0,
            active=True
        )

        self.improvement_loops[loop_id] = loop

        # Ejecutar iteraciones
        for iteration in range(max_iterations):
            if not loop.active:
                logger.info(f"Bucle {loop_id} detenido por límites de seguridad")
                break

            loop.current_iteration = iteration + 1
            loop.last_iteration_at = datetime.now()

            logger.info(f"📊 Iteración {loop.current_iteration}/{max_iterations} del bucle {loop_id}")

            # Análisis REAL
            analysis_result = await self._analyze_code_directory(target_dir, loop_type)
            
            # Encontrar mejoras REALES
            improvements = await self._find_improvements(analysis_result, loop_type)
            
            if not improvements:
                logger.info("No se encontraron más mejoras, finalizando bucle")
                break

            loop.improvements_found += len(improvements)
            self.improvements_found.extend(improvements)

            # Aplicar mejoras (limitado por seguridad)
            applied_count = 0
            for improvement in improvements[:self.safety_boundaries.get('max_files_per_iteration', 100)]:
                if await self._should_apply_improvement(improvement):
                    success = await self._apply_improvement(improvement)
                    if success:
                        loop.improvements_applied += 1
                        loop.improvements_successful += 1
                        applied_count += 1
                    else:
                        loop.improvements_applied += 1

            if applied_count == 0:
                logger.info("No se aplicaron mejoras en esta iteración")
                break

            logger.info(f"✅ Iteración {loop.current_iteration} completada: {applied_count} mejoras aplicadas")

        loop.active = False
        logger.info(f"🎉 Bucle {loop_id} completado")
        logger.info(f"   Mejoras encontradas: {loop.improvements_found}")
        logger.info(f"   Mejoras aplicadas: {loop.improvements_applied}")
        logger.info(f"   Mejoras exitosas: {loop.improvements_successful}")

        # Guardar estado
        await self._save_loop_state(loop)

        return loop

    async def _analyze_code_directory(self, directory: Path, analysis_type: str) -> Dict[str, Any]:
        """Analiza directorio de código REALMENTE usando AST"""
        logger.info(f"🔍 Analizando directorio: {directory}")

        analysis_result = {
            'files_analyzed': [],
            'code_issues': [],
            'performance_issues': [],
            'security_issues': [],
            'quality_metrics': {},
            'complexity_scores': {}
        }

        # Encontrar archivos Python
        python_files = list(directory.rglob('*.py'))
        
        # Filtrar archivos a ignorar
        ignore_patterns = ['__pycache__', '.git', 'venv', 'env', '.pytest_cache', 'node_modules']
        python_files = [
            f for f in python_files 
            if not any(pattern in str(f) for pattern in ignore_patterns)
        ]

        logger.info(f"   Encontrados {len(python_files)} archivos Python")

        for file_path in python_files:
            try:
                file_analysis = await self._analyze_python_file(file_path, analysis_type)
                analysis_result['files_analyzed'].append(str(file_path))
                
                # Agregar issues encontrados
                analysis_result['code_issues'].extend(file_analysis.get('code_issues', []))
                analysis_result['performance_issues'].extend(file_analysis.get('performance_issues', []))
                analysis_result['security_issues'].extend(file_analysis.get('security_issues', []))
                
                # Métricas
                file_metrics = file_analysis.get('metrics', {})
                analysis_result['quality_metrics'][str(file_path)] = file_metrics
                analysis_result['complexity_scores'][str(file_path)] = file_analysis.get('complexity_score', 0.0)

                self.metrics['total_files_analyzed'] += 1

            except Exception as e:
                logger.error(f"Error analizando {file_path}: {e}", exc_info=True)

        return analysis_result

    async def _analyze_python_file(self, file_path: Path, analysis_type: str) -> Dict[str, Any]:
        """Analiza un archivo Python REALMENTE usando AST"""
        result = {
            'file_path': str(file_path),
            'code_issues': [],
            'performance_issues': [],
            'security_issues': [],
            'metrics': {},
            'complexity_score': 0.0
        }

        try:
            code_content = file_path.read_text(encoding='utf-8')
            
            # Parsear AST
            try:
                tree = ast.parse(code_content)
            except SyntaxError as e:
                result['code_issues'].append({
                    'type': 'syntax_error',
                    'severity': 'critical',
                    'message': f"Error de sintaxis: {e.msg} en línea {e.lineno}",
                    'line': e.lineno,
                    'file_path': str(file_path)
                })
                return result

            # Análisis REAL de código
            issues, metrics, complexity = await self._analyze_ast_tree(tree, code_content, file_path)
            
            # Agregar file_path a todos los issues
            for issue in issues:
                issue['file_path'] = str(file_path)
            
            result['code_issues'].extend(issues)
            result['metrics'] = metrics
            result['complexity_score'] = complexity

            # Análisis de patrones problemáticos
            pattern_issues = await self._analyze_code_patterns(code_content, file_path)
            result['code_issues'].extend(pattern_issues)

            # Análisis de seguridad
            if analysis_type in ['code_analysis', 'security_hardening']:
                security_issues = await self._analyze_security(code_content, file_path)
                result['security_issues'].extend(security_issues)

            # Análisis de rendimiento
            if analysis_type in ['code_analysis', 'performance_optimization']:
                perf_issues = await self._analyze_performance(code_content, file_path, tree)
                result['performance_issues'].extend(perf_issues)

        except Exception as e:
            logger.error(f"Error analizando archivo {file_path}: {e}", exc_info=True)
            result['code_issues'].append({
                'type': 'analysis_error',
                'severity': 'medium',
                'message': f"Error durante análisis: {e}",
                'file_path': str(file_path)
            })

        return result

    async def _analyze_ast_tree(self, tree: ast.AST, code_content: str, file_path: Path) -> Tuple[List[Dict], Dict[str, Any], float]:
        """Analiza el AST del código REALMENTE"""
        issues = []
        metrics = {
            'total_lines': len(code_content.split('\n')),
            'total_functions': 0,
            'total_classes': 0,
            'max_function_length': 0,
            'max_class_length': 0,
            'nested_depth': 0,
            'imports': []
        }
        
        complexity_score = 0.0

        # Recorrer AST
        for node in ast.walk(tree):
            # Contar funciones
            if isinstance(node, ast.FunctionDef):
                metrics['total_functions'] += 1
                if hasattr(node, 'end_lineno') and node.end_lineno and node.lineno:
                    func_lines = node.end_lineno - node.lineno
                else:
                    func_lines = 50  # Estimación por defecto
                metrics['max_function_length'] = max(metrics['max_function_length'], func_lines)
                
                # Detectar funciones muy largas
                if func_lines > 100:
                    issues.append({
                        'type': 'long_function',
                        'severity': 'medium',
                        'message': f"Función '{node.name}' muy larga ({func_lines} líneas), considerar dividirla",
                        'line': node.lineno,
                        'function_name': node.name,
                        'file_path': str(file_path)
                    })

                # Calcular complejidad ciclomática básica
                complexity_score += self._calculate_function_complexity(node)

            # Contar clases
            if isinstance(node, ast.ClassDef):
                metrics['total_classes'] += 1
                if hasattr(node, 'end_lineno') and node.end_lineno and node.lineno:
                    class_lines = node.end_lineno - node.lineno
                else:
                    class_lines = 50  # Estimación por defecto
                metrics['max_class_length'] = max(metrics['max_class_length'], class_lines)

            # Detectar imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    metrics['imports'].append(node.module)
                else:
                    for alias in node.names:
                        metrics['imports'].append(alias.name)

        # Calcular profundidad de anidación
        metrics['nested_depth'] = self._calculate_max_nesting_depth(tree)

        if metrics['nested_depth'] > 5:
            issues.append({
                'type': 'deep_nesting',
                'severity': 'medium',
                'message': f"Anidación profunda detectada ({metrics['nested_depth']} niveles)",
                'line': 1,
                'file_path': str(file_path)
            })

        # Normalizar complejidad
        complexity_score = min(100.0, complexity_score / max(metrics['total_functions'], 1) * 10)

        return issues, metrics, complexity_score

    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calcula complejidad ciclomática básica de una función"""
        complexity = 1  # Base
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity += 1
            if isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity

    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calcula máxima profundidad de anidación"""
        max_depth = 0
        
        def visit(node: ast.AST, depth: int):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.Try, ast.With)):
                    visit(child, depth + 1)
                else:
                    visit(child, depth)
        
        visit(tree, 0)
        return max_depth

    async def _analyze_code_patterns(self, code_content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Analiza patrones problemáticos en el código"""
        issues = []
        lines = code_content.split('\n')

        # Patrón 1: print() en lugar de logging
        for i, line in enumerate(lines, 1):
            if re.search(r'\bprint\s*\(', line) and 'logging' not in line.lower():
                # Verificar que no sea un comentario
                stripped = line.strip()
                if not stripped.startswith('#'):
                    issues.append({
                        'type': 'print_instead_of_logging',
                        'severity': 'low',
                        'message': "Uso de print() en lugar de logging",
                        'line': i,
                        'code_snippet': line.strip()[:100],
                        'file_path': str(file_path)
                    })

        # Patrón 2: except Exception genérico
        for i, line in enumerate(lines, 1):
            if re.search(r'except\s+Exception\s*:', line):
                issues.append({
                    'type': 'generic_exception',
                    'severity': 'medium',
                    'message': "Captura genérica de Exception, ser más específico",
                    'line': i,
                    'code_snippet': line.strip()[:100],
                    'file_path': str(file_path)
                })

        # Patrón 3: TODO/FIXME/HACK sin resolver
        for i, line in enumerate(lines, 1):
            if re.search(r'\b(TODO|FIXME|HACK|XXX)\b', line, re.IGNORECASE):
                issues.append({
                    'type': 'unresolved_todo',
                    'severity': 'low',
                    'message': f"Marcador encontrado: {line.strip()[:100]}",
                    'line': i,
                    'code_snippet': line.strip()[:100],
                    'file_path': str(file_path)
                })

        # Patrón 4: Imports no utilizados (básico)
        # Esto requiere análisis más profundo que se puede mejorar

        return issues

    async def _analyze_security(self, code_content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Analiza problemas de seguridad REALES"""
        issues = []
        lines = code_content.split('\n')

        # Patrón 1: eval() o exec()
        for i, line in enumerate(lines, 1):
            if re.search(r'\beval\s*\(|\bexec\s*\(', line):
                issues.append({
                    'type': 'dangerous_eval',
                    'severity': 'high',
                    'message': "Uso de eval() o exec() puede ser peligroso",
                    'line': i,
                    'code_snippet': line.strip()[:100],
                    'file_path': str(file_path)
                })

        # Patrón 2: Hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']{20,}["\']'  # Tokens largos probablemente secretos
        ]
        
        for pattern in secret_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        'type': 'hardcoded_secret',
                        'severity': 'critical',
                        'message': "Posible secreto hardcodeado en el código",
                        'line': i,
                        'code_snippet': line.strip()[:50],  # No mostrar el secreto completo
                        'file_path': str(file_path)
                    })

        # Patrón 3: SQL injection vulnerable
        for i, line in enumerate(lines, 1):
            if re.search(r'execute\s*\(.*\+|execute\s*\(.*%', line):
                if 'sql' in line.lower() or 'query' in line.lower():
                    issues.append({
                        'type': 'sql_injection_risk',
                        'severity': 'high',
                        'message': "Posible riesgo de SQL injection",
                        'line': i,
                        'code_snippet': line.strip()[:100],
                        'file_path': str(file_path)
                    })

        return issues

    async def _analyze_performance(self, code_content: str, file_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analiza problemas de rendimiento REALES"""
        issues = []

        # Patrón 1: Loops con operaciones pesadas
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Detectar loops anidados
                nested_loops = sum(1 for n in ast.walk(node) if isinstance(n, (ast.For, ast.While)))
                if nested_loops > 3:
                    issues.append({
                        'type': 'nested_loops',
                        'severity': 'medium',
                        'message': f"Loop con {nested_loops} niveles de anidación puede ser lento",
                        'line': node.lineno,
                        'file_path': str(file_path)
                    })

        # Patrón 2: List comprehension vs loop (recomendación)
        # Esto es más complejo y requiere análisis más profundo

        return issues

    async def _find_improvements(self, analysis_result: Dict[str, Any], improvement_type: str) -> List[CodeImprovement]:
        """Encuentra mejoras aplicables basadas en análisis"""
        improvements = []

        # Procesar code_issues
        for issue in analysis_result.get('code_issues', []):
            improvement = await self._create_improvement_from_issue(issue, improvement_type)
            if improvement:
                improvements.append(improvement)

        # Procesar security_issues
        for issue in analysis_result.get('security_issues', []):
            improvement = await self._create_improvement_from_issue(issue, 'security')
            if improvement:
                improvements.append(improvement)

        # Procesar performance_issues
        for issue in analysis_result.get('performance_issues', []):
            improvement = await self._create_improvement_from_issue(issue, 'performance')
            if improvement:
                improvements.append(improvement)

        # Ordenar por severidad y confianza
        improvements.sort(key=lambda x: (
            ['low', 'medium', 'high', 'critical'].index(x.severity),
            -x.confidence_score
        ), reverse=True)

        return improvements

    async def _create_improvement_from_issue(self, issue: Dict[str, Any], improvement_type: str) -> Optional[CodeImprovement]:
        """Crea un CodeImprovement desde un issue encontrado"""
        issue_type = issue.get('type', 'unknown')
        file_path = issue.get('file_path', 'unknown')
        
        # Si no hay file_path, no podemos crear la mejora
        if file_path == 'unknown' or not file_path:
            logger.warning(f"Issue sin file_path: {issue_type}")
            return None
        
        # Verificar que el archivo existe
        if not Path(file_path).exists():
            logger.warning(f"Archivo no existe para mejora: {file_path}")
            return None
        
        # Generar código mejorado basado en el tipo de issue
        improved_code = None
        confidence = 0.5

        if issue_type == 'print_instead_of_logging':
            original_line = issue.get('code_snippet', '')
            if original_line:
                # Intentar convertir print() a logging
                improved_code = await self._fix_print_to_logging(original_line)
                confidence = 0.8

        elif issue_type == 'generic_exception':
            original_line = issue.get('code_snippet', '')
            if original_line:
                improved_code = await self._fix_generic_exception(original_line)
                confidence = 0.6
        
        # Solo crear mejora si hay código mejorado o si es un issue informativo
        if not improved_code and issue_type not in ['unresolved_todo', 'long_function', 'deep_nesting', 'nested_loops']:
            # No crear mejora si no podemos mejorarla automáticamente
            return None

        # Crear improvement
        improvement_id = hashlib.md5(
            f"{file_path}:{issue.get('line', 0)}:{issue_type}".encode()
        ).hexdigest()[:12]

        return CodeImprovement(
            improvement_id=improvement_id,
            file_path=file_path,
            improvement_type=improvement_type,
            description=issue.get('message', ''),
            severity=issue.get('severity', 'medium'),
            line_number=issue.get('line'),
            original_code=issue.get('code_snippet'),
            improved_code=improved_code,
            confidence_score=confidence
        )

    async def _fix_print_to_logging(self, original_line: str) -> Optional[str]:
        """Convierte print() a logging"""
        # Extraer el contenido del print
        match = re.search(r'print\s*\((.*?)\)', original_line)
        if not match:
            return None
        
        content = match.group(1)
        
        # Detectar nivel de log apropiado
        if 'error' in original_line.lower() or '❌' in original_line:
            log_level = 'logger.error'
        elif 'warning' in original_line.lower() or '⚠️' in original_line:
            log_level = 'logger.warning'
        elif 'info' in original_line.lower() or '✅' in original_line:
            log_level = 'logger.info'
        elif 'debug' in original_line.lower():
            log_level = 'logger.debug'
        else:
            log_level = 'logger.info'
        
        # Construir línea mejorada
        # Intentar preservar formato
        improved = f"{log_level}({content})"
        
        return improved

    async def _fix_generic_exception(self, original_line: str) -> Optional[str]:
        """Sugiere excepciones más específicas"""
        # Esto es una sugerencia, la aplicación real requiere más contexto
        if 'except Exception:' in original_line:
            improved = original_line.replace('except Exception:', 'except (ValueError, KeyError, TypeError):  # Especificar excepciones')
            return improved
        return None

    async def _should_apply_improvement(self, improvement: CodeImprovement) -> bool:
        """Determina si una mejora debe aplicarse basado en límites de seguridad"""
        # Verificar nivel de riesgo
        risk_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        max_risk = risk_levels.get(self.safety_boundaries.get('max_risk_level', 'medium'), 2)
        improvement_risk = risk_levels.get(improvement.severity, 2)

        if improvement_risk > max_risk:
            logger.warning(f"Mejora {improvement.improvement_id} rechazada por nivel de riesgo")
            return False

        # Verificar confianza
        if improvement.confidence_score < 0.5:
            logger.warning(f"Mejora {improvement.improvement_id} rechazada por baja confianza")
            return False

        # Verificar aprobación humana si es requerida
        if self.safety_boundaries.get('require_human_approval', False):
            # En una implementación completa, esto requeriría interacción
            logger.info(f"Mejora {improvement.improvement_id} requiere aprobación humana")
            return False

        return True

    async def _apply_improvement(self, improvement: CodeImprovement) -> bool:
        """Aplica una mejora REAL al código con validación y backup"""
        file_path = Path(improvement.file_path)
        backup_path = None  # Inicializar antes del try
        
        if not file_path.exists():
            logger.error(f"Archivo no encontrado: {file_path}")
            improvement.error_message = "File not found"
            return False

        try:
            # Crear backup si está habilitado
            if self.safety_boundaries.get('backup_before_modify', True):
                backup_path = await self._create_backup(file_path)
                logger.info(f"Backup creado: {backup_path}")

            # Leer archivo original
            original_content = file_path.read_text(encoding='utf-8')
            lines = original_content.split('\n')

            # Aplicar mejora
            if not improvement.line_number or not improvement.improved_code:
                logger.warning(f"Mejora {improvement.improvement_id} no tiene código mejorado o línea válida")
                improvement.error_message = "No improved code or line number"
                return False

            # Reemplazar línea específica
            line_idx = improvement.line_number - 1
            if line_idx < 0 or line_idx >= len(lines):
                logger.error(f"Índice de línea {improvement.line_number} fuera de rango en {file_path} (archivo tiene {len(lines)} líneas)")
                improvement.error_message = f"Line number {improvement.line_number} out of range"
                self.metrics['total_improvements_failed'] += 1
                return False

            original_line = lines[line_idx]
            
            # Aplicar reemplazo
            improved_content = None
            if improvement.improvement_type == 'print_instead_of_logging':
                # Asegurar que logging esté importado
                improved_content = await self._apply_logging_improvement(original_content, line_idx, improvement.improved_code)
                if not improved_content:
                    logger.error(f"No se pudo generar contenido mejorado para {file_path}")
                    improvement.error_message = "Failed to generate improved content"
                    self.metrics['total_improvements_failed'] += 1
                    return False
            else:
                # Reemplazo simple
                lines[line_idx] = improvement.improved_code
                improved_content = '\n'.join(lines)

            # Verificar que el contenido mejorado es válido
            if not improved_content or len(improved_content.strip()) == 0:
                logger.error(f"Contenido mejorado vacío para {file_path}")
                improvement.error_message = "Improved content is empty"
                self.metrics['total_improvements_failed'] += 1
                return False

            # Escribir archivo modificado
            file_path.write_text(improved_content, encoding='utf-8')
            logger.info(f"Mejora aplicada: {improvement.improvement_id} en {file_path}:{improvement.line_number}")

            # Validar modificación
            validation_passed = False
            if self.safety_boundaries.get('validate_after_modify', True):
                validation_passed = await self._validate_modification(file_path)

            if validation_passed:
                improvement.applied = True
                improvement.applied_at = datetime.now()
                improvement.success = True

                # Crear SelfModification record
                mod = SelfModification(
                    mod_id=improvement.improvement_id,
                    target_file=str(file_path),
                    modification_type=improvement.improvement_type,
                    description=improvement.description,
                    confidence_score=improvement.confidence_score,
                    risk_level=improvement.severity,
                    backup_path=str(backup_path) if backup_path else None,
                    reversible=True,
                    applied_at=datetime.now(),
                    success=True,
                    validation_passed=True,
                    rollback_available=backup_path is not None
                )
                self.self_modifications.append(mod)

                self.metrics['total_improvements_applied'] += 1
                self.metrics['total_improvements_successful'] += 1

                return True
            else:
                # Revertir si falla validación
                logger.warning(f"Validación falló, revirtiendo: {file_path}")
                if backup_path and backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                    logger.info(f"Archivo revertido desde backup")
                improvement.error_message = "Validation failed"
                self.metrics['total_improvements_failed'] += 1
                return False

        except Exception as e:
            logger.error(f"Error aplicando mejora {improvement.improvement_id}: {e}", exc_info=True)
            improvement.error_message = str(e)
            self.metrics['total_improvements_failed'] += 1
            
            # Revertir si hay backup
            if backup_path and backup_path.exists():
                shutil.copy2(backup_path, file_path)
                logger.info("Archivo revertido debido a error")

            return False

    async def _apply_logging_improvement(self, content: str, line_idx: int, improved_code: str) -> str:
        """Aplica mejora de logging asegurando que el import esté presente"""
        lines = content.split('\n')
        
        # Verificar si logging está importado
        has_logging_import = any('import logging' in line or 'from logging import' in line for line in lines[:20])
        has_logger = any('logger = logging.getLogger' in line for line in lines[:30])

        # Agregar import si falta
        if not has_logging_import:
            # Encontrar última línea de imports
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_end = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            lines.insert(import_end, 'import logging')
            import_end += 1
            line_idx += 1  # Ajustar índice

        # Agregar logger si falta (solo si agregamos el import)
        if not has_logger:
            # Buscar después de imports o después del import que acabamos de agregar
            insert_idx = import_end if not has_logging_import else None
            
            if insert_idx is None:
                # Buscar después de todos los imports
                for i in range(len(lines)):
                    if lines[i].strip().startswith(('import ', 'from ')):
                        insert_idx = i + 1
                    elif lines[i].strip() and not lines[i].strip().startswith('#'):
                        if insert_idx is None:
                            insert_idx = i
                        break
            
            if insert_idx is not None and insert_idx < len(lines):
                lines.insert(insert_idx, 'logger = logging.getLogger(__name__)')
                if insert_idx <= line_idx:
                    line_idx += 1  # Ajustar índice solo si se insertó antes de nuestra línea

        # Aplicar reemplazo
        if 0 <= line_idx < len(lines):
            lines[line_idx] = improved_code
        else:
            # Si el índice está fuera de rango después de los ajustes, mantener contenido original
            logger.warning(f"Índice de línea {line_idx} fuera de rango después de ajustes")
            # No aplicar cambio si el índice es inválido

        return '\n'.join(lines)

    async def _create_backup(self, file_path: Path) -> Path:
        """Crea backup de un archivo"""
        backup_dir = self.improvement_dir / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name

        shutil.copy2(file_path, backup_path)
        return backup_path

    async def _validate_modification(self, file_path: Path) -> bool:
        """Valida que la modificación no rompa el código"""
        try:
            # Intentar parsear el archivo
            code_content = file_path.read_text(encoding='utf-8')
            ast.parse(code_content)
            
            # Intentar compilar
            compile(code_content, str(file_path), 'exec')
            
            logger.info(f"Validación exitosa: {file_path}")
            return True

        except SyntaxError as e:
            logger.error(f"Error de sintaxis en {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validando {file_path}: {e}")
            return False

    async def _save_loop_state(self, loop: ImprovementLoop):
        """Guarda estado del bucle"""
        try:
            state_file = self.improvement_dir / f"loop_{loop.loop_id}.json"
            state = {
                'loop_id': loop.loop_id,
                'loop_type': loop.loop_type,
                'target_directory': loop.target_directory,
                'current_iteration': loop.current_iteration,
                'improvements_found': loop.improvements_found,
                'improvements_applied': loop.improvements_applied,
                'improvements_successful': loop.improvements_successful,
                'active': loop.active,
                'started_at': loop.started_at.isoformat(),
                'last_iteration_at': loop.last_iteration_at.isoformat() if loop.last_iteration_at else None
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Estado guardado: {state_file}")

        except Exception as e:
            logger.error(f"Error guardando estado: {e}", exc_info=True)

    def analyze_architecture(self) -> Dict[str, Any]:
        """Analiza arquitectura REAL del sistema"""
        logger.info("🔍 Analizando arquitectura del sistema...")

        analysis = {
            'system_resources': {},
            'code_metrics': self.metrics.copy(),
            'potential_improvements': []
        }

        # Análisis de recursos del sistema
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)
                
                analysis['system_resources'] = {
                    'memory_usage_percent': mem.percent,
                    'cpu_usage_percent': cpu,
                    'available_memory_gb': mem.available / (1024**3)
                }

                # Sugerencias basadas en recursos
                if mem.percent > 80:
                    analysis['potential_improvements'].append({
                        'type': 'memory_optimization',
                        'severity': 'medium',
                        'description': 'Uso de memoria alto, considerar optimización'
                    })
                
                if cpu > 70:
                    analysis['potential_improvements'].append({
                        'type': 'cpu_optimization',
                        'severity': 'medium',
                        'description': 'Uso de CPU alto, considerar paralelización'
                    })

            except Exception as e:
                logger.warning(f"Error analizando recursos: {e}")
        else:
            analysis['system_resources'] = {
                'status': 'unavailable',
                'message': 'psutil no disponible'
            }

        return analysis

    def calculate_improvement_score(self, metrics: Dict[str, Any]) -> float:
        """Calcula score de mejora REAL basado en métricas"""
        if not metrics:
            return 0.0

        # Validar métricas
        completion_rate = max(0.0, min(1.0, metrics.get('completion_rate', 0.0)))
        quality_delta = max(-1.0, min(1.0, metrics.get('quality_delta', 0.0)))
        efficiency_score = max(0.0, min(1.0, metrics.get('learning_rate', 0.0)))

        # Calcular score ponderado
        base_score = completion_rate * 0.4
        quality_score = (quality_delta + 1.0) / 2.0 * 0.3  # Normalizar a 0-1
        efficiency_score = efficiency_score * 0.3

        total_score = (base_score + quality_score + efficiency_score) * 100.0
        return min(100.0, max(0.0, total_score))

    def get_improvement_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas REALES de mejoras"""
        return {
            'metrics': self.metrics.copy(),
            'active_loops': len([l for l in self.improvement_loops.values() if l.active]),
            'total_loops': len(self.improvement_loops),
            'total_improvements_found': len(self.improvements_found),
            'total_modifications_applied': len([m for m in self.self_modifications if m.success]),
            'improvement_rate': (
                self.metrics['total_improvements_successful'] / 
                max(1, self.metrics['total_improvements_applied'])
            ) * 100.0 if self.metrics['total_improvements_applied'] > 0 else 0.0
        }


