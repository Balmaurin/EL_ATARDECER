#!/usr/bin/env python3
"""
ENTERPRISE PROJECT AUDITOR
==========================

Auditoría completa del proyecto enterprise con análisis de:
- Cobertura de tests
- Calidad de código
- Seguridad
- Performance
- Compliance enterprise

CRÍTICO: Full project assessment, security analysis, enterprise readiness.
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import os
import ast
import re

class EnterpriseProjectAuditor:
    """Comprehensive enterprise project auditor"""
    
    def __init__(self):
        self.audit_start_time = time.time()
        self.project_root = Path(".")
        self.results_dir = Path("audit_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.audit_results = {
            'project_info': {},
            'code_quality': {},
            'test_coverage': {},
            'security_analysis': {},
            'performance_metrics': {},
            'enterprise_compliance': {},
            'recommendations': []
        }
    
    def run_full_audit(self):
        """Execute comprehensive enterprise audit"""
        print("🔍 ENTERPRISE PROJECT AUDITOR")
        print("=" * 60)
        print("🏢 COMPREHENSIVE QUALITY ASSURANCE AUDIT")
        print("=" * 60)
        
        audit_phases = [
            ("📊 Project Structure Analysis", self._audit_project_structure),
            ("🧪 Test Suite Execution", self._audit_test_suites),
            ("📈 Code Quality Analysis", self._audit_code_quality),
            ("🔒 Security Assessment", self._audit_security),
            ("⚡ Performance Analysis", self._audit_performance),
            ("📋 Enterprise Compliance", self._audit_enterprise_compliance)
        ]
        
        for phase_name, phase_func in audit_phases:
            print(f"\n{phase_name}")
            print("-" * 50)
            try:
                phase_func()
                print(f"✅ {phase_name}: COMPLETED")
            except Exception as e:
                print(f"❌ {phase_name}: FAILED - {e}")
                self.audit_results['recommendations'].append(f"Fix {phase_name}: {e}")
        
        self._generate_executive_audit_report()

    def _generate_executive_audit_report(self):
        """Generate and print the final executive audit report."""
        print("\n" + "=" * 60)
        print("📝 GENERATING EXECUTIVE AUDIT REPORT")
        print("=" * 60)

        # Calculate scores
        scores = {
            'test_coverage': self.audit_results['test_coverage'].get('test_success_rate', 0),
            'code_quality': self.audit_results['code_quality'].get('documentation_ratio', 0),
            'security': self.audit_results['security_analysis'].get('security_score', 0),
            'performance': 100 if self.audit_results['performance_metrics'].get('performance_grade') == 'A' else 60,
            'compliance': self.audit_results['enterprise_compliance'].get('compliance_score', 0)
        }

        # Calculate overall score (weighted average)
        weights = {'test_coverage': 0.25, 'code_quality': 0.15, 'security': 0.3, 'performance': 0.1, 'compliance': 0.2}
        overall_score = sum(scores[cat] * weights[cat] for cat in scores)

        # Generate recommendations and identify critical issues
        self.audit_results['recommendations'].extend(self._generate_audit_recommendations(scores))
        critical_issues = self._identify_critical_issues()

        # Prepare summary
        audit_duration = time.time() - self.audit_start_time
        summary = {
            'overall_score': overall_score,
            'audit_duration': audit_duration,
            'enterprise_readiness': self.audit_results['enterprise_compliance'].get('enterprise_ready', False),
            'individual_scores': scores,
            'critical_issues': critical_issues
        }

        # Finalize audit results
        self.audit_results['executive_summary'] = summary
        
        # Save detailed report
        report_file = self.results_dir / "enterprise_audit_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.audit_results, f, indent=4)

        # Print summary to console
        self._print_executive_summary(summary, report_file)
    
    def _audit_project_structure(self):
        """Analyze project structure and organization"""
        print("📁 Analyzing project structure...")
        
        # Count files by type
        file_counts = {}
        total_lines = 0
        test_files = []
        source_files = []
        
        for file_path in self.project_root.rglob("*.py"):
            if file_path.is_file():
                file_type = "test" if "test" in str(file_path) else "source"
                file_counts[file_type] = file_counts.get(file_type, 0) + 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        
                        if file_type == "test":
                            test_files.append({'path': str(file_path), 'lines': lines})
                        else:
                            source_files.append({'path': str(file_path), 'lines': lines})
                except:
                    pass
        
        # Analyze directory structure
        directories = []
        for dir_path in self.project_root.rglob("*"):
            if dir_path.is_dir() and not str(dir_path).startswith('.'):
                directories.append(str(dir_path))
        
        self.audit_results['project_info'] = {
            'total_python_files': file_counts.get('source', 0) + file_counts.get('test', 0),
            'source_files': file_counts.get('source', 0),
            'test_files': file_counts.get('test', 0),
            'total_lines_of_code': total_lines,
            'directories': len(directories),
            'test_to_source_ratio': file_counts.get('test', 0) / max(file_counts.get('source', 1), 1),
            'enterprise_test_suites': len([f for f in test_files if 'enterprise' in f['path']]),
            'largest_files': sorted(source_files + test_files, key=lambda x: x['lines'], reverse=True)[:5]
        }
        
        print(f"📊 Python files: {self.audit_results['project_info']['total_python_files']}")
        print(f"📊 Test coverage ratio: {self.audit_results['project_info']['test_to_source_ratio']:.2f}")
    
    def _audit_test_suites(self):
        """Execute and analyze all test suites"""
        print("🧪 Executing enterprise test suites...")
        
        test_suites = [
            'tests/enterprise/test_api_enterprise_suites.py',
            'tests/enterprise/test_blockchain_enterprise.py', 
            'tests/enterprise/test_rag_system_enterprise.py'
        ]
        
        suite_results = {}
        total_tests = 0
        passed_tests = 0
        
        for suite_file in test_suites:
            if not Path(suite_file).exists():
                print(f"⚠️ Test suite not found: {suite_file}")
                continue
            
            print(f"🔍 Executing: {suite_file}")
            
            start_time = time.time()
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest",
                    suite_file,
                    "-v",
                    "--tb=short",
                    "--disable-warnings"
                ], capture_output=True, text=True, timeout=180)
                
                duration = time.time() - start_time
                success = result.returncode == 0
                
                # Parse test results
                output_lines = result.stdout.split('\n')
                test_count = self._count_tests_in_output(output_lines)
                
                suite_results[suite_file] = {
                    'success': success,
                    'duration': duration,
                    'test_count': test_count,
                    'return_code': result.returncode
                }
                
                if success:
                    passed_tests += test_count
                total_tests += test_count
                
                print(f"{'✅' if success else '❌'} {suite_file}: {test_count} tests in {duration:.1f}s")
                
            except subprocess.TimeoutExpired:
                suite_results[suite_file] = {'success': False, 'error': 'timeout'}
                print(f"⏰ {suite_file}: TIMEOUT")
            except Exception as e:
                suite_results[suite_file] = {'success': False, 'error': str(e)}
                print(f"💥 {suite_file}: ERROR - {e}")
        
        self.audit_results['test_coverage'] = {
            'total_test_suites': len(test_suites),
            'executed_suites': len([r for r in suite_results.values() if 'error' not in r]),
            'passed_suites': len([r for r in suite_results.values() if r.get('success', False)]),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'test_success_rate': (passed_tests / max(total_tests, 1)) * 100,
            'suite_results': suite_results
        }
    
    def _count_tests_in_output(self, output_lines: List[str]) -> int:
        """Count tests from pytest output"""
        for line in output_lines:
            if 'passed' in line or 'failed' in line:
                # Look for patterns like "12 passed" or "5 failed, 7 passed"
                # Use raw string for regex pattern
                match = re.search(r'(\d+)\s+(?:passed|failed)', line)
                if match:
                    try:
                        return int(match.group(1))
                    except ValueError:
                        continue
        return 0
    
    def _audit_code_quality(self):
        """Analyze code quality metrics"""
        print("📈 Analyzing code quality...")
        
        quality_metrics = {
            'complexity_score': 0,
            'documentation_ratio': 0,
            'enterprise_patterns': 0,
            'code_smells': []
        }
        
        python_files = list(self.project_root.rglob("*.py"))
        total_functions = 0
        documented_functions = 0
        complex_functions = 0
        enterprise_patterns_found = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Parse AST for analysis
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        # Count functions and documentation
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                            
                            # Check complexity (simplified)
                            if len(node.body) > 20:
                                complex_functions += 1
                        
                        # Check for enterprise patterns
                        if isinstance(node, ast.ClassDef):
                            if 'Enterprise' in node.name or 'Test' in node.name:
                                enterprise_patterns_found += 1
                
                except SyntaxError:
                    quality_metrics['code_smells'].append(f"Syntax error in {file_path}")
                    
            except Exception as e:
                quality_metrics['code_smells'].append(f"Could not analyze {file_path}: {e}")
        
        quality_metrics.update({
            'complexity_score': (complex_functions / max(total_functions, 1)) * 100,
            'documentation_ratio': (documented_functions / max(total_functions, 1)) * 100,
            'enterprise_patterns': enterprise_patterns_found,
            'total_functions': total_functions,
            'documented_functions': documented_functions
        })
        
        self.audit_results['code_quality'] = quality_metrics
        print(f"📊 Documentation ratio: {quality_metrics['documentation_ratio']:.1f}%")
        print(f"📊 Enterprise patterns: {enterprise_patterns_found}")
    
    def _audit_security(self):
        """Perform security analysis"""
        print("🔒 Performing security analysis...")
        
        security_issues = []
        sensitive_patterns = [
            'password', 'secret', 'key', 'token', 'api_key',
            'private_key', 'SECRET', 'PASSWORD', 'TOKEN'
        ]
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for line_num, line in enumerate(lines, 1):
                        # Check for hardcoded secrets (excluding tests and comments)
                        if any(pattern in line for pattern in sensitive_patterns):
                            if not line.strip().startswith('#') and 'test' not in str(file_path).lower():
                                if '=' in line and not line.strip().startswith('def'):
                                    security_issues.append({
                                        'file': str(file_path),
                                        'line': line_num,
                                        'issue': 'Potential hardcoded secret',
                                        'severity': 'medium'
                                    })
                        
                        # Check for SQL injection patterns using raw string
                        if re.search(r'execute\([^)]*\+', line):
                            security_issues.append({
                                'file': str(file_path),
                                'line': line_num,
                                'issue': 'Potential SQL injection',
                                'severity': 'high'
                            })
                            
            except Exception:
                pass
        
        # Check for security test coverage
        security_test_files = [f for f in python_files if 'security' in str(f).lower()]
        
        self.audit_results['security_analysis'] = {
            'total_security_issues': len(security_issues),
            'high_severity_issues': len([i for i in security_issues if i['severity'] == 'high']),
            'medium_severity_issues': len([i for i in security_issues if i['severity'] == 'medium']),
            'security_test_files': len(security_test_files),
            'issues': security_issues[:10],  # Top 10 issues
            'security_score': max(0, 100 - len(security_issues) * 5)
        }
        
        print(f"🔒 Security issues found: {len(security_issues)}")
        print(f"🔒 Security score: {self.audit_results['security_analysis']['security_score']}/100")
    
    def _audit_performance(self):
        """Analyze performance characteristics"""
        print("⚡ Analyzing performance...")
        
        # Run a performance test of the test suites
        performance_results = {}
        
        test_file = "tests/enterprise/test_api_enterprise_suites.py"
        if Path(test_file).exists():
            start_time = time.time()
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest",
                    test_file,
                    "--durations=0",
                    "-q"
                ], capture_output=True, text=True, timeout=60)
                
                execution_time = time.time() - start_time
                performance_results[test_file] = {
                    'execution_time': execution_time,
                    'success': result.returncode == 0
                }
                
            except subprocess.TimeoutExpired:
                performance_results[test_file] = {'execution_time': 60, 'timeout': True}
        
        self.audit_results['performance_metrics'] = {
            'test_execution_speed': performance_results,
            'average_test_time': sum(r.get('execution_time', 0) for r in performance_results.values()) / max(len(performance_results), 1),
            'performance_grade': 'A' if all(r.get('execution_time', 999) < 30 for r in performance_results.values()) else 'B'
        }
        
        print(f"⚡ Average test time: {self.audit_results['performance_metrics']['average_test_time']:.1f}s")
    
    def _audit_enterprise_compliance(self):
        """Check enterprise compliance requirements"""
        print("📋 Checking enterprise compliance...")
        
        compliance_checks = {
            'has_enterprise_tests': len(list(self.project_root.rglob("*enterprise*.py"))) >= 3,
            'has_security_tests': len([f for f in self.project_root.rglob("*.py") if 'security' in str(f).lower()]) > 0,
            'has_performance_tests': len([f for f in self.project_root.rglob("*.py") if 'performance' in str(f).lower()]) > 0,
            'has_documentation': len(list(self.project_root.rglob("*.md"))) > 0,
            'test_coverage_adequate': self.audit_results.get('test_coverage', {}).get('test_success_rate', 0) >= 80,
            'security_score_adequate': self.audit_results.get('security_analysis', {}).get('security_score', 0) >= 80,
            'code_quality_adequate': self.audit_results.get('code_quality', {}).get('documentation_ratio', 0) >= 50
        }
        
        compliance_score = (sum(compliance_checks.values()) / len(compliance_checks)) * 100
        
        self.audit_results['enterprise_compliance'] = {
            'individual_checks': compliance_checks,
            'compliance_score': compliance_score,
            'enterprise_ready': compliance_score >= 85
        }
        
        print(f"📋 Enterprise compliance: {compliance_score:.1f}%")

    def _generate_audit_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if scores['test_coverage'] < 90:
            recommendations.append("Improve test coverage to achieve >90% pass rate")
        
        if scores['code_quality'] < 70:
            recommendations.append("Increase code documentation and reduce complexity")
        
        if scores['security'] < 90:
            recommendations.append("Address security vulnerabilities and add security tests")
        
        if scores['compliance'] < 85:
            recommendations.append("Meet enterprise compliance requirements")
        
        return recommendations

    def _identify_critical_issues(self) -> List[str]:
        """Identify critical issues requiring immediate attention"""
        critical_issues = []
        
        # High severity security issues
        high_security = self.audit_results['security_analysis'].get('high_severity_issues', 0)
        if high_security > 0:
            critical_issues.append(f"{high_security} high-severity security issues found")
        
        # Failed test suites
        failed_suites = len([r for r in self.audit_results['test_coverage']['suite_results'].values() 
                           if not r.get('success', False)])
        if failed_suites > 0:
            critical_issues.append(f"{failed_suites} test suites failing")
        
        # Low enterprise compliance
        compliance_score = self.audit_results['enterprise_compliance'].get('compliance_score', 100)
        if compliance_score < 75:
            critical_issues.append("Enterprise compliance below minimum threshold")
        
        return critical_issues

    def _print_executive_summary(self, summary: Dict, report_file: Path):
        """Print formatted executive summary"""
        print("\n" + "=" * 60)
        print("🏆 ENTERPRISE PROJECT AUDIT SUMMARY")
        print("=" * 60)
        
        print(f"📊 Overall Score: {summary['overall_score']:.1f}/100")
        print(f"⏱️  Audit Duration: {summary['audit_duration']:.1f}s")
        print(f"🏢 Enterprise Ready: {'YES' if summary['enterprise_readiness'] else 'NO'}")
        
        print(f"\n📈 DETAILED SCORES:")
        for category, score in summary['individual_scores'].items():
            status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            print(f"   {status} {category.title()}: {score:.1f}/100")
        
        if summary['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES:")
            for i, issue in enumerate(summary['critical_issues'], 1):
                print(f"   {i}. {issue}")
        
        if self.audit_results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.audit_results['recommendations'][:5], 1):
                print(f"   {i}. {rec}")
        
        # Final assessment
        if summary['overall_score'] >= 85:
            status = "🏅 ENTERPRISE GRADE"
        elif summary['overall_score'] >= 70:
            status = "📋 REVIEW REQUIRED"
        else:
            status = "🔧 MAJOR IMPROVEMENTS NEEDED"
        
        print(f"\n🎯 FINAL ASSESSMENT: {status}")
        print(f"📄 Detailed Report: {report_file}")
        
        return summary


def main():
    """Execute enterprise project audit"""
    auditor = EnterpriseProjectAuditor()
    auditor.run_full_audit()
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Audit interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Audit failed: {e}")
        sys.exit(1)
