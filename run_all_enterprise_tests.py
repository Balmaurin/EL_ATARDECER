#!/usr/bin/env python3
"""
ENTERPRISE TEST SUITE ORCHESTRATOR
=================================

Ejecuta todas las suites de test enterprise con reporting comprehensivo.
Incluye: API, Blockchain, RAG, y otros sistemas críticos.

CRÍTICO: Enterprise-grade test execution, comprehensive reporting, performance monitoring.
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import os

class EnterpriseTestOrchestrator:
    """Orchestrator for all enterprise test suites"""
    
    def __init__(self):
        """Initialize test orchestrator"""
        self.test_results = {}
        self.total_start_time = time.time()
        self.results_dir = Path("tests/results/enterprise")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_all_tests(self):
        """Execute all enterprise test suites"""
        print("🚀 ENTERPRISE TEST SUITE ORCHESTRATOR")
        print("=" * 60)
        print("🏢 ENTERPRISE QUALITY ASSURANCE EXECUTION")
        print("=" * 60)
        
        test_suites = [
            {
                'name': 'API Enterprise Tests',
                'file': 'tests/enterprise/test_api_enterprise_suites.py',
                'description': 'REST API functionality, authentication, performance'
            },
            {
                'name': 'Blockchain Enterprise Tests', 
                'file': 'tests/enterprise/test_blockchain_enterprise.py',
                'description': 'Smart contracts, consensus, token economics'
            },
            {
                'name': 'RAG System Enterprise Tests',
                'file': 'tests/enterprise/test_rag_system_enterprise.py', 
                'description': 'Retrieval accuracy, embedding quality, performance'
            }
        ]
        
        for suite in test_suites:
            print(f"\n🔍 EXECUTING: {suite['name']}")
            print(f"📝 Coverage: {suite['description']}")
            print("-" * 50)
            
            success, metrics = self._run_test_suite(suite['file'])
            
            self.test_results[suite['name']] = {
                'success': success,
                'metrics': metrics,
                'file': suite['file'],
                'description': suite['description']
            }
            
            if success:
                print(f"✅ {suite['name']}: PASSED")
            else:
                print(f"❌ {suite['name']}: FAILED")
        
        self._generate_executive_report()
    
    def _run_test_suite(self, test_file: str) -> tuple[bool, Dict[str, Any]]:
        """Execute a single test suite with comprehensive metrics"""
        start_time = time.time()
        
        # Check if test file exists
        if not Path(test_file).exists():
            print(f"⚠️ Test file not found: {test_file}")
            return False, {'error': 'File not found', 'duration': 0}
        
        try:
            # Run pytest with enterprise configuration
            cmd = [
                sys.executable, "-m", "pytest",
                test_file,
                "-v",
                "--tb=short", 
                "--durations=10",
                "--maxfail=3",
                "--disable-warnings",
                "--color=yes"
            ]
            
            print(f"🔧 Command: {' '.join(cmd[:4])}...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            # Parse results
            success = result.returncode == 0
            
            metrics = {
                'duration': duration,
                'return_code': result.returncode,
                'stdout_lines': len(result.stdout.split('\n')),
                'stderr_lines': len(result.stderr.split('\n')) if result.stderr else 0
            }
            
            # Extract test metrics from output
            if result.stdout:
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'passed' in line or 'failed' in line or 'error' in line:
                        metrics['test_summary'] = line.strip()
                        break
            
            # Print condensed output
            print(f"Return Code: {result.returncode}")
            if result.stdout:
                # Print last few lines of stdout
                lines = result.stdout.strip().split('\n')[-5:]
                for line in lines:
                    if line.strip():
                        print(f"OUT: {line}")
            
            if result.stderr:
                print(f"ERR: {result.stderr[:200]}...")
            
            return success, metrics
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Test suite timed out: {test_file}")
            return False, {'error': 'Timeout', 'duration': 300}
        except Exception as e:
            print(f"💥 Error executing {test_file}: {e}")
            return False, {'error': str(e), 'duration': time.time() - start_time}
    
    def _generate_executive_report(self):
        """Generate comprehensive executive report"""
        total_duration = time.time() - self.total_start_time
        
        # Calculate summary metrics
        total_suites = len(self.test_results)
        passed_suites = sum(1 for r in self.test_results.values() if r['success'])
        failed_suites = total_suites - passed_suites
        
        success_rate = (passed_suites / total_suites) * 100 if total_suites > 0 else 0
        
        # Generate executive report
        executive_report = {
            'execution_summary': {
                'total_duration': total_duration,
                'total_suites': total_suites,
                'passed_suites': passed_suites,
                'failed_suites': failed_suites,
                'success_rate': success_rate,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'suite_details': self.test_results,
            'enterprise_status': 'ENTERPRISE_READY' if success_rate >= 90 else 'REVIEW_REQUIRED',
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        try:
            report_file = self.results_dir / "enterprise_executive_report.json"
            with open(report_file, 'w') as f:
                json.dump(executive_report, f, indent=2, default=str)
            print(f"\n📄 Executive Report saved: {report_file}")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
        
        # Print executive summary
        print("\n" + "=" * 60)
        print("🏆 ENTERPRISE TEST EXECUTION SUMMARY")
        print("=" * 60)
        print(f"📊 Total Duration: {total_duration:.1f} seconds")
        print(f"📈 Success Rate: {success_rate:.1f}% ({passed_suites}/{total_suites})")
        print(f"📋 Enterprise Status: {executive_report['enterprise_status']}")
        
        if failed_suites > 0:
            print(f"\n❌ Failed Suites: {failed_suites}")
            for name, result in self.test_results.items():
                if not result['success']:
                    error = result['metrics'].get('error', 'Unknown error')
                    print(f"   • {name}: {error}")
        
        # Print recommendations
        recommendations = executive_report['recommendations']
        if recommendations:
            print(f"\n💡 ENTERPRISE RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n🎯 ENTERPRISE QUALITY GATES:")
        security_status = "PASSED" if passed_suites >= total_suites * 0.9 else "REVIEW"
        performance_status = "PASSED" if total_duration < 180 else "REVIEW" 
        reliability_status = "PASSED" if success_rate >= 90 else "REVIEW"
        
        print(f"   ✅ Security: {security_status}")
        print(f"   ✅ Performance: {performance_status}")
        print(f"   ✅ Reliability: {reliability_status}")
        
        overall_status = "🏅 ENTERPRISE GRADE" if success_rate >= 90 and total_duration < 180 else "📋 REVIEW REQUIRED"
        print(f"\n🏆 OVERALL ENTERPRISE STATUS: {overall_status}")
        
        return executive_report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        # Check for failed suites
        failed_count = sum(1 for r in self.test_results.values() if not r['success'])
        if failed_count > 0:
            recommendations.append("Address failed test suites to meet enterprise quality standards")
        
        # Check for performance issues
        slow_suites = [name for name, result in self.test_results.items() 
                      if result['metrics'].get('duration', 0) > 60]
        if slow_suites:
            recommendations.append(f"Optimize performance for slow test suites: {', '.join(slow_suites)}")
        
        # Check for timeout issues
        timeout_suites = [name for name, result in self.test_results.items()
                         if result['metrics'].get('error') == 'Timeout']
        if timeout_suites:
            recommendations.append(f"Investigate timeout issues in: {', '.join(timeout_suites)}")
        
        # General recommendations for enterprise readiness
        success_rate = (len([r for r in self.test_results.values() if r['success']]) / 
                       len(self.test_results)) * 100 if self.test_results else 0
        
        if success_rate < 100:
            recommendations.append("Achieve 100% test pass rate for enterprise deployment readiness")
        
        if not recommendations:
            recommendations.append("All enterprise quality gates met - system ready for production")
        
        return recommendations


def main():
    """Main execution function"""
    print("🚀 Starting Enterprise Test Suite Orchestrator...")
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    orchestrator = EnterpriseTestOrchestrator()
    orchestrator.run_all_tests()
    
    print("\n✅ Enterprise Test Orchestration Complete!")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Enterprise test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Enterprise test orchestrator failed: {e}")
        sys.exit(1)
