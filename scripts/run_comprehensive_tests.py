#!/usr/bin/env python3
"""
Comprehensive Test Runner for Shagun Intelligence Trading Platform
Runs all tests with coverage reporting and performance benchmarks
"""

import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Comprehensive test runner with coverage and reporting"""

    def __init__(self):
        self.project_root = project_root
        self.test_results: dict[str, dict] = {}
        self.total_start_time = time.time()

    def run_command(
        self, command: list[str], description: str
    ) -> tuple[bool, str, str]:
        """Run a command and capture output"""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(command)}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            print(f"Duration: {duration:.2f}s")
            print(f"Status: {'✅ PASSED' if success else '❌ FAILED'}")

            if result.stdout:
                print(f"\nSTDOUT:\n{result.stdout}")

            if result.stderr and not success:
                print(f"\nSTDERR:\n{result.stderr}")

            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            print("❌ TIMEOUT after 5 minutes")
            return False, "", "Command timed out"
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False, "", str(e)

    def run_unit_tests(self) -> bool:
        """Run unit tests with coverage"""
        command = [
            "python",
            "-m",
            "pytest",
            "tests/unit/",
            "-v",
            "--cov=app",
            "--cov=agents",
            "--cov=services",
            "--cov-report=html:reports/coverage_unit",
            "--cov-report=term-missing",
            "--cov-fail-under=70",
            "--junit-xml=reports/junit_unit.xml",
        ]

        success, stdout, stderr = self.run_command(command, "Unit Tests")

        self.test_results["unit_tests"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
        }

        return success

    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        command = [
            "python",
            "-m",
            "pytest",
            "tests/integration/",
            "-v",
            "--cov=app",
            "--cov-append",
            "--cov-report=html:reports/coverage_integration",
            "--cov-report=term-missing",
            "--junit-xml=reports/junit_integration.xml",
        ]

        success, stdout, stderr = self.run_command(command, "Integration Tests")

        self.test_results["integration_tests"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
        }

        return success

    def run_performance_tests(self) -> bool:
        """Run performance tests"""
        command = [
            "python",
            "-m",
            "pytest",
            "tests/performance/",
            "-v",
            "-s",  # Don't capture output for performance metrics
            "--junit-xml=reports/junit_performance.xml",
        ]

        success, stdout, stderr = self.run_command(command, "Performance Tests")

        self.test_results["performance_tests"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
        }

        return success

    def run_validation_tests(self) -> bool:
        """Run validation tests"""
        command = [
            "python",
            "-m",
            "pytest",
            "tests/validation/",
            "-v",
            "--junit-xml=reports/junit_validation.xml",
        ]

        success, stdout, stderr = self.run_command(command, "Validation Tests")

        self.test_results["validation_tests"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
        }

        return success

    def run_linting(self) -> bool:
        """Run code linting"""
        command = ["python", "-m", "ruff", "check", ".", "--output-format=json"]

        success, stdout, stderr = self.run_command(command, "Code Linting (Ruff)")

        self.test_results["linting"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
        }

        return success

    def run_type_checking(self) -> bool:
        """Run type checking"""
        command = ["python", "-m", "mypy", "app/", "--ignore-missing-imports"]

        success, stdout, stderr = self.run_command(command, "Type Checking (MyPy)")

        self.test_results["type_checking"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
        }

        return success

    def run_security_scan(self) -> bool:
        """Run security scanning"""
        command = [
            "python",
            "-m",
            "bandit",
            "-r",
            "app/",
            "-f",
            "json",
            "-o",
            "reports/bandit_report.json",
        ]

        success, stdout, stderr = self.run_command(command, "Security Scan (Bandit)")

        self.test_results["security_scan"] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
        }

        return success

    def generate_coverage_report(self) -> bool:
        """Generate final coverage report"""
        command = ["python", "-m", "coverage", "html", "-d", "reports/coverage_final"]

        success, stdout, stderr = self.run_command(command, "Final Coverage Report")

        return success

    def create_reports_directory(self):
        """Create reports directory"""
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        print(f"Created reports directory: {reports_dir}")

    def print_summary(self):
        """Print test summary"""
        total_duration = time.time() - self.total_start_time

        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total Duration: {total_duration:.2f}s")
        print()

        passed = 0
        failed = 0

        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title():<20} {status}")

            if result["success"]:
                passed += 1
            else:
                failed += 1

        print(f"\n{'='*80}")
        print(f"OVERALL RESULT: {passed} passed, {failed} failed")

        if failed == 0:
            print("🎉 ALL TESTS PASSED! System is ready for production.")
        else:
            print("⚠️  Some tests failed. Please review the results above.")

        print(f"{'='*80}")

        # Print report locations
        print(f"\nReports generated in: {self.project_root}/reports/")
        print("- coverage_final/index.html - Coverage report")
        print("- junit_*.xml - JUnit test results")
        print("- bandit_report.json - Security scan results")

    def run_all_tests(self) -> bool:
        """Run all tests and generate reports"""
        print(
            "🚀 Starting comprehensive test suite for Shagun Intelligence Trading Platform"
        )

        # Create reports directory
        self.create_reports_directory()

        # Run all test suites
        test_suites = [
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Validation Tests", self.run_validation_tests),
            ("Code Linting", self.run_linting),
            ("Type Checking", self.run_type_checking),
            ("Security Scan", self.run_security_scan),
        ]

        all_passed = True

        for suite_name, suite_func in test_suites:
            try:
                success = suite_func()
                if not success:
                    all_passed = False
            except Exception as e:
                print(f"❌ Error running {suite_name}: {e}")
                all_passed = False

        # Generate final coverage report
        self.generate_coverage_report()

        # Print summary
        self.print_summary()

        return all_passed


def main():
    """Main entry point"""
    runner = TestRunner()
    success = runner.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
