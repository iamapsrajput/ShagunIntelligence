#!/usr/bin/env python3
"""
Shagun Intelligence Test Runner
Orchestrates and runs all tests with comprehensive reporting
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
import json


def run_command(cmd: str, description: str) -> tuple[bool, str]:
    """Run a command and return success status and output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stdout}")
        print(f"Error details: {e.stderr}")
        return False, e.stderr


def main():
    parser = argparse.ArgumentParser(description="Shagun Intelligence Test Runner")
    parser.add_argument(
        "--suite",
        choices=["unit", "integration", "performance", "validation", "all"],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage analysis"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed HTML report"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Ensure test directories exist
    test_dirs = ["tests/unit", "tests/integration", "tests/performance", "tests/validation", "tests/simulation"]
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create report directory
    report_dir = Path("test_reports")
    report_dir.mkdir(exist_ok=True)
    
    # Build base pytest command
    base_cmd = "pytest"
    
    if args.verbose:
        base_cmd += " -v"
    
    if args.parallel:
        base_cmd += " -n auto"
    
    if args.coverage:
        base_cmd += " --cov=app --cov=agents --cov=services --cov-report=html --cov-report=term"
    
    # Test results
    results = {
        "timestamp": datetime.now().isoformat(),
        "suite": args.suite,
        "tests": {}
    }
    
    # Define test suites
    test_suites = {
        "unit": [
            ("Unit Tests - Market Analyst", f"{base_cmd} tests/unit/test_market_analyst.py"),
            ("Unit Tests - Risk Manager", f"{base_cmd} tests/unit/test_risk_manager.py"),
            ("Unit Tests - Trade Executor", f"{base_cmd} tests/unit/test_trade_executor.py"),
        ],
        "integration": [
            ("Integration Tests - Historical Data", f"{base_cmd} tests/integration/test_historical_data_integration.py"),
        ],
        "performance": [
            ("Performance Benchmarks", f"{base_cmd} tests/performance/test_benchmarks.py"),
        ],
        "validation": [
            ("Risk Management Validation", f"{base_cmd} tests/validation/test_risk_management.py"),
        ],
        "simulation": [
            ("Paper Trading Simulation", f"{base_cmd} tests/simulation/test_paper_trading.py"),
        ]
    }
    
    # Determine which suites to run
    if args.suite == "all":
        suites_to_run = list(test_suites.keys())
    else:
        suites_to_run = [args.suite]
    
    # Run tests
    total_passed = 0
    total_failed = 0
    
    for suite_name in suites_to_run:
        if suite_name in test_suites:
            for test_name, test_cmd in test_suites[suite_name]:
                success, output = run_command(test_cmd, test_name)
                
                results["tests"][test_name] = {
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                }
                
                if success:
                    total_passed += 1
                else:
                    total_failed += 1
    
    # Generate summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"Total test suites run: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {(total_passed/(total_passed+total_failed)*100):.1f}%" if total_passed+total_failed > 0 else "N/A")
    
    # Save results
    results_file = report_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate HTML report if requested
    if args.report:
        print("\nGenerating HTML report...")
        report_cmd = f"python -m pytest tests/ --html={report_dir}/report.html --self-contained-html"
        success, _ = run_command(report_cmd, "HTML Report Generation")
        
        if success:
            print(f"HTML report generated: {report_dir}/report.html")
    
    # Show coverage report location
    if args.coverage:
        print(f"\nCoverage report generated: htmlcov/index.html")
    
    # Additional checks
    print(f"\n{'='*60}")
    print("ADDITIONAL CHECKS")
    print('='*60)
    
    # Code quality checks
    print("\nRunning code quality checks...")
    
    # Black formatting check
    run_command("black --check .", "Code Formatting Check (Black)")
    
    # Flake8 linting
    run_command("flake8 --max-line-length=100 --exclude=venv,__pycache__ .", "Linting (Flake8)")
    
    # Type checking with mypy
    run_command("mypy app/ agents/ services/ --ignore-missing-imports", "Type Checking (MyPy)")
    
    # Exit with appropriate code
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()