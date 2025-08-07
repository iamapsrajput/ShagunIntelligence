import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template


class TestReporter:
    """Automated test reporting system for Shagun Intelligence"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.test_results = []
        self.performance_metrics = {}
        self.coverage_data = {}
        self.timestamp = datetime.now()

    def add_test_result(self, result: dict[str, Any]):
        """Add a test result to the report"""
        self.test_results.append({**result, "timestamp": datetime.now().isoformat()})

    def add_performance_metric(self, name: str, value: Any, unit: str = ""):
        """Add a performance metric to the report"""
        if name not in self.performance_metrics:
            self.performance_metrics[name] = []

        self.performance_metrics[name].append(
            {"value": value, "unit": unit, "timestamp": datetime.now().isoformat()}
        )

    def add_coverage_data(self, module: str, coverage_percent: float):
        """Add code coverage data"""
        self.coverage_data[module] = coverage_percent

    def generate_html_report(self):
        """Generate comprehensive HTML test report"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Shagun Intelligence Test Report - {{ timestamp }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .metrics { display: flex; justify-content: space-around; margin: 20px 0; }
        .metric-card {
            background: #fff;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            min-width: 150px;
        }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { color: #666; margin-top: 5px; }
        .test-results { margin: 20px 0; }
        .test-pass { color: green; }
        .test-fail { color: red; }
        .test-skip { color: orange; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .chart { margin: 20px 0; text-align: center; }
        .coverage-bar {
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            height: 20px;
            margin: 5px 0;
        }
        .coverage-fill {
            background-color: #4CAF50;
            height: 100%;
            text-align: center;
            line-height: 20px;
            color: white;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Shagun Intelligence Test Report</h1>
        <p>Generated: {{ timestamp }}</p>
        <p>Duration: {{ duration }}</p>
    </div>

    <div class="summary">
        <h2>Test Summary</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{{ total_tests }}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value test-pass">{{ passed_tests }}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value test-fail">{{ failed_tests }}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value test-skip">{{ skipped_tests }}</div>
                <div class="metric-label">Skipped</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ pass_rate }}%</div>
                <div class="metric-label">Pass Rate</div>
            </div>
        </div>
    </div>

    <div class="coverage">
        <h2>Code Coverage</h2>
        <table>
            <tr>
                <th>Module</th>
                <th>Coverage</th>
                <th>Visual</th>
            </tr>
            {% for module, coverage in coverage_data.items() %}
            <tr>
                <td>{{ module }}</td>
                <td>{{ "%.1f" | format(coverage) }}%</td>
                <td>
                    <div class="coverage-bar">
                        <div class="coverage-fill" style="width: {{ coverage }}%">
                            {{ "%.1f" | format(coverage) }}%
                        </div>
                    </div>
                </td>
            </tr>
            {% endfor %}
        </table>
        <p><strong>Overall Coverage: {{ "%.1f" | format(overall_coverage) }}%</strong></p>
    </div>

    <div class="performance">
        <h2>Performance Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Unit</th>
            </tr>
            {% for name, metrics in performance_metrics.items() %}
            <tr>
                <td>{{ name }}</td>
                <td>{{ "%.3f" | format(metrics[-1].value) }}</td>
                <td>{{ metrics[-1].unit }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="test-results">
        <h2>Detailed Test Results</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Module</th>
                <th>Status</th>
                <th>Duration (s)</th>
                <th>Message</th>
            </tr>
            {% for test in test_results %}
            <tr>
                <td>{{ test.name }}</td>
                <td>{{ test.module }}</td>
                <td class="test-{{ test.status }}">{{ test.status }}</td>
                <td>{{ "%.3f" | format(test.duration) }}</td>
                <td>{{ test.message or "-" }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="charts">
        <h2>Test Trends</h2>
        <div class="chart">
            <img src="test_trends.png" alt="Test Trends">
        </div>
        <div class="chart">
            <img src="performance_chart.png" alt="Performance Metrics">
        </div>
    </div>

    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
            {% for rec in recommendations %}
            <li>{{ rec }}</li>
            {% endfor %}
        </ul>
    </div>
</body>
</html>
        """

        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results if t.get("status") == "pass")
        failed_tests = sum(1 for t in self.test_results if t.get("status") == "fail")
        skipped_tests = sum(1 for t in self.test_results if t.get("status") == "skip")
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Calculate overall coverage
        overall_coverage = (
            sum(self.coverage_data.values()) / len(self.coverage_data)
            if self.coverage_data
            else 0
        )

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Calculate duration
        if self.test_results:
            start_time = datetime.fromisoformat(self.test_results[0]["timestamp"])
            end_time = datetime.fromisoformat(self.test_results[-1]["timestamp"])
            duration = str(end_time - start_time)
        else:
            duration = "N/A"

        # Render HTML
        template = Template(html_template)
        html_content = template.render(
            timestamp=self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            duration=duration,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            pass_rate=round(pass_rate, 1),
            coverage_data=self.coverage_data,
            overall_coverage=overall_coverage,
            performance_metrics=self.performance_metrics,
            test_results=self.test_results,
            recommendations=recommendations,
        )

        # Save HTML report
        report_path = self.output_dir / "test_report.html"
        with open(report_path, "w") as f:
            f.write(html_content)

        # Generate charts
        self._generate_charts()

        return report_path

    def generate_junit_xml(self):
        """Generate JUnit XML report for CI/CD integration"""
        testsuites = ET.Element("testsuites")

        # Group tests by module
        modules = {}
        for test in self.test_results:
            module = test.get("module", "unknown")
            if module not in modules:
                modules[module] = []
            modules[module].append(test)

        # Create test suite for each module
        for module, tests in modules.items():
            testsuite = ET.SubElement(testsuites, "testsuite")
            testsuite.set("name", module)
            testsuite.set("tests", str(len(tests)))
            testsuite.set(
                "failures", str(sum(1 for t in tests if t["status"] == "fail"))
            )
            testsuite.set(
                "skipped", str(sum(1 for t in tests if t["status"] == "skip"))
            )

            for test in tests:
                testcase = ET.SubElement(testsuite, "testcase")
                testcase.set("name", test["name"])
                testcase.set("classname", module)
                testcase.set("time", str(test.get("duration", 0)))

                if test["status"] == "fail":
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("message", test.get("message", "Test failed"))
                    failure.text = test.get("traceback", "")
                elif test["status"] == "skip":
                    skipped = ET.SubElement(testcase, "skipped")
                    skipped.set("message", test.get("message", "Test skipped"))

        # Save XML
        tree = ET.ElementTree(testsuites)
        xml_path = self.output_dir / "junit_report.xml"
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        return xml_path

    def generate_json_report(self):
        """Generate JSON report for further processing"""
        report_data = {
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(1 for t in self.test_results if t["status"] == "pass"),
                "failed": sum(1 for t in self.test_results if t["status"] == "fail"),
                "skipped": sum(1 for t in self.test_results if t["status"] == "skip"),
            },
            "coverage": self.coverage_data,
            "performance_metrics": self.performance_metrics,
            "test_results": self.test_results,
            "recommendations": self._generate_recommendations(),
        }

        json_path = self.output_dir / "test_report.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)

        return json_path

    def _generate_charts(self):
        """Generate visualization charts"""
        # Test trends chart
        if self.test_results:
            plt.figure(figsize=(10, 6))

            # Group by status
            status_counts = {"pass": 0, "fail": 0, "skip": 0}
            for test in self.test_results:
                status = test.get("status", "unknown")
                if status in status_counts:
                    status_counts[status] += 1

            # Pie chart
            plt.subplot(1, 2, 1)
            colors = ["green", "red", "orange"]
            plt.pie(
                status_counts.values(),
                labels=status_counts.keys(),
                colors=colors,
                autopct="%1.1f%%",
            )
            plt.title("Test Status Distribution")

            # Duration histogram
            plt.subplot(1, 2, 2)
            durations = [t.get("duration", 0) for t in self.test_results]
            plt.hist(durations, bins=20, color="blue", alpha=0.7)
            plt.xlabel("Duration (seconds)")
            plt.ylabel("Number of Tests")
            plt.title("Test Duration Distribution")

            plt.tight_layout()
            plt.savefig(self.output_dir / "test_trends.png")
            plt.close()

        # Performance metrics chart
        if self.performance_metrics:
            plt.figure(figsize=(12, 8))

            num_metrics = len(self.performance_metrics)
            for i, (name, metrics) in enumerate(self.performance_metrics.items()):
                plt.subplot(2, (num_metrics + 1) // 2, i + 1)

                values = [m["value"] for m in metrics]
                plt.plot(values, marker="o")
                plt.title(name)
                plt.xlabel("Measurement")
                plt.ylabel(metrics[0].get("unit", "Value"))
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / "performance_chart.png")
            plt.close()

    def _generate_recommendations(self) -> list[str]:
        """Generate test recommendations based on results"""
        recommendations = []

        # Check test pass rate
        if self.test_results:
            pass_rate = sum(
                1 for t in self.test_results if t["status"] == "pass"
            ) / len(self.test_results)
            if pass_rate < 0.9:
                recommendations.append(
                    f"Test pass rate is {pass_rate * 100:.1f}%. Investigate failing tests."
                )

        # Check coverage
        if self.coverage_data:
            low_coverage_modules = [
                module for module, cov in self.coverage_data.items() if cov < 70
            ]
            if low_coverage_modules:
                recommendations.append(
                    f"Improve test coverage for: {', '.join(low_coverage_modules)}"
                )

        # Check performance
        if "execution_time" in self.performance_metrics:
            avg_time = np.mean(
                [m["value"] for m in self.performance_metrics["execution_time"]]
            )
            if avg_time > 2.0:
                recommendations.append(
                    f"Average execution time is {avg_time:.2f}s. Consider optimization."
                )

        # Check for flaky tests
        if len(self.test_results) > 10:
            # Simple flakiness detection would require multiple runs
            recommendations.append("Run tests multiple times to detect flaky tests.")

        if not recommendations:
            recommendations.append(
                "All tests are performing well. Keep up the good work!"
            )

        return recommendations


# Pytest plugin to automatically generate reports
class ShagunintelligenceTestPlugin:
    """Pytest plugin for automated test reporting"""

    def __init__(self):
        self.reporter = TestReporter(Path("test_reports"))

    def pytest_runtest_logreport(self, report):
        """Hook to capture test results"""
        if report.when == "call":
            result = {
                "name": report.nodeid.split("::")[-1],
                "module": report.nodeid.split("::")[0],
                "status": (
                    "pass" if report.passed else "fail" if report.failed else "skip"
                ),
                "duration": report.duration,
                "message": str(report.longrepr) if report.failed else None,
            }
            self.reporter.add_test_result(result)

    def pytest_sessionfinish(self):
        """Generate reports at the end of test session"""
        self.reporter.generate_html_report()
        self.reporter.generate_junit_xml()
        self.reporter.generate_json_report()
        print(f"\nTest reports generated in: {self.reporter.output_dir}")


# Example usage in test file
def test_reporter_example():
    """Example test using the reporter"""
    reporter = TestReporter(Path("test_output"))

    # Add some test results
    reporter.add_test_result(
        {
            "name": "test_market_analysis",
            "module": "test_market_analyst",
            "status": "pass",
            "duration": 1.234,
        }
    )

    reporter.add_test_result(
        {
            "name": "test_risk_calculation",
            "module": "test_risk_manager",
            "status": "fail",
            "duration": 0.567,
            "message": "Assertion error: Expected 100, got 99",
        }
    )

    # Add performance metrics
    reporter.add_performance_metric("api_response_time", 0.150, "seconds")
    reporter.add_performance_metric("memory_usage", 45.6, "MB")

    # Add coverage data
    reporter.add_coverage_data("agents.market_analyst", 85.5)
    reporter.add_coverage_data("agents.risk_manager", 92.3)
    reporter.add_coverage_data("services.kite", 78.9)

    # Generate reports
    html_path = reporter.generate_html_report()
    xml_path = reporter.generate_junit_xml()
    json_path = reporter.generate_json_report()

    assert html_path.exists()
    assert xml_path.exists()
    assert json_path.exists()
