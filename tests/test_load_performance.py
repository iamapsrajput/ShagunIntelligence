"""
Load Testing and Performance Benchmarks for Trading System
"""

import asyncio
import os
import statistics
import time
from datetime import datetime
from typing import Any

import psutil
import pytest

from app.services.advanced_order_management import (
    AdvancedOrderManager,
    AdvancedOrderRequest,
    AdvancedOrderType,
)
from app.services.enhanced_risk_management import EnhancedRiskManager
from app.services.multi_timeframe_analysis import MultiTimeFrameEngine, TimeFrame


class LoadTestMetrics:
    """Collect and analyze load test metrics"""

    def __init__(self):
        self.response_times = []
        self.error_count = 0
        self.success_count = 0
        self.memory_usage = []
        self.cpu_usage = []
        self.start_time = None
        self.end_time = None

    def add_response_time(self, response_time: float):
        """Add response time measurement"""
        self.response_times.append(response_time)

    def add_success(self):
        """Record successful operation"""
        self.success_count += 1

    def add_error(self):
        """Record failed operation"""
        self.error_count += 1

    def record_system_metrics(self):
        """Record current system metrics"""
        process = psutil.Process(os.getpid())
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_usage.append(process.cpu_percent())

    def start_test(self):
        """Mark test start time"""
        self.start_time = time.time()

    def end_test(self):
        """Mark test end time"""
        self.end_time = time.time()

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive test summary"""
        if not self.response_times:
            return {"error": "No response times recorded"}

        total_requests = self.success_count + self.error_count
        duration = (
            self.end_time - self.start_time if self.end_time and self.start_time else 0
        )

        return {
            "total_requests": total_requests,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "success_rate": (
                (self.success_count / total_requests * 100) if total_requests > 0 else 0
            ),
            "duration_seconds": duration,
            "requests_per_second": total_requests / duration if duration > 0 else 0,
            "response_times": {
                "min": min(self.response_times),
                "max": max(self.response_times),
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": self._percentile(self.response_times, 95),
                "p99": self._percentile(self.response_times, 99),
            },
            "memory_usage": {
                "min_mb": min(self.memory_usage) if self.memory_usage else 0,
                "max_mb": max(self.memory_usage) if self.memory_usage else 0,
                "avg_mb": (
                    statistics.mean(self.memory_usage) if self.memory_usage else 0
                ),
            },
            "cpu_usage": {
                "min_percent": min(self.cpu_usage) if self.cpu_usage else 0,
                "max_percent": max(self.cpu_usage) if self.cpu_usage else 0,
                "avg_percent": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            },
        }

    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestAdvancedOrderManagementLoad:
    """Load tests for Advanced Order Management System"""

    def setup_method(self):
        self.manager = AdvancedOrderManager()
        self.metrics = LoadTestMetrics()

    @pytest.mark.asyncio
    async def test_concurrent_order_placement(self):
        """Test concurrent order placement under load"""
        concurrent_orders = 50

        async def place_order(order_id: int):
            """Place a single order and measure performance"""
            start_time = time.time()

            try:
                order_request = AdvancedOrderRequest(
                    symbol=f"TEST_{order_id % 10}",  # Rotate through 10 symbols
                    exchange="NSE",
                    transaction_type="BUY",
                    quantity=100 + (order_id % 1000),  # Vary quantity
                    order_type=AdvancedOrderType.TWAP,
                    twap_intervals=5,
                    time_horizon=300,
                )

                # Simulate order placement (without actual broker calls)
                result = await self.manager._validate_advanced_order(order_request)

                response_time = time.time() - start_time
                self.metrics.add_response_time(response_time)

                if result.get("valid", False):
                    self.metrics.add_success()
                else:
                    self.metrics.add_error()

            except Exception:
                self.metrics.add_error()
                response_time = time.time() - start_time
                self.metrics.add_response_time(response_time)

        self.metrics.start_test()

        # Execute concurrent orders
        tasks = [place_order(i) for i in range(concurrent_orders)]
        await asyncio.gather(*tasks)

        self.metrics.end_test()

        # Analyze results
        summary = self.metrics.get_summary()

        # Performance assertions
        assert summary["success_rate"] > 95  # At least 95% success rate
        assert summary["response_times"]["p95"] < 1.0  # 95th percentile under 1 second
        assert summary["response_times"]["mean"] < 0.5  # Average under 500ms

        print("Order Management Load Test Results:")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Avg Response Time: {summary['response_times']['mean']:.3f}s")
        print(f"  P95 Response Time: {summary['response_times']['p95']:.3f}s")
        print(f"  Requests/Second: {summary['requests_per_second']:.1f}")

    @pytest.mark.asyncio
    async def test_algorithm_performance_under_load(self):
        """Test algorithm execution performance under load"""
        from app.services.advanced_order_management import (
            IcebergAlgorithm,
            TWAPAlgorithm,
            VWAPAlgorithm,
        )

        algorithms = [
            ("TWAP", TWAPAlgorithm()),
            ("VWAP", VWAPAlgorithm()),
            ("ICEBERG", IcebergAlgorithm()),
        ]

        concurrent_executions = 30

        async def execute_algorithm(algo_name: str, algorithm, execution_id: int):
            """Execute algorithm and measure performance"""
            start_time = time.time()

            try:
                order_request = AdvancedOrderRequest(
                    symbol=f"TEST_{execution_id % 5}",
                    exchange="NSE",
                    transaction_type="BUY",
                    quantity=1000 + (execution_id % 5000),
                    order_type=getattr(AdvancedOrderType, algo_name),
                    twap_intervals=10 if algo_name == "TWAP" else None,
                    max_participation_rate=0.1 if algo_name == "VWAP" else None,
                    iceberg_visible_quantity=200 if algo_name == "ICEBERG" else None,
                )

                # Mock market data
                market_data = {
                    "quote": {"price": 2500, "volume": 100000},
                    "volume_profile": [5000] * 20,
                }

                result = await algorithm.execute(order_request, market_data)

                response_time = time.time() - start_time
                self.metrics.add_response_time(response_time)

                if "execution_plan" in result:
                    self.metrics.add_success()
                else:
                    self.metrics.add_error()

            except Exception:
                self.metrics.add_error()
                response_time = time.time() - start_time
                self.metrics.add_response_time(response_time)

        self.metrics.start_test()

        # Execute algorithms concurrently
        tasks = []
        for i in range(concurrent_executions):
            algo_name, algorithm = algorithms[i % len(algorithms)]
            tasks.append(execute_algorithm(algo_name, algorithm, i))

        await asyncio.gather(*tasks)

        self.metrics.end_test()

        summary = self.metrics.get_summary()

        # Performance assertions
        assert summary["success_rate"] > 98  # Very high success rate for algorithms
        assert summary["response_times"]["p95"] < 2.0  # P95 under 2 seconds

        print("Algorithm Performance Load Test Results:")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Avg Response Time: {summary['response_times']['mean']:.3f}s")
        print(f"  P95 Response Time: {summary['response_times']['p95']:.3f}s")


class TestRiskManagementLoad:
    """Load tests for Enhanced Risk Management System"""

    def setup_method(self):
        self.manager = EnhancedRiskManager()
        self.metrics = LoadTestMetrics()

    @pytest.mark.asyncio
    async def test_concurrent_risk_calculations(self):
        """Test concurrent risk metric calculations"""
        concurrent_calculations = 40

        async def calculate_risk_metrics(calc_id: int):
            """Calculate risk metrics and measure performance"""
            start_time = time.time()

            try:
                # Create mock portfolio
                positions = {
                    f"STOCK_{calc_id % 10}": {"shares": 100, "market_value": 25000},
                    f"STOCK_{(calc_id + 1) % 10}": {
                        "shares": 50,
                        "market_value": 18000,
                    },
                    f"STOCK_{(calc_id + 2) % 10}": {
                        "shares": 75,
                        "market_value": 12000,
                    },
                }

                # Create mock market data
                market_data = {}
                for symbol in positions.keys():
                    import numpy as np
                    import pandas as pd

                    dates = pd.date_range(end=datetime.now(), periods=50, freq="1D")
                    np.random.seed(calc_id + hash(symbol) % 1000)

                    returns = np.random.normal(0, 0.02, 50)
                    prices = 2500 * np.exp(np.cumsum(returns))

                    market_data[symbol] = pd.DataFrame(
                        {
                            "close": prices,
                            "volume": np.random.randint(10000, 100000, 50),
                        },
                        index=dates,
                    )

                # Calculate portfolio metrics
                metrics = await self.manager.calculate_portfolio_metrics(
                    positions, market_data
                )

                response_time = time.time() - start_time
                self.metrics.add_response_time(response_time)

                if metrics.total_value > 0:
                    self.metrics.add_success()
                else:
                    self.metrics.add_error()

            except Exception:
                self.metrics.add_error()
                response_time = time.time() - start_time
                self.metrics.add_response_time(response_time)

        self.metrics.start_test()

        # Execute concurrent risk calculations
        tasks = [calculate_risk_metrics(i) for i in range(concurrent_calculations)]
        await asyncio.gather(*tasks)

        self.metrics.end_test()

        summary = self.metrics.get_summary()

        # Performance assertions
        assert summary["success_rate"] > 90  # At least 90% success rate
        assert summary["response_times"]["p95"] < 3.0  # P95 under 3 seconds

        print("Risk Management Load Test Results:")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Avg Response Time: {summary['response_times']['mean']:.3f}s")
        print(f"  P95 Response Time: {summary['response_times']['p95']:.3f}s")


class TestMultiTimeFrameLoad:
    """Load tests for Multi-Timeframe Analysis Engine"""

    def setup_method(self):
        self.engine = MultiTimeFrameEngine()
        self.metrics = LoadTestMetrics()

    @pytest.mark.asyncio
    async def test_concurrent_analysis(self):
        """Test concurrent multi-timeframe analysis"""
        concurrent_analyses = 25
        symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

        async def analyze_symbol(symbol: str, analysis_id: int):
            """Analyze symbol and measure performance"""
            start_time = time.time()

            try:
                timeframes = [TimeFrame.M15, TimeFrame.H1, TimeFrame.H4, TimeFrame.D1]
                analysis = await self.engine.analyze_symbol(symbol, timeframes)

                response_time = time.time() - start_time
                self.metrics.add_response_time(response_time)

                if analysis.consensus_signal in ["BUY", "SELL", "HOLD"]:
                    self.metrics.add_success()
                else:
                    self.metrics.add_error()

            except Exception:
                self.metrics.add_error()
                response_time = time.time() - start_time
                self.metrics.add_response_time(response_time)

        self.metrics.start_test()

        # Execute concurrent analyses
        tasks = []
        for i in range(concurrent_analyses):
            symbol = symbols[i % len(symbols)]
            tasks.append(analyze_symbol(symbol, i))

        await asyncio.gather(*tasks)

        self.metrics.end_test()

        summary = self.metrics.get_summary()

        # Performance assertions
        assert summary["success_rate"] > 95  # At least 95% success rate
        assert summary["response_times"]["p95"] < 2.0  # P95 under 2 seconds

        print("Multi-Timeframe Analysis Load Test Results:")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Avg Response Time: {summary['response_times']['mean']:.3f}s")
        print(f"  P95 Response Time: {summary['response_times']['p95']:.3f}s")


class TestSystemIntegrationLoad:
    """Integration load tests across all systems"""

    @pytest.mark.asyncio
    async def test_full_system_load(self):
        """Test full system under realistic load"""
        order_manager = AdvancedOrderManager()
        risk_manager = EnhancedRiskManager()
        analysis_engine = MultiTimeFrameEngine()

        metrics = LoadTestMetrics()
        concurrent_operations = 20

        async def full_trading_workflow(workflow_id: int):
            """Execute complete trading workflow"""
            start_time = time.time()

            try:
                symbol = f"STOCK_{workflow_id % 5}"

                # 1. Multi-timeframe analysis
                analysis = await analysis_engine.analyze_symbol(symbol)

                # 2. Risk assessment
                positions = {symbol: {"shares": 100, "market_value": 25000}}
                market_data = {symbol: self._create_mock_data()}

                portfolio_metrics = await risk_manager.calculate_portfolio_metrics(
                    positions, market_data
                )
                risk_check = await risk_manager.check_risk_limits(portfolio_metrics)

                # 3. Order placement (if signals align)
                if (
                    analysis.consensus_signal == "BUY"
                    and risk_check["overall_status"] == "OK"
                ):
                    order_request = AdvancedOrderRequest(
                        symbol=symbol,
                        exchange="NSE",
                        transaction_type="BUY",
                        quantity=100,
                        order_type=AdvancedOrderType.TWAP,
                        twap_intervals=5,
                        time_horizon=300,
                    )

                    validation = await order_manager._validate_advanced_order(
                        order_request
                    )

                response_time = time.time() - start_time
                metrics.add_response_time(response_time)
                metrics.add_success()

            except Exception:
                metrics.add_error()
                response_time = time.time() - start_time
                metrics.add_response_time(response_time)

        metrics.start_test()

        # Execute concurrent workflows
        tasks = [full_trading_workflow(i) for i in range(concurrent_operations)]
        await asyncio.gather(*tasks)

        metrics.end_test()

        summary = metrics.get_summary()

        # Performance assertions for full system
        assert summary["success_rate"] > 85  # At least 85% success rate
        assert summary["response_times"]["p95"] < 5.0  # P95 under 5 seconds
        assert summary["response_times"]["mean"] < 2.0  # Average under 2 seconds

        print("Full System Integration Load Test Results:")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Avg Response Time: {summary['response_times']['mean']:.3f}s")
        print(f"  P95 Response Time: {summary['response_times']['p95']:.3f}s")
        print(f"  Requests/Second: {summary['requests_per_second']:.1f}")

    def _create_mock_data(self):
        """Create mock market data"""
        import numpy as np
        import pandas as pd

        dates = pd.date_range(end=datetime.now(), periods=50, freq="1D")
        np.random.seed(42)

        returns = np.random.normal(0, 0.02, 50)
        prices = 2500 * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {"close": prices, "volume": np.random.randint(10000, 100000, 50)},
            index=dates,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
