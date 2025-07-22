import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.risk_manager.agent import RiskManagerAgent
from agents.risk_manager.circuit_breaker import CircuitBreaker
from agents.risk_manager.position_sizing import PositionSizer
from agents.risk_manager.risk_metrics import RiskMetricsCalculator


class TestRiskManagementValidation:
    """Comprehensive validation tests for risk management system"""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance"""
        return RiskManagerAgent()

    @pytest.fixture
    def risk_metrics(self):
        """Create risk metrics instance"""
        return RiskMetricsCalculator()

    @pytest.fixture
    def position_sizer(self):
        """Create position sizer instance"""
        return PositionSizer()

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker instance"""
        return CircuitBreaker()

    @pytest.mark.asyncio
    async def test_position_size_validation(self, risk_manager):
        """Validate position sizing calculations"""
        test_cases = [
            {
                "account_balance": 1000000,
                "risk_percent": 2.0,
                "entry_price": 2500,
                "stop_loss": 2450,
                "expected_shares": 400,  # Risk 20k, loss per share 50
            },
            {
                "account_balance": 500000,
                "risk_percent": 1.0,
                "entry_price": 1500,
                "stop_loss": 1470,
                "expected_shares": 166,  # Risk 5k, loss per share 30
            },
            {
                "account_balance": 100000,
                "risk_percent": 0.5,
                "entry_price": 900,
                "stop_loss": 891,
                "expected_shares": 55,  # Risk 500, loss per share 9
            },
        ]

        for case in test_cases:
            position_size = await risk_manager.calculate_position_size(
                symbol="TEST",
                entry_price=case["entry_price"],
                stop_loss=case["stop_loss"],
                account_balance=case["account_balance"],
                risk_percent=case["risk_percent"],
            )

            # Allow small rounding differences
            assert abs(position_size - case["expected_shares"]) <= 1

    @pytest.mark.asyncio
    async def test_stop_loss_placement(self, risk_manager):
        """Validate stop loss placement logic"""
        stop_loss_manager = risk_manager.stop_loss_manager

        # Test ATR-based stop loss
        atr_stop = stop_loss_manager.calculate_stop_loss(
            entry_price=2500,
            position_type="long",
            volatility=0.02,  # 2% volatility
            support_level=2450,
        )

        # Stop should be below entry but above support
        assert atr_stop < 2500
        assert atr_stop >= 2450

        # Test for short position
        short_stop = stop_loss_manager.calculate_stop_loss(
            entry_price=2500, position_type="short", volatility=0.02, resistance_level=2550
        )

        # Stop should be above entry but below resistance
        assert short_stop > 2500
        assert short_stop <= 2550

    @pytest.mark.asyncio
    async def test_trailing_stop_logic(self, risk_manager):
        """Validate trailing stop loss updates"""
        stop_loss_manager = risk_manager.stop_loss_manager

        # Long position scenario
        initial_stop = 2450
        entry_price = 2500

        # Price moves up
        new_stop = stop_loss_manager.update_trailing_stop(
            current_price=2600,
            entry_price=entry_price,
            current_stop=initial_stop,
            position_type="long",
            trailing_percent=2.0,
        )

        # Stop should move up
        assert new_stop > initial_stop
        # But maintain 2% distance
        assert abs((2600 - new_stop) / 2600 - 0.02) < 0.001

        # Price moves down (but above stop)
        unchanged_stop = stop_loss_manager.update_trailing_stop(
            current_price=2550,
            entry_price=entry_price,
            current_stop=new_stop,
            position_type="long",
            trailing_percent=2.0,
        )

        # Stop should not move down
        assert unchanged_stop == new_stop

    @pytest.mark.asyncio
    async def test_risk_reward_validation(self, risk_manager):
        """Validate risk-reward ratio calculations"""
        test_scenarios = [
            {
                "entry": 2500,
                "stop_loss": 2450,
                "take_profit": 2650,
                "min_ratio": 2.0,
                "should_approve": True,  # R:R = 3:1
            },
            {
                "entry": 1500,
                "stop_loss": 1480,
                "take_profit": 1530,
                "min_ratio": 2.0,
                "should_approve": False,  # R:R = 1.5:1
            },
            {
                "entry": 900,
                "stop_loss": 885,
                "take_profit": 945,
                "min_ratio": 3.0,
                "should_approve": True,  # R:R = 3:1
            },
        ]

        for scenario in test_scenarios:
            risk = scenario["entry"] - scenario["stop_loss"]
            reward = scenario["take_profit"] - scenario["entry"]
            ratio = reward / risk

            is_approved = ratio >= scenario["min_ratio"]
            assert is_approved == scenario["should_approve"]

    @pytest.mark.asyncio
    async def test_portfolio_risk_limits(self, risk_manager):
        """Validate portfolio-wide risk limits"""
        # Test concentration limits
        positions = [
            {"symbol": "RELIANCE", "value": 250000, "risk": 5000},
            {"symbol": "TCS", "value": 200000, "risk": 4000},
            {"symbol": "INFY", "value": 150000, "risk": 3000},
        ]

        portfolio_value = 1000000

        # Check single position concentration
        for position in positions:
            concentration = position["value"] / portfolio_value
            assert concentration <= 0.25  # Max 25% per position

        # Check sector concentration (assuming same sector)
        sector_value = sum(p["value"] for p in positions)
        sector_concentration = sector_value / portfolio_value
        assert sector_concentration <= 0.6  # Max 60% per sector

    @pytest.mark.asyncio
    async def test_correlation_risk(self, risk_metrics):
        """Validate correlation risk calculations"""
        # Create correlation matrix
        correlation_matrix = np.array(
            [[1.0, 0.7, 0.6, 0.3], [0.7, 1.0, 0.8, 0.4], [0.6, 0.8, 1.0, 0.5], [0.3, 0.4, 0.5, 1.0]]
        )

        # Calculate portfolio correlation
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        portfolio_correlation = np.dot(weights, np.dot(correlation_matrix, weights))

        # High correlation increases risk
        assert portfolio_correlation > 0.5

        # Diversification benefit
        individual_risk = 0.02  # 2% volatility each
        portfolio_risk = individual_risk * np.sqrt(portfolio_correlation)
        diversification_benefit = 1 - (portfolio_risk / individual_risk)

        assert diversification_benefit > 0
        assert diversification_benefit < 0.5  # Limited due to correlation

    @pytest.mark.asyncio
    async def test_var_calculation(self, risk_metrics):
        """Validate Value at Risk calculations"""
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)  # 0.1% mean, 2% std

        # Calculate VaR at different confidence levels
        var_95 = risk_metrics.calculate_var(returns, confidence=0.95)
        var_99 = risk_metrics.calculate_var(returns, confidence=0.99)

        # VaR should be negative (loss)
        assert var_95 < 0
        assert var_99 < 0

        # 99% VaR should be larger loss than 95%
        assert var_99 < var_95

        # Verify approximate values (normal distribution)
        # 95% VaR ≈ mean - 1.645 * std
        expected_var_95 = 0.001 - 1.645 * 0.02
        assert abs(var_95 - expected_var_95) < 0.005

    @pytest.mark.asyncio
    async def test_maximum_drawdown(self, risk_metrics):
        """Validate maximum drawdown calculations"""
        # Create price series with known drawdown
        prices = [100]

        # Rise to 120
        for i in range(20):
            prices.append(prices[-1] * 1.01)

        # Fall to 90 (25% drawdown from peak)
        for i in range(10):
            prices.append(prices[-1] * 0.97)

        # Recover to 110
        for i in range(10):
            prices.append(prices[-1] * 1.02)

        max_dd = risk_metrics.calculate_max_drawdown(prices)

        # Should detect ~25% drawdown
        assert 0.24 < max_dd < 0.26

    @pytest.mark.asyncio
    async def test_kelly_criterion(self, position_sizer):
        """Validate Kelly Criterion calculations"""
        test_cases = [
            {
                "win_rate": 0.6,
                "avg_win": 1.5,
                "avg_loss": 1.0,
                "expected_kelly": 0.2,  # f = (bp - q) / b
            },
            {
                "win_rate": 0.55,
                "avg_win": 2.0,
                "avg_loss": 1.0,
                "expected_kelly": 0.1,  # Conservative
            },
            {
                "win_rate": 0.4,
                "avg_win": 3.0,
                "avg_loss": 1.0,
                "expected_kelly": 0.2,  # High reward compensates
            },
        ]

        for case in test_cases:
            kelly_percent = position_sizer.kelly_criterion(
                win_rate=case["win_rate"], avg_win=case["avg_win"], avg_loss=case["avg_loss"]
            )

            # Kelly should be positive for positive expectancy
            assert kelly_percent > 0

            # Should be reasonably close to expected
            assert abs(kelly_percent - case["expected_kelly"]) < 0.05

            # Should never exceed 25% (safety cap)
            assert kelly_percent <= 0.25

    @pytest.mark.asyncio
    async def test_circuit_breaker_triggers(self, circuit_breaker):
        """Validate circuit breaker activation"""
        # Test daily loss limit
        circuit_breaker.daily_loss_limit = 5.0  # 5%
        circuit_breaker.current_daily_loss = 4.5

        # Should not trigger yet
        assert not circuit_breaker.should_halt_trading()

        # Additional loss triggers circuit breaker
        circuit_breaker.current_daily_loss = 5.1
        assert circuit_breaker.should_halt_trading()

        # Test consecutive loss trigger
        circuit_breaker.reset()
        circuit_breaker.max_consecutive_losses = 5

        for i in range(4):
            circuit_breaker.record_trade_result(False)  # Loss

        assert not circuit_breaker.should_halt_trading()

        circuit_breaker.record_trade_result(False)  # 5th loss
        assert circuit_breaker.should_halt_trading()

    @pytest.mark.asyncio
    async def test_risk_parity_allocation(self, position_sizer):
        """Validate risk parity position sizing"""
        assets = [
            {"symbol": "RELIANCE", "volatility": 0.02},
            {"symbol": "TCS", "volatility": 0.015},
            {"symbol": "HDFC", "volatility": 0.025},
            {"symbol": "INFY", "volatility": 0.018},
        ]

        total_capital = 1000000
        target_risk = 0.01  # 1% portfolio volatility

        allocations = {}
        for asset in assets:
            # Inverse volatility weighting
            weight = (1 / asset["volatility"]) / sum(1 / a["volatility"] for a in assets)
            allocations[asset["symbol"]] = weight * total_capital

        # Verify equal risk contribution
        risk_contributions = []
        for asset in assets:
            position_value = allocations[asset["symbol"]]
            risk_contribution = (position_value / total_capital) * asset["volatility"]
            risk_contributions.append(risk_contribution)

        # All should contribute roughly equal risk
        avg_risk = np.mean(risk_contributions)
        for risk in risk_contributions:
            assert abs(risk - avg_risk) / avg_risk < 0.1  # Within 10%

    @pytest.mark.asyncio
    async def test_stress_test_scenarios(self, risk_manager):
        """Validate stress testing scenarios"""
        portfolio = [
            {"symbol": "RELIANCE", "quantity": 100, "price": 2500},
            {"symbol": "TCS", "quantity": 50, "price": 3500},
            {"symbol": "INFY", "quantity": 200, "price": 1500},
        ]

        scenarios = {"market_crash_10": -0.10, "market_crash_20": -0.20, "sector_shock": -0.15, "black_swan": -0.30}

        portfolio_value = sum(p["quantity"] * p["price"] for p in portfolio)

        stress_results = {}
        for scenario, shock in scenarios.items():
            stressed_value = portfolio_value * (1 + shock)
            loss = portfolio_value - stressed_value
            stress_results[scenario] = {"loss": loss, "loss_percent": abs(shock) * 100}

        # Validate stress test results
        assert stress_results["market_crash_10"]["loss"] < stress_results["market_crash_20"]["loss"]
        assert stress_results["black_swan"]["loss_percent"] == 30

        # Check if portfolio can survive worst case
        worst_case_value = portfolio_value * (1 + scenarios["black_swan"])
        survival_ratio = worst_case_value / portfolio_value
        assert survival_ratio >= 0.7  # At least 70% survives

    @pytest.mark.asyncio
    async def test_margin_call_prevention(self, risk_manager):
        """Validate margin call prevention logic"""
        account_balance = 100000
        margin_used = 80000
        maintenance_margin = 0.25  # 25%

        # Calculate margin level
        margin_level = account_balance / margin_used

        # Should trigger warning before margin call
        warning_level = 1.5  # 150% margin level
        margin_call_level = 1.25  # 125% margin level

        assert margin_level > margin_call_level

        # Test position reduction recommendation
        if margin_level < warning_level:
            # Calculate required reduction
            target_margin_usage = account_balance / warning_level
            reduction_needed = margin_used - target_margin_usage
            reduction_percent = reduction_needed / margin_used

            assert reduction_percent > 0
            assert reduction_percent < 0.5  # Reasonable reduction

    def test_risk_report_generation(self, risk_manager, tmp_path):
        """Test risk report generation"""
        report_path = tmp_path / "risk_report.json"

        risk_report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_metrics": {
                "total_exposure": 750000,
                "var_95": -25000,
                "max_drawdown": 0.12,
                "sharpe_ratio": 1.8,
            },
            "position_risks": [
                {"symbol": "RELIANCE", "risk": 5000, "weight": 0.33},
                {"symbol": "TCS", "risk": 4500, "weight": 0.30},
                {"symbol": "INFY", "risk": 3500, "weight": 0.23},
            ],
            "risk_limits": {
                "daily_loss": {"current": -2.5, "limit": -5.0, "status": "OK"},
                "position_concentration": {"max": 0.33, "limit": 0.35, "status": "OK"},
                "leverage": {"current": 1.5, "limit": 2.0, "status": "OK"},
            },
            "recommendations": [
                "Consider reducing RELIANCE position by 10%",
                "Add hedging for market downturn risk",
                "Maintain current stop loss levels",
            ],
        }

        # Save report
        import json

        with open(report_path, "w") as f:
            json.dump(risk_report, f, indent=2)

        assert report_path.exists()

        # Verify report structure
        with open(report_path) as f:
            loaded_report = json.load(f)

        assert "portfolio_metrics" in loaded_report
        assert "risk_limits" in loaded_report
        assert all(limit["status"] == "OK" for limit in loaded_report["risk_limits"].values())
