"""
Tests for Enhanced Risk Management Framework
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.services.enhanced_risk_management import (
    AdvancedVaRCalculator,
    CorrelationAnalyzer,
    DynamicPositionSizer,
    EnhancedRiskManager,
    PortfolioMetrics,
    RiskLevel,
    RiskLimits,
    SectorExposureAnalyzer,
)


class TestAdvancedVaRCalculator:
    """Test advanced VaR calculation methods"""

    def setup_method(self):
        self.calculator = AdvancedVaRCalculator()

        # Generate test returns data
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0, 0.02, 100))

    def test_historical_var_calculation(self):
        """Test historical VaR calculation accuracy"""
        result = self.calculator.calculate_historical_var(self.returns, 0.95, 1)

        assert "var" in result
        assert "cvar" in result
        assert "confidence" in result
        assert "method" in result

        assert result["confidence"] == 0.95
        assert result["method"] == "historical"
        assert result["var"] > 0
        assert result["cvar"] >= result["var"]  # CVaR should be >= VaR

    def test_parametric_var_calculation(self):
        """Test parametric VaR calculation"""
        result = self.calculator.calculate_parametric_var(self.returns, 0.99, 5)

        assert result["confidence"] == 0.99
        assert result["time_horizon"] == 5
        assert result["method"] == "parametric"
        assert "mean" in result
        assert "std" in result

    def test_monte_carlo_var_calculation(self):
        """Test Monte Carlo VaR calculation"""
        result = self.calculator.calculate_monte_carlo_var(self.returns, 0.95, 1, 1000)

        assert result["method"] == "monte_carlo"
        assert result["simulations"] == 1000
        assert result["var"] > 0
        assert result["cvar"] > 0

    def test_var_scaling_with_time_horizon(self):
        """Test that VaR scales properly with time horizon"""
        var_1d = self.calculator.calculate_historical_var(self.returns, 0.95, 1)
        var_5d = self.calculator.calculate_historical_var(self.returns, 0.95, 5)

        # 5-day VaR should be approximately sqrt(5) times 1-day VaR
        expected_ratio = np.sqrt(5)
        actual_ratio = var_5d["var"] / var_1d["var"]

        # Allow 20% tolerance due to randomness
        assert 0.8 * expected_ratio <= actual_ratio <= 1.2 * expected_ratio

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        short_returns = pd.Series([0.01, -0.02, 0.005])
        result = self.calculator.calculate_historical_var(short_returns, 0.95, 1)

        assert result["var"] == 0.0
        assert result["method"] == "insufficient_data"


class TestCorrelationAnalyzer:
    """Test correlation analysis functionality"""

    def setup_method(self):
        self.analyzer = CorrelationAnalyzer()

        # Generate correlated returns data
        np.random.seed(42)
        n_assets = 5
        n_periods = 100

        # Create correlation structure
        base_returns = np.random.normal(0, 0.02, n_periods)

        self.returns_data = {
            "ASSET1": pd.Series(base_returns + np.random.normal(0, 0.01, n_periods)),
            "ASSET2": pd.Series(
                base_returns + np.random.normal(0, 0.01, n_periods)
            ),  # Highly correlated with ASSET1
            "ASSET3": pd.Series(
                -base_returns + np.random.normal(0, 0.01, n_periods)
            ),  # Negatively correlated
            "ASSET4": pd.Series(np.random.normal(0, 0.02, n_periods)),  # Independent
            "ASSET5": pd.Series(np.random.normal(0, 0.02, n_periods)),  # Independent
        }

    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation"""
        corr_matrix = self.analyzer.calculate_correlation_matrix(self.returns_data)

        assert len(corr_matrix) == 5
        assert "ASSET1" in corr_matrix
        assert "ASSET2" in corr_matrix

        # Diagonal should be 1.0
        for asset in corr_matrix:
            assert abs(corr_matrix[asset][asset] - 1.0) < 0.01

        # ASSET1 and ASSET2 should be highly correlated
        assert corr_matrix["ASSET1"]["ASSET2"] > 0.5

    def test_high_correlation_detection(self):
        """Test detection of high correlations"""
        corr_matrix = self.analyzer.calculate_correlation_matrix(self.returns_data)
        high_corrs = self.analyzer.detect_high_correlations(corr_matrix)

        # Should detect high correlation between ASSET1 and ASSET2
        assert len(high_corrs) > 0

        # Check that detected pairs have high correlation
        for asset1, asset2, corr_value in high_corrs:
            assert abs(corr_value) > self.analyzer.correlation_threshold

    def test_diversification_ratio_calculation(self):
        """Test portfolio diversification ratio calculation"""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights

        # Create correlation matrix
        corr_matrix = np.array(
            [
                [1.0, 0.8, -0.3, 0.1, 0.0],
                [0.8, 1.0, -0.2, 0.0, 0.1],
                [-0.3, -0.2, 1.0, 0.1, 0.0],
                [0.1, 0.0, 0.1, 1.0, 0.2],
                [0.0, 0.1, 0.0, 0.2, 1.0],
            ]
        )

        div_ratio = self.analyzer.calculate_portfolio_diversification_ratio(
            weights, corr_matrix
        )

        assert div_ratio > 1.0  # Should be > 1 for diversified portfolio
        assert div_ratio < 5.0  # Reasonable upper bound


class TestSectorExposureAnalyzer:
    """Test sector exposure analysis"""

    def setup_method(self):
        self.analyzer = SectorExposureAnalyzer()

    def test_sector_exposure_calculation(self):
        """Test sector exposure calculation"""
        positions = {
            "RELIANCE": {"market_value": 100000},
            "TCS": {"market_value": 80000},
            "INFY": {"market_value": 70000},
            "HDFCBANK": {"market_value": 90000},
            "ICICIBANK": {"market_value": 60000},
        }

        exposures = self.analyzer.calculate_sector_exposures(positions)

        # Check that exposures sum to 1.0
        total_exposure = sum(exposures.values())
        assert abs(total_exposure - 1.0) < 0.01

        # Check specific sectors
        assert "IT" in exposures  # TCS and INFY
        assert "Banking" in exposures  # HDFCBANK and ICICIBANK
        assert "Energy" in exposures  # RELIANCE

        # IT sector should have significant exposure
        assert exposures["IT"] > 0.3  # (80000 + 70000) / 400000 = 0.375

    def test_sector_concentration_check(self):
        """Test sector concentration risk detection"""
        # Create concentrated portfolio
        sector_exposures = {
            "Banking": 0.6,  # Over-concentrated
            "IT": 0.3,
            "Energy": 0.1,
        }

        concentration_check = self.analyzer.check_sector_concentration(
            sector_exposures, max_sector_exposure=0.5
        )

        assert concentration_check["concentration_risk"] is True
        assert len(concentration_check["violations"]) == 1
        assert concentration_check["violations"][0]["sector"] == "Banking"
        assert concentration_check["max_sector"] == "Banking"
        assert concentration_check["max_exposure"] == 0.6

    def test_herfindahl_index_calculation(self):
        """Test Herfindahl index for concentration measurement"""
        # Highly concentrated portfolio
        concentrated_exposures = {"Banking": 0.8, "IT": 0.2}

        # Well-diversified portfolio
        diversified_exposures = {
            "Banking": 0.2,
            "IT": 0.2,
            "Energy": 0.2,
            "FMCG": 0.2,
            "Auto": 0.2,
        }

        conc_check = self.analyzer.check_sector_concentration(concentrated_exposures)
        div_check = self.analyzer.check_sector_concentration(diversified_exposures)

        # Concentrated portfolio should have higher Herfindahl index
        assert conc_check["herfindahl_index"] > div_check["herfindahl_index"]


class TestDynamicPositionSizer:
    """Test dynamic position sizing functionality"""

    def setup_method(self):
        self.sizer = DynamicPositionSizer(base_risk_per_trade=0.02)

    def test_basic_position_sizing(self):
        """Test basic position size calculation"""
        result = self.sizer.calculate_dynamic_position_size(
            symbol="RELIANCE",
            entry_price=2500,
            stop_loss=2400,
            portfolio_state={"total_value": 100000, "current_drawdown": 0.0},
            market_conditions={"volatility": 0.02, "stress_level": 0.0},
        )

        assert "shares" in result
        assert "position_value" in result
        assert "risk_amount" in result
        assert "risk_percentage" in result

        # Risk amount should be approximately 2% of portfolio
        expected_risk = 100000 * 0.02
        assert abs(result["risk_amount"] - expected_risk) < expected_risk * 0.1

    def test_volatility_adjustment(self):
        """Test position sizing adjustment for volatility"""
        base_conditions = {"volatility": 0.02, "stress_level": 0.0}
        high_vol_conditions = {
            "volatility": 0.06,
            "stress_level": 0.0,
        }  # 3x normal volatility

        portfolio_state = {"total_value": 100000, "current_drawdown": 0.0}

        base_result = self.sizer.calculate_dynamic_position_size(
            "RELIANCE", 2500, 2400, portfolio_state, base_conditions
        )

        high_vol_result = self.sizer.calculate_dynamic_position_size(
            "RELIANCE", 2500, 2400, portfolio_state, high_vol_conditions
        )

        # High volatility should result in smaller position size
        assert high_vol_result["shares"] < base_result["shares"]
        assert high_vol_result["risk_multiplier"] < base_result["risk_multiplier"]

    def test_drawdown_adjustment(self):
        """Test position sizing adjustment for portfolio drawdown"""
        normal_state = {"total_value": 100000, "current_drawdown": 0.0}
        drawdown_state = {
            "total_value": 100000,
            "current_drawdown": -0.08,
        }  # 8% drawdown

        market_conditions = {"volatility": 0.02, "stress_level": 0.0}

        normal_result = self.sizer.calculate_dynamic_position_size(
            "RELIANCE", 2500, 2400, normal_state, market_conditions
        )

        drawdown_result = self.sizer.calculate_dynamic_position_size(
            "RELIANCE", 2500, 2400, drawdown_state, market_conditions
        )

        # Drawdown should result in smaller position size
        assert drawdown_result["shares"] < normal_result["shares"]

    def test_market_stress_adjustment(self):
        """Test position sizing adjustment for market stress"""
        normal_conditions = {"volatility": 0.02, "stress_level": 0.0}
        stress_conditions = {"volatility": 0.02, "stress_level": 0.8}  # High stress

        portfolio_state = {"total_value": 100000, "current_drawdown": 0.0}

        normal_result = self.sizer.calculate_dynamic_position_size(
            "RELIANCE", 2500, 2400, portfolio_state, normal_conditions
        )

        stress_result = self.sizer.calculate_dynamic_position_size(
            "RELIANCE", 2500, 2400, portfolio_state, stress_conditions
        )

        # Market stress should result in smaller position size
        assert stress_result["shares"] < normal_result["shares"]


class TestEnhancedRiskManagerIntegration:
    """Integration tests for Enhanced Risk Manager"""

    def setup_method(self):
        self.risk_limits = RiskLimits(
            max_portfolio_var=0.05,
            max_sector_exposure=0.30,
            max_single_position=0.10,
            max_leverage=2.0,
            max_drawdown=0.15,
        )
        self.manager = EnhancedRiskManager(self.risk_limits)

    @pytest.mark.asyncio
    async def test_comprehensive_portfolio_analysis(self):
        """Test comprehensive portfolio risk analysis"""
        # Create test portfolio
        positions = {
            "RELIANCE": {"shares": 100, "market_value": 250000},
            "TCS": {"shares": 50, "market_value": 180000},
            "HDFCBANK": {"shares": 75, "market_value": 120000},
            "INFY": {"shares": 60, "market_value": 150000},
        }

        # Create test market data
        market_data = {}
        for symbol in positions.keys():
            dates = pd.date_range(end=datetime.now(), periods=100, freq="1D")
            np.random.seed(hash(symbol) % 1000)  # Different seed per symbol

            returns = np.random.normal(0, 0.02, 100)
            prices = 2500 * np.exp(np.cumsum(returns))

            market_data[symbol] = pd.DataFrame(
                {"close": prices, "volume": np.random.randint(10000, 100000, 100)},
                index=dates,
            )

        # Calculate portfolio metrics
        metrics = await self.manager.calculate_portfolio_metrics(positions, market_data)

        # Verify metrics structure
        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.total_value > 0
        assert metrics.total_exposure > 0
        assert metrics.leverage >= 0
        assert isinstance(metrics.risk_level, RiskLevel)

        # Verify sector exposures
        assert len(metrics.sector_exposures) > 0
        assert "IT" in metrics.sector_exposures  # TCS and INFY
        assert "Banking" in metrics.sector_exposures  # HDFCBANK

    @pytest.mark.asyncio
    async def test_risk_limit_violations(self):
        """Test risk limit violation detection"""
        # Create portfolio that violates limits
        positions = {
            "RELIANCE": {"shares": 1000, "market_value": 2500000},  # Large position
            "TCS": {"shares": 100, "market_value": 360000},
            "INFY": {"shares": 200, "market_value": 720000},  # High IT concentration
        }

        market_data = self._create_mock_market_data(positions.keys())

        # Calculate metrics
        metrics = await self.manager.calculate_portfolio_metrics(positions, market_data)

        # Check risk limits
        risk_check = await self.manager.check_risk_limits(metrics)

        # Should have violations
        assert len(risk_check["violations"]) > 0
        assert risk_check["overall_status"] in ["VIOLATION", "WARNING"]

        # Check for specific violations
        violation_types = [v["type"] for v in risk_check["violations"]]
        assert "sector_concentration" in violation_types  # IT over-concentration

    @pytest.mark.asyncio
    async def test_risk_recommendations_generation(self):
        """Test generation of risk management recommendations"""
        # Create portfolio with issues
        positions = {
            "TCS": {"shares": 200, "market_value": 720000},
            "INFY": {"shares": 150, "market_value": 540000},
            "WIPRO": {"shares": 100, "market_value": 240000},  # All IT stocks
        }

        market_data = self._create_mock_market_data(positions.keys())

        metrics = await self.manager.calculate_portfolio_metrics(positions, market_data)
        risk_check = await self.manager.check_risk_limits(metrics)

        # Generate recommendations
        recommendations = await self.manager.generate_risk_recommendations(
            metrics, risk_check
        )

        assert len(recommendations) > 0

        # Should recommend diversification due to IT concentration
        diversification_recs = [
            r for r in recommendations if r["category"] == "diversification"
        ]
        assert len(diversification_recs) > 0

        # Check recommendation structure
        for rec in recommendations:
            assert "priority" in rec
            assert "category" in rec
            assert "action" in rec
            assert "description" in rec
            assert "timeline" in rec

    def _create_mock_market_data(self, symbols):
        """Create mock market data for testing"""
        market_data = {}

        for i, symbol in enumerate(symbols):
            dates = pd.date_range(end=datetime.now(), periods=50, freq="1D")
            np.random.seed(i * 42)  # Different seed per symbol

            returns = np.random.normal(0, 0.02, 50)
            prices = 2500 * np.exp(np.cumsum(returns))

            market_data[symbol] = pd.DataFrame(
                {"close": prices, "volume": np.random.randint(10000, 100000, 50)},
                index=dates,
            )

        return market_data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
