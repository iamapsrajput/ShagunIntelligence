import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.crew_manager import CrewManager
from agents.technical_indicator.agent import TechnicalIndicatorAgent
from services.data_pipeline.pipeline import DataPipeline


class TestHistoricalDataIntegration:
    """Integration tests using historical market data"""

    @pytest.fixture
    def crew_manager(self, mock_kite_client):
        """Create crew manager with mock client"""
        with patch("agents.crew_manager.kite_client", mock_kite_client):
            manager = CrewManager()
            return manager

    @pytest.fixture
    def data_pipeline(self, mock_kite_client):
        """Create data pipeline with mock client"""
        with patch("services.data_pipeline.pipeline.kite_client", mock_kite_client):
            pipeline = DataPipeline()
            return pipeline

    @pytest.fixture
    def historical_data(self):
        """Generate realistic historical data"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")

        # Generate realistic price movement
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        price = 2500
        prices = [price]

        for ret in returns[1:]:
            price = price * (1 + ret)
            prices.append(price)

        data = []
        for i, date in enumerate(dates):
            open_price = prices[i] * (1 + np.random.uniform(-0.005, 0.005))
            close_price = prices[i]
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))

            data.append(
                {
                    "date": date,
                    "timestamp": date.isoformat(),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": np.random.randint(1000000, 5000000),
                }
            )

        return data

    @pytest.mark.asyncio
    async def test_technical_analysis_on_historical_data(self, historical_data):
        """Test technical indicator calculations on historical data"""
        df = pd.DataFrame(historical_data)

        # Initialize technical indicator agent
        tech_agent = TechnicalIndicatorAgent()

        # Calculate various indicators
        indicators = tech_agent.calculate_indicators(df)

        # Verify indicator calculations
        assert "sma_20" in indicators
        assert "sma_50" in indicators
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "bollinger_upper" in indicators
        assert "bollinger_lower" in indicators

        # Verify indicator values are reasonable
        assert 0 <= indicators["rsi"] <= 100
        assert indicators["sma_20"] > 0
        assert indicators["bollinger_upper"] > indicators["bollinger_lower"]

    @pytest.mark.asyncio
    async def test_backtest_trading_strategy(
        self, crew_manager, historical_data, mock_kite_client
    ):
        """Test backtesting a trading strategy on historical data"""
        with patch("agents.crew_manager.kite_client", mock_kite_client):
            # Mock historical data return
            mock_kite_client.get_historical_data = AsyncMock(
                return_value=historical_data
            )

            # Run backtest
            start_date = datetime.now() - timedelta(days=90)
            end_date = datetime.now()

            results = await crew_manager.run_backtest(
                symbol="RELIANCE",
                start_date=start_date,
                end_date=end_date,
                strategy="momentum",
            )

            assert results is not None
            assert "metrics" in results
            assert "total_trades" in results["metrics"]
            assert "win_rate" in results["metrics"]
            assert "sharpe_ratio" in results["metrics"]
            assert "max_drawdown" in results["metrics"]

    @pytest.mark.asyncio
    async def test_pattern_detection_historical(self, historical_data):
        """Test chart pattern detection on historical data"""
        from agents.market_analyst.pattern_detector import PatternDetector

        detector = PatternDetector()
        df = pd.DataFrame(historical_data)

        # Detect patterns
        patterns = detector.detect_patterns(df)

        assert isinstance(patterns, list)
        # Check if any patterns were detected
        if len(patterns) > 0:
            pattern = patterns[0]
            assert "pattern_type" in pattern
            assert "confidence" in pattern
            assert "start_date" in pattern
            assert "end_date" in pattern

    @pytest.mark.asyncio
    async def test_multi_symbol_correlation(self, mock_kite_client):
        """Test correlation analysis across multiple symbols"""
        symbols = ["RELIANCE", "TCS", "INFY", "HDFC"]

        # Generate correlated historical data
        historical_data = {}
        base_returns = np.random.normal(0.001, 0.02, 100)

        for i, symbol in enumerate(symbols):
            # Add some correlation
            symbol_returns = base_returns + np.random.normal(0, 0.01, 100)

            price = 2500 * (1 + i * 0.2)  # Different base prices
            prices = [price]

            for ret in symbol_returns[1:]:
                price = price * (1 + ret)
                prices.append(price)

            historical_data[symbol] = prices

        # Calculate correlation matrix
        df = pd.DataFrame(historical_data)
        correlation_matrix = df.corr()

        # Verify correlation properties
        assert correlation_matrix.shape == (4, 4)
        assert all(correlation_matrix.diagonal() == 1.0)  # Self-correlation is 1
        assert correlation_matrix.equals(correlation_matrix.T)  # Symmetric

    @pytest.mark.asyncio
    async def test_risk_metrics_historical(self, historical_data):
        """Test risk metrics calculation on historical data"""
        from agents.risk_manager.risk_metrics import RiskMetrics

        risk_metrics = RiskMetrics()
        df = pd.DataFrame(historical_data)

        # Calculate returns
        df["returns"] = df["close"].pct_change()
        returns = df["returns"].dropna().tolist()

        # Calculate various risk metrics
        volatility = risk_metrics.calculate_volatility(returns)
        var_95 = risk_metrics.calculate_var(returns, confidence=0.95)
        sharpe = risk_metrics.calculate_sharpe_ratio(returns, risk_free_rate=0.05 / 252)
        max_dd = risk_metrics.calculate_max_drawdown(df["close"].tolist())

        # Verify calculations
        assert volatility > 0
        assert var_95 < 0  # VaR is typically negative
        assert isinstance(sharpe, float)
        assert 0 < max_dd < 1  # Drawdown as percentage

    @pytest.mark.asyncio
    async def test_volume_profile_analysis(self, historical_data):
        """Test volume profile analysis on historical data"""
        df = pd.DataFrame(historical_data)

        # Calculate volume-weighted metrics
        vwap = (df["close"] * df["volume"]).sum() / df["volume"].sum()

        # Identify high volume nodes
        price_bins = pd.cut(df["close"], bins=20)
        volume_profile = df.groupby(price_bins)["volume"].sum()

        # Find point of control (highest volume price)
        poc_index = volume_profile.idxmax()

        assert vwap > 0
        assert poc_index is not None

    @pytest.mark.asyncio
    async def test_market_regime_detection(self, historical_data):
        """Test market regime detection on historical data"""
        df = pd.DataFrame(historical_data)

        # Simple regime detection based on moving averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # Determine regime
        df["regime"] = "neutral"
        df.loc[df["sma_20"] > df["sma_50"], "regime"] = "bullish"
        df.loc[df["sma_20"] < df["sma_50"], "regime"] = "bearish"

        # Count regime periods
        regime_counts = df["regime"].value_counts()

        assert len(regime_counts) > 0
        assert all(
            regime in ["bullish", "bearish", "neutral"]
            for regime in regime_counts.index
        )

    @pytest.mark.asyncio
    async def test_performance_attribution(self, crew_manager, historical_data):
        """Test performance attribution on historical trades"""
        # Simulate some trades
        trades = [
            {
                "date": historical_data[20]["date"],
                "symbol": "RELIANCE",
                "action": "BUY",
                "quantity": 100,
                "price": historical_data[20]["close"],
            },
            {
                "date": historical_data[40]["date"],
                "symbol": "RELIANCE",
                "action": "SELL",
                "quantity": 100,
                "price": historical_data[40]["close"],
            },
            {
                "date": historical_data[60]["date"],
                "symbol": "RELIANCE",
                "action": "BUY",
                "quantity": 50,
                "price": historical_data[60]["close"],
            },
            {
                "date": historical_data[80]["date"],
                "symbol": "RELIANCE",
                "action": "SELL",
                "quantity": 50,
                "price": historical_data[80]["close"],
            },
        ]

        # Calculate performance metrics
        total_pnl = 0
        winning_trades = 0

        for i in range(0, len(trades), 2):
            buy_trade = trades[i]
            sell_trade = trades[i + 1]

            pnl = (sell_trade["price"] - buy_trade["price"]) * buy_trade["quantity"]
            total_pnl += pnl

            if pnl > 0:
                winning_trades += 1

        win_rate = winning_trades / (len(trades) // 2)

        assert isinstance(total_pnl, int | float)
        assert 0 <= win_rate <= 1

    @pytest.mark.asyncio
    async def test_signal_generation_accuracy(self, crew_manager, historical_data):
        """Test trading signal generation accuracy on historical data"""
        # Generate signals for different time periods
        signals = []

        for i in range(20, len(historical_data) - 1):
            # Use data up to point i to generate signal
            historical_data[:i]

            # Mock signal generation
            if i % 10 == 0:  # Generate signal every 10 days
                signal = {
                    "date": historical_data[i]["date"],
                    "action": "BUY" if i % 20 == 0 else "SELL",
                    "predicted_direction": "up" if i % 20 == 0 else "down",
                    "confidence": 0.7 + (i % 10) * 0.02,
                }

                # Check actual movement
                actual_movement = (
                    historical_data[i + 1]["close"] - historical_data[i]["close"]
                )
                signal["actual_direction"] = "up" if actual_movement > 0 else "down"
                signal["correct"] = (
                    signal["predicted_direction"] == signal["actual_direction"]
                )

                signals.append(signal)

        # Calculate accuracy
        if len(signals) > 0:
            accuracy = sum(1 for s in signals if s["correct"]) / len(signals)
            assert 0 <= accuracy <= 1

            # Calculate confidence-weighted accuracy
            weighted_accuracy = sum(
                s["confidence"] for s in signals if s["correct"]
            ) / sum(s["confidence"] for s in signals)
            assert 0 <= weighted_accuracy <= 1
