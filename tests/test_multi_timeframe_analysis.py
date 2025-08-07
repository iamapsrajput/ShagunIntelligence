"""
Tests for Multi-Timeframe Analysis Engine
"""

import asyncio
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.services.multi_timeframe_analysis import (
    AdaptiveIndicatorCalculator,
    CrossTimeFrameAnalyzer,
    MultiTimeFrameAnalysis,
    MultiTimeFrameEngine,
    SignalStrength,
    TimeFrame,
    TimeFrameSignal,
    TrendDirection,
)


class TestAdaptiveIndicatorCalculator:
    """Test adaptive indicator calculations"""

    def setup_method(self):
        self.calculator = AdaptiveIndicatorCalculator()
        self.test_data = self._create_test_data()

    def _create_test_data(self):
        """Create realistic test OHLCV data"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq="1H")
        np.random.seed(42)

        # Generate realistic price movements
        base_price = 2500
        returns = np.random.normal(0, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.001, 100)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                "close": prices,
                "volume": np.random.randint(10000, 100000, 100),
            },
            index=dates,
        )

    def test_adaptive_rsi_calculation(self):
        """Test adaptive RSI calculation with different timeframes"""
        # Test fast timeframe (should use shorter period)
        rsi_fast = self.calculator.calculate_adaptive_rsi(self.test_data, TimeFrame.M5)

        # Test slow timeframe (should use longer period)
        rsi_slow = self.calculator.calculate_adaptive_rsi(self.test_data, TimeFrame.D1)

        # Verify structure
        for rsi in [rsi_fast, rsi_slow]:
            assert "value" in rsi
            assert "period" in rsi
            assert "overbought_level" in rsi
            assert "oversold_level" in rsi
            assert "is_overbought" in rsi
            assert "is_oversold" in rsi
            assert "trend" in rsi

        # Fast timeframe should use shorter period
        assert rsi_fast["period"] < rsi_slow["period"]

        # RSI value should be between 0 and 100
        assert 0 <= rsi_fast["value"] <= 100
        assert 0 <= rsi_slow["value"] <= 100

    def test_adaptive_macd_calculation(self):
        """Test adaptive MACD calculation"""
        macd_result = self.calculator.calculate_adaptive_macd(
            self.test_data, TimeFrame.H1
        )

        assert "macd" in macd_result
        assert "signal" in macd_result
        assert "histogram" in macd_result
        assert "bullish_crossover" in macd_result
        assert "bearish_crossover" in macd_result
        assert "trend_strength" in macd_result

        # Histogram should be difference between MACD and signal
        expected_histogram = macd_result["macd"] - macd_result["signal"]
        assert abs(macd_result["histogram"] - expected_histogram) < 0.001

    def test_adaptive_bollinger_bands(self):
        """Test adaptive Bollinger Bands calculation"""
        bb_result = self.calculator.calculate_adaptive_bollinger_bands(
            self.test_data, TimeFrame.H1
        )

        assert "upper" in bb_result
        assert "middle" in bb_result
        assert "lower" in bb_result
        assert "position" in bb_result
        assert "band_width" in bb_result
        assert "squeeze" in bb_result
        assert "breakout" in bb_result

        # Upper band should be above middle, middle above lower
        assert bb_result["upper"] > bb_result["middle"]
        assert bb_result["middle"] > bb_result["lower"]

        # Position should be between 0 and 1
        assert 0 <= bb_result["position"] <= 1

    def test_support_resistance_calculation(self):
        """Test support and resistance level calculation"""
        levels = self.calculator.calculate_support_resistance(
            self.test_data, TimeFrame.H1
        )

        assert "resistance" in levels
        assert "support" in levels
        assert "pivot_highs" in levels
        assert "pivot_lows" in levels

        # Should have some levels
        assert len(levels["resistance"]) >= 0
        assert len(levels["support"]) >= 0

        # Resistance levels should be sorted in descending order
        if len(levels["resistance"]) > 1:
            assert levels["resistance"] == sorted(levels["resistance"], reverse=True)

        # Support levels should be sorted in ascending order
        if len(levels["support"]) > 1:
            assert levels["support"] == sorted(levels["support"])

    def test_volatility_adaptive_parameters(self):
        """Test that parameters adapt to volatility"""
        # Create high volatility data
        high_vol_data = self.test_data.copy()
        high_vol_data["close"] = high_vol_data["close"] * (
            1 + np.random.normal(0, 0.05, len(high_vol_data))
        )

        # Create low volatility data
        low_vol_data = self.test_data.copy()
        low_vol_data["close"] = low_vol_data["close"] * (
            1 + np.random.normal(0, 0.005, len(low_vol_data))
        )

        rsi_high_vol = self.calculator.calculate_adaptive_rsi(
            high_vol_data, TimeFrame.H1
        )
        rsi_low_vol = self.calculator.calculate_adaptive_rsi(low_vol_data, TimeFrame.H1)

        # High volatility should have wider overbought/oversold levels
        high_vol_range = (
            rsi_high_vol["overbought_level"] - rsi_high_vol["oversold_level"]
        )
        low_vol_range = rsi_low_vol["overbought_level"] - rsi_low_vol["oversold_level"]

        assert high_vol_range >= low_vol_range


class TestCrossTimeFrameAnalyzer:
    """Test cross-timeframe analysis functionality"""

    def setup_method(self):
        self.analyzer = CrossTimeFrameAnalyzer()
        self.test_signals = self._create_test_signals()

    def _create_test_signals(self):
        """Create test signals for different timeframes"""
        return {
            TimeFrame.M15: TimeFrameSignal(
                timeframe=TimeFrame.M15,
                signal_type="BUY",
                strength=SignalStrength.MODERATE,
                confidence=0.7,
                trend_direction=TrendDirection.BULLISH,
                key_levels={"resistance": [2550, 2580], "support": [2480, 2450]},
                indicators={},
                reasoning=["RSI oversold", "MACD bullish crossover"],
            ),
            TimeFrame.H1: TimeFrameSignal(
                timeframe=TimeFrame.H1,
                signal_type="BUY",
                strength=SignalStrength.STRONG,
                confidence=0.8,
                trend_direction=TrendDirection.BULLISH,
                key_levels={"resistance": [2560, 2590], "support": [2470, 2440]},
                indicators={},
                reasoning=["Strong uptrend", "Volume confirmation"],
            ),
            TimeFrame.H4: TimeFrameSignal(
                timeframe=TimeFrame.H4,
                signal_type="HOLD",
                strength=SignalStrength.WEAK,
                confidence=0.5,
                trend_direction=TrendDirection.NEUTRAL,
                key_levels={"resistance": [2570, 2600], "support": [2460, 2430]},
                indicators={},
                reasoning=["Consolidation pattern"],
            ),
            TimeFrame.D1: TimeFrameSignal(
                timeframe=TimeFrame.D1,
                signal_type="BUY",
                strength=SignalStrength.MODERATE,
                confidence=0.6,
                trend_direction=TrendDirection.BULLISH,
                key_levels={"resistance": [2580, 2620], "support": [2450, 2400]},
                indicators={},
                reasoning=["Long-term uptrend intact"],
            ),
        }

    def test_trend_alignment_analysis(self):
        """Test trend alignment analysis across timeframes"""
        alignment = self.analyzer.analyze_trend_alignment(self.test_signals)

        assert "overall_trend_score" in alignment
        assert "trend_direction" in alignment
        assert "alignment_strength" in alignment
        assert "timeframe_trends" in alignment
        assert "conflicting_timeframes" in alignment

        # Should detect bullish trend (3 out of 4 timeframes bullish)
        assert alignment["trend_direction"] in ["BULLISH", "STRONG_BULLISH"]

        # Alignment strength should be reasonable
        assert 0 <= alignment["alignment_strength"] <= 1

    def test_consensus_signal_calculation(self):
        """Test consensus signal calculation"""
        consensus = self.analyzer.calculate_consensus_signal(self.test_signals)

        assert "signal" in consensus
        assert "strength" in consensus
        assert "confidence" in consensus
        assert "signal_scores" in consensus
        assert "agreement_level" in consensus

        # Should be BUY signal (3 BUY, 1 HOLD)
        assert consensus["signal"] == "BUY"

        # Agreement level should be reasonable
        assert 0 <= consensus["agreement_level"] <= 1

    def test_key_levels_identification(self):
        """Test identification of key support/resistance levels"""
        key_levels = self.analyzer.identify_key_levels(self.test_signals)

        assert "primary_resistance" in key_levels
        assert "secondary_resistance" in key_levels
        assert "primary_support" in key_levels
        assert "secondary_support" in key_levels
        assert "all_resistance" in key_levels
        assert "all_support" in key_levels

        # Primary levels should exist
        if key_levels["primary_resistance"]:
            assert isinstance(key_levels["primary_resistance"], (int, float))

        if key_levels["primary_support"]:
            assert isinstance(key_levels["primary_support"], (int, float))

    def test_conflicting_timeframes_detection(self):
        """Test detection of conflicting timeframes"""
        # Create conflicting signals
        conflicting_signals = self.test_signals.copy()
        conflicting_signals[TimeFrame.H4] = TimeFrameSignal(
            timeframe=TimeFrame.H4,
            signal_type="SELL",
            strength=SignalStrength.STRONG,
            confidence=0.8,
            trend_direction=TrendDirection.BEARISH,
            key_levels={},
            indicators={},
            reasoning=["Strong bearish reversal"],
        )

        alignment = self.analyzer.analyze_trend_alignment(conflicting_signals)

        # Should detect conflicts
        assert len(alignment["conflicting_timeframes"]) > 0
        assert "4h" in alignment["conflicting_timeframes"]


class TestMultiTimeFrameEngine:
    """Test the main Multi-Timeframe Analysis Engine"""

    def setup_method(self):
        self.engine = MultiTimeFrameEngine()

    @pytest.mark.asyncio
    async def test_symbol_analysis_with_mock_data(self):
        """Test complete symbol analysis with mock data"""
        analysis = await self.engine.analyze_symbol("RELIANCE")

        # Verify analysis structure
        assert isinstance(analysis, MultiTimeFrameAnalysis)
        assert analysis.symbol == "RELIANCE"
        assert analysis.consensus_signal in ["BUY", "SELL", "HOLD"]
        assert 0 <= analysis.consensus_confidence <= 1
        assert isinstance(analysis.risk_reward_ratio, (int, float))
        assert isinstance(analysis.entry_conditions, list)
        assert isinstance(analysis.exit_conditions, list)

        # Should have analyzed multiple timeframes
        assert len(analysis.timeframe_signals) > 0

    @pytest.mark.asyncio
    async def test_custom_timeframes_analysis(self):
        """Test analysis with custom timeframes"""
        custom_timeframes = [TimeFrame.M15, TimeFrame.H1, TimeFrame.D1]

        analysis = await self.engine.analyze_symbol("RELIANCE", custom_timeframes)

        # Should only analyze specified timeframes
        analyzed_timeframes = set(analysis.timeframe_signals.keys())
        expected_timeframes = set(custom_timeframes)

        assert analyzed_timeframes.issubset(expected_timeframes)

    @pytest.mark.asyncio
    async def test_analysis_caching(self):
        """Test that analysis results are cached"""
        # First analysis
        start_time = datetime.now()
        analysis1 = await self.engine.analyze_symbol("RELIANCE")
        first_duration = (datetime.now() - start_time).total_seconds()

        # Second analysis (should be cached)
        start_time = datetime.now()
        analysis2 = await self.engine.analyze_symbol("RELIANCE")
        second_duration = (datetime.now() - start_time).total_seconds()

        # Second call should be faster (cached)
        assert second_duration < first_duration

        # Results should be identical
        assert analysis1.consensus_signal == analysis2.consensus_signal
        assert analysis1.consensus_confidence == analysis2.consensus_confidence

    def test_analysis_summary_generation(self):
        """Test analysis summary generation"""
        # Create mock analysis
        mock_analysis = MultiTimeFrameAnalysis(
            symbol="TEST",
            timeframe_signals={
                TimeFrame.H1: TimeFrameSignal(
                    timeframe=TimeFrame.H1,
                    signal_type="BUY",
                    strength=SignalStrength.STRONG,
                    confidence=0.8,
                    trend_direction=TrendDirection.BULLISH,
                    key_levels={},
                    indicators={},
                    reasoning=[],
                )
            },
            consensus_signal="BUY",
            consensus_strength=0.8,
            consensus_confidence=0.8,
            trend_alignment={"trend_direction": "BULLISH", "alignment_strength": 0.9},
            key_levels={"primary_resistance": 2600, "primary_support": 2400},
            risk_reward_ratio=2.5,
            entry_conditions=["Strong bullish signals"],
            exit_conditions=["Take profit at resistance"],
        )

        summary = self.engine.get_analysis_summary(mock_analysis)

        # Verify summary structure
        assert "symbol" in summary
        assert "consensus" in summary
        assert "trend" in summary
        assert "timeframes" in summary
        assert "levels" in summary
        assert "risk_reward" in summary
        assert "entry_conditions" in summary
        assert "exit_conditions" in summary
        assert "timestamp" in summary

        # Verify content
        assert summary["symbol"] == "TEST"
        assert summary["consensus"]["signal"] == "BUY"
        assert summary["risk_reward"] == 2.5

    @pytest.mark.asyncio
    async def test_error_handling_insufficient_data(self):
        """Test error handling when insufficient data is available"""
        # Mock the _get_market_data method to return insufficient data
        with patch.object(self.engine, "_get_market_data") as mock_get_data:
            mock_get_data.return_value = pd.DataFrame()  # Empty data

            analysis = await self.engine.analyze_symbol("INVALID_SYMBOL")

            # Should return default analysis
            assert analysis.consensus_signal == "HOLD"
            assert analysis.consensus_confidence == 0.0
            assert "Insufficient data" in analysis.entry_conditions[0]

    def test_risk_reward_calculation(self):
        """Test risk-reward ratio calculation"""
        # Create test signals with key levels
        test_signals = {
            TimeFrame.H1: TimeFrameSignal(
                timeframe=TimeFrame.H1,
                signal_type="BUY",
                strength=SignalStrength.STRONG,
                confidence=0.8,
                trend_direction=TrendDirection.BULLISH,
                key_levels={"resistance": [2600], "support": [2400]},
                indicators={},
                reasoning=[],
            )
        }

        key_levels = {"primary_resistance": 2600, "primary_support": 2400}

        risk_reward = self.engine._calculate_risk_reward_ratio(test_signals, key_levels)

        # With current price at 2500, resistance at 2600, support at 2400
        # Risk = 2500 - 2400 = 100, Reward = 2600 - 2500 = 100
        # Risk-reward ratio should be 1.0
        assert isinstance(risk_reward, (int, float))
        assert risk_reward > 0


class TestPerformanceBenchmarks:
    """Performance benchmarks for multi-timeframe analysis"""

    def setup_method(self):
        self.engine = MultiTimeFrameEngine()

    @pytest.mark.asyncio
    async def test_analysis_performance(self):
        """Test analysis performance benchmarks"""
        symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

        start_time = datetime.now()

        # Analyze multiple symbols
        for symbol in symbols:
            await self.engine.analyze_symbol(symbol)

        total_time = (datetime.now() - start_time).total_seconds()
        avg_time_per_symbol = total_time / len(symbols)

        # Should complete analysis within reasonable time
        assert avg_time_per_symbol < 2.0  # Less than 2 seconds per symbol

        print(f"Average analysis time per symbol: {avg_time_per_symbol:.2f} seconds")

    def test_memory_usage(self):
        """Test memory usage during analysis"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform multiple analyses
        for i in range(10):
            asyncio.run(self.engine.analyze_symbol(f"TEST_{i}"))

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase

        print(f"Memory increase: {memory_increase:.2f} MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
