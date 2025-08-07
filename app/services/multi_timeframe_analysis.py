"""
Multi-Timeframe Analysis Engine
Advanced technical analysis system supporting multiple timeframes with cross-timeframe signal confirmation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Using pandas implementations.")

from app.core.resilience import with_circuit_breaker, with_retry


class TimeFrame(Enum):
    """Supported timeframes for analysis"""

    M1 = "1m"  # 1 minute
    M5 = "5m"  # 5 minutes
    M15 = "15m"  # 15 minutes
    M30 = "30m"  # 30 minutes
    H1 = "1h"  # 1 hour
    H4 = "4h"  # 4 hours
    D1 = "1d"  # 1 day
    W1 = "1w"  # 1 week


class SignalStrength(Enum):
    """Signal strength levels"""

    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


class TrendDirection(Enum):
    """Trend direction classification"""

    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class TimeFrameSignal:
    """Signal from a specific timeframe"""

    timeframe: TimeFrame
    signal_type: str  # BUY, SELL, HOLD
    strength: SignalStrength
    confidence: float
    trend_direction: TrendDirection
    key_levels: dict[str, float]
    indicators: dict[str, Any]
    reasoning: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MultiTimeFrameAnalysis:
    """Comprehensive multi-timeframe analysis result"""

    symbol: str
    timeframe_signals: dict[TimeFrame, TimeFrameSignal]
    consensus_signal: str
    consensus_strength: float
    consensus_confidence: float
    trend_alignment: dict[str, Any]
    key_levels: dict[str, float]
    risk_reward_ratio: float
    entry_conditions: list[str]
    exit_conditions: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


class AdaptiveIndicatorCalculator:
    """Advanced indicator calculator with adaptive parameters"""

    def __init__(self):
        self.use_talib = TALIB_AVAILABLE

        # Adaptive parameter ranges
        self.rsi_periods = {"fast": 9, "medium": 14, "slow": 21}
        self.ma_periods = {
            "fast": [5, 10, 20],
            "medium": [10, 20, 50],
            "slow": [20, 50, 200],
        }
        self.bb_periods = {"fast": 10, "medium": 20, "slow": 30}

    def calculate_adaptive_rsi(
        self, data: pd.DataFrame, timeframe: TimeFrame
    ) -> dict[str, Any]:
        """Calculate RSI with adaptive periods based on timeframe"""
        try:
            close_prices = data["close"].values

            # Select period based on timeframe
            if timeframe in [TimeFrame.M1, TimeFrame.M5]:
                period = self.rsi_periods["fast"]
            elif timeframe in [TimeFrame.M15, TimeFrame.M30, TimeFrame.H1]:
                period = self.rsi_periods["medium"]
            else:
                period = self.rsi_periods["slow"]

            if self.use_talib:
                rsi = talib.RSI(close_prices, timeperiod=period)
            else:
                rsi = self._calculate_rsi_pandas(close_prices, period)

            current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50

            # Dynamic overbought/oversold levels based on volatility
            volatility = data["close"].pct_change().std() * 100
            if volatility > 3:  # High volatility
                overbought, oversold = 75, 25
            elif volatility > 1.5:  # Medium volatility
                overbought, oversold = 70, 30
            else:  # Low volatility
                overbought, oversold = 65, 35

            return {
                "value": current_rsi,
                "period": period,
                "overbought_level": overbought,
                "oversold_level": oversold,
                "is_overbought": current_rsi > overbought,
                "is_oversold": current_rsi < oversold,
                "divergence": self._detect_rsi_divergence(data, rsi),
                "trend": "bullish" if current_rsi > 50 else "bearish",
            }

        except Exception as e:
            logger.error(f"Error calculating adaptive RSI: {str(e)}")
            return {"value": 50, "period": 14, "error": str(e)}

    def calculate_adaptive_macd(
        self, data: pd.DataFrame, timeframe: TimeFrame
    ) -> dict[str, Any]:
        """Calculate MACD with adaptive parameters"""
        try:
            close_prices = data["close"].values

            # Adaptive MACD parameters based on timeframe
            if timeframe in [TimeFrame.M1, TimeFrame.M5]:
                fast, slow, signal = 8, 17, 9
            elif timeframe in [TimeFrame.M15, TimeFrame.M30]:
                fast, slow, signal = 12, 26, 9
            else:
                fast, slow, signal = 12, 26, 9

            if self.use_talib:
                macd_line, macd_signal, macd_histogram = talib.MACD(
                    close_prices, fastperiod=fast, slowperiod=slow, signalperiod=signal
                )
            else:
                macd_line, macd_signal, macd_histogram = self._calculate_macd_pandas(
                    close_prices, fast, slow, signal
                )

            current_macd = macd_line[-1] if not np.isnan(macd_line[-1]) else 0
            current_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            current_histogram = (
                macd_histogram[-1] if not np.isnan(macd_histogram[-1]) else 0
            )

            # Detect crossovers
            bullish_crossover = (
                current_macd > current_signal and macd_line[-2] <= macd_signal[-2]
            )
            bearish_crossover = (
                current_macd < current_signal and macd_line[-2] >= macd_signal[-2]
            )

            return {
                "macd": current_macd,
                "signal": current_signal,
                "histogram": current_histogram,
                "bullish_crossover": bullish_crossover,
                "bearish_crossover": bearish_crossover,
                "divergence": self._detect_macd_divergence(data, macd_line),
                "trend_strength": abs(current_histogram)
                / max(abs(current_macd), 0.001),
            }

        except Exception as e:
            logger.error(f"Error calculating adaptive MACD: {str(e)}")
            return {"macd": 0, "signal": 0, "histogram": 0, "error": str(e)}

    def calculate_adaptive_bollinger_bands(
        self, data: pd.DataFrame, timeframe: TimeFrame
    ) -> dict[str, Any]:
        """Calculate Bollinger Bands with adaptive parameters"""
        try:
            close_prices = data["close"].values

            # Adaptive period based on timeframe
            if timeframe in [TimeFrame.M1, TimeFrame.M5]:
                period = self.bb_periods["fast"]
            elif timeframe in [TimeFrame.M15, TimeFrame.M30, TimeFrame.H1]:
                period = self.bb_periods["medium"]
            else:
                period = self.bb_periods["slow"]

            # Dynamic standard deviation multiplier based on volatility
            volatility = data["close"].pct_change().std()
            if volatility > 0.03:  # High volatility
                std_dev = 2.5
            elif volatility > 0.015:  # Medium volatility
                std_dev = 2.0
            else:  # Low volatility
                std_dev = 1.5

            if self.use_talib:
                upper_band, middle_band, lower_band = talib.BBANDS(
                    close_prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
                )
            else:
                upper_band, middle_band, lower_band = self._calculate_bb_pandas(
                    close_prices, period, std_dev
                )

            current_price = close_prices[-1]
            current_upper = upper_band[-1]
            current_middle = middle_band[-1]
            current_lower = lower_band[-1]

            # Calculate position within bands
            band_width = current_upper - current_lower
            position = (
                (current_price - current_lower) / band_width if band_width > 0 else 0.5
            )

            return {
                "upper": current_upper,
                "middle": current_middle,
                "lower": current_lower,
                "position": position,
                "band_width": band_width,
                "squeeze": band_width
                < (current_middle * 0.1),  # Band squeeze detection
                "breakout": current_price > current_upper
                or current_price < current_lower,
                "trend": (
                    "bullish"
                    if position > 0.8
                    else "bearish" if position < 0.2 else "neutral"
                ),
            }

        except Exception as e:
            logger.error(f"Error calculating adaptive Bollinger Bands: {str(e)}")
            return {"upper": 0, "middle": 0, "lower": 0, "error": str(e)}

    def calculate_support_resistance(
        self, data: pd.DataFrame, timeframe: TimeFrame
    ) -> dict[str, list[float]]:
        """Calculate dynamic support and resistance levels"""
        try:
            # Adaptive lookback period based on timeframe
            if timeframe in [TimeFrame.M1, TimeFrame.M5]:
                lookback = 20
            elif timeframe in [TimeFrame.M15, TimeFrame.M30]:
                lookback = 50
            else:
                lookback = 100

            lookback = min(lookback, len(data) - 1)

            highs = data["high"].rolling(window=5, center=True).max()
            lows = data["low"].rolling(window=5, center=True).min()

            # Find pivot points
            pivot_highs = data[data["high"] == highs]["high"].dropna()
            pivot_lows = data[data["low"] == lows]["low"].dropna()

            # Get recent pivots
            recent_highs = pivot_highs.tail(lookback // 5).tolist()
            recent_lows = pivot_lows.tail(lookback // 5).tolist()

            # Cluster levels (group nearby levels)
            resistance_levels = self._cluster_levels(recent_highs)
            support_levels = self._cluster_levels(recent_lows)

            return {
                "resistance": sorted(resistance_levels, reverse=True)[:5],
                "support": sorted(support_levels)[:5],
                "pivot_highs": recent_highs,
                "pivot_lows": recent_lows,
            }

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return {
                "resistance": [],
                "support": [],
                "pivot_highs": [],
                "pivot_lows": [],
            }

    def _calculate_rsi_pandas(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Pandas implementation of RSI"""
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    def _calculate_macd_pandas(
        self, prices: np.ndarray, fast: int, slow: int, signal: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pandas implementation of MACD"""
        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=fast).mean()
        ema_slow = prices_series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line.values, signal_line.values, histogram.values

    def _calculate_bb_pandas(
        self, prices: np.ndarray, period: int, std_dev: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pandas implementation of Bollinger Bands"""
        prices_series = pd.Series(prices)
        middle = prices_series.rolling(window=period).mean()
        std = prices_series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper.values, middle.values, lower.values

    def _detect_rsi_divergence(self, data: pd.DataFrame, rsi: np.ndarray) -> bool:
        """Detect RSI divergence with price"""
        try:
            if len(data) < 10:
                return False

            # Simple divergence detection (can be enhanced)
            price_trend = data["close"].iloc[-5:].is_monotonic_increasing
            rsi_trend = pd.Series(rsi[-5:]).is_monotonic_increasing

            return price_trend != rsi_trend
        except:
            return False

    def _detect_macd_divergence(self, data: pd.DataFrame, macd: np.ndarray) -> bool:
        """Detect MACD divergence with price"""
        try:
            if len(data) < 10:
                return False

            price_trend = data["close"].iloc[-5:].is_monotonic_increasing
            macd_trend = pd.Series(macd[-5:]).is_monotonic_increasing

            return price_trend != macd_trend
        except:
            return False

    def _cluster_levels(
        self, levels: list[float], threshold: float = 0.02
    ) -> list[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []

        clustered = []
        sorted_levels = sorted(levels)

        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= threshold:
                current_cluster.append(level)
            else:
                # Average the cluster
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        # Add the last cluster
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))

        return clustered


class CrossTimeFrameAnalyzer:
    """Analyzes signals across multiple timeframes for confirmation"""

    def __init__(self):
        self.timeframe_weights = {
            TimeFrame.M1: 0.05,
            TimeFrame.M5: 0.10,
            TimeFrame.M15: 0.15,
            TimeFrame.M30: 0.20,
            TimeFrame.H1: 0.25,
            TimeFrame.H4: 0.30,
            TimeFrame.D1: 0.40,
            TimeFrame.W1: 0.50,
        }

    def analyze_trend_alignment(
        self, signals: dict[TimeFrame, TimeFrameSignal]
    ) -> dict[str, Any]:
        """Analyze trend alignment across timeframes"""
        try:
            trend_scores = {}
            weighted_score = 0
            total_weight = 0

            for timeframe, signal in signals.items():
                weight = self.timeframe_weights.get(timeframe, 0.1)

                # Convert trend to numeric score
                trend_score = self._trend_to_score(signal.trend_direction)
                trend_scores[timeframe.value] = trend_score

                weighted_score += trend_score * weight
                total_weight += weight

            overall_trend_score = (
                weighted_score / total_weight if total_weight > 0 else 0
            )

            # Determine alignment quality
            trend_values = list(trend_scores.values())
            alignment_strength = (
                1.0 - (np.std(trend_values) / 2.0) if trend_values else 0
            )

            return {
                "overall_trend_score": overall_trend_score,
                "trend_direction": self._score_to_trend(overall_trend_score),
                "alignment_strength": max(0, min(1, alignment_strength)),
                "timeframe_trends": trend_scores,
                "conflicting_timeframes": self._find_conflicting_timeframes(
                    trend_scores
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing trend alignment: {str(e)}")
            return {
                "overall_trend_score": 0,
                "trend_direction": "NEUTRAL",
                "alignment_strength": 0,
            }

    def calculate_consensus_signal(
        self, signals: dict[TimeFrame, TimeFrameSignal]
    ) -> dict[str, Any]:
        """Calculate consensus signal from multiple timeframes"""
        try:
            signal_scores = {"BUY": 0, "SELL": 0, "HOLD": 0}
            total_weight = 0
            confidence_sum = 0

            for timeframe, signal in signals.items():
                weight = self.timeframe_weights.get(timeframe, 0.1)
                confidence_weight = weight * signal.confidence

                signal_scores[signal.signal_type] += confidence_weight
                total_weight += weight
                confidence_sum += signal.confidence * weight

            # Normalize scores
            if total_weight > 0:
                for signal_type in signal_scores:
                    signal_scores[signal_type] /= total_weight

                consensus_confidence = confidence_sum / total_weight
            else:
                consensus_confidence = 0

            # Determine consensus signal
            consensus_signal = max(signal_scores, key=signal_scores.get)
            consensus_strength = signal_scores[consensus_signal]

            return {
                "signal": consensus_signal,
                "strength": consensus_strength,
                "confidence": consensus_confidence,
                "signal_scores": signal_scores,
                "agreement_level": self._calculate_agreement_level(signals),
            }

        except Exception as e:
            logger.error(f"Error calculating consensus signal: {str(e)}")
            return {"signal": "HOLD", "strength": 0, "confidence": 0}

    def identify_key_levels(
        self, signals: dict[TimeFrame, TimeFrameSignal]
    ) -> dict[str, float]:
        """Identify key support and resistance levels across timeframes"""
        try:
            all_resistance = []
            all_support = []

            for timeframe, signal in signals.items():
                weight = self.timeframe_weights.get(timeframe, 0.1)

                # Weight levels by timeframe importance
                resistance_levels = signal.key_levels.get("resistance", [])
                support_levels = signal.key_levels.get("support", [])

                for level in resistance_levels:
                    all_resistance.append((level, weight))

                for level in support_levels:
                    all_support.append((level, weight))

            # Cluster and weight levels
            key_resistance = self._weight_and_cluster_levels(all_resistance)
            key_support = self._weight_and_cluster_levels(all_support)

            return {
                "primary_resistance": key_resistance[0] if key_resistance else None,
                "secondary_resistance": (
                    key_resistance[1] if len(key_resistance) > 1 else None
                ),
                "primary_support": key_support[0] if key_support else None,
                "secondary_support": key_support[1] if len(key_support) > 1 else None,
                "all_resistance": key_resistance[:5],
                "all_support": key_support[:5],
            }

        except Exception as e:
            logger.error(f"Error identifying key levels: {str(e)}")
            return {}

    def _trend_to_score(self, trend: TrendDirection) -> float:
        """Convert trend direction to numeric score"""
        trend_map = {
            TrendDirection.STRONG_BEARISH: -2.0,
            TrendDirection.BEARISH: -1.0,
            TrendDirection.NEUTRAL: 0.0,
            TrendDirection.BULLISH: 1.0,
            TrendDirection.STRONG_BULLISH: 2.0,
        }
        return trend_map.get(trend, 0.0)

    def _score_to_trend(self, score: float) -> str:
        """Convert numeric score to trend direction"""
        if score >= 1.5:
            return "STRONG_BULLISH"
        elif score >= 0.5:
            return "BULLISH"
        elif score <= -1.5:
            return "STRONG_BEARISH"
        elif score <= -0.5:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def _find_conflicting_timeframes(self, trend_scores: dict[str, float]) -> list[str]:
        """Find timeframes with conflicting trends"""
        conflicting = []

        for tf1, score1 in trend_scores.items():
            for tf2, score2 in trend_scores.items():
                if tf1 != tf2 and score1 * score2 < -0.5:  # Opposite trends
                    if tf1 not in conflicting:
                        conflicting.append(tf1)
                    if tf2 not in conflicting:
                        conflicting.append(tf2)

        return conflicting

    def _calculate_agreement_level(
        self, signals: dict[TimeFrame, TimeFrameSignal]
    ) -> float:
        """Calculate agreement level between timeframes"""
        if len(signals) < 2:
            return 1.0

        signal_types = [signal.signal_type for signal in signals.values()]
        most_common = max(set(signal_types), key=signal_types.count)
        agreement_count = signal_types.count(most_common)

        return agreement_count / len(signal_types)

    def _weight_and_cluster_levels(
        self, weighted_levels: list[tuple[float, float]], threshold: float = 0.02
    ) -> list[float]:
        """Weight and cluster price levels"""
        if not weighted_levels:
            return []

        # Group levels by proximity
        level_groups = {}

        for level, weight in weighted_levels:
            # Find existing group or create new one
            group_key = None
            for existing_level in level_groups:
                if abs(level - existing_level) / existing_level <= threshold:
                    group_key = existing_level
                    break

            if group_key is None:
                group_key = level
                level_groups[group_key] = []

            level_groups[group_key].append((level, weight))

        # Calculate weighted average for each group
        weighted_levels_final = []
        for group_levels in level_groups.values():
            total_weight = sum(weight for _, weight in group_levels)
            weighted_avg = (
                sum(level * weight for level, weight in group_levels) / total_weight
            )
            weighted_levels_final.append((weighted_avg, total_weight))

        # Sort by weight (importance) and return levels
        weighted_levels_final.sort(key=lambda x: x[1], reverse=True)
        return [level for level, _ in weighted_levels_final]


class MultiTimeFrameEngine:
    """Main Multi-Timeframe Analysis Engine"""

    def __init__(self, market_data_service=None):
        self.market_data_service = market_data_service
        self.indicator_calculator = AdaptiveIndicatorCalculator()
        self.cross_analyzer = CrossTimeFrameAnalyzer()

        # Default timeframes for analysis
        self.default_timeframes = [
            TimeFrame.M15,
            TimeFrame.H1,
            TimeFrame.H4,
            TimeFrame.D1,
        ]

        # Analysis cache
        self.analysis_cache = {}
        self.cache_expiry = timedelta(minutes=5)

        logger.info("Multi-Timeframe Analysis Engine initialized")

    @with_circuit_breaker("multi_timeframe_analysis")
    @with_retry(max_retries=2, delay=1.0)
    async def analyze_symbol(
        self, symbol: str, timeframes: list[TimeFrame] | None = None
    ) -> MultiTimeFrameAnalysis:
        """Perform comprehensive multi-timeframe analysis"""
        try:
            timeframes = timeframes or self.default_timeframes

            # Check cache
            cache_key = f"{symbol}_{'-'.join([tf.value for tf in timeframes])}"
            if cache_key in self.analysis_cache:
                cached_analysis, cache_time = self.analysis_cache[cache_key]
                if datetime.now() - cache_time < self.cache_expiry:
                    return cached_analysis

            # Analyze each timeframe
            timeframe_signals = {}

            for timeframe in timeframes:
                signal = await self._analyze_single_timeframe(symbol, timeframe)
                if signal:
                    timeframe_signals[timeframe] = signal

            if not timeframe_signals:
                return self._create_default_analysis(symbol)

            # Cross-timeframe analysis
            trend_alignment = self.cross_analyzer.analyze_trend_alignment(
                timeframe_signals
            )
            consensus = self.cross_analyzer.calculate_consensus_signal(
                timeframe_signals
            )
            key_levels = self.cross_analyzer.identify_key_levels(timeframe_signals)

            # Calculate risk-reward ratio
            risk_reward = self._calculate_risk_reward_ratio(
                timeframe_signals, key_levels
            )

            # Generate entry/exit conditions
            entry_conditions = self._generate_entry_conditions(
                timeframe_signals, trend_alignment, consensus
            )
            exit_conditions = self._generate_exit_conditions(
                timeframe_signals, key_levels
            )

            # Create comprehensive analysis
            analysis = MultiTimeFrameAnalysis(
                symbol=symbol,
                timeframe_signals=timeframe_signals,
                consensus_signal=consensus["signal"],
                consensus_strength=consensus["strength"],
                consensus_confidence=consensus["confidence"],
                trend_alignment=trend_alignment,
                key_levels=key_levels,
                risk_reward_ratio=risk_reward,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
            )

            # Cache the analysis
            self.analysis_cache[cache_key] = (analysis, datetime.now())

            return analysis

        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {str(e)}")
            return self._create_default_analysis(symbol)

    async def _analyze_single_timeframe(
        self, symbol: str, timeframe: TimeFrame
    ) -> TimeFrameSignal | None:
        """Analyze a single timeframe for the symbol"""
        try:
            # Get market data for the timeframe
            data = await self._get_market_data(symbol, timeframe)
            if data is None or len(data) < 50:
                return None

            # Calculate adaptive indicators
            rsi = self.indicator_calculator.calculate_adaptive_rsi(data, timeframe)
            macd = self.indicator_calculator.calculate_adaptive_macd(data, timeframe)
            bb = self.indicator_calculator.calculate_adaptive_bollinger_bands(
                data, timeframe
            )
            levels = self.indicator_calculator.calculate_support_resistance(
                data, timeframe
            )

            # Generate signal
            (
                signal_type,
                strength,
                confidence,
                trend,
                reasoning,
            ) = self._generate_timeframe_signal(data, rsi, macd, bb, timeframe)

            return TimeFrameSignal(
                timeframe=timeframe,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                trend_direction=trend,
                key_levels=levels,
                indicators={"rsi": rsi, "macd": macd, "bb": bb},
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"Error analyzing {timeframe.value} for {symbol}: {str(e)}")
            return None

    def _generate_timeframe_signal(
        self,
        data: pd.DataFrame,
        rsi: dict[str, Any],
        macd: dict[str, Any],
        bb: dict[str, Any],
        timeframe: TimeFrame,
    ) -> tuple[str, SignalStrength, float, TrendDirection, list[str]]:
        """Generate trading signal for a specific timeframe"""
        try:
            signal_score = 0
            confidence_factors = []
            reasoning = []

            # RSI analysis
            if rsi.get("is_oversold"):
                signal_score += 2
                confidence_factors.append(0.8)
                reasoning.append(f"RSI oversold ({rsi['value']:.1f})")
            elif rsi.get("is_overbought"):
                signal_score -= 2
                confidence_factors.append(0.8)
                reasoning.append(f"RSI overbought ({rsi['value']:.1f})")
            elif rsi.get("divergence"):
                signal_score += 1 if rsi["trend"] == "bullish" else -1
                confidence_factors.append(0.6)
                reasoning.append("RSI divergence detected")

            # MACD analysis
            if macd.get("bullish_crossover"):
                signal_score += 2
                confidence_factors.append(0.9)
                reasoning.append("MACD bullish crossover")
            elif macd.get("bearish_crossover"):
                signal_score -= 2
                confidence_factors.append(0.9)
                reasoning.append("MACD bearish crossover")
            elif macd.get("histogram", 0) > 0:
                signal_score += 1
                confidence_factors.append(0.5)
                reasoning.append("MACD histogram positive")
            elif macd.get("histogram", 0) < 0:
                signal_score -= 1
                confidence_factors.append(0.5)
                reasoning.append("MACD histogram negative")

            # Bollinger Bands analysis
            bb_position = bb.get("position", 0.5)
            if bb_position > 0.9:
                signal_score -= 1
                confidence_factors.append(0.6)
                reasoning.append("Price near upper Bollinger Band")
            elif bb_position < 0.1:
                signal_score += 1
                confidence_factors.append(0.6)
                reasoning.append("Price near lower Bollinger Band")

            if bb.get("squeeze"):
                confidence_factors.append(0.3)  # Lower confidence during squeeze
                reasoning.append("Bollinger Band squeeze detected")
            elif bb.get("breakout"):
                signal_score += 1 if bb_position > 0.5 else -1
                confidence_factors.append(0.8)
                reasoning.append("Bollinger Band breakout")

            # Determine signal type
            if signal_score >= 3:
                signal_type = "BUY"
                strength = (
                    SignalStrength.STRONG
                    if signal_score >= 4
                    else SignalStrength.MODERATE
                )
            elif signal_score <= -3:
                signal_type = "SELL"
                strength = (
                    SignalStrength.STRONG
                    if signal_score <= -4
                    else SignalStrength.MODERATE
                )
            elif abs(signal_score) >= 1:
                signal_type = "BUY" if signal_score > 0 else "SELL"
                strength = SignalStrength.WEAK
            else:
                signal_type = "HOLD"
                strength = SignalStrength.VERY_WEAK

            # Calculate confidence
            confidence = np.mean(confidence_factors) if confidence_factors else 0.5

            # Determine trend direction
            trend_score = signal_score / 5.0  # Normalize to -1 to 1
            if trend_score >= 0.6:
                trend = TrendDirection.STRONG_BULLISH
            elif trend_score >= 0.2:
                trend = TrendDirection.BULLISH
            elif trend_score <= -0.6:
                trend = TrendDirection.STRONG_BEARISH
            elif trend_score <= -0.2:
                trend = TrendDirection.BEARISH
            else:
                trend = TrendDirection.NEUTRAL

            return signal_type, strength, confidence, trend, reasoning

        except Exception as e:
            logger.error(f"Error generating timeframe signal: {str(e)}")
            return (
                "HOLD",
                SignalStrength.VERY_WEAK,
                0.0,
                TrendDirection.NEUTRAL,
                ["Error in analysis"],
            )

    async def _get_market_data(
        self, symbol: str, timeframe: TimeFrame
    ) -> pd.DataFrame | None:
        """Get market data for symbol and timeframe"""
        try:
            if self.market_data_service:
                # Use actual market data service
                data = await self.market_data_service.get_historical_data(
                    symbol, timeframe.value, limit=200
                )
                return data
            else:
                # Mock data for testing
                dates = pd.date_range(end=datetime.now(), periods=100, freq="1H")
                np.random.seed(42)  # For reproducible results

                # Generate realistic OHLCV data
                base_price = 2500
                returns = np.random.normal(0, 0.02, 100)
                prices = base_price * np.exp(np.cumsum(returns))

                data = pd.DataFrame(
                    {
                        "open": prices * (1 + np.random.normal(0, 0.001, 100)),
                        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                        "close": prices,
                        "volume": np.random.randint(10000, 100000, 100),
                    },
                    index=dates,
                )

                return data

        except Exception as e:
            logger.error(
                f"Error getting market data for {symbol} {timeframe.value}: {str(e)}"
            )
            return None

    def _calculate_risk_reward_ratio(
        self, signals: dict[TimeFrame, TimeFrameSignal], key_levels: dict[str, float]
    ) -> float:
        """Calculate risk-reward ratio based on key levels"""
        try:
            # Get current price from the most recent signal
            current_price = 2500  # Would get from actual data

            primary_resistance = key_levels.get("primary_resistance")
            primary_support = key_levels.get("primary_support")

            if not primary_resistance or not primary_support:
                return 1.0  # Default ratio

            # Calculate potential reward and risk
            if current_price < primary_resistance and current_price > primary_support:
                potential_reward = primary_resistance - current_price
                potential_risk = current_price - primary_support

                if potential_risk > 0:
                    return potential_reward / potential_risk

            return 1.0

        except Exception as e:
            logger.error(f"Error calculating risk-reward ratio: {str(e)}")
            return 1.0

    def _generate_entry_conditions(
        self,
        signals: dict[TimeFrame, TimeFrameSignal],
        trend_alignment: dict[str, Any],
        consensus: dict[str, Any],
    ) -> list[str]:
        """Generate entry conditions based on analysis"""
        conditions = []

        try:
            # Trend alignment conditions
            if trend_alignment.get("alignment_strength", 0) > 0.7:
                conditions.append("Strong trend alignment across timeframes")

            # Consensus conditions
            if consensus.get("confidence", 0) > 0.8:
                conditions.append(f"High confidence {consensus['signal']} signal")

            # Individual timeframe conditions
            for timeframe, signal in signals.items():
                if signal.confidence > 0.8:
                    conditions.append(
                        f"{timeframe.value}: {signal.signal_type} with high confidence"
                    )

            # Technical conditions
            strong_signals = [
                s
                for s in signals.values()
                if s.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]
            ]
            if len(strong_signals) >= 2:
                conditions.append("Multiple strong signals detected")

            if not conditions:
                conditions.append("Wait for clearer signals")

            return conditions

        except Exception as e:
            logger.error(f"Error generating entry conditions: {str(e)}")
            return ["Error generating conditions"]

    def _generate_exit_conditions(
        self, signals: dict[TimeFrame, TimeFrameSignal], key_levels: dict[str, float]
    ) -> list[str]:
        """Generate exit conditions based on analysis"""
        conditions = []

        try:
            # Level-based exits
            if key_levels.get("primary_resistance"):
                conditions.append(
                    f"Take profit near resistance: {key_levels['primary_resistance']:.2f}"
                )

            if key_levels.get("primary_support"):
                conditions.append(
                    f"Stop loss below support: {key_levels['primary_support']:.2f}"
                )

            # Signal-based exits
            bearish_signals = [
                s
                for s in signals.values()
                if s.signal_type == "SELL" and s.confidence > 0.7
            ]
            if bearish_signals:
                conditions.append("Exit on strong bearish signals")

            # Risk management exits
            conditions.append("Exit if risk-reward ratio deteriorates below 1:2")
            conditions.append("Exit if trend alignment breaks down")

            return conditions

        except Exception as e:
            logger.error(f"Error generating exit conditions: {str(e)}")
            return ["Use standard risk management rules"]

    def _create_default_analysis(self, symbol: str) -> MultiTimeFrameAnalysis:
        """Create default analysis when data is insufficient"""
        return MultiTimeFrameAnalysis(
            symbol=symbol,
            timeframe_signals={},
            consensus_signal="HOLD",
            consensus_strength=0.0,
            consensus_confidence=0.0,
            trend_alignment={"trend_direction": "NEUTRAL", "alignment_strength": 0.0},
            key_levels={},
            risk_reward_ratio=1.0,
            entry_conditions=["Insufficient data for analysis"],
            exit_conditions=["Use standard risk management"],
        )

    def get_analysis_summary(self, analysis: MultiTimeFrameAnalysis) -> dict[str, Any]:
        """Get formatted summary of multi-timeframe analysis"""
        return {
            "symbol": analysis.symbol,
            "consensus": {
                "signal": analysis.consensus_signal,
                "strength": analysis.consensus_strength,
                "confidence": analysis.consensus_confidence,
            },
            "trend": {
                "direction": analysis.trend_alignment.get("trend_direction"),
                "alignment": analysis.trend_alignment.get("alignment_strength", 0),
            },
            "timeframes": {
                tf.value: {
                    "signal": signal.signal_type,
                    "strength": signal.strength.value,
                    "confidence": signal.confidence,
                    "trend": signal.trend_direction.value,
                }
                for tf, signal in analysis.timeframe_signals.items()
            },
            "levels": analysis.key_levels,
            "risk_reward": analysis.risk_reward_ratio,
            "entry_conditions": analysis.entry_conditions[:3],  # Top 3
            "exit_conditions": analysis.exit_conditions[:3],  # Top 3
            "timestamp": analysis.timestamp,
        }
