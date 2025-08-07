"""Pattern recognition system for technical analysis"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import argrelextrema

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available. Pattern recognition will be limited.")
    TALIB_AVAILABLE = False

from .data_processor import RealTimeDataProcessor


class PatternMatch(NamedTuple):
    """Pattern match result"""

    pattern_name: str
    confidence: float
    start_index: int
    end_index: int
    key_levels: list[float]
    direction: str  # 'bullish', 'bearish', 'neutral'
    description: str


@dataclass
class BreakoutSignal:
    """Breakout detection signal"""

    symbol: str
    breakout_type: str  # 'resistance', 'support', 'range'
    direction: str  # 'up', 'down'
    level: float
    current_price: float
    volume_confirmation: bool
    strength: float
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "breakout_type": self.breakout_type,
            "direction": self.direction,
            "level": self.level,
            "current_price": self.current_price,
            "volume_confirmation": self.volume_confirmation,
            "strength": self.strength,
            "timestamp": self.timestamp,
        }


class CandlestickPatternRecognizer:
    """Recognizes candlestick patterns using TA-Lib"""

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> dict[str, list[int]]:
        """Detect various candlestick patterns"""
        if df.empty or len(df) < 10:
            return {}

        if not TALIB_AVAILABLE:
            logger.warning("TA-Lib not available. Using basic pattern detection.")
            return CandlestickPatternRecognizer._detect_basic_patterns(df)

        try:
            open_prices = df["open"].values
            high_prices = df["high"].values
            low_prices = df["low"].values
            close_prices = df["close"].values

            patterns = {}

            # Bullish patterns
            patterns["hammer"] = talib.CDLHAMMER(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["inverted_hammer"] = talib.CDLINVERTEDHAMMER(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["morning_star"] = talib.CDLMORNINGSTAR(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["bullish_engulfing"] = talib.CDLENGULFING(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["piercing_pattern"] = talib.CDLPIERCING(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["three_white_soldiers"] = talib.CDL3WHITESOLDIERS(
                open_prices, high_prices, low_prices, close_prices
            )

            # Bearish patterns
            patterns["shooting_star"] = talib.CDLSHOOTINGSTAR(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["hanging_man"] = talib.CDLHANGINGMAN(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["evening_star"] = talib.CDLEVENINGSTAR(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["bearish_engulfing"] = talib.CDLENGULFING(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["dark_cloud_cover"] = talib.CDLDARKCLOUDCOVER(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["three_black_crows"] = talib.CDL3BLACKCROWS(
                open_prices, high_prices, low_prices, close_prices
            )

            # Reversal patterns
            patterns["doji"] = talib.CDLDOJI(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["harami"] = talib.CDLHARAMI(
                open_prices, high_prices, low_prices, close_prices
            )
            patterns["spinning_top"] = talib.CDLSPINNINGTOP(
                open_prices, high_prices, low_prices, close_prices
            )

            # Filter and return recent patterns
            recent_patterns = {}
            for pattern_name, pattern_data in patterns.items():
                recent_indices = np.where(pattern_data[-20:] != 0)[0]  # Last 20 bars
                if len(recent_indices) > 0:
                    recent_patterns[pattern_name] = [
                        len(df) - 20 + idx for idx in recent_indices
                    ]

            return recent_patterns

        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {str(e)}")
            return CandlestickPatternRecognizer._detect_basic_patterns(df)

    @staticmethod
    def _detect_basic_patterns(df: pd.DataFrame) -> dict[str, list[int]]:
        """Basic pattern detection when TA-Lib is not available"""
        patterns = {}

        try:
            # Simple doji detection
            body_size = abs(df["close"] - df["open"])
            wick_size = df["high"] - df["low"]
            doji_threshold = 0.1  # Body is less than 10% of total range

            doji_indices = []
            for i in range(len(df)):
                if body_size.iloc[i] < (wick_size.iloc[i] * doji_threshold):
                    doji_indices.append(i)

            if doji_indices:
                patterns["doji"] = doji_indices

            # Simple hammer detection
            hammer_indices = []
            for i in range(len(df)):
                body = abs(df["close"].iloc[i] - df["open"].iloc[i])
                lower_wick = (
                    min(df["open"].iloc[i], df["close"].iloc[i]) - df["low"].iloc[i]
                )
                upper_wick = df["high"].iloc[i] - max(
                    df["open"].iloc[i], df["close"].iloc[i]
                )

                # Hammer: small body, long lower wick, short upper wick
                if lower_wick > 2 * body and upper_wick < body * 0.5:
                    hammer_indices.append(i)

            if hammer_indices:
                patterns["hammer"] = hammer_indices

            return patterns

        except Exception as e:
            logger.error(f"Error in basic pattern detection: {str(e)}")
            return {}


class ChartPatternRecognizer:
    """Recognizes chart patterns like triangles, flags, head and shoulders"""

    def __init__(self):
        self.min_pattern_length = 10
        self.max_pattern_length = 50

    def detect_patterns(self, df: pd.DataFrame) -> list[PatternMatch]:
        """Detect various chart patterns"""
        if df.empty or len(df) < self.min_pattern_length:
            return []

        patterns = []

        try:
            # Detect different pattern types
            patterns.extend(self._detect_triangle_patterns(df))
            patterns.extend(self._detect_flag_patterns(df))
            patterns.extend(self._detect_head_and_shoulders(df))
            patterns.extend(self._detect_double_tops_bottoms(df))
            patterns.extend(self._detect_channel_patterns(df))

            return patterns

        except Exception as e:
            logger.error(f"Error detecting chart patterns: {str(e)}")
            return []

    def _detect_triangle_patterns(self, df: pd.DataFrame) -> list[PatternMatch]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        patterns = []

        try:
            # Find peaks and troughs
            highs = df["high"].values
            lows = df["low"].values

            # Find local maxima and minima
            peak_indices = argrelextrema(highs, np.greater, order=3)[0]
            trough_indices = argrelextrema(lows, np.less, order=3)[0]

            if len(peak_indices) < 2 or len(trough_indices) < 2:
                return patterns

            # Check for ascending triangle (horizontal resistance, rising support)
            recent_peaks = peak_indices[-3:] if len(peak_indices) >= 3 else peak_indices
            recent_troughs = (
                trough_indices[-3:] if len(trough_indices) >= 3 else trough_indices
            )

            if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
                # Ascending triangle: resistance is horizontal, support is rising
                peak_prices = [highs[i] for i in recent_peaks]
                trough_prices = [lows[i] for i in recent_troughs]

                # Check if peaks are roughly at same level (horizontal resistance)
                peak_std = np.std(peak_prices)
                peak_mean = np.mean(peak_prices)

                if peak_std / peak_mean < 0.02:  # Less than 2% variation
                    # Check if troughs are rising
                    if len(recent_troughs) >= 2:
                        trough_slope = (trough_prices[-1] - trough_prices[0]) / (
                            recent_troughs[-1] - recent_troughs[0]
                        )
                        if trough_slope > 0:
                            patterns.append(
                                PatternMatch(
                                    pattern_name="ascending_triangle",
                                    confidence=0.7,
                                    start_index=min(recent_peaks[0], recent_troughs[0]),
                                    end_index=max(recent_peaks[-1], recent_troughs[-1]),
                                    key_levels=[peak_mean, trough_prices[-1]],
                                    direction="bullish",
                                    description=f"Ascending triangle with resistance at {peak_mean:.2f}",
                                )
                            )

                # Descending triangle: support is horizontal, resistance is falling
                trough_std = np.std(trough_prices)
                trough_mean = np.mean(trough_prices)

                if trough_std / trough_mean < 0.02:  # Less than 2% variation
                    # Check if peaks are falling
                    if len(recent_peaks) >= 2:
                        peak_slope = (peak_prices[-1] - peak_prices[0]) / (
                            recent_peaks[-1] - recent_peaks[0]
                        )
                        if peak_slope < 0:
                            patterns.append(
                                PatternMatch(
                                    pattern_name="descending_triangle",
                                    confidence=0.7,
                                    start_index=min(recent_peaks[0], recent_troughs[0]),
                                    end_index=max(recent_peaks[-1], recent_troughs[-1]),
                                    key_levels=[peak_prices[-1], trough_mean],
                                    direction="bearish",
                                    description=f"Descending triangle with support at {trough_mean:.2f}",
                                )
                            )

            return patterns

        except Exception as e:
            logger.error(f"Error detecting triangle patterns: {str(e)}")
            return []

    def _detect_flag_patterns(self, df: pd.DataFrame) -> list[PatternMatch]:
        """Detect flag and pennant patterns"""
        patterns = []

        try:
            # Flag pattern: strong move followed by small consolidation
            if len(df) < 20:
                return patterns

            close_prices = df["close"].values
            volume = df["volume"].values

            # Look for strong initial move (flagpole)
            for i in range(10, len(df) - 10):
                # Check for strong move in previous 5-10 bars
                lookback = min(10, i)
                price_change = (
                    close_prices[i] - close_prices[i - lookback]
                ) / close_prices[i - lookback]

                if abs(price_change) > 0.05:  # At least 5% move
                    # Check for consolidation in next 5-15 bars
                    consolidation_end = min(i + 15, len(df))
                    consolidation_prices = close_prices[i:consolidation_end]

                    if len(consolidation_prices) > 5:
                        consolidation_range = (
                            np.max(consolidation_prices) - np.min(consolidation_prices)
                        ) / np.mean(consolidation_prices)

                        if consolidation_range < 0.03:  # Less than 3% range
                            # Check volume pattern (should decrease during consolidation)
                            flagpole_volume = np.mean(volume[i - lookback : i])
                            consolidation_volume = np.mean(volume[i:consolidation_end])

                            if (
                                consolidation_volume < flagpole_volume * 0.7
                            ):  # Volume decreases
                                direction = "bullish" if price_change > 0 else "bearish"
                                patterns.append(
                                    PatternMatch(
                                        pattern_name="flag",
                                        confidence=0.65,
                                        start_index=i - lookback,
                                        end_index=consolidation_end - 1,
                                        key_levels=[
                                            close_prices[i - lookback],
                                            close_prices[i],
                                            np.mean(consolidation_prices),
                                        ],
                                        direction=direction,
                                        description=f"Flag pattern after {price_change*100:.1f}% move",
                                    )
                                )

            return patterns

        except Exception as e:
            logger.error(f"Error detecting flag patterns: {str(e)}")
            return []

    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> list[PatternMatch]:
        """Detect head and shoulders patterns"""
        patterns = []

        try:
            if len(df) < 30:
                return patterns

            highs = df["high"].values
            lows = df["low"].values

            # Find significant peaks
            peak_indices = argrelextrema(highs, np.greater, order=5)[0]

            if len(peak_indices) < 3:
                return patterns

            # Look for head and shoulders pattern in recent peaks
            for i in range(len(peak_indices) - 2):
                left_shoulder = peak_indices[i]
                head = peak_indices[i + 1]
                right_shoulder = peak_indices[i + 2]

                left_shoulder_price = highs[left_shoulder]
                head_price = highs[head]
                right_shoulder_price = highs[right_shoulder]

                # Head should be higher than both shoulders
                if (
                    head_price > left_shoulder_price
                    and head_price > right_shoulder_price
                ):
                    # Shoulders should be roughly at same level
                    shoulder_diff = abs(
                        left_shoulder_price - right_shoulder_price
                    ) / max(left_shoulder_price, right_shoulder_price)

                    if shoulder_diff < 0.05:  # Less than 5% difference
                        # Find neckline (lows between shoulders and head)
                        neckline_start = (
                            np.argmin(lows[left_shoulder:head]) + left_shoulder
                        )
                        neckline_end = np.argmin(lows[head:right_shoulder]) + head
                        neckline_level = (lows[neckline_start] + lows[neckline_end]) / 2

                        patterns.append(
                            PatternMatch(
                                pattern_name="head_and_shoulders",
                                confidence=0.75,
                                start_index=left_shoulder,
                                end_index=right_shoulder,
                                key_levels=[
                                    left_shoulder_price,
                                    head_price,
                                    right_shoulder_price,
                                    neckline_level,
                                ],
                                direction="bearish",
                                description=f"Head and shoulders with neckline at {neckline_level:.2f}",
                            )
                        )

            return patterns

        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {str(e)}")
            return []

    def _detect_double_tops_bottoms(self, df: pd.DataFrame) -> list[PatternMatch]:
        """Detect double top and double bottom patterns"""
        patterns = []

        try:
            highs = df["high"].values
            lows = df["low"].values

            # Find peaks and troughs
            peak_indices = argrelextrema(highs, np.greater, order=3)[0]
            trough_indices = argrelextrema(lows, np.less, order=3)[0]

            # Double tops
            if len(peak_indices) >= 2:
                for i in range(len(peak_indices) - 1):
                    peak1_idx = peak_indices[i]
                    peak2_idx = peak_indices[i + 1]

                    peak1_price = highs[peak1_idx]
                    peak2_price = highs[peak2_idx]

                    # Peaks should be at similar levels
                    price_diff = abs(peak1_price - peak2_price) / max(
                        peak1_price, peak2_price
                    )

                    if price_diff < 0.03:  # Less than 3% difference
                        # Find valley between peaks
                        valley_idx = np.argmin(lows[peak1_idx:peak2_idx]) + peak1_idx
                        valley_price = lows[valley_idx]

                        # Valley should be significantly lower
                        valley_drop = (
                            min(peak1_price, peak2_price) - valley_price
                        ) / min(peak1_price, peak2_price)

                        if valley_drop > 0.05:  # At least 5% drop
                            patterns.append(
                                PatternMatch(
                                    pattern_name="double_top",
                                    confidence=0.7,
                                    start_index=peak1_idx,
                                    end_index=peak2_idx,
                                    key_levels=[peak1_price, valley_price, peak2_price],
                                    direction="bearish",
                                    description=f"Double top at {(peak1_price + peak2_price)/2:.2f}",
                                )
                            )

            # Double bottoms
            if len(trough_indices) >= 2:
                for i in range(len(trough_indices) - 1):
                    trough1_idx = trough_indices[i]
                    trough2_idx = trough_indices[i + 1]

                    trough1_price = lows[trough1_idx]
                    trough2_price = lows[trough2_idx]

                    # Troughs should be at similar levels
                    price_diff = abs(trough1_price - trough2_price) / max(
                        trough1_price, trough2_price
                    )

                    if price_diff < 0.03:  # Less than 3% difference
                        # Find peak between troughs
                        peak_idx = (
                            np.argmax(highs[trough1_idx:trough2_idx]) + trough1_idx
                        )
                        peak_price = highs[peak_idx]

                        # Peak should be significantly higher
                        peak_rise = (
                            peak_price - max(trough1_price, trough2_price)
                        ) / max(trough1_price, trough2_price)

                        if peak_rise > 0.05:  # At least 5% rise
                            patterns.append(
                                PatternMatch(
                                    pattern_name="double_bottom",
                                    confidence=0.7,
                                    start_index=trough1_idx,
                                    end_index=trough2_idx,
                                    key_levels=[
                                        trough1_price,
                                        peak_price,
                                        trough2_price,
                                    ],
                                    direction="bullish",
                                    description=f"Double bottom at {(trough1_price + trough2_price)/2:.2f}",
                                )
                            )

            return patterns

        except Exception as e:
            logger.error(f"Error detecting double tops/bottoms: {str(e)}")
            return []

    def _detect_channel_patterns(self, df: pd.DataFrame) -> list[PatternMatch]:
        """Detect channel patterns (parallel support and resistance)"""
        patterns = []

        try:
            if len(df) < 20:
                return patterns

            highs = df["high"].values
            lows = df["low"].values

            # Find multiple peaks and troughs
            peak_indices = argrelextrema(highs, np.greater, order=2)[0]
            trough_indices = argrelextrema(lows, np.less, order=2)[0]

            if len(peak_indices) < 3 or len(trough_indices) < 3:
                return patterns

            # Check for parallel lines in recent data
            recent_peaks = peak_indices[-5:] if len(peak_indices) >= 5 else peak_indices
            recent_troughs = (
                trough_indices[-5:] if len(trough_indices) >= 5 else trough_indices
            )

            if len(recent_peaks) >= 3 and len(recent_troughs) >= 3:
                # Calculate trend lines
                peak_prices = [highs[i] for i in recent_peaks]
                trough_prices = [lows[i] for i in recent_troughs]

                # Linear regression for trend lines
                peak_slope = np.polyfit(recent_peaks, peak_prices, 1)[0]
                trough_slope = np.polyfit(recent_troughs, trough_prices, 1)[0]

                # Check if slopes are similar (parallel channel)
                slope_diff = abs(peak_slope - trough_slope)
                avg_slope = (abs(peak_slope) + abs(trough_slope)) / 2

                if avg_slope > 0 and slope_diff / avg_slope < 0.3:  # Similar slopes
                    # Determine channel type
                    if peak_slope > 0 and trough_slope > 0:
                        direction = "bullish"
                        pattern_name = "ascending_channel"
                    elif peak_slope < 0 and trough_slope < 0:
                        direction = "bearish"
                        pattern_name = "descending_channel"
                    else:
                        direction = "neutral"
                        pattern_name = "horizontal_channel"

                    patterns.append(
                        PatternMatch(
                            pattern_name=pattern_name,
                            confidence=0.6,
                            start_index=min(recent_peaks[0], recent_troughs[0]),
                            end_index=max(recent_peaks[-1], recent_troughs[-1]),
                            key_levels=[np.mean(peak_prices), np.mean(trough_prices)],
                            direction=direction,
                            description=f"{pattern_name.replace('_', ' ').title()}",
                        )
                    )

            return patterns

        except Exception as e:
            logger.error(f"Error detecting channel patterns: {str(e)}")
            return []


class BreakoutDetector:
    """Detects breakouts from support/resistance levels"""

    def __init__(self, data_processor: RealTimeDataProcessor):
        self.data_processor = data_processor
        self.breakout_threshold = 0.002  # 0.2% minimum breakout
        self.volume_threshold = 1.5  # 1.5x average volume for confirmation

    def detect_breakouts(
        self, symbol: str, timeframes: list[str] = None
    ) -> list[BreakoutSignal]:
        """Detect breakouts across multiple timeframes"""
        if timeframes is None:
            timeframes = ["5min", "15min"]

        breakouts = []

        for timeframe in timeframes:
            try:
                breakout = self._detect_breakout_timeframe(symbol, timeframe)
                if breakout:
                    breakouts.append(breakout)
            except Exception as e:
                logger.error(
                    f"Error detecting breakout for {symbol} on {timeframe}: {str(e)}"
                )
                continue

        return breakouts

    def _detect_breakout_timeframe(
        self, symbol: str, timeframe: str
    ) -> BreakoutSignal | None:
        """Detect breakout for specific timeframe"""
        try:
            df = self.data_processor.get_candle_data(symbol, timeframe, 50)
            if df.empty or len(df) < 20:
                return None

            current_price = df["close"].iloc[-1]
            current_volume = df["volume"].iloc[-1]
            avg_volume = df["volume"].rolling(20).mean().iloc[-1]

            # Find recent support and resistance levels
            highs = df["high"].values
            lows = df["low"].values

            # Recent peaks and troughs
            peak_indices = argrelextrema(highs, np.greater, order=2)[0]
            trough_indices = argrelextrema(lows, np.less, order=2)[0]

            if len(peak_indices) == 0 and len(trough_indices) == 0:
                return None

            # Check for resistance breakout
            if len(peak_indices) > 0:
                recent_resistance = np.max([highs[i] for i in peak_indices[-3:]])
                if current_price > recent_resistance * (1 + self.breakout_threshold):
                    volume_confirmation = (
                        current_volume > avg_volume * self.volume_threshold
                    )
                    strength = min(
                        (current_price - recent_resistance) / recent_resistance * 100,
                        5.0,
                    )

                    return BreakoutSignal(
                        symbol=symbol,
                        breakout_type="resistance",
                        direction="up",
                        level=recent_resistance,
                        current_price=current_price,
                        volume_confirmation=volume_confirmation,
                        strength=strength,
                        timestamp=datetime.now(),
                    )

            # Check for support breakdown
            if len(trough_indices) > 0:
                recent_support = np.min([lows[i] for i in trough_indices[-3:]])
                if current_price < recent_support * (1 - self.breakout_threshold):
                    volume_confirmation = (
                        current_volume > avg_volume * self.volume_threshold
                    )
                    strength = min(
                        (recent_support - current_price) / recent_support * 100, 5.0
                    )

                    return BreakoutSignal(
                        symbol=symbol,
                        breakout_type="support",
                        direction="down",
                        level=recent_support,
                        current_price=current_price,
                        volume_confirmation=volume_confirmation,
                        strength=strength,
                        timestamp=datetime.now(),
                    )

            # Check for range breakout
            if len(peak_indices) > 0 and len(trough_indices) > 0:
                recent_high = np.max([highs[i] for i in peak_indices[-2:]])
                recent_low = np.min([lows[i] for i in trough_indices[-2:]])
                range_size = recent_high - recent_low

                # Check if we're in a narrow range
                if range_size / current_price < 0.05:  # Less than 5% range
                    if current_price > recent_high:
                        volume_confirmation = (
                            current_volume > avg_volume * self.volume_threshold
                        )
                        strength = min(
                            (current_price - recent_high) / recent_high * 100, 5.0
                        )

                        return BreakoutSignal(
                            symbol=symbol,
                            breakout_type="range",
                            direction="up",
                            level=recent_high,
                            current_price=current_price,
                            volume_confirmation=volume_confirmation,
                            strength=strength,
                            timestamp=datetime.now(),
                        )
                    elif current_price < recent_low:
                        volume_confirmation = (
                            current_volume > avg_volume * self.volume_threshold
                        )
                        strength = min(
                            (recent_low - current_price) / recent_low * 100, 5.0
                        )

                        return BreakoutSignal(
                            symbol=symbol,
                            breakout_type="range",
                            direction="down",
                            level=recent_low,
                            current_price=current_price,
                            volume_confirmation=volume_confirmation,
                            strength=strength,
                            timestamp=datetime.now(),
                        )

            return None

        except Exception as e:
            logger.error(f"Error detecting breakout for {symbol}: {str(e)}")
            return None


class PatternRecognitionEngine:
    """Main pattern recognition engine that combines all pattern detectors"""

    def __init__(self, data_processor: RealTimeDataProcessor):
        self.data_processor = data_processor
        self.candlestick_recognizer = CandlestickPatternRecognizer()
        self.chart_recognizer = ChartPatternRecognizer()
        self.breakout_detector = BreakoutDetector(data_processor)

    def analyze_patterns(
        self, symbol: str, timeframes: list[str] = None
    ) -> dict[str, Any]:
        """Comprehensive pattern analysis for a symbol"""
        if timeframes is None:
            timeframes = ["5min", "15min"]

        analysis_results = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "candlestick_patterns": {},
            "chart_patterns": {},
            "breakouts": [],
            "pattern_signals": [],
        }

        try:
            for timeframe in timeframes:
                df = self.data_processor.get_candle_data(symbol, timeframe, 100)
                if df.empty:
                    continue

                # Detect candlestick patterns
                candlestick_patterns = self.candlestick_recognizer.detect_patterns(df)
                analysis_results["candlestick_patterns"][
                    timeframe
                ] = candlestick_patterns

                # Detect chart patterns
                chart_patterns = self.chart_recognizer.detect_patterns(df)
                analysis_results["chart_patterns"][timeframe] = chart_patterns

            # Detect breakouts
            breakouts = self.breakout_detector.detect_breakouts(symbol, timeframes)
            analysis_results["breakouts"] = [
                breakout.to_dict() for breakout in breakouts
            ]

            # Generate pattern-based signals
            pattern_signals = self._generate_pattern_signals(analysis_results)
            analysis_results["pattern_signals"] = pattern_signals

            return analysis_results

        except Exception as e:
            logger.error(f"Error in pattern analysis for {symbol}: {str(e)}")
            analysis_results["error"] = str(e)
            return analysis_results

    def _generate_pattern_signals(
        self, analysis_results: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate trading signals based on detected patterns"""
        signals = []

        try:
            # Process candlestick patterns
            for timeframe, patterns in analysis_results["candlestick_patterns"].items():
                for pattern_name, indices in patterns.items():
                    if indices and len(indices) > 0:
                        # Determine if pattern is bullish or bearish
                        bullish_patterns = [
                            "hammer",
                            "inverted_hammer",
                            "morning_star",
                            "bullish_engulfing",
                            "piercing_pattern",
                            "three_white_soldiers",
                        ]
                        bearish_patterns = [
                            "shooting_star",
                            "hanging_man",
                            "evening_star",
                            "dark_cloud_cover",
                            "three_black_crows",
                        ]

                        if pattern_name in bullish_patterns:
                            signals.append(
                                {
                                    "type": "candlestick",
                                    "pattern": pattern_name,
                                    "direction": "bullish",
                                    "timeframe": timeframe,
                                    "strength": 0.6,
                                    "recent": max(indices)
                                    >= len(analysis_results.get("data", [])) - 5,
                                }
                            )
                        elif pattern_name in bearish_patterns:
                            signals.append(
                                {
                                    "type": "candlestick",
                                    "pattern": pattern_name,
                                    "direction": "bearish",
                                    "timeframe": timeframe,
                                    "strength": 0.6,
                                    "recent": max(indices)
                                    >= len(analysis_results.get("data", [])) - 5,
                                }
                            )

            # Process chart patterns
            for timeframe, patterns in analysis_results["chart_patterns"].items():
                for pattern in patterns:
                    signals.append(
                        {
                            "type": "chart_pattern",
                            "pattern": pattern.pattern_name,
                            "direction": pattern.direction,
                            "timeframe": timeframe,
                            "strength": pattern.confidence,
                            "description": pattern.description,
                            "key_levels": pattern.key_levels,
                        }
                    )

            # Process breakouts
            for breakout in analysis_results["breakouts"]:
                strength = breakout["strength"] / 5.0  # Normalize to 0-1
                signals.append(
                    {
                        "type": "breakout",
                        "pattern": f"{breakout['breakout_type']}_breakout",
                        "direction": (
                            "bullish" if breakout["direction"] == "up" else "bearish"
                        ),
                        "strength": strength,
                        "volume_confirmed": breakout["volume_confirmation"],
                        "level": breakout["level"],
                        "current_price": breakout["current_price"],
                    }
                )

            return signals

        except Exception as e:
            logger.error(f"Error generating pattern signals: {str(e)}")
            return []

    def get_pattern_summary(self, symbol: str) -> dict[str, Any]:
        """Get a summary of all patterns for a symbol"""
        try:
            analysis = self.analyze_patterns(symbol)

            summary = {
                "symbol": symbol,
                "total_patterns": 0,
                "bullish_signals": 0,
                "bearish_signals": 0,
                "breakout_signals": 0,
                "strongest_signal": None,
                "key_levels": [],
                "timestamp": datetime.now(),
            }

            # Count signals
            for signal in analysis["pattern_signals"]:
                summary["total_patterns"] += 1

                if signal["direction"] == "bullish":
                    summary["bullish_signals"] += 1
                elif signal["direction"] == "bearish":
                    summary["bearish_signals"] += 1

                if signal["type"] == "breakout":
                    summary["breakout_signals"] += 1

                # Track strongest signal
                if (
                    not summary["strongest_signal"]
                    or signal["strength"] > summary["strongest_signal"]["strength"]
                ):
                    summary["strongest_signal"] = signal

                # Collect key levels
                if "key_levels" in signal:
                    summary["key_levels"].extend(signal["key_levels"])

            # Remove duplicate levels and sort
            if summary["key_levels"]:
                summary["key_levels"] = sorted(set(summary["key_levels"]))

            return summary

        except Exception as e:
            logger.error(f"Error creating pattern summary for {symbol}: {str(e)}")
            return {"symbol": symbol, "error": str(e), "timestamp": datetime.now()}
