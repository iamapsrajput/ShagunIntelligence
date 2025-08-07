"""Dynamic stop loss management using ATR and support/resistance levels."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class StopLossManager:
    """Manage dynamic stop losses using various technical methods."""

    def __init__(self):
        """Initialize stop loss manager."""
        self.stop_history = []
        self.atr_multiplier = 2.0  # Default ATR multiplier
        self.trailing_percentage = 0.02  # Default 2% trailing stop

        logger.info("StopLossManager initialized")

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).

        Args:
            data: OHLC price data
            period: ATR period (default 14)

        Returns:
            Current ATR value
        """
        try:
            # Calculate True Range
            high_low = data["high"] - data["low"]
            high_close = abs(data["high"] - data["close"].shift())
            low_close = abs(data["low"] - data["close"].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                axis=1
            )

            # Calculate ATR
            atr = true_range.rolling(window=period).mean()

            current_atr = atr.iloc[-1]

            return current_atr if not pd.isna(current_atr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return 0.0

    def find_support_resistance(
        self,
        data: pd.DataFrame,
        current_price: float,
        lookback: int = 50,
        min_touches: int = 2,
    ) -> tuple[float, float]:
        """
        Find nearby support and resistance levels.

        Args:
            data: OHLC price data
            current_price: Current price
            lookback: Number of periods to look back
            min_touches: Minimum touches to confirm level

        Returns:
            Tuple of (support_level, resistance_level)
        """
        try:
            # Get recent data
            recent_data = data.tail(lookback)

            # Find swing highs and lows
            highs = self._find_swing_points(recent_data["high"], "high")
            lows = self._find_swing_points(recent_data["low"], "low")

            # Cluster similar levels
            resistance_levels = self._cluster_levels(highs, tolerance=0.002)
            support_levels = self._cluster_levels(lows, tolerance=0.002)

            # Find nearest support and resistance
            support = self._find_nearest_support(support_levels, current_price)
            resistance = self._find_nearest_resistance(resistance_levels, current_price)

            # Fallback to percentage-based levels if none found
            if support == 0:
                support = current_price * 0.97  # 3% below
            if resistance == float("inf"):
                resistance = current_price * 1.03  # 3% above

            return support, resistance

        except Exception as e:
            logger.error(f"Error finding support/resistance: {str(e)}")
            return current_price * 0.97, current_price * 1.03

    def calculate_dynamic_stop(
        self,
        entry_price: float,
        atr: float,
        support_level: float,
        trade_direction: str = "long",
        atr_multiplier: float | None = None,
    ) -> float:
        """
        Calculate dynamic stop loss level.

        Args:
            entry_price: Entry price
            atr: Current ATR value
            support_level: Nearest support level
            trade_direction: 'long' or 'short'
            atr_multiplier: ATR multiplier (uses default if None)

        Returns:
            Stop loss price
        """
        try:
            multiplier = atr_multiplier or self.atr_multiplier

            if trade_direction == "long":
                # ATR-based stop
                atr_stop = entry_price - (atr * multiplier)

                # Support-based stop (slightly below support)
                support_stop = support_level * 0.99

                # Use the higher of the two (tighter stop)
                stop_loss = max(atr_stop, support_stop)

                # Ensure stop is below entry
                stop_loss = min(stop_loss, entry_price * 0.98)

            else:  # short
                # ATR-based stop
                atr_stop = entry_price + (atr * multiplier)

                # Resistance-based stop (slightly above resistance)
                # Note: support_level here represents resistance for shorts
                resistance_stop = support_level * 1.01

                # Use the lower of the two (tighter stop)
                stop_loss = min(atr_stop, resistance_stop)

                # Ensure stop is above entry
                stop_loss = max(stop_loss, entry_price * 1.02)

            # Store in history
            self.stop_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "atr": atr,
                    "support_level": support_level,
                    "trade_direction": trade_direction,
                    "method": "dynamic",
                }
            )

            return stop_loss

        except Exception as e:
            logger.error(f"Error calculating dynamic stop: {str(e)}")
            # Fallback to percentage-based stop
            return (
                entry_price * 0.98 if trade_direction == "long" else entry_price * 1.02
            )

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        atr: float,
        position_type: str = "long",
        trailing_method: str = "atr",
    ) -> float:
        """
        Calculate trailing stop loss.

        Args:
            entry_price: Original entry price
            current_price: Current market price
            atr: Current ATR value
            position_type: 'long' or 'short'
            trailing_method: 'atr' or 'percentage'

        Returns:
            Trailing stop price
        """
        try:
            if position_type == "long":
                if current_price <= entry_price:
                    # No trailing if price hasn't moved favorably
                    return self.calculate_dynamic_stop(
                        entry_price, atr, entry_price * 0.97
                    )

                if trailing_method == "atr":
                    # ATR-based trailing stop
                    trailing_stop = current_price - (atr * self.atr_multiplier)
                else:
                    # Percentage-based trailing stop
                    trailing_stop = current_price * (1 - self.trailing_percentage)

                # Never let stop go below breakeven after 2% profit
                if current_price > entry_price * 1.02:
                    trailing_stop = max(trailing_stop, entry_price)

            else:  # short
                if current_price >= entry_price:
                    # No trailing if price hasn't moved favorably
                    return self.calculate_dynamic_stop(
                        entry_price, atr, entry_price * 1.03, "short"
                    )

                if trailing_method == "atr":
                    # ATR-based trailing stop
                    trailing_stop = current_price + (atr * self.atr_multiplier)
                else:
                    # Percentage-based trailing stop
                    trailing_stop = current_price * (1 + self.trailing_percentage)

                # Never let stop go above breakeven after 2% profit
                if current_price < entry_price * 0.98:
                    trailing_stop = min(trailing_stop, entry_price)

            return trailing_stop

        except Exception as e:
            logger.error(f"Error calculating trailing stop: {str(e)}")
            return (
                current_price * 0.98
                if position_type == "long"
                else current_price * 1.02
            )

    def breakeven_stop(
        self,
        entry_price: float,
        current_price: float,
        position_type: str = "long",
        trigger_profit: float = 0.01,
    ) -> float | None:
        """
        Calculate breakeven stop loss.

        Args:
            entry_price: Original entry price
            current_price: Current market price
            position_type: 'long' or 'short'
            trigger_profit: Profit percentage to trigger breakeven stop

        Returns:
            Breakeven stop price or None if not triggered
        """
        try:
            if position_type == "long":
                profit_percentage = (current_price - entry_price) / entry_price

                if profit_percentage >= trigger_profit:
                    # Set stop at breakeven plus small buffer for fees
                    return entry_price * 1.001

            else:  # short
                profit_percentage = (entry_price - current_price) / entry_price

                if profit_percentage >= trigger_profit:
                    # Set stop at breakeven minus small buffer for fees
                    return entry_price * 0.999

            return None

        except Exception as e:
            logger.error(f"Error calculating breakeven stop: {str(e)}")
            return None

    def chandelier_exit(
        self,
        data: pd.DataFrame,
        atr_period: int = 22,
        multiplier: float = 3.0,
        position_type: str = "long",
    ) -> float:
        """
        Calculate Chandelier Exit stop loss.

        Args:
            data: OHLC price data
            atr_period: ATR period for calculation
            multiplier: ATR multiplier
            position_type: 'long' or 'short'

        Returns:
            Chandelier exit stop level
        """
        try:
            # Calculate ATR
            atr = self.calculate_atr(data, atr_period)

            if position_type == "long":
                # Highest high over the period
                highest_high = data["high"].rolling(window=atr_period).max().iloc[-1]
                chandelier_stop = highest_high - (atr * multiplier)
            else:
                # Lowest low over the period
                lowest_low = data["low"].rolling(window=atr_period).min().iloc[-1]
                chandelier_stop = lowest_low + (atr * multiplier)

            return chandelier_stop

        except Exception as e:
            logger.error(f"Error calculating Chandelier Exit: {str(e)}")
            current_price = data["close"].iloc[-1]
            return (
                current_price * 0.95
                if position_type == "long"
                else current_price * 1.05
            )

    def volatility_stop(
        self, data: pd.DataFrame, lookback: int = 20, volatility_factor: float = 2.5
    ) -> tuple[float, float]:
        """
        Calculate volatility-based stop levels.

        Args:
            data: OHLC price data
            lookback: Period for volatility calculation
            volatility_factor: Standard deviation multiplier

        Returns:
            Tuple of (lower_band, upper_band) for stops
        """
        try:
            # Calculate returns and volatility
            returns = data["close"].pct_change()
            volatility = returns.rolling(window=lookback).std()

            # Current price and volatility
            current_price = data["close"].iloc[-1]
            current_vol = volatility.iloc[-1]

            # Calculate bands
            price_std = current_price * current_vol * np.sqrt(lookback)

            lower_band = current_price - (price_std * volatility_factor)
            upper_band = current_price + (price_std * volatility_factor)

            return lower_band, upper_band

        except Exception as e:
            logger.error(f"Error calculating volatility stop: {str(e)}")
            current_price = data["close"].iloc[-1]
            return current_price * 0.95, current_price * 1.05

    def _find_swing_points(self, series: pd.Series, point_type: str) -> list[float]:
        """Find swing highs or lows in price series."""
        points = []

        for i in range(2, len(series) - 2):
            if point_type == "high":
                if (
                    series.iloc[i] > series.iloc[i - 1]
                    and series.iloc[i] > series.iloc[i - 2]
                    and series.iloc[i] > series.iloc[i + 1]
                    and series.iloc[i] > series.iloc[i + 2]
                ):
                    points.append(series.iloc[i])
            else:  # low
                if (
                    series.iloc[i] < series.iloc[i - 1]
                    and series.iloc[i] < series.iloc[i - 2]
                    and series.iloc[i] < series.iloc[i + 1]
                    and series.iloc[i] < series.iloc[i + 2]
                ):
                    points.append(series.iloc[i])

        return points

    def _cluster_levels(
        self, levels: list[float], tolerance: float = 0.002
    ) -> list[dict[str, Any]]:
        """Cluster similar price levels together."""
        if not levels:
            return []

        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(
                    {
                        "level": np.mean(current_cluster),
                        "strength": len(current_cluster),
                    }
                )
                current_cluster = [level]

        # Add last cluster
        if current_cluster:
            clusters.append(
                {"level": np.mean(current_cluster), "strength": len(current_cluster)}
            )

        return clusters

    def _find_nearest_support(
        self, levels: list[dict[str, Any]], current_price: float
    ) -> float:
        """Find nearest support level below current price."""
        supports = [lvl["level"] for lvl in levels if lvl["level"] < current_price]
        return max(supports) if supports else 0

    def _find_nearest_resistance(
        self, levels: list[dict[str, Any]], current_price: float
    ) -> float:
        """Find nearest resistance level above current price."""
        resistances = [lvl["level"] for lvl in levels if lvl["level"] > current_price]
        return min(resistances) if resistances else float("inf")

    def update_settings(
        self,
        atr_multiplier: float | None = None,
        trailing_percentage: float | None = None,
    ):
        """Update stop loss settings."""
        if atr_multiplier is not None:
            self.atr_multiplier = atr_multiplier
        if trailing_percentage is not None:
            self.trailing_percentage = trailing_percentage

        logger.info(
            f"Stop loss settings updated: ATR multiplier={self.atr_multiplier}, "
            f"Trailing percentage={self.trailing_percentage}"
        )

    def get_stop_statistics(self) -> dict[str, Any]:
        """Get statistics on stop loss history."""
        if not self.stop_history:
            return {"message": "No stop loss history available"}

        stop_distances = []
        for stop in self.stop_history:
            if stop["trade_direction"] == "long":
                distance = (stop["entry_price"] - stop["stop_loss"]) / stop[
                    "entry_price"
                ]
            else:
                distance = (stop["stop_loss"] - stop["entry_price"]) / stop[
                    "entry_price"
                ]
            stop_distances.append(distance)

        return {
            "total_stops_set": len(self.stop_history),
            "average_stop_distance": np.mean(stop_distances),
            "min_stop_distance": min(stop_distances),
            "max_stop_distance": max(stop_distances),
            "methods_used": list(
                {s.get("method", "unknown") for s in self.stop_history}
            ),
        }
