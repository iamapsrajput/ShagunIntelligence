"""Format technical indicator data for frontend visualization."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class VisualizationFormatter:
    """Format indicator data for chart libraries and frontend visualization."""

    def __init__(self):
        """Initialize the visualization formatter."""
        self.chart_colors = {
            "price": "#2196F3",
            "sma": "#FFC107",
            "ema": "#FF9800",
            "bb_upper": "#E91E63",
            "bb_middle": "#9C27B0",
            "bb_lower": "#E91E63",
            "volume": "#607D8B",
            "macd": "#4CAF50",
            "macd_signal": "#FF5722",
            "rsi": "#00BCD4",
        }
        logger.info("VisualizationFormatter initialized")

    def format_data(
        self,
        price_data: pd.DataFrame,
        indicators: dict[str, Any],
        signals: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Format all data for frontend visualization.

        Args:
            price_data: OHLCV price data
            indicators: Calculated indicator values
            signals: Generated trading signals

        Returns:
            Formatted data ready for charting
        """
        try:
            formatted_data = {
                "candlestick": self._format_candlestick_data(price_data),
                "volume": self._format_volume_data(price_data),
                "indicators": {},
                "signals": self._format_signals(signals, price_data),
                "metadata": self._generate_metadata(price_data, indicators),
            }

            # Format each indicator
            for indicator_name, indicator_data in indicators.items():
                if indicator_name == "RSI":
                    formatted_data["indicators"]["rsi"] = self._format_rsi(
                        indicator_data
                    )
                elif indicator_name == "MACD":
                    formatted_data["indicators"]["macd"] = self._format_macd(
                        indicator_data
                    )
                elif indicator_name == "BB":
                    formatted_data["indicators"]["bollinger_bands"] = (
                        self._format_bollinger_bands(indicator_data, price_data)
                    )
                elif indicator_name == "SMA":
                    formatted_data["indicators"]["sma"] = self._format_moving_averages(
                        indicator_data, "SMA"
                    )
                elif indicator_name == "EMA":
                    formatted_data["indicators"]["ema"] = self._format_moving_averages(
                        indicator_data, "EMA"
                    )

            return formatted_data

        except Exception as e:
            logger.error(f"Error formatting visualization data: {str(e)}")
            raise

    def _format_candlestick_data(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Format OHLCV data for candlestick charts."""
        candlestick_data = []

        for idx, row in data.iterrows():
            candlestick_data.append(
                {
                    "timestamp": (
                        idx.isoformat() if isinstance(idx, datetime) else str(idx)
                    ),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "color": "#4CAF50" if row["close"] >= row["open"] else "#F44336",
                }
            )

        return candlestick_data

    def _format_volume_data(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Format volume data for bar charts."""
        volume_data = []

        for idx, row in data.iterrows():
            volume_data.append(
                {
                    "timestamp": (
                        idx.isoformat() if isinstance(idx, datetime) else str(idx)
                    ),
                    "volume": int(row["volume"]),
                    "color": "#4CAF50" if row["close"] >= row["open"] else "#F44336",
                }
            )

        return volume_data

    def _format_rsi(self, rsi_data: dict[str, Any]) -> dict[str, Any]:
        """Format RSI data for line charts."""
        values = rsi_data.get("values", [])
        timestamps = self._generate_timestamps(len(values))

        formatted_rsi = {
            "type": "line",
            "data": [],
            "current_value": rsi_data.get("current"),
            "overbought_level": 70,
            "oversold_level": 30,
            "color": self.chart_colors["rsi"],
            "zones": [
                {"start": 0, "end": 30, "color": "#4CAF5020", "label": "Oversold"},
                {"start": 70, "end": 100, "color": "#F4433620", "label": "Overbought"},
            ],
        }

        for _i, (timestamp, value) in enumerate(zip(timestamps, values, strict=False)):
            if not np.isnan(value):
                formatted_rsi["data"].append(
                    {"timestamp": timestamp, "value": float(value)}
                )

        return formatted_rsi

    def _format_macd(self, macd_data: dict[str, Any]) -> dict[str, Any]:
        """Format MACD data for combination charts."""
        macd_line = macd_data.get("macd", [])
        signal_line = macd_data.get("signal", [])
        histogram = macd_data.get("histogram", [])
        timestamps = self._generate_timestamps(len(macd_line))

        formatted_macd = {
            "macd_line": {
                "type": "line",
                "data": [],
                "color": self.chart_colors["macd"],
            },
            "signal_line": {
                "type": "line",
                "data": [],
                "color": self.chart_colors["macd_signal"],
            },
            "histogram": {
                "type": "bar",
                "data": [],
                "positive_color": "#4CAF50",
                "negative_color": "#F44336",
            },
            "crossover": macd_data.get("crossover"),
        }

        for i, timestamp in enumerate(timestamps):
            if i < len(macd_line) and not np.isnan(macd_line[i]):
                formatted_macd["macd_line"]["data"].append(
                    {"timestamp": timestamp, "value": float(macd_line[i])}
                )

            if i < len(signal_line) and not np.isnan(signal_line[i]):
                formatted_macd["signal_line"]["data"].append(
                    {"timestamp": timestamp, "value": float(signal_line[i])}
                )

            if i < len(histogram) and not np.isnan(histogram[i]):
                formatted_macd["histogram"]["data"].append(
                    {
                        "timestamp": timestamp,
                        "value": float(histogram[i]),
                        "color": "#4CAF50" if histogram[i] > 0 else "#F44336",
                    }
                )

        return formatted_macd

    def _format_bollinger_bands(
        self, bb_data: dict[str, Any], price_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Format Bollinger Bands data for area charts."""
        upper = bb_data.get("upper", [])
        middle = bb_data.get("middle", [])
        lower = bb_data.get("lower", [])

        formatted_bb = {
            "type": "area",
            "upper_band": {
                "data": [],
                "color": self.chart_colors["bb_upper"],
                "lineStyle": "dashed",
            },
            "middle_band": {
                "data": [],
                "color": self.chart_colors["bb_middle"],
                "lineStyle": "solid",
            },
            "lower_band": {
                "data": [],
                "color": self.chart_colors["bb_lower"],
                "lineStyle": "dashed",
            },
            "fill": {"color": "#E91E6310", "between": ["upper_band", "lower_band"]},
        }

        for idx, _row in price_data.iterrows():
            timestamp = idx.isoformat() if isinstance(idx, datetime) else str(idx)
            i = price_data.index.get_loc(idx)

            if i < len(upper) and not np.isnan(upper[i]):
                formatted_bb["upper_band"]["data"].append(
                    {"timestamp": timestamp, "value": float(upper[i])}
                )

            if i < len(middle) and not np.isnan(middle[i]):
                formatted_bb["middle_band"]["data"].append(
                    {"timestamp": timestamp, "value": float(middle[i])}
                )

            if i < len(lower) and not np.isnan(lower[i]):
                formatted_bb["lower_band"]["data"].append(
                    {"timestamp": timestamp, "value": float(lower[i])}
                )

        return formatted_bb

    def _format_moving_averages(
        self, ma_data: dict[str, Any], ma_type: str
    ) -> dict[str, Any]:
        """Format moving average data for line charts."""
        formatted_ma = {"type": "line", "series": []}

        for key, values in ma_data.items():
            if key.startswith(ma_type):
                period = values.get("period")
                data_values = values.get("values", [])
                timestamps = self._generate_timestamps(len(data_values))

                series_data = {
                    "name": key,
                    "period": period,
                    "data": [],
                    "color": self._get_ma_color(period),
                    "current_value": values.get("current"),
                }

                for timestamp, value in zip(timestamps, data_values, strict=False):
                    if not np.isnan(value):
                        series_data["data"].append(
                            {"timestamp": timestamp, "value": float(value)}
                        )

                formatted_ma["series"].append(series_data)

        return formatted_ma

    def _format_signals(
        self, signals: dict[str, Any], price_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Format trading signals for chart annotations."""
        if not signals or "combined_signal" not in signals:
            return {"annotations": []}

        combined_signal = signals["combined_signal"]
        signal_type = combined_signal.get("signal", "NEUTRAL")
        confidence = combined_signal.get("confidence", 0)

        # Get last price for signal annotation
        last_price = price_data["close"].iloc[-1]
        last_timestamp = price_data.index[-1]

        annotation = {
            "timestamp": (
                last_timestamp.isoformat()
                if isinstance(last_timestamp, datetime)
                else str(last_timestamp)
            ),
            "price": float(last_price),
            "type": signal_type,
            "confidence": float(confidence),
            "label": f"{signal_type} ({confidence:.0%})",
            "color": self._get_signal_color(signal_type),
            "shape": self._get_signal_shape(signal_type),
        }

        # Add individual signal indicators
        individual_signals = signals.get("individual_signals", {})
        annotation["indicators"] = []

        for signal_name, signal_data in individual_signals.items():
            annotation["indicators"].append(
                {
                    "name": signal_name.replace("_signal", "").upper(),
                    "signal": (
                        signal_data.get("signal", "").value
                        if hasattr(signal_data.get("signal", ""), "value")
                        else str(signal_data.get("signal", ""))
                    ),
                    "confidence": signal_data.get("confidence", 0),
                }
            )

        return {
            "annotations": [annotation],
            "current_signal": signal_type,
            "confidence": confidence,
            "reasons": combined_signal.get("reasons", []),
        }

    def _generate_metadata(
        self, price_data: pd.DataFrame, indicators: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate metadata for the visualization."""
        return {
            "data_points": len(price_data),
            "timeframe": self._detect_timeframe(price_data),
            "start_time": (
                price_data.index[0].isoformat()
                if isinstance(price_data.index[0], datetime)
                else str(price_data.index[0])
            ),
            "end_time": (
                price_data.index[-1].isoformat()
                if isinstance(price_data.index[-1], datetime)
                else str(price_data.index[-1])
            ),
            "indicators_calculated": list(indicators.keys()),
            "chart_type": "candlestick",
            "last_update": datetime.now().isoformat(),
        }

    def _generate_timestamps(self, length: int) -> list[str]:
        """Generate placeholder timestamps for data without explicit timestamps."""
        # This would be replaced with actual timestamps from the data
        return [f"T{i}" for i in range(length)]

    def _get_ma_color(self, period: int) -> str:
        """Get color for moving average based on period."""
        color_map = {
            10: "#2196F3",
            20: "#FFC107",
            50: "#FF9800",
            100: "#9C27B0",
            200: "#E91E63",
        }
        return color_map.get(period, "#607D8B")

    def _get_signal_color(self, signal_type: str) -> str:
        """Get color for signal type."""
        color_map = {
            "STRONG_BUY": "#1B5E20",
            "BUY": "#4CAF50",
            "NEUTRAL": "#FFC107",
            "SELL": "#FF5722",
            "STRONG_SELL": "#B71C1C",
        }
        return color_map.get(signal_type, "#607D8B")

    def _get_signal_shape(self, signal_type: str) -> str:
        """Get shape for signal type."""
        shape_map = {
            "STRONG_BUY": "triangle-up",
            "BUY": "arrow-up",
            "NEUTRAL": "circle",
            "SELL": "arrow-down",
            "STRONG_SELL": "triangle-down",
        }
        return shape_map.get(signal_type, "circle")

    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """Detect timeframe from data timestamps."""
        if len(data) < 2:
            return "unknown"

        # Calculate average time difference
        if hasattr(data.index, "to_series"):
            time_diffs = data.index.to_series().diff().dropna()
            avg_diff = time_diffs.mean()

            if avg_diff <= pd.Timedelta(minutes=2):
                return "1min"
            elif avg_diff <= pd.Timedelta(minutes=10):
                return "5min"
            elif avg_diff <= pd.Timedelta(minutes=30):
                return "15min"
            elif avg_diff <= pd.Timedelta(hours=2):
                return "1hour"
            else:
                return "daily"

        return "unknown"

    def format_for_api_response(
        self, visualization_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Format visualization data for API response.

        Args:
            visualization_data: Formatted visualization data

        Returns:
            API-ready response format
        """
        return {
            "success": True,
            "data": visualization_data,
            "timestamp": datetime.now().isoformat(),
        }
