"""Efficient rolling window data management for technical indicators."""

import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger


class RollingDataManager:
    """Manage rolling windows of market data for efficient indicator calculations."""

    def __init__(self, max_window_size: int = 500):
        """
        Initialize the rolling data manager.

        Args:
            max_window_size: Maximum number of data points to keep per symbol/timeframe
        """
        self.max_window_size = max_window_size
        self.data_store: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
        self.last_update: dict[str, dict[str, datetime]] = defaultdict(dict)
        self.lock = threading.Lock()

        # Cache for calculated values
        self.indicator_cache: dict[str, dict[str, Any]] = defaultdict(dict)
        self.cache_expiry = timedelta(seconds=30)  # Cache for 30 seconds

        logger.info(
            f"RollingDataManager initialized with window size: {max_window_size}"
        )

    def update_data(self, symbol: str, timeframe: str, new_data: pd.DataFrame) -> None:
        """
        Update rolling data for a symbol/timeframe combination.

        Args:
            symbol: Stock symbol
            timeframe: Time interval (1min, 5min, 15min)
            new_data: New OHLCV data to add
        """
        with self.lock:
            try:
                key = f"{symbol}_{timeframe}"

                if key in self.data_store[symbol]:
                    # Append new data and maintain window size
                    existing_data = self.data_store[symbol][key]
                    combined_data = pd.concat([existing_data, new_data])

                    # Remove duplicates based on index (timestamp)
                    combined_data = combined_data[
                        ~combined_data.index.duplicated(keep="last")
                    ]

                    # Keep only the most recent data points
                    if len(combined_data) > self.max_window_size:
                        combined_data = combined_data.iloc[-self.max_window_size :]

                    self.data_store[symbol][key] = combined_data
                else:
                    # First time storing data for this symbol/timeframe
                    if len(new_data) > self.max_window_size:
                        new_data = new_data.iloc[-self.max_window_size :]
                    self.data_store[symbol][key] = new_data

                self.last_update[symbol][key] = datetime.now()

                # Invalidate cache for this symbol/timeframe
                cache_key = f"{symbol}_{timeframe}"
                if cache_key in self.indicator_cache:
                    del self.indicator_cache[cache_key]

                logger.debug(
                    f"Updated data for {symbol} {timeframe}, "
                    f"current size: {len(self.data_store[symbol][key])}"
                )

            except Exception as e:
                logger.error(f"Error updating data for {symbol} {timeframe}: {str(e)}")
                raise

    def get_calculation_window(
        self, symbol: str, timeframe: str, window_size: int | None = None
    ) -> pd.DataFrame | None:
        """
        Get optimized data window for calculations.

        Args:
            symbol: Stock symbol
            timeframe: Time interval
            window_size: Specific window size (uses all available if None)

        Returns:
            DataFrame with requested window of data
        """
        with self.lock:
            try:
                key = f"{symbol}_{timeframe}"

                if symbol not in self.data_store or key not in self.data_store[symbol]:
                    logger.warning(f"No data available for {symbol} {timeframe}")
                    return None

                data = self.data_store[symbol][key]

                if window_size and len(data) > window_size:
                    return data.iloc[-window_size:].copy()

                return data.copy()

            except Exception as e:
                logger.error(f"Error getting calculation window: {str(e)}")
                return None

    def get_latest_data(
        self, symbol: str, timeframe: str, num_points: int = 1
    ) -> pd.DataFrame | None:
        """
        Get the most recent data points.

        Args:
            symbol: Stock symbol
            timeframe: Time interval
            num_points: Number of recent points to return

        Returns:
            DataFrame with recent data points
        """
        with self.lock:
            try:
                key = f"{symbol}_{timeframe}"

                if symbol not in self.data_store or key not in self.data_store[symbol]:
                    return None

                data = self.data_store[symbol][key]

                if len(data) >= num_points:
                    return data.iloc[-num_points:].copy()

                return data.copy()

            except Exception as e:
                logger.error(f"Error getting latest data: {str(e)}")
                return None

    def get_data_stats(self, symbol: str, timeframe: str) -> dict[str, Any]:
        """
        Get statistics about stored data.

        Args:
            symbol: Stock symbol
            timeframe: Time interval

        Returns:
            Statistics about the data
        """
        with self.lock:
            key = f"{symbol}_{timeframe}"

            if symbol not in self.data_store or key not in self.data_store[symbol]:
                return {"available": False, "data_points": 0}

            data = self.data_store[symbol][key]
            last_update = self.last_update[symbol].get(key)

            return {
                "available": True,
                "data_points": len(data),
                "first_timestamp": data.index[0] if len(data) > 0 else None,
                "last_timestamp": data.index[-1] if len(data) > 0 else None,
                "last_update": last_update.isoformat() if last_update else None,
                "memory_usage_bytes": data.memory_usage(deep=True).sum(),
            }

    def cache_indicator_value(
        self, symbol: str, timeframe: str, indicator: str, value: Any
    ) -> None:
        """
        Cache calculated indicator value.

        Args:
            symbol: Stock symbol
            timeframe: Time interval
            indicator: Indicator name
            value: Calculated value to cache
        """
        cache_key = f"{symbol}_{timeframe}"
        if cache_key not in self.indicator_cache:
            self.indicator_cache[cache_key] = {}

        self.indicator_cache[cache_key][indicator] = {
            "value": value,
            "timestamp": datetime.now(),
        }

    def get_cached_indicator(
        self, symbol: str, timeframe: str, indicator: str
    ) -> Any | None:
        """
        Get cached indicator value if still valid.

        Args:
            symbol: Stock symbol
            timeframe: Time interval
            indicator: Indicator name

        Returns:
            Cached value if valid, None otherwise
        """
        cache_key = f"{symbol}_{timeframe}"

        if (
            cache_key in self.indicator_cache
            and indicator in self.indicator_cache[cache_key]
        ):
            cached = self.indicator_cache[cache_key][indicator]
            if datetime.now() - cached["timestamp"] < self.cache_expiry:
                return cached["value"]

        return None

    def optimize_memory(self) -> dict[str, int]:
        """
        Optimize memory usage by removing old data.

        Returns:
            Statistics about cleaned data
        """
        with self.lock:
            stats = {"symbols_cleaned": 0, "data_points_removed": 0}

            try:
                current_time = datetime.now()

                # Remove data not updated in the last hour
                symbols_to_remove = []

                for symbol in list(self.data_store.keys()):
                    timeframes_to_remove = []

                    for key in list(self.data_store[symbol].keys()):
                        key.replace(f"{symbol}_", "")
                        last_update = self.last_update[symbol].get(key)

                        if last_update and (current_time - last_update) > timedelta(
                            hours=1
                        ):
                            timeframes_to_remove.append(key)
                            stats["data_points_removed"] += len(
                                self.data_store[symbol][key]
                            )

                    # Remove old timeframes
                    for key in timeframes_to_remove:
                        del self.data_store[symbol][key]
                        if key in self.last_update[symbol]:
                            del self.last_update[symbol][key]

                    # Remove symbol if no data left
                    if not self.data_store[symbol]:
                        symbols_to_remove.append(symbol)

                # Clean up empty symbols
                for symbol in symbols_to_remove:
                    del self.data_store[symbol]
                    del self.last_update[symbol]
                    stats["symbols_cleaned"] += 1

                # Clear old cache entries
                cache_keys_to_remove = []
                for cache_key, indicators in self.indicator_cache.items():
                    indicators_to_remove = []

                    for indicator, cached in indicators.items():
                        if current_time - cached["timestamp"] > self.cache_expiry:
                            indicators_to_remove.append(indicator)

                    for indicator in indicators_to_remove:
                        del indicators[indicator]

                    if not indicators:
                        cache_keys_to_remove.append(cache_key)

                for cache_key in cache_keys_to_remove:
                    del self.indicator_cache[cache_key]

                logger.info(f"Memory optimization completed: {stats}")
                return stats

            except Exception as e:
                logger.error(f"Error during memory optimization: {str(e)}")
                return stats

    def get_status(self) -> dict[str, Any]:
        """Get the status of the rolling data manager."""
        with self.lock:
            total_data_points = 0
            total_memory = 0

            for symbol_data in self.data_store.values():
                for data in symbol_data.values():
                    total_data_points += len(data)
                    total_memory += data.memory_usage(deep=True).sum()

            return {
                "status": "active",
                "max_window_size": self.max_window_size,
                "active_symbols": len(self.data_store),
                "total_data_points": total_data_points,
                "total_memory_mb": total_memory / (1024 * 1024),
                "cache_size": len(self.indicator_cache),
                "cache_expiry_seconds": self.cache_expiry.total_seconds(),
            }

    def clear_symbol_data(self, symbol: str) -> None:
        """
        Clear all data for a specific symbol.

        Args:
            symbol: Stock symbol to clear
        """
        with self.lock:
            if symbol in self.data_store:
                del self.data_store[symbol]
            if symbol in self.last_update:
                del self.last_update[symbol]

            # Clear cache entries for this symbol
            cache_keys_to_remove = [
                key
                for key in self.indicator_cache.keys()
                if key.startswith(f"{symbol}_")
            ]
            for key in cache_keys_to_remove:
                del self.indicator_cache[key]

            logger.info(f"Cleared all data for symbol: {symbol}")
