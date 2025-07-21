"""Module for calculating technical indicators using TA-Lib and pandas."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from loguru import logger

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available. Using pandas implementations.")
    TALIB_AVAILABLE = False


class IndicatorCalculator:
    """Calculate various technical indicators efficiently."""

    def __init__(self):
        """Initialize the indicator calculator."""
        self.use_talib = TALIB_AVAILABLE
        logger.info(f"IndicatorCalculator initialized (TA-Lib: {self.use_talib})")

    def calculate_rsi(
        self, 
        data: pd.DataFrame, 
        period: int = 14,
        price_column: str = "close"
    ) -> Dict[str, Any]:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price data with OHLCV columns
            period: RSI period (default: 14)
            price_column: Column to use for calculation
            
        Returns:
            RSI values and metadata
        """
        try:
            prices = data[price_column].values
            
            if self.use_talib:
                rsi_values = talib.RSI(prices, timeperiod=period)
            else:
                rsi_values = self._calculate_rsi_pandas(prices, period)
            
            current_rsi = rsi_values[-1] if not np.isnan(rsi_values[-1]) else None
            
            return {
                "values": rsi_values,
                "current": current_rsi,
                "period": period,
                "overbought": current_rsi > 70 if current_rsi else False,
                "oversold": current_rsi < 30 if current_rsi else False,
                "timestamp": data.index[-1] if hasattr(data, 'index') else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            raise

    def calculate_macd(
        self, 
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_column: str = "close"
    ) -> Dict[str, Any]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price data with OHLCV columns
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
            price_column: Column to use for calculation
            
        Returns:
            MACD values, signal line, and histogram
        """
        try:
            prices = data[price_column].values
            
            if self.use_talib:
                macd, signal, histogram = talib.MACD(
                    prices, 
                    fastperiod=fast_period,
                    slowperiod=slow_period, 
                    signalperiod=signal_period
                )
            else:
                macd, signal, histogram = self._calculate_macd_pandas(
                    prices, fast_period, slow_period, signal_period
                )
            
            # Check for crossovers
            crossover = self._check_crossover(macd, signal)
            
            return {
                "macd": macd,
                "signal": signal,
                "histogram": histogram,
                "current_macd": macd[-1] if not np.isnan(macd[-1]) else None,
                "current_signal": signal[-1] if not np.isnan(signal[-1]) else None,
                "current_histogram": histogram[-1] if not np.isnan(histogram[-1]) else None,
                "crossover": crossover,
                "parameters": {
                    "fast": fast_period,
                    "slow": slow_period,
                    "signal": signal_period
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            raise

    def calculate_bollinger_bands(
        self, 
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        price_column: str = "close"
    ) -> Dict[str, Any]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price data with OHLCV columns
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            price_column: Column to use for calculation
            
        Returns:
            Upper band, middle band (SMA), and lower band
        """
        try:
            prices = data[price_column].values
            
            if self.use_talib:
                upper, middle, lower = talib.BBANDS(
                    prices,
                    timeperiod=period,
                    nbdevup=std_dev,
                    nbdevdn=std_dev,
                    matype=0
                )
            else:
                upper, middle, lower = self._calculate_bollinger_bands_pandas(
                    prices, period, std_dev
                )
            
            current_price = prices[-1]
            position = self._get_band_position(current_price, upper[-1], middle[-1], lower[-1])
            
            return {
                "upper": upper,
                "middle": middle,
                "lower": lower,
                "current_upper": upper[-1] if not np.isnan(upper[-1]) else None,
                "current_middle": middle[-1] if not np.isnan(middle[-1]) else None,
                "current_lower": lower[-1] if not np.isnan(lower[-1]) else None,
                "band_width": (upper[-1] - lower[-1]) if not np.isnan(upper[-1]) else None,
                "position": position,
                "parameters": {
                    "period": period,
                    "std_dev": std_dev
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise

    def calculate_sma(
        self, 
        data: pd.DataFrame,
        periods: Union[int, list] = [10, 20, 50],
        price_column: str = "close"
    ) -> Dict[str, Any]:
        """
        Calculate Simple Moving Average(s).
        
        Args:
            data: Price data with OHLCV columns
            periods: Period(s) for SMA calculation
            price_column: Column to use for calculation
            
        Returns:
            SMA values for each period
        """
        try:
            prices = data[price_column].values
            if isinstance(periods, int):
                periods = [periods]
            
            sma_results = {}
            for period in periods:
                if self.use_talib:
                    sma = talib.SMA(prices, timeperiod=period)
                else:
                    sma = pd.Series(prices).rolling(window=period).mean().values
                
                sma_results[f"SMA_{period}"] = {
                    "values": sma,
                    "current": sma[-1] if not np.isnan(sma[-1]) else None,
                    "period": period
                }
            
            # Check for golden/death crosses
            if 50 in periods and 200 in periods:
                cross = self._check_ma_cross(
                    sma_results["SMA_50"]["values"],
                    sma_results["SMA_200"]["values"]
                )
                sma_results["golden_cross"] = cross == "golden"
                sma_results["death_cross"] = cross == "death"
            
            return sma_results
            
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            raise

    def calculate_ema(
        self, 
        data: pd.DataFrame,
        periods: Union[int, list] = [12, 26],
        price_column: str = "close"
    ) -> Dict[str, Any]:
        """
        Calculate Exponential Moving Average(s).
        
        Args:
            data: Price data with OHLCV columns
            periods: Period(s) for EMA calculation
            price_column: Column to use for calculation
            
        Returns:
            EMA values for each period
        """
        try:
            prices = data[price_column].values
            if isinstance(periods, int):
                periods = [periods]
            
            ema_results = {}
            for period in periods:
                if self.use_talib:
                    ema = talib.EMA(prices, timeperiod=period)
                else:
                    ema = pd.Series(prices).ewm(span=period, adjust=False).mean().values
                
                ema_results[f"EMA_{period}"] = {
                    "values": ema,
                    "current": ema[-1] if not np.isnan(ema[-1]) else None,
                    "period": period
                }
            
            return ema_results
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            raise

    # Helper methods for pandas-based calculations
    def _calculate_rsi_pandas(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI using pandas when TA-Lib is not available."""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:period] = np.nan
        rsi[period] = 100. - 100. / (1. + rs)
        
        for i in range(period + 1, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi

    def _calculate_macd_pandas(
        self, 
        prices: np.ndarray, 
        fast: int, 
        slow: int, 
        signal: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD using pandas when TA-Lib is not available."""
        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=fast, adjust=False).mean()
        ema_slow = prices_series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line.values, signal_line.values, histogram.values

    def _calculate_bollinger_bands_pandas(
        self, 
        prices: np.ndarray, 
        period: int, 
        std_dev: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands using pandas when TA-Lib is not available."""
        prices_series = pd.Series(prices)
        middle = prices_series.rolling(window=period).mean()
        std = prices_series.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper.values, middle.values, lower.values

    def _check_crossover(self, line1: np.ndarray, line2: np.ndarray) -> Optional[str]:
        """Check for crossover between two lines."""
        if len(line1) < 2 or np.isnan(line1[-1]) or np.isnan(line2[-1]):
            return None
        
        if line1[-2] < line2[-2] and line1[-1] > line2[-1]:
            return "bullish"
        elif line1[-2] > line2[-2] and line1[-1] < line2[-1]:
            return "bearish"
        
        return None

    def _get_band_position(
        self, 
        price: float, 
        upper: float, 
        middle: float, 
        lower: float
    ) -> str:
        """Determine price position relative to Bollinger Bands."""
        if np.isnan(upper) or np.isnan(lower):
            return "unknown"
        
        if price > upper:
            return "above_upper"
        elif price < lower:
            return "below_lower"
        elif price > middle:
            return "upper_half"
        else:
            return "lower_half"

    def _check_ma_cross(self, fast_ma: np.ndarray, slow_ma: np.ndarray) -> Optional[str]:
        """Check for golden cross or death cross."""
        if len(fast_ma) < 2:
            return None
        
        # Remove NaN values for comparison
        valid_idx = ~(np.isnan(fast_ma) | np.isnan(slow_ma))
        if np.sum(valid_idx) < 2:
            return None
        
        valid_fast = fast_ma[valid_idx]
        valid_slow = slow_ma[valid_idx]
        
        if len(valid_fast) < 2:
            return None
        
        if valid_fast[-2] < valid_slow[-2] and valid_fast[-1] > valid_slow[-1]:
            return "golden"
        elif valid_fast[-2] > valid_slow[-2] and valid_fast[-1] < valid_slow[-1]:
            return "death"
        
        return None