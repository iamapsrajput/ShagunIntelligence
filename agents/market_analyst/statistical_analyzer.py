"""Statistical analysis engine for identifying trading opportunities"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from loguru import logger

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available. Using pandas implementations for technical indicators.")
    TALIB_AVAILABLE = False

from .data_processor import TickData, CandleData, RealTimeDataProcessor


@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    reasoning: List[str]
    technical_data: Dict[str, Any] = field(default_factory=dict)
    risk_reward: Optional[Tuple[float, float]] = None  # (risk, reward)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'reasoning': self.reasoning,
            'technical_data': self.technical_data,
            'risk_reward': self.risk_reward
        }


@dataclass
class MarketStats:
    """Market statistics structure"""
    symbol: str
    timestamp: datetime
    price_stats: Dict[str, float]
    volume_stats: Dict[str, float]
    volatility_stats: Dict[str, float]
    trend_stats: Dict[str, float]
    momentum_stats: Dict[str, float]


class TechnicalIndicators:
    """Technical indicators calculation engine"""

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for OHLC data"""
        if df.empty or len(df) < 20:
            return df

        try:
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning("Missing required OHLC columns")
                return df

            df = df.copy()

            # Convert to numpy arrays for TA-Lib
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            volume = df['volume'].values

                        # Moving Averages
            if TALIB_AVAILABLE:
                df['sma_9'] = talib.SMA(close_prices, timeperiod=9)
                df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
                df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
                df['ema_9'] = talib.EMA(close_prices, timeperiod=9)
                df['ema_20'] = talib.EMA(close_prices, timeperiod=20)
                df['ema_50'] = talib.EMA(close_prices, timeperiod=50)

                # Bollinger Bands
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                    close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
                )

                # MACD
                df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(
                    close_prices, fastperiod=12, slowperiod=26, signalperiod=9
                )

                # RSI
                df['rsi'] = talib.RSI(close_prices, timeperiod=14)
            else:
                # Pandas fallback implementations
                df['sma_9'] = df['close'].rolling(window=9).mean()
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['ema_9'] = df['close'].ewm(span=9).mean()
                df['ema_20'] = df['close'].ewm(span=20).mean()
                df['ema_50'] = df['close'].ewm(span=50).mean()

                # Bollinger Bands
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                bb_std = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
                df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

                # MACD
                ema12 = df['close'].ewm(span=12).mean()
                ema26 = df['close'].ewm(span=26).mean()
                df['macd'] = ema12 - ema26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']

                # RSI
                df['rsi'] = TechnicalIndicators._calculate_rsi_pandas(df['close'])

            # Common calculations for both TA-Lib and pandas
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
            df['bb_position'] = (close_prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['rsi_oversold'] = df['rsi'] < 30
            df['rsi_overbought'] = df['rsi'] > 70

            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=14, slowk_period=3, slowd_period=3
            )

            # ADX (Trend Strength)
            if TALIB_AVAILABLE:
                df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
                df['di_plus'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
                df['di_minus'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)

                # Commodity Channel Index
                df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)

                # Williams %R
                df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)

                # Average True Range (Volatility)
                df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)

                # Volume indicators
                df['volume_sma'] = talib.SMA(volume, timeperiod=20)
                df['ad_line'] = talib.AD(high_prices, low_prices, close_prices, volume)
                df['obv'] = talib.OBV(close_prices, volume)

                # Price change and momentum
                df['momentum'] = talib.MOM(close_prices, timeperiod=10)
                df['roc'] = talib.ROC(close_prices, timeperiod=10)
            else:
                # Pandas fallback implementations
                df['adx'] = TechnicalIndicators._calculate_adx_pandas(df)
                df['di_plus'] = TechnicalIndicators._calculate_di_plus_pandas(df)
                df['di_minus'] = TechnicalIndicators._calculate_di_minus_pandas(df)

                # CCI
                df['cci'] = TechnicalIndicators._calculate_cci_pandas(df)

                # Williams %R
                df['williams_r'] = TechnicalIndicators._calculate_williams_r_pandas(df)

                # ATR
                df['atr'] = TechnicalIndicators._calculate_atr_pandas(df)

                # Volume indicators
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                df['ad_line'] = TechnicalIndicators._calculate_ad_pandas(df)
                df['obv'] = TechnicalIndicators._calculate_obv_pandas(df)

                # Momentum
                df['momentum'] = df['close'] - df['close'].shift(10)
                df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100

            # Common calculations
            df['atr_percent'] = df['atr'] / close_prices * 100
            df['volume_ratio'] = volume / df['volume_sma']
            df['price_change'] = close_prices - df['close'].shift(1)
            df['price_change_percent'] = df['price_change'] / df['close'].shift(1) * 100

            # Support and Resistance levels
            df['pivot_high'] = df['high'].rolling(window=5, center=True).max() == df['high']
            df['pivot_low'] = df['low'].rolling(window=5, center=True).min() == df['low']

            # Trend detection
            df['trend_sma'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
            df['trend_ema'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)

            logger.debug(f"Calculated technical indicators for {len(df)} bars")
            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df

    @staticmethod
    def _calculate_rsi_pandas(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using pandas"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)

    @staticmethod
    def _calculate_adx_pandas(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX using pandas"""
        try:
            # True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement
            dm_plus = df['high'] - df['high'].shift(1)
            dm_minus = df['low'].shift(1) - df['low']
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

            # Smoothed values
            tr_smooth = tr.rolling(window=period).mean()
            dm_plus_smooth = dm_plus.rolling(window=period).mean()
            dm_minus_smooth = dm_minus.rolling(window=period).mean()

            # DI+ and DI-
            di_plus = 100 * dm_plus_smooth / tr_smooth
            di_minus = 100 * dm_minus_smooth / tr_smooth

            # DX and ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window=period).mean()

            return adx
        except:
            return pd.Series([25] * len(df), index=df.index)

    @staticmethod
    def _calculate_di_plus_pandas(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate DI+ using pandas"""
        try:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            dm_plus = df['high'] - df['high'].shift(1)
            dm_minus = df['low'].shift(1) - df['low']
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)

            tr_smooth = tr.rolling(window=period).mean()
            dm_plus_smooth = dm_plus.rolling(window=period).mean()

            di_plus = 100 * dm_plus_smooth / tr_smooth
            return di_plus
        except:
            return pd.Series([25] * len(df), index=df.index)

    @staticmethod
    def _calculate_di_minus_pandas(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate DI- using pandas"""
        try:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            dm_plus = df['high'] - df['high'].shift(1)
            dm_minus = df['low'].shift(1) - df['low']
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

            tr_smooth = tr.rolling(window=period).mean()
            dm_minus_smooth = dm_minus.rolling(window=period).mean()

            di_minus = 100 * dm_minus_smooth / tr_smooth
            return di_minus
        except:
            return pd.Series([25] * len(df), index=df.index)

    @staticmethod
    def _calculate_cci_pandas(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate CCI using pandas"""
        try:
            tp = (df['high'] + df['low'] + df['close']) / 3
            tp_sma = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (tp - tp_sma) / (0.015 * mad)
            return cci
        except:
            return pd.Series([0] * len(df), index=df.index)

    @staticmethod
    def _calculate_williams_r_pandas(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R using pandas"""
        try:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
            return williams_r
        except:
            return pd.Series([-50] * len(df), index=df.index)

    @staticmethod
    def _calculate_atr_pandas(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR using pandas"""
        try:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
        except:
            return pd.Series([0] * len(df), index=df.index)

    @staticmethod
    def _calculate_ad_pandas(df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line using pandas"""
        try:
            mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mfm = mfm.fillna(0)
            mfv = mfm * df['volume']
            ad_line = mfv.cumsum()
            return ad_line
        except:
            return pd.Series([0] * len(df), index=df.index)

    @staticmethod
    def _calculate_obv_pandas(df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume using pandas"""
        try:
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = df['volume'].iloc[0]

            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]

            return obv
        except:
            return pd.Series([0] * len(df), index=df.index)


class StatisticalAnalyzer:
    """Main statistical analysis engine"""

    def __init__(self, data_processor: RealTimeDataProcessor):
        self.data_processor = data_processor
        self.scaler = StandardScaler()

        # Analysis parameters
        self.analysis_config = {
            'min_data_points': 50,
            'volatility_threshold': 2.0,  # Standard deviations
            'volume_anomaly_threshold': 2.5,
            'momentum_threshold': 0.7,
            'trend_strength_threshold': 25,  # ADX threshold
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }

        # Signal weights for different factors
        self.signal_weights = {
            'trend': 0.25,
            'momentum': 0.20,
            'volume': 0.15,
            'volatility': 0.10,
            'support_resistance': 0.15,
            'technical_indicators': 0.15
        }

    def analyze_symbol(self, symbol: str, timeframes: List[str] = None) -> TradingSignal:
        """Comprehensive analysis of a symbol"""
        if timeframes is None:
            timeframes = ['1min', '5min', '15min']

        try:
            signals = []
            combined_reasoning = []
            combined_technical_data = {}

            for timeframe in timeframes:
                signal = self._analyze_timeframe(symbol, timeframe)
                if signal:
                    signals.append(signal)
                    combined_reasoning.extend(signal.reasoning)
                    combined_technical_data[f'{timeframe}_data'] = signal.technical_data

            if not signals:
                return TradingSignal(
                    symbol=symbol,
                    signal_type='HOLD',
                    strength=0.0,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    reasoning=['Insufficient data for analysis'],
                    technical_data={}
                )

            # Combine signals from different timeframes
            final_signal = self._combine_signals(signals, symbol)
            final_signal.reasoning = list(set(combined_reasoning))  # Remove duplicates
            final_signal.technical_data = combined_technical_data

            return final_signal

        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {str(e)}")
            return TradingSignal(
                symbol=symbol,
                signal_type='HOLD',
                strength=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                reasoning=[f'Analysis error: {str(e)}'],
                technical_data={}
            )

    def _analyze_timeframe(self, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        """Analyze a specific timeframe for a symbol"""
        try:
            # Get candle data
            df = self.data_processor.get_candle_data(symbol, timeframe, 100)

            if df.empty or len(df) < self.analysis_config['min_data_points']:
                return None

            # Calculate technical indicators
            df = TechnicalIndicators.calculate_all_indicators(df)

            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            # Analyze different components
            trend_signal = self._analyze_trend(df, latest)
            momentum_signal = self._analyze_momentum(df, latest)
            volume_signal = self._analyze_volume(df, latest)
            volatility_signal = self._analyze_volatility(df, latest)
            support_resistance_signal = self._analyze_support_resistance(df, latest)
            technical_signal = self._analyze_technical_indicators(df, latest)

            # Combine signals
            signals = {
                'trend': trend_signal,
                'momentum': momentum_signal,
                'volume': volume_signal,
                'volatility': volatility_signal,
                'support_resistance': support_resistance_signal,
                'technical_indicators': technical_signal
            }

            # Calculate weighted score
            total_score = 0
            total_weight = 0
            reasoning = []

            for component, (score, reason) in signals.items():
                if score is not None:
                    weight = self.signal_weights[component]
                    total_score += score * weight
                    total_weight += weight
                    if reason:
                        reasoning.append(f"{component.title()}: {reason}")

            if total_weight == 0:
                return None

            final_score = total_score / total_weight

            # Determine signal type and strength
            if final_score > 0.3:
                signal_type = 'BUY'
                strength = min(final_score, 1.0)
            elif final_score < -0.3:
                signal_type = 'SELL'
                strength = min(abs(final_score), 1.0)
            else:
                signal_type = 'HOLD'
                strength = abs(final_score)

            # Calculate confidence based on data quality and consensus
            confidence = self._calculate_confidence(df, signals)

            # Calculate risk-reward ratio
            risk_reward = self._calculate_risk_reward(df, latest, signal_type)

            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                reasoning=reasoning,
                technical_data={
                    'timeframe': timeframe,
                    'current_price': latest['close'],
                    'rsi': latest.get('rsi'),
                    'macd': latest.get('macd'),
                    'volume_ratio': latest.get('volume_ratio'),
                    'atr_percent': latest.get('atr_percent'),
                    'adx': latest.get('adx')
                },
                risk_reward=risk_reward
            )

        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe}: {str(e)}")
            return None

    def _analyze_trend(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[Optional[float], str]:
        """Analyze trend strength and direction"""
        try:
            # Multiple trend indicators
            sma_trend = 1 if latest.get('sma_20', 0) > latest.get('sma_50', 0) else -1
            ema_trend = 1 if latest.get('ema_20', 0) > latest.get('ema_50', 0) else -1
            price_above_sma20 = 1 if latest['close'] > latest.get('sma_20', 0) else -1

            adx = latest.get('adx', 0)
            trend_strength = min(adx / 50, 1.0) if adx else 0

            # Combine trend signals
            trend_consensus = (sma_trend + ema_trend + price_above_sma20) / 3

            # Strong trend if ADX > 25 and trend consensus
            if adx > self.analysis_config['trend_strength_threshold']:
                score = trend_consensus * trend_strength
                reason = f"Strong {'uptrend' if trend_consensus > 0 else 'downtrend'} (ADX: {adx:.1f})"
            else:
                score = trend_consensus * 0.3  # Weak trend
                reason = f"Weak trend (ADX: {adx:.1f})"

            return score, reason

        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return None, ""

    def _analyze_momentum(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[Optional[float], str]:
        """Analyze momentum indicators"""
        try:
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            stoch_k = latest.get('stoch_k', 50)

            # RSI momentum
            if rsi > 70:
                rsi_score = -0.8  # Overbought
                rsi_reason = "Overbought"
            elif rsi < 30:
                rsi_score = 0.8  # Oversold
                rsi_reason = "Oversold"
            else:
                rsi_score = (rsi - 50) / 50 * 0.5  # Neutral momentum
                rsi_reason = "Neutral"

            # MACD momentum
            macd_bullish = macd > macd_signal
            macd_score = 0.5 if macd_bullish else -0.5

            # Stochastic momentum
            stoch_score = (stoch_k - 50) / 50 * 0.3

            # Combined momentum score
            momentum_score = (rsi_score * 0.5 + macd_score * 0.3 + stoch_score * 0.2)

            reason = f"RSI {rsi_reason} ({rsi:.1f}), MACD {'Bullish' if macd_bullish else 'Bearish'}"

            return momentum_score, reason

        except Exception as e:
            logger.error(f"Error in momentum analysis: {str(e)}")
            return None, ""

    def _analyze_volume(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[Optional[float], str]:
        """Analyze volume patterns and anomalies"""
        try:
            volume_ratio = latest.get('volume_ratio', 1.0)
            volume = latest.get('volume', 0)

            # Volume surge detection
            if volume_ratio > self.analysis_config['volume_anomaly_threshold']:
                score = min(volume_ratio / 5, 1.0)  # Positive volume surge
                reason = f"High volume surge ({volume_ratio:.1f}x average)"
            elif volume_ratio < 0.5:
                score = -0.3  # Low volume
                reason = f"Below average volume ({volume_ratio:.1f}x)"
            else:
                score = 0.1  # Normal volume
                reason = f"Normal volume ({volume_ratio:.1f}x)"

            return score, reason

        except Exception as e:
            logger.error(f"Error in volume analysis: {str(e)}")
            return None, ""

    def _analyze_volatility(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[Optional[float], str]:
        """Analyze volatility patterns"""
        try:
            atr_percent = latest.get('atr_percent', 0)
            bb_width = latest.get('bb_width', 0)

            # Historical volatility comparison
            if len(df) > 20:
                recent_volatility = df['atr_percent'].tail(5).mean()
                avg_volatility = df['atr_percent'].mean()

                volatility_ratio = recent_volatility / avg_volatility if avg_volatility > 0 else 1

                if volatility_ratio > 1.5:
                    score = -0.3  # High volatility is generally negative for stability
                    reason = f"High volatility ({volatility_ratio:.1f}x average)"
                elif volatility_ratio < 0.7:
                    score = 0.2  # Low volatility can be positive for breakouts
                    reason = f"Low volatility ({volatility_ratio:.1f}x average)"
                else:
                    score = 0.0
                    reason = f"Normal volatility ({volatility_ratio:.1f}x)"
            else:
                score = 0.0
                reason = "Insufficient data for volatility analysis"

            return score, reason

        except Exception as e:
            logger.error(f"Error in volatility analysis: {str(e)}")
            return None, ""

    def _analyze_support_resistance(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[Optional[float], str]:
        """Analyze support and resistance levels"""
        try:
            current_price = latest['close']

            # Bollinger Bands position
            bb_position = latest.get('bb_position', 0.5)

            if bb_position > 0.8:
                score = -0.5  # Near resistance
                reason = "Near Bollinger Band resistance"
            elif bb_position < 0.2:
                score = 0.5   # Near support
                reason = "Near Bollinger Band support"
            else:
                score = 0.0
                reason = "Middle of Bollinger Bands"

            # Pivot points analysis
            if 'pivot_high' in df.columns and 'pivot_low' in df.columns:
                recent_highs = df[df['pivot_high']]['high'].tail(5)
                recent_lows = df[df['pivot_low']]['low'].tail(5)

                if len(recent_highs) > 0:
                    resistance = recent_highs.max()
                    if current_price >= resistance * 0.99:  # Within 1% of resistance
                        score += -0.3
                        reason += ", Near resistance level"

                if len(recent_lows) > 0:
                    support = recent_lows.min()
                    if current_price <= support * 1.01:  # Within 1% of support
                        score += 0.3
                        reason += ", Near support level"

            return score, reason

        except Exception as e:
            logger.error(f"Error in support/resistance analysis: {str(e)}")
            return None, ""

    def _analyze_technical_indicators(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[Optional[float], str]:
        """Analyze various technical indicators"""
        try:
            cci = latest.get('cci', 0)
            williams_r = latest.get('williams_r', -50)

            score = 0
            reasons = []

            # CCI analysis
            if cci > 100:
                score += -0.3  # Overbought
                reasons.append("CCI overbought")
            elif cci < -100:
                score += 0.3   # Oversold
                reasons.append("CCI oversold")

            # Williams %R analysis
            if williams_r > -20:
                score += -0.3  # Overbought
                reasons.append("Williams %R overbought")
            elif williams_r < -80:
                score += 0.3   # Oversold
                reasons.append("Williams %R oversold")

            reason = ", ".join(reasons) if reasons else "Technical indicators neutral"

            return score, reason

        except Exception as e:
            logger.error(f"Error in technical indicators analysis: {str(e)}")
            return None, ""

    def _combine_signals(self, signals: List[TradingSignal], symbol: str) -> TradingSignal:
        """Combine signals from multiple timeframes"""
        try:
            # Weight signals by timeframe (longer timeframes have more weight)
            timeframe_weights = {'1min': 0.2, '5min': 0.3, '15min': 0.5}

            total_score = 0
            total_weight = 0
            combined_confidence = 0
            combined_reasoning = []

            for signal in signals:
                timeframe = signal.technical_data.get('timeframe', '5min')
                weight = timeframe_weights.get(timeframe, 0.3)

                signal_score = signal.strength if signal.signal_type == 'BUY' else -signal.strength if signal.signal_type == 'SELL' else 0

                total_score += signal_score * weight
                total_weight += weight
                combined_confidence += signal.confidence * weight
                combined_reasoning.extend(signal.reasoning)

            if total_weight == 0:
                return signals[0]  # Return first signal if no weights

            final_score = total_score / total_weight
            final_confidence = combined_confidence / total_weight

            # Determine final signal
            if final_score > 0.3:
                signal_type = 'BUY'
                strength = min(final_score, 1.0)
            elif final_score < -0.3:
                signal_type = 'SELL'
                strength = min(abs(final_score), 1.0)
            else:
                signal_type = 'HOLD'
                strength = abs(final_score)

            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=final_confidence,
                timestamp=datetime.now(),
                reasoning=list(set(combined_reasoning))
            )

        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
            return signals[0] if signals else TradingSignal(
                symbol=symbol,
                signal_type='HOLD',
                strength=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                reasoning=['Error combining signals']
            )

    def _calculate_confidence(self, df: pd.DataFrame, signals: Dict[str, Tuple]) -> float:
        """Calculate confidence based on data quality and signal consensus"""
        try:
            # Data quality factors
            data_quality = min(len(df) / 100, 1.0)  # More data = higher confidence

            # Signal consensus
            valid_signals = [score for score, _ in signals.values() if score is not None]
            if not valid_signals:
                return 0.0

            # Check if signals agree
            positive_signals = len([s for s in valid_signals if s > 0])
            negative_signals = len([s for s in valid_signals if s < 0])
            total_signals = len(valid_signals)

            # Consensus = how much signals agree
            consensus = max(positive_signals, negative_signals) / total_signals

            # Volatility factor (lower volatility = higher confidence)
            atr_percent = df['atr_percent'].iloc[-1] if 'atr_percent' in df.columns else 2.0
            volatility_factor = max(0.3, 1.0 - (atr_percent / 5.0))

            # Combined confidence
            confidence = data_quality * consensus * volatility_factor

            return min(confidence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    def _calculate_risk_reward(self, df: pd.DataFrame, latest: pd.Series, signal_type: str) -> Optional[Tuple[float, float]]:
        """Calculate risk-reward ratio based on support/resistance and ATR"""
        try:
            current_price = latest['close']
            atr = latest.get('atr', current_price * 0.02)  # Default 2% ATR

            if signal_type == 'BUY':
                # Risk: to nearest support or 2x ATR
                risk = min(atr * 2, current_price * 0.03)  # Max 3% risk
                # Reward: to nearest resistance or 3x ATR
                reward = atr * 3
            elif signal_type == 'SELL':
                # Risk: to nearest resistance or 2x ATR
                risk = min(atr * 2, current_price * 0.03)  # Max 3% risk
                # Reward: to nearest support or 3x ATR
                reward = atr * 3
            else:
                return None

            return (risk / current_price * 100, reward / current_price * 100)  # Return as percentages

        except Exception as e:
            logger.error(f"Error calculating risk-reward: {str(e)}")
            return None

    def get_market_statistics(self, symbols: List[str]) -> Dict[str, MarketStats]:
        """Get comprehensive market statistics for symbols"""
        stats = {}

        for symbol in symbols:
            try:
                df = self.data_processor.get_candle_data(symbol, "5min", 100)
                if df.empty:
                    continue

                df = TechnicalIndicators.calculate_all_indicators(df)
                latest = df.iloc[-1]

                price_stats = {
                    'current_price': latest['close'],
                    'daily_high': df['high'].max(),
                    'daily_low': df['low'].min(),
                    'price_change': latest.get('price_change', 0),
                    'price_change_percent': latest.get('price_change_percent', 0)
                }

                volume_stats = {
                    'current_volume': latest['volume'],
                    'avg_volume': df['volume'].mean(),
                    'volume_ratio': latest.get('volume_ratio', 1.0),
                    'total_volume': df['volume'].sum()
                }

                volatility_stats = {
                    'atr': latest.get('atr', 0),
                    'atr_percent': latest.get('atr_percent', 0),
                    'bb_width': latest.get('bb_width', 0),
                    'price_volatility': df['close'].std()
                }

                trend_stats = {
                    'adx': latest.get('adx', 0),
                    'trend_direction': latest.get('trend_sma', 0),
                    'sma20': latest.get('sma_20', 0),
                    'sma50': latest.get('sma_50', 0)
                }

                momentum_stats = {
                    'rsi': latest.get('rsi', 50),
                    'macd': latest.get('macd', 0),
                    'stoch_k': latest.get('stoch_k', 50),
                    'momentum': latest.get('momentum', 0)
                }

                stats[symbol] = MarketStats(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price_stats=price_stats,
                    volume_stats=volume_stats,
                    volatility_stats=volatility_stats,
                    trend_stats=trend_stats,
                    momentum_stats=momentum_stats
                )

            except Exception as e:
                logger.error(f"Error calculating stats for {symbol}: {str(e)}")
                continue

        return stats