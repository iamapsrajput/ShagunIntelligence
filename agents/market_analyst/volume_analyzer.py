"""Volume analysis and anomaly detection system"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import talib
from loguru import logger

from .data_processor import RealTimeDataProcessor, TickData


@dataclass
class VolumeAnomaly:
    """Volume anomaly detection result"""
    symbol: str
    anomaly_type: str  # 'spike', 'drought', 'unusual_pattern'
    severity: float  # 0.0 to 1.0
    current_volume: int
    expected_volume: int
    volume_ratio: float
    timestamp: datetime
    description: str
    supporting_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'anomaly_type': self.anomaly_type,
            'severity': self.severity,
            'current_volume': self.current_volume,
            'expected_volume': self.expected_volume,
            'volume_ratio': self.volume_ratio,
            'timestamp': self.timestamp,
            'description': self.description,
            'supporting_data': self.supporting_data
        }


@dataclass
class VolumeProfile:
    """Volume profile analysis"""
    symbol: str
    price_levels: List[float]
    volume_at_price: List[int]
    poc: float  # Point of Control (price with highest volume)
    vah: float  # Value Area High
    val: float  # Value Area Low
    value_area_volume_percent: float
    timestamp: datetime


@dataclass
class VolumeSignal:
    """Volume-based trading signal"""
    symbol: str
    signal_type: str  # 'accumulation', 'distribution', 'breakout_volume', 'climax'
    strength: float
    volume_metrics: Dict[str, float]
    price_volume_relationship: str
    timestamp: datetime
    reasoning: List[str]


class VolumeIndicators:
    """Volume-based technical indicators"""
    
    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various volume indicators"""
        if df.empty or len(df) < 20:
            return df
        
        try:
            df = df.copy()
            close_prices = df['close'].values
            volume = df['volume'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            # Basic volume indicators
            df['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
            df['volume_ratio'] = volume / df['volume_sma_20']
            
            # On-Balance Volume (OBV)
            df['obv'] = talib.OBV(close_prices, volume)
            df['obv_sma'] = talib.SMA(df['obv'].values, timeperiod=20)
            
            # Accumulation/Distribution Line
            df['ad_line'] = talib.AD(high_prices, low_prices, close_prices, volume)
            df['ad_sma'] = talib.SMA(df['ad_line'].values, timeperiod=20)
            
            # Chaikin Money Flow
            df['cmf'] = VolumeIndicators._calculate_cmf(df, 20)
            
            # Volume Price Trend
            df['vpt'] = VolumeIndicators._calculate_vpt(df)
            
            # Money Flow Index
            df['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
            
            # Volume-Weighted Average Price (VWAP)
            df['vwap'] = VolumeIndicators._calculate_vwap(df)
            
            # Volume Rate of Change
            df['volume_roc'] = talib.ROC(volume, timeperiod=10)
            
            # Ease of Movement
            df['eom'] = VolumeIndicators._calculate_ease_of_movement(df)
            
            # Force Index
            df['force_index'] = VolumeIndicators._calculate_force_index(df)
            
            # Klinger Oscillator
            df['klinger'] = VolumeIndicators._calculate_klinger_oscillator(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return df
    
    @staticmethod
    def _calculate_cmf(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        try:
            mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mf_multiplier = mf_multiplier.fillna(0)
            mf_volume = mf_multiplier * df['volume']
            
            cmf = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
            return cmf.fillna(0)
        except:
            return pd.Series([0] * len(df), index=df.index)
    
    @staticmethod
    def _calculate_vpt(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        try:
            price_change_pct = df['close'].pct_change()
            vpt = (price_change_pct * df['volume']).cumsum()
            return vpt.fillna(0)
        except:
            return pd.Series([0] * len(df), index=df.index)
    
    @staticmethod
    def _calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            return vwap.fillna(df['close'])
        except:
            return df['close']
    
    @staticmethod
    def _calculate_ease_of_movement(df: pd.DataFrame) -> pd.Series:
        """Calculate Ease of Movement"""
        try:
            distance_moved = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
            box_height = df['volume'] / (df['high'] - df['low'])
            box_height = box_height.replace([np.inf, -np.inf], 0)
            
            emv = distance_moved / box_height
            return emv.rolling(window=14).mean().fillna(0)
        except:
            return pd.Series([0] * len(df), index=df.index)
    
    @staticmethod
    def _calculate_force_index(df: pd.DataFrame) -> pd.Series:
        """Calculate Force Index"""
        try:
            price_change = df['close'] - df['close'].shift(1)
            force_index = price_change * df['volume']
            return force_index.fillna(0)
        except:
            return pd.Series([0] * len(df), index=df.index)
    
    @staticmethod
    def _calculate_klinger_oscillator(df: pd.DataFrame) -> pd.Series:
        """Calculate Klinger Oscillator"""
        try:
            hlc3 = (df['high'] + df['low'] + df['close']) / 3
            hlc3_prev = hlc3.shift(1)
            
            trend = np.where(hlc3 > hlc3_prev, 1, -1)
            
            dm = df['high'] - df['low']
            cm = np.where(trend == trend.shift(1), 
                         dm + (dm.shift(1) if len(dm) > 1 else 0), 
                         dm)
            
            vf = df['volume'] * trend * np.abs(2 * ((dm / cm) - 1)) * 100
            vf = pd.Series(vf, index=df.index).fillna(0)
            
            klinger = talib.EMA(vf.values, timeperiod=34) - talib.EMA(vf.values, timeperiod=55)
            return pd.Series(klinger, index=df.index).fillna(0)
        except:
            return pd.Series([0] * len(df), index=df.index)


class VolumeAnomalyDetector:
    """Detects volume anomalies and unusual patterns"""
    
    def __init__(self, data_processor: RealTimeDataProcessor):
        self.data_processor = data_processor
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
        # Anomaly detection parameters
        self.spike_threshold = 3.0  # Standard deviations
        self.drought_threshold = 0.3  # Fraction of normal volume
        self.pattern_window = 20
        
    def detect_anomalies(self, symbol: str, timeframes: List[str] = None) -> List[VolumeAnomaly]:
        """Detect volume anomalies across timeframes"""
        if timeframes is None:
            timeframes = ['1min', '5min', '15min']
        
        anomalies = []
        
        for timeframe in timeframes:
            try:
                timeframe_anomalies = self._detect_timeframe_anomalies(symbol, timeframe)
                anomalies.extend(timeframe_anomalies)
            except Exception as e:
                logger.error(f"Error detecting anomalies for {symbol} on {timeframe}: {str(e)}")
                continue
        
        return anomalies
    
    def _detect_timeframe_anomalies(self, symbol: str, timeframe: str) -> List[VolumeAnomaly]:
        """Detect anomalies for specific timeframe"""
        anomalies = []
        
        try:
            df = self.data_processor.get_candle_data(symbol, timeframe, 100)
            if df.empty or len(df) < 20:
                return anomalies
            
            # Calculate volume indicators
            df = VolumeIndicators.calculate_volume_indicators(df)
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            volume_std = df['volume'].rolling(20).std().iloc[-1]
            
            # Detect volume spikes
            if current_volume > avg_volume + (self.spike_threshold * volume_std):
                severity = min((current_volume - avg_volume) / (self.spike_threshold * volume_std), 1.0)
                
                anomalies.append(VolumeAnomaly(
                    symbol=symbol,
                    anomaly_type='spike',
                    severity=severity,
                    current_volume=int(current_volume),
                    expected_volume=int(avg_volume),
                    volume_ratio=current_volume / avg_volume,
                    timestamp=datetime.now(),
                    description=f"Volume spike: {current_volume/avg_volume:.1f}x normal volume",
                    supporting_data={
                        'timeframe': timeframe,
                        'std_deviations': (current_volume - avg_volume) / volume_std,
                        'price_change': df['close'].iloc[-1] - df['close'].iloc[-2],
                        'price_change_percent': ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                    }
                ))
            
            # Detect volume droughts
            elif current_volume < avg_volume * self.drought_threshold:
                severity = 1.0 - (current_volume / (avg_volume * self.drought_threshold))
                
                anomalies.append(VolumeAnomaly(
                    symbol=symbol,
                    anomaly_type='drought',
                    severity=severity,
                    current_volume=int(current_volume),
                    expected_volume=int(avg_volume),
                    volume_ratio=current_volume / avg_volume,
                    timestamp=datetime.now(),
                    description=f"Volume drought: {current_volume/avg_volume:.1f}x normal volume",
                    supporting_data={
                        'timeframe': timeframe,
                        'price_volatility': df['close'].rolling(5).std().iloc[-1]
                    }
                ))
            
            # Detect unusual volume patterns using isolation forest
            if len(df) >= 50:
                volume_features = self._extract_volume_features(df)
                if len(volume_features) > 0:
                    try:
                        # Fit isolation forest on historical data
                        historical_features = volume_features[:-5]  # Exclude recent data
                        self.isolation_forest.fit(historical_features)
                        
                        # Check recent patterns
                        recent_features = volume_features[-5:]
                        anomaly_scores = self.isolation_forest.decision_function(recent_features)
                        outliers = self.isolation_forest.predict(recent_features)
                        
                        # Find the most anomalous recent pattern
                        if any(outliers == -1):
                            most_anomalous_idx = np.argmin(anomaly_scores)
                            if outliers[most_anomalous_idx] == -1:
                                severity = min(abs(anomaly_scores[most_anomalous_idx]) / 0.5, 1.0)
                                
                                anomalies.append(VolumeAnomaly(
                                    symbol=symbol,
                                    anomaly_type='unusual_pattern',
                                    severity=severity,
                                    current_volume=int(current_volume),
                                    expected_volume=int(avg_volume),
                                    volume_ratio=current_volume / avg_volume,
                                    timestamp=datetime.now(),
                                    description="Unusual volume pattern detected",
                                    supporting_data={
                                        'timeframe': timeframe,
                                        'anomaly_score': anomaly_scores[most_anomalous_idx],
                                        'pattern_features': recent_features[most_anomalous_idx].tolist()
                                    }
                                ))
                    except Exception as e:
                        logger.debug(f"Isolation forest failed: {str(e)}")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting timeframe anomalies: {str(e)}")
            return []
    
    def _extract_volume_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract volume features for anomaly detection"""
        try:
            features = []
            window_size = 5
            
            for i in range(window_size, len(df)):
                window_data = df.iloc[i-window_size:i]
                
                # Volume statistics
                vol_mean = window_data['volume'].mean()
                vol_std = window_data['volume'].std()
                vol_trend = np.polyfit(range(window_size), window_data['volume'], 1)[0]
                
                # Volume ratios
                vol_ratio = window_data['volume'].iloc[-1] / vol_mean if vol_mean > 0 else 1
                
                # Price-volume relationship
                price_change = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
                volume_change = (window_data['volume'].iloc[-1] - window_data['volume'].iloc[0]) / window_data['volume'].iloc[0] if window_data['volume'].iloc[0] > 0 else 0
                
                # OBV trend
                obv_change = (window_data['obv'].iloc[-1] - window_data['obv'].iloc[0]) / abs(window_data['obv'].iloc[0]) if window_data['obv'].iloc[0] != 0 else 0
                
                feature_vector = [
                    vol_ratio,
                    vol_trend,
                    price_change,
                    volume_change,
                    obv_change,
                    vol_std / vol_mean if vol_mean > 0 else 0
                ]
                
                features.append(feature_vector)
            
            return np.array(features) if features else np.array([])
            
        except Exception as e:
            logger.error(f"Error extracting volume features: {str(e)}")
            return np.array([])


class VolumeProfileAnalyzer:
    """Analyzes volume distribution across price levels"""
    
    def __init__(self, data_processor: RealTimeDataProcessor):
        self.data_processor = data_processor
        
    def calculate_volume_profile(self, symbol: str, timeframe: str = "5min", lookback_periods: int = 100) -> Optional[VolumeProfile]:
        """Calculate volume profile for a symbol"""
        try:
            df = self.data_processor.get_candle_data(symbol, timeframe, lookback_periods)
            if df.empty or len(df) < 20:
                return None
            
            # Create price levels (bins)
            price_min = df['low'].min()
            price_max = df['high'].max()
            num_bins = min(50, len(df) // 2)  # Adaptive number of bins
            
            price_levels = np.linspace(price_min, price_max, num_bins)
            volume_at_price = np.zeros(num_bins - 1)
            
            # Distribute volume across price levels for each candle
            for _, row in df.iterrows():
                # Simple distribution: assign volume proportionally across the candle's range
                candle_low = row['low']
                candle_high = row['high']
                candle_volume = row['volume']
                
                # Find relevant price level indices
                start_idx = np.searchsorted(price_levels, candle_low)
                end_idx = np.searchsorted(price_levels, candle_high)
                
                if start_idx == end_idx and start_idx < len(volume_at_price):
                    volume_at_price[start_idx] += candle_volume
                else:
                    # Distribute volume across multiple price levels
                    relevant_levels = max(1, end_idx - start_idx)
                    volume_per_level = candle_volume / relevant_levels
                    
                    for idx in range(start_idx, min(end_idx, len(volume_at_price))):
                        volume_at_price[idx] += volume_per_level
            
            # Calculate key levels
            poc_index = np.argmax(volume_at_price)
            poc = (price_levels[poc_index] + price_levels[poc_index + 1]) / 2
            
            # Calculate Value Area (70% of volume)
            total_volume = np.sum(volume_at_price)
            target_volume = total_volume * 0.7
            
            # Find value area by expanding from POC
            va_indices = [poc_index]
            va_volume = volume_at_price[poc_index]
            
            while va_volume < target_volume and (min(va_indices) > 0 or max(va_indices) < len(volume_at_price) - 1):
                # Check adjacent levels
                left_volume = volume_at_price[min(va_indices) - 1] if min(va_indices) > 0 else 0
                right_volume = volume_at_price[max(va_indices) + 1] if max(va_indices) < len(volume_at_price) - 1 else 0
                
                if left_volume >= right_volume and min(va_indices) > 0:
                    va_indices.append(min(va_indices) - 1)
                    va_volume += left_volume
                elif right_volume > 0 and max(va_indices) < len(volume_at_price) - 1:
                    va_indices.append(max(va_indices) + 1)
                    va_volume += right_volume
                else:
                    break
            
            va_indices.sort()
            vah = (price_levels[max(va_indices)] + price_levels[max(va_indices) + 1]) / 2
            val = (price_levels[min(va_indices)] + price_levels[min(va_indices) + 1]) / 2
            
            return VolumeProfile(
                symbol=symbol,
                price_levels=price_levels[:-1].tolist(),  # Remove last level (upper bound)
                volume_at_price=volume_at_price.tolist(),
                poc=poc,
                vah=vah,
                val=val,
                value_area_volume_percent=va_volume / total_volume * 100,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating volume profile for {symbol}: {str(e)}")
            return None


class VolumeSignalGenerator:
    """Generates trading signals based on volume analysis"""
    
    def __init__(self, data_processor: RealTimeDataProcessor):
        self.data_processor = data_processor
        self.anomaly_detector = VolumeAnomalyDetector(data_processor)
        self.profile_analyzer = VolumeProfileAnalyzer(data_processor)
        
    def generate_volume_signals(self, symbol: str, timeframes: List[str] = None) -> List[VolumeSignal]:
        """Generate comprehensive volume-based signals"""
        if timeframes is None:
            timeframes = ['5min', '15min']
        
        signals = []
        
        for timeframe in timeframes:
            try:
                timeframe_signals = self._analyze_timeframe_volume(symbol, timeframe)
                signals.extend(timeframe_signals)
            except Exception as e:
                logger.error(f"Error generating volume signals for {symbol} on {timeframe}: {str(e)}")
                continue
        
        return signals
    
    def _analyze_timeframe_volume(self, symbol: str, timeframe: str) -> List[VolumeSignal]:
        """Analyze volume for specific timeframe"""
        signals = []
        
        try:
            df = self.data_processor.get_candle_data(symbol, timeframe, 50)
            if df.empty or len(df) < 20:
                return signals
            
            # Calculate volume indicators
            df = VolumeIndicators.calculate_volume_indicators(df)
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            # Accumulation/Distribution signals
            ad_signal = self._analyze_accumulation_distribution(df, latest)
            if ad_signal:
                signals.append(ad_signal)
            
            # Volume breakout signals
            breakout_signal = self._analyze_volume_breakout(df, latest, symbol)
            if breakout_signal:
                signals.append(breakout_signal)
            
            # Climax volume signals
            climax_signal = self._analyze_volume_climax(df, latest, symbol)
            if climax_signal:
                signals.append(climax_signal)
            
            # Money flow signals
            money_flow_signal = self._analyze_money_flow(df, latest, symbol)
            if money_flow_signal:
                signals.append(money_flow_signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe volume: {str(e)}")
            return []
    
    def _analyze_accumulation_distribution(self, df: pd.DataFrame, latest: pd.Series) -> Optional[VolumeSignal]:
        """Analyze accumulation/distribution patterns"""
        try:
            if len(df) < 20:
                return None
            
            # Compare OBV with price movement
            obv_trend = latest['obv'] - df['obv'].iloc[-10]
            price_trend = latest['close'] - df['close'].iloc[-10]
            
            # Compare A/D line with price
            ad_trend = latest['ad_line'] - df['ad_line'].iloc[-10]
            
            # Chaikin Money Flow analysis
            cmf = latest['cmf']
            
            reasoning = []
            signal_strength = 0
            
            # OBV divergence analysis
            if price_trend > 0 and obv_trend > 0:
                signal_strength += 0.3
                reasoning.append("OBV confirms price uptrend")
                signal_type = "accumulation"
            elif price_trend < 0 and obv_trend < 0:
                signal_strength += 0.3
                reasoning.append("OBV confirms price downtrend")
                signal_type = "distribution"
            elif price_trend > 0 and obv_trend < 0:
                signal_strength += 0.5
                reasoning.append("Bearish OBV divergence")
                signal_type = "distribution"
            elif price_trend < 0 and obv_trend > 0:
                signal_strength += 0.5
                reasoning.append("Bullish OBV divergence")
                signal_type = "accumulation"
            else:
                signal_type = "accumulation" if obv_trend > 0 else "distribution"
            
            # A/D line confirmation
            if (signal_type == "accumulation" and ad_trend > 0) or (signal_type == "distribution" and ad_trend < 0):
                signal_strength += 0.2
                reasoning.append("A/D line confirms signal")
            
            # CMF analysis
            if cmf > 0.2:
                signal_strength += 0.3 if signal_type == "accumulation" else -0.2
                reasoning.append(f"Strong money flow inflow (CMF: {cmf:.2f})")
            elif cmf < -0.2:
                signal_strength += 0.3 if signal_type == "distribution" else -0.2
                reasoning.append(f"Strong money flow outflow (CMF: {cmf:.2f})")
            
            if signal_strength > 0.3:
                return VolumeSignal(
                    symbol=latest.name if hasattr(latest, 'name') else "Unknown",
                    signal_type=signal_type,
                    strength=min(signal_strength, 1.0),
                    volume_metrics={
                        'obv_trend': obv_trend,
                        'ad_trend': ad_trend,
                        'cmf': cmf,
                        'volume_ratio': latest.get('volume_ratio', 1.0)
                    },
                    price_volume_relationship="confirmative" if abs(obv_trend * price_trend) > 0 else "divergent",
                    timestamp=datetime.now(),
                    reasoning=reasoning
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing accumulation/distribution: {str(e)}")
            return None
    
    def _analyze_volume_breakout(self, df: pd.DataFrame, latest: pd.Series, symbol: str) -> Optional[VolumeSignal]:
        """Analyze volume breakout patterns"""
        try:
            volume_ratio = latest.get('volume_ratio', 1.0)
            price_change_percent = ((latest['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            
            # High volume with significant price movement
            if volume_ratio > 2.0 and abs(price_change_percent) > 1.0:
                strength = min(volume_ratio / 3.0, 1.0)
                
                return VolumeSignal(
                    symbol=symbol,
                    signal_type="breakout_volume",
                    strength=strength,
                    volume_metrics={
                        'volume_ratio': volume_ratio,
                        'price_change_percent': price_change_percent,
                        'volume': latest['volume']
                    },
                    price_volume_relationship="supportive",
                    timestamp=datetime.now(),
                    reasoning=[
                        f"High volume breakout: {volume_ratio:.1f}x normal volume",
                        f"Price change: {price_change_percent:.1f}%"
                    ]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing volume breakout: {str(e)}")
            return None
    
    def _analyze_volume_climax(self, df: pd.DataFrame, latest: pd.Series, symbol: str) -> Optional[VolumeSignal]:
        """Analyze volume climax patterns (exhaustion)"""
        try:
            if len(df) < 10:
                return None
            
            # Look for volume spikes with price exhaustion
            volume_ratio = latest.get('volume_ratio', 1.0)
            recent_volumes = df['volume_ratio'].tail(5)
            recent_prices = df['close'].tail(5)
            
            # Check for volume climax (high volume with price reversal)
            if volume_ratio > 2.5:
                # Check if price is showing signs of exhaustion
                price_momentum = talib.MOM(recent_prices.values, timeperiod=3)[-1] if len(recent_prices) >= 4 else 0
                price_range = (df['high'].tail(5).max() - df['low'].tail(5).min()) / df['close'].tail(5).mean()
                
                if abs(price_momentum) < 0.5 and price_range > 0.03:  # Low momentum, high volatility
                    strength = min(volume_ratio / 4.0, 1.0)
                    
                    return VolumeSignal(
                        symbol=symbol,
                        signal_type="climax",
                        strength=strength,
                        volume_metrics={
                            'volume_ratio': volume_ratio,
                            'price_momentum': price_momentum,
                            'price_range': price_range
                        },
                        price_volume_relationship="exhaustion",
                        timestamp=datetime.now(),
                        reasoning=[
                            f"Volume climax detected: {volume_ratio:.1f}x normal volume",
                            f"Price momentum weakening: {price_momentum:.2f}",
                            "Potential reversal signal"
                        ]
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing volume climax: {str(e)}")
            return None
    
    def _analyze_money_flow(self, df: pd.DataFrame, latest: pd.Series, symbol: str) -> Optional[VolumeSignal]:
        """Analyze money flow indicators"""
        try:
            mfi = latest.get('mfi', 50)
            force_index = latest.get('force_index', 0)
            
            reasoning = []
            signal_strength = 0
            signal_type = "accumulation"
            
            # Money Flow Index analysis
            if mfi > 80:
                signal_strength += 0.4
                signal_type = "distribution"
                reasoning.append(f"MFI overbought: {mfi:.1f}")
            elif mfi < 20:
                signal_strength += 0.4
                signal_type = "accumulation"
                reasoning.append(f"MFI oversold: {mfi:.1f}")
            
            # Force Index analysis
            if len(df) > 1:
                prev_force = df['force_index'].iloc[-2]
                if force_index > 0 and prev_force <= 0:
                    signal_strength += 0.3
                    signal_type = "accumulation"
                    reasoning.append("Force Index turning positive")
                elif force_index < 0 and prev_force >= 0:
                    signal_strength += 0.3
                    signal_type = "distribution"
                    reasoning.append("Force Index turning negative")
            
            if signal_strength > 0.3:
                return VolumeSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=min(signal_strength, 1.0),
                    volume_metrics={
                        'mfi': mfi,
                        'force_index': force_index,
                        'volume_ratio': latest.get('volume_ratio', 1.0)
                    },
                    price_volume_relationship="money_flow_based",
                    timestamp=datetime.now(),
                    reasoning=reasoning
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing money flow: {str(e)}")
            return None