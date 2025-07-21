"""Module for generating trading signals with confidence levels."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from loguru import logger
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalGenerator:
    """Generate trading signals based on technical indicators."""

    def __init__(self):
        """Initialize the signal generator."""
        self.signal_weights = {
            "RSI": 0.25,
            "MACD": 0.30,
            "BB": 0.20,
            "MA_CROSS": 0.25
        }
        logger.info("SignalGenerator initialized")

    def generate_signals(
        self, 
        data: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trading signals from indicators.
        
        Args:
            data: Price data
            indicators: Calculated indicator values
            
        Returns:
            Trading signals with confidence levels
        """
        try:
            signals = {}
            
            # RSI signals
            if "RSI" in indicators:
                signals["rsi_signal"] = self._generate_rsi_signal(indicators["RSI"])
            
            # MACD signals
            if "MACD" in indicators:
                signals["macd_signal"] = self._generate_macd_signal(indicators["MACD"])
            
            # Bollinger Bands signals
            if "BB" in indicators:
                signals["bb_signal"] = self._generate_bb_signal(
                    indicators["BB"], data["close"].iloc[-1]
                )
            
            # Moving Average signals
            ma_signal = self._generate_ma_signal(indicators)
            if ma_signal:
                signals["ma_signal"] = ma_signal
            
            # Combine signals
            combined_signal = self._combine_signals(signals)
            
            # Add volume confirmation
            volume_confirmation = self._check_volume_confirmation(data)
            
            # Calculate final confidence
            if volume_confirmation:
                combined_signal["confidence"] *= 1.1  # 10% boost for volume confirmation
                combined_signal["confidence"] = min(combined_signal["confidence"], 1.0)
            
            return {
                "individual_signals": signals,
                "combined_signal": combined_signal,
                "volume_confirmation": volume_confirmation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise

    def get_current_signal(
        self, 
        data: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get the current trading signal with confidence.
        
        Args:
            data: Price data
            indicators: Calculated indicator values
            
        Returns:
            Current signal with confidence level
        """
        try:
            signals = self.generate_signals(data, indicators)
            combined = signals["combined_signal"]
            
            return {
                "signal": combined["signal"],
                "confidence": combined["confidence"],
                "strength": combined["strength"],
                "reasons": combined["reasons"]
            }
            
        except Exception as e:
            logger.error(f"Error getting current signal: {str(e)}")
            return {
                "signal": SignalType.NEUTRAL.value,
                "confidence": 0.0,
                "strength": 0.0,
                "reasons": ["Error in signal generation"]
            }

    def _generate_rsi_signal(self, rsi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal from RSI indicator."""
        current_rsi = rsi_data.get("current")
        if current_rsi is None:
            return {"signal": SignalType.NEUTRAL, "confidence": 0.0}
        
        # Check RSI values history for divergence
        rsi_values = rsi_data.get("values", [])
        divergence = self._check_rsi_divergence(rsi_values) if len(rsi_values) > 20 else None
        
        if current_rsi < 30:
            signal = SignalType.BUY if current_rsi < 25 else SignalType.BUY
            confidence = 0.8 if current_rsi < 25 else 0.6
            if divergence == "bullish":
                confidence += 0.1
        elif current_rsi > 70:
            signal = SignalType.SELL if current_rsi > 75 else SignalType.SELL
            confidence = 0.8 if current_rsi > 75 else 0.6
            if divergence == "bearish":
                confidence += 0.1
        else:
            signal = SignalType.NEUTRAL
            confidence = 0.3
        
        return {
            "signal": signal,
            "confidence": min(confidence, 1.0),
            "value": current_rsi,
            "divergence": divergence
        }

    def _generate_macd_signal(self, macd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal from MACD indicator."""
        crossover = macd_data.get("crossover")
        histogram = macd_data.get("current_histogram")
        
        if crossover == "bullish":
            signal = SignalType.BUY
            confidence = 0.8
        elif crossover == "bearish":
            signal = SignalType.SELL
            confidence = 0.8
        elif histogram is not None:
            if histogram > 0:
                signal = SignalType.BUY if histogram > 0.5 else SignalType.NEUTRAL
                confidence = 0.5 if histogram > 0.5 else 0.3
            else:
                signal = SignalType.SELL if histogram < -0.5 else SignalType.NEUTRAL
                confidence = 0.5 if histogram < -0.5 else 0.3
        else:
            signal = SignalType.NEUTRAL
            confidence = 0.0
        
        return {
            "signal": signal,
            "confidence": confidence,
            "crossover": crossover,
            "histogram": histogram
        }

    def _generate_bb_signal(
        self, 
        bb_data: Dict[str, Any], 
        current_price: float
    ) -> Dict[str, Any]:
        """Generate signal from Bollinger Bands."""
        position = bb_data.get("position")
        band_width = bb_data.get("band_width")
        
        if position == "below_lower":
            signal = SignalType.BUY
            confidence = 0.7
        elif position == "above_upper":
            signal = SignalType.SELL
            confidence = 0.7
        else:
            signal = SignalType.NEUTRAL
            confidence = 0.3
        
        # Adjust confidence based on band width (volatility)
        if band_width and bb_data.get("current_middle"):
            volatility_ratio = band_width / bb_data["current_middle"]
            if volatility_ratio > 0.04:  # High volatility
                confidence *= 0.8
            elif volatility_ratio < 0.02:  # Low volatility
                confidence *= 1.2
        
        return {
            "signal": signal,
            "confidence": min(confidence, 1.0),
            "position": position,
            "band_width": band_width
        }

    def _generate_ma_signal(self, indicators: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate signal from moving averages."""
        sma_data = indicators.get("SMA", {})
        ema_data = indicators.get("EMA", {})
        
        # Check for golden/death cross
        if sma_data and "golden_cross" in sma_data:
            if sma_data["golden_cross"]:
                return {
                    "signal": SignalType.STRONG_BUY,
                    "confidence": 0.9,
                    "type": "golden_cross"
                }
            elif sma_data["death_cross"]:
                return {
                    "signal": SignalType.STRONG_SELL,
                    "confidence": 0.9,
                    "type": "death_cross"
                }
        
        # Check EMA trends
        if ema_data:
            ema_12 = ema_data.get("EMA_12", {}).get("current")
            ema_26 = ema_data.get("EMA_26", {}).get("current")
            
            if ema_12 and ema_26:
                if ema_12 > ema_26:
                    return {
                        "signal": SignalType.BUY,
                        "confidence": 0.6,
                        "type": "ema_bullish"
                    }
                else:
                    return {
                        "signal": SignalType.SELL,
                        "confidence": 0.6,
                        "type": "ema_bearish"
                    }
        
        return None

    def _combine_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple signals into a final signal with confidence."""
        if not signals:
            return {
                "signal": SignalType.NEUTRAL.value,
                "confidence": 0.0,
                "strength": 0.0,
                "reasons": ["No signals available"]
            }
        
        # Calculate weighted scores
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        reasons = []
        
        signal_mapping = {
            "rsi_signal": "RSI",
            "macd_signal": "MACD",
            "bb_signal": "BB",
            "ma_signal": "MA_CROSS"
        }
        
        for signal_key, signal_data in signals.items():
            if signal_key in signal_mapping:
                weight = self.signal_weights.get(signal_mapping[signal_key], 0.25)
                signal_type = signal_data["signal"]
                confidence = signal_data["confidence"]
                
                if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    buy_score += weight * confidence
                    if confidence > 0.5:
                        reasons.append(f"{signal_mapping[signal_key]} bullish")
                elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    sell_score += weight * confidence
                    if confidence > 0.5:
                        reasons.append(f"{signal_mapping[signal_key]} bearish")
                
                total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine final signal
        strength = abs(buy_score - sell_score)
        
        if buy_score > sell_score:
            if strength > 0.7:
                final_signal = SignalType.STRONG_BUY
            elif strength > 0.3:
                final_signal = SignalType.BUY
            else:
                final_signal = SignalType.NEUTRAL
        elif sell_score > buy_score:
            if strength > 0.7:
                final_signal = SignalType.STRONG_SELL
            elif strength > 0.3:
                final_signal = SignalType.SELL
            else:
                final_signal = SignalType.NEUTRAL
        else:
            final_signal = SignalType.NEUTRAL
        
        confidence = strength * 0.8 + 0.2  # Base confidence of 20%
        
        return {
            "signal": final_signal.value,
            "confidence": min(confidence, 1.0),
            "strength": strength,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "reasons": reasons if reasons else ["Mixed signals"]
        }

    def _check_volume_confirmation(self, data: pd.DataFrame) -> bool:
        """Check if volume confirms the price movement."""
        if len(data) < 10:
            return False
        
        try:
            # Calculate average volume
            avg_volume = data["volume"].iloc[-20:-1].mean()
            current_volume = data["volume"].iloc[-1]
            
            # Check if current volume is above average
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Price movement
            price_change = (data["close"].iloc[-1] - data["close"].iloc[-2]) / data["close"].iloc[-2]
            
            # Volume should confirm price movement
            if abs(price_change) > 0.01 and volume_ratio > 1.2:
                return True
                
            return False
            
        except Exception:
            return False

    def _check_rsi_divergence(self, rsi_values: np.ndarray) -> Optional[str]:
        """Check for RSI divergence patterns."""
        if len(rsi_values) < 20:
            return None
        
        try:
            # Get recent peaks and troughs
            recent_rsi = rsi_values[-20:]
            
            # Simple divergence check (can be enhanced)
            rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
            
            if rsi_trend > 0 and recent_rsi[-1] < 30:
                return "bullish"
            elif rsi_trend < 0 and recent_rsi[-1] > 70:
                return "bearish"
                
            return None
            
        except Exception:
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the signal generator."""
        return {
            "status": "active",
            "signal_weights": self.signal_weights,
            "supported_signals": [s.value for s in SignalType]
        }