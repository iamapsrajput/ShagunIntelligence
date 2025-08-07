import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength levels"""

    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class FusedSignal:
    """Represents a fused trading signal from multiple sources"""

    symbol: str
    action: str  # BUY or SELL
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    expected_return: float
    risk_score: float  # 0 to 1
    contributing_factors: dict[str, float]
    timestamp: datetime


class DecisionFusionEngine:
    """Fuses insights from multiple agents to create unified trading decisions"""

    def __init__(self):
        # Weight configuration for different analysis types
        self.default_weights = {
            "market": 0.3,
            "technical": 0.4,
            "sentiment": 0.2,
            "fundamental": 0.1,
        }

        self.weights = self.default_weights.copy()

        # Confidence thresholds
        self.min_agreement_threshold = 0.6  # 60% agreement needed
        self.strong_signal_threshold = 0.8  # 80% for strong signals

        # Risk adjustment factors
        self.risk_multipliers = {
            "high_volatility": 0.7,
            "low_liquidity": 0.8,
            "correlation_risk": 0.85,
            "news_uncertainty": 0.9,
        }

        # Historical performance tracking
        self.fusion_history = []
        self.performance_metrics = {
            "successful_fusions": 0,
            "failed_fusions": 0,
            "average_confidence": 0.0,
        }

    def fuse_insights(
        self,
        symbol: str,
        market_analysis: dict[str, Any],
        technical_analysis: dict[str, Any],
        sentiment_analysis: dict[str, Any],
        fundamental_analysis: dict[str, Any] | None = None,
    ) -> Any | None:
        """Fuse insights from multiple agents into a unified decision"""
        try:
            # Extract signals from each analysis
            market_signal = self._extract_market_signal(market_analysis)
            technical_signal = self._extract_technical_signal(technical_analysis)
            sentiment_signal = self._extract_sentiment_signal(sentiment_analysis)

            signals = {
                "market": market_signal,
                "technical": technical_signal,
                "sentiment": sentiment_signal,
            }

            if fundamental_analysis:
                fundamental_signal = self._extract_fundamental_signal(
                    fundamental_analysis
                )
                signals["fundamental"] = fundamental_signal

            # Check if we have sufficient data
            valid_signals = {k: v for k, v in signals.items() if v is not None}
            if len(valid_signals) < 2:
                logger.warning(
                    f"Insufficient signals for {symbol}: only {len(valid_signals)} available"
                )
                return None

            # Calculate weighted consensus
            consensus = self._calculate_weighted_consensus(valid_signals)

            # Determine action and strength
            action, strength = self._determine_action(consensus)

            if action == "NEUTRAL":
                return None  # No trade signal

            # Calculate confidence
            confidence = self._calculate_confidence(valid_signals, consensus)

            # Estimate expected return
            expected_return = self._estimate_expected_return(
                valid_signals, action, strength
            )

            # Calculate risk score
            risk_score = self._calculate_risk_score(
                symbol, valid_signals, market_analysis
            )

            # Create fused signal
            fused_signal = FusedSignal(
                symbol=symbol,
                action=action,
                strength=strength,
                confidence=confidence,
                expected_return=expected_return,
                risk_score=risk_score,
                contributing_factors=self._get_contributing_factors(valid_signals),
                timestamp=datetime.now(),
            )

            # Apply risk adjustments
            adjusted_signal = self._apply_risk_adjustments(
                fused_signal, market_analysis
            )

            # Record fusion
            self._record_fusion(adjusted_signal)

            # Convert to TradingOpportunity
            from .agent import TradingOpportunity

            opportunity = TradingOpportunity(
                id=f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                symbol=symbol,
                action=adjusted_signal.action,
                confidence=adjusted_signal.confidence,
                expected_return=adjusted_signal.expected_return,
                risk_score=adjusted_signal.risk_score,
                priority=0.0,  # Will be set by ranking
                source_agents=list(valid_signals.keys()),
                analysis={
                    "fused_signal": adjusted_signal,
                    "raw_signals": valid_signals,
                },
                timestamp=adjusted_signal.timestamp,
            )

            return opportunity

        except Exception as e:
            logger.error(f"Error fusing insights for {symbol}: {str(e)}")
            return None

    def _extract_market_signal(self, analysis: dict[str, Any]) -> dict[str, Any] | None:
        """Extract trading signal from market analysis"""
        if not analysis:
            return None

        # Look for trend, momentum, and market structure
        trend = analysis.get("trend", {})
        momentum = analysis.get("momentum", {})

        signal_value = 0
        confidence = 0

        # Process trend
        if trend.get("direction") == "up":
            signal_value += 0.5
            confidence += trend.get("strength", 0.5)
        elif trend.get("direction") == "down":
            signal_value -= 0.5
            confidence += trend.get("strength", 0.5)

        # Process momentum
        if momentum.get("rsi", 50) > 70:
            signal_value -= 0.3  # Overbought
            confidence += 0.3
        elif momentum.get("rsi", 50) < 30:
            signal_value += 0.3  # Oversold
            confidence += 0.3

        return {
            "signal": signal_value,
            "confidence": min(confidence, 1.0),
            "factors": {"trend": trend, "momentum": momentum},
        }

    def _extract_technical_signal(
        self, analysis: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract trading signal from technical analysis"""
        if not analysis:
            return None

        signal_value = 0
        confidence = 0
        factors = {}

        # Process technical indicators
        indicators = analysis.get("indicators", {})

        # Moving averages
        if "ma_crossover" in indicators:
            if indicators["ma_crossover"] == "bullish":
                signal_value += 0.4
                confidence += 0.4
            elif indicators["ma_crossover"] == "bearish":
                signal_value -= 0.4
                confidence += 0.4
            factors["ma_crossover"] = indicators["ma_crossover"]

        # MACD
        if "macd" in indicators:
            macd = indicators["macd"]
            if macd.get("signal") == "buy":
                signal_value += 0.3
                confidence += 0.3
            elif macd.get("signal") == "sell":
                signal_value -= 0.3
                confidence += 0.3
            factors["macd"] = macd

        # Bollinger Bands
        if "bollinger" in indicators:
            bb = indicators["bollinger"]
            if bb.get("position") == "below_lower":
                signal_value += 0.3  # Potential bounce
                confidence += 0.2
            elif bb.get("position") == "above_upper":
                signal_value -= 0.3  # Potential reversal
                confidence += 0.2
            factors["bollinger"] = bb

        # Support/Resistance
        if "support_resistance" in analysis:
            sr = analysis["support_resistance"]
            if sr.get("near_support"):
                signal_value += 0.2
                confidence += 0.3
            elif sr.get("near_resistance"):
                signal_value -= 0.2
                confidence += 0.3
            factors["support_resistance"] = sr

        return {
            "signal": np.clip(signal_value, -1, 1),
            "confidence": min(confidence, 1.0),
            "factors": factors,
        }

    def _extract_sentiment_signal(
        self, analysis: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract trading signal from sentiment analysis"""
        if not analysis:
            return None

        sentiment_score = analysis.get("sentiment_score", 0)
        confidence = analysis.get("confidence", 0.5)

        # Convert sentiment to signal (-1 to 1)
        signal_value = np.clip(sentiment_score, -1, 1)

        # Adjust confidence based on data sources
        sources = analysis.get("sources", [])
        if len(sources) > 3:
            confidence *= 1.2
        elif len(sources) < 2:
            confidence *= 0.8

        return {
            "signal": signal_value,
            "confidence": min(confidence, 1.0),
            "factors": {
                "sentiment_score": sentiment_score,
                "sources": sources,
                "keywords": analysis.get("keywords", []),
            },
        }

    def _extract_fundamental_signal(
        self, analysis: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract trading signal from fundamental analysis"""
        if not analysis:
            return None

        # This would process fundamental data like P/E ratios, earnings, etc.
        # For now, simplified implementation
        valuation = analysis.get("valuation", {})

        signal_value = 0
        if valuation.get("undervalued"):
            signal_value = 0.6
        elif valuation.get("overvalued"):
            signal_value = -0.6

        return {
            "signal": signal_value,
            "confidence": analysis.get("confidence", 0.6),
            "factors": valuation,
        }

    def _calculate_weighted_consensus(
        self, signals: dict[str, dict[str, Any]]
    ) -> float:
        """Calculate weighted consensus from multiple signals"""
        weighted_sum = 0
        weight_sum = 0

        for signal_type, signal_data in signals.items():
            weight = self.weights.get(signal_type, 0.25)
            signal_value = signal_data["signal"]
            confidence = signal_data["confidence"]

            # Weight by both configured weight and confidence
            effective_weight = weight * confidence
            weighted_sum += signal_value * effective_weight
            weight_sum += effective_weight

        if weight_sum > 0:
            return weighted_sum / weight_sum

        return 0

    def _determine_action(self, consensus: float) -> tuple[str, float]:
        """Determine action and strength from consensus"""
        if consensus > 0.2:
            action = "BUY"
            strength = min(consensus, 1.0)
        elif consensus < -0.2:
            action = "SELL"
            strength = min(abs(consensus), 1.0)
        else:
            action = "NEUTRAL"
            strength = 0

        return action, strength

    def _calculate_confidence(
        self, signals: dict[str, dict[str, Any]], consensus: float
    ) -> float:
        """Calculate overall confidence in the decision"""
        # Factor 1: Average confidence of individual signals
        avg_confidence = np.mean([s["confidence"] for s in signals.values()])

        # Factor 2: Agreement between signals
        signal_values = [s["signal"] for s in signals.values()]

        # All signals should have same sign as consensus for high agreement
        agreement_scores = []
        for signal in signal_values:
            if consensus > 0 and signal > 0:
                agreement_scores.append(1)
            elif consensus < 0 and signal < 0:
                agreement_scores.append(1)
            elif abs(signal) < 0.2:  # Neutral signal
                agreement_scores.append(0.5)
            else:
                agreement_scores.append(0)

        agreement_factor = np.mean(agreement_scores)

        # Factor 3: Strength of consensus
        strength_factor = min(abs(consensus) * 2, 1.0)

        # Combine factors
        confidence = (
            avg_confidence * 0.4 + agreement_factor * 0.4 + strength_factor * 0.2
        )

        return min(confidence, 1.0)

    def _estimate_expected_return(
        self, signals: dict[str, dict[str, Any]], action: str, strength: float
    ) -> float:
        """Estimate expected return based on signals"""
        # Base expected return on signal strength
        base_return = strength * 0.02  # 2% max base return

        # Adjust based on technical factors
        technical = signals.get("technical", {})
        if technical:
            factors = technical.get("factors", {})

            # Boost return if near support (for buy) or resistance (for sell)
            sr = factors.get("support_resistance", {})
            if action == "BUY" and sr.get("near_support"):
                base_return *= 1.5
            elif action == "SELL" and sr.get("near_resistance"):
                base_return *= 1.5

        # Adjust based on sentiment
        sentiment = signals.get("sentiment", {})
        if sentiment and sentiment["confidence"] > 0.7:
            if (action == "BUY" and sentiment["signal"] > 0.5) or (
                action == "SELL" and sentiment["signal"] < -0.5
            ):
                base_return *= 1.2

        return min(base_return, 0.05)  # Cap at 5%

    def _calculate_risk_score(
        self,
        symbol: str,
        signals: dict[str, dict[str, Any]],
        market_analysis: dict[str, Any],
    ) -> float:
        """Calculate risk score for the opportunity"""
        risk_factors = []

        # Market volatility
        volatility = market_analysis.get("volatility", {}).get("current", 0.02)
        volatility_risk = min(volatility / 0.05, 1.0)  # Normalize to 0-1
        risk_factors.append(volatility_risk * 0.3)

        # Signal disagreement
        signal_values = [s["signal"] for s in signals.values()]
        signal_std = np.std(signal_values)
        disagreement_risk = min(signal_std, 1.0)
        risk_factors.append(disagreement_risk * 0.2)

        # Low confidence
        avg_confidence = np.mean([s["confidence"] for s in signals.values()])
        confidence_risk = 1 - avg_confidence
        risk_factors.append(confidence_risk * 0.2)

        # Technical risk (overbought/oversold)
        technical = signals.get("technical", {})
        if technical:
            factors = technical.get("factors", {})
            if "momentum" in factors:
                rsi = factors["momentum"].get("rsi", 50)
                if rsi > 70 or rsi < 30:
                    risk_factors.append(0.3)  # Extreme RSI risk

        # Sentiment uncertainty
        sentiment = signals.get("sentiment", {})
        if sentiment and sentiment["confidence"] < 0.5:
            risk_factors.append(0.2)

        return min(sum(risk_factors), 1.0)

    def _get_contributing_factors(
        self, signals: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Get contribution of each signal type to final decision"""
        contributions = {}
        total_weight = 0

        for signal_type, signal_data in signals.items():
            weight = self.weights.get(signal_type, 0.25)
            confidence = signal_data["confidence"]

            contribution = weight * confidence
            contributions[signal_type] = contribution
            total_weight += contribution

        # Normalize to percentages
        if total_weight > 0:
            for signal_type in contributions:
                contributions[signal_type] = contributions[signal_type] / total_weight

        return contributions

    def _apply_risk_adjustments(
        self, signal: FusedSignal, market_analysis: dict[str, Any]
    ) -> FusedSignal:
        """Apply risk-based adjustments to signal"""
        adjusted_confidence = signal.confidence

        # Apply risk multipliers
        market_conditions = market_analysis.get("conditions", {})

        if market_conditions.get("high_volatility"):
            adjusted_confidence *= self.risk_multipliers["high_volatility"]

        if market_conditions.get("low_liquidity"):
            adjusted_confidence *= self.risk_multipliers["low_liquidity"]

        if market_conditions.get("high_correlation"):
            adjusted_confidence *= self.risk_multipliers["correlation_risk"]

        # Create adjusted signal
        signal.confidence = adjusted_confidence

        return signal

    def _record_fusion(self, signal: FusedSignal) -> None:
        """Record fusion for performance tracking"""
        self.fusion_history.append(
            {
                "timestamp": signal.timestamp,
                "symbol": signal.symbol,
                "action": signal.action,
                "confidence": signal.confidence,
                "risk_score": signal.risk_score,
            }
        )

        # Keep only recent history
        if len(self.fusion_history) > 1000:
            self.fusion_history = self.fusion_history[-1000:]

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update fusion weights"""
        self.weights.update(new_weights)

        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total

    def get_weights(self) -> dict[str, float]:
        """Get current fusion weights"""
        return self.weights.copy()

    def get_fusion_statistics(self) -> dict[str, Any]:
        """Get statistics about fusion performance"""
        if not self.fusion_history:
            return {"message": "No fusion history available"}

        recent_fusions = self.fusion_history[-100:]  # Last 100 fusions

        return {
            "total_fusions": len(self.fusion_history),
            "recent_fusions": len(recent_fusions),
            "average_confidence": np.mean([f["confidence"] for f in recent_fusions]),
            "average_risk_score": np.mean([f["risk_score"] for f in recent_fusions]),
            "action_distribution": {
                "BUY": len([f for f in recent_fusions if f["action"] == "BUY"]),
                "SELL": len([f for f in recent_fusions if f["action"] == "SELL"]),
            },
            "current_weights": self.weights,
        }
