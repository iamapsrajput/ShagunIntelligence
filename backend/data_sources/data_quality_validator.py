from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from backend.data_sources.market.models import MarketData
from loguru import logger


class DataSourceType(str, Enum):
    """Data source types for quality tracking"""

    KITE_CONNECT = "kite_connect"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    YAHOO_FINANCE = "yahoo_finance"
    NSE_OFFICIAL = "nse_official"
    GLOBAL_DATAFEEDS = "global_datafeeds"
    EODHD = "eodhd"
    MARKETAUX = "marketaux"
    TWITTER = "twitter"
    GROK = "grok"
    NEWSAPI = "newsapi"


class QualityGrade(Enum):
    EXCELLENT = "Excellent"
    GOOD = "Good"
    ACCEPTABLE = "Acceptable"
    POOR = "Poor"
    FAILED = "Failed"


class QualityMetrics:
    def __init__(
        self,
        freshness_score: float = 0.0,
        accuracy_score: float = 0.0,
        completeness_score: float = 0.0,
        reliability_score: float = 0.0,
        overall_score: float = 0.0,
        grade: QualityGrade = QualityGrade.FAILED,
        anomalies: List[str] = None,
        timestamp: datetime = None,
    ):
        self.freshness_score = freshness_score
        self.accuracy_score = accuracy_score
        self.completeness_score = completeness_score
        self.reliability_score = reliability_score
        self.overall_score = overall_score
        self.grade = grade
        self.anomalies = anomalies or []
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "freshness_score": self.freshness_score,
            "accuracy_score": self.accuracy_score,
            "completeness_score": self.completeness_score,
            "reliability_score": self.reliability_score,
            "overall_score": self.overall_score,
            "grade": self.grade.value,
            "anomalies": self.anomalies,
            "timestamp": self.timestamp.isoformat(),
        }


class SourceReliabilityTracker:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.quality_history = defaultdict(lambda: deque(maxlen=window_size))
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.last_update = defaultdict(lambda: datetime.utcnow())

    def update(self, source: DataSourceType, quality_score: float, success: bool = True):
        self.quality_history[source].append(quality_score)
        self.last_update[source] = datetime.utcnow()

        if success:
            self.success_counts[source] += 1
        else:
            self.error_counts[source] += 1

    def get_reliability_score(self, source: DataSourceType) -> float:
        if source not in self.quality_history:
            return 0.5

        history = list(self.quality_history[source])
        if not history:
            return 0.5

        # Calculate weighted average with recent scores having more weight
        weights = np.linspace(0.5, 1.0, len(history))
        weighted_avg = np.average(history, weights=weights)

        # Factor in success rate
        total_requests = self.success_counts[source] + self.error_counts[source]
        if total_requests > 0:
            success_rate = self.success_counts[source] / total_requests
            reliability = 0.7 * weighted_avg + 0.3 * success_rate
        else:
            reliability = weighted_avg

        return min(max(reliability, 0.0), 1.0)


class DataQualityValidator:
    def __init__(
        self,
        freshness_threshold_seconds: int = 5,
        price_deviation_threshold: float = 0.02,
        volume_spike_threshold: float = 3.0,
        min_required_fields: List[str] = None,
    ):
        self.freshness_threshold = freshness_threshold_seconds
        self.price_deviation_threshold = price_deviation_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.min_required_fields = min_required_fields or ["symbol", "current_price", "volume", "timestamp"]

        self.reliability_tracker = SourceReliabilityTracker()
        self.price_history = defaultdict(lambda: deque(maxlen=100))
        self.volume_history = defaultdict(lambda: deque(maxlen=100))
        self.last_known_good = {}

        # Quality thresholds for grading
        self.grade_thresholds = {
            QualityGrade.EXCELLENT: 0.9,
            QualityGrade.GOOD: 0.75,
            QualityGrade.ACCEPTABLE: 0.6,
            QualityGrade.POOR: 0.4,
            QualityGrade.FAILED: 0.0,
        }

        # Alerting thresholds
        self.alert_callbacks = []
        self.alert_threshold = 0.6

    def validate_data(
        self, data: MarketData, source: DataSourceType, reference_data: Optional[Dict[str, MarketData]] = None
    ) -> QualityMetrics:
        """
        Validate market data and return quality metrics
        """
        anomalies = []

        # Calculate individual scores
        freshness_score = self._calculate_freshness_score(data)
        completeness_score = self._calculate_completeness_score(data)

        # Accuracy score requires reference data for comparison
        if reference_data:
            accuracy_score, accuracy_anomalies = self._calculate_accuracy_score(data, reference_data)
            anomalies.extend(accuracy_anomalies)
        else:
            accuracy_score = self._estimate_accuracy_score(data)

        # Check for anomalies
        price_anomalies = self._detect_price_anomalies(data)
        volume_anomalies = self._detect_volume_anomalies(data)
        anomalies.extend(price_anomalies + volume_anomalies)

        # Get historical reliability score
        reliability_score = self.reliability_tracker.get_reliability_score(source)

        # Calculate overall score (weighted average)
        weights = {"freshness": 0.3, "accuracy": 0.3, "completeness": 0.2, "reliability": 0.2}

        overall_score = (
            weights["freshness"] * freshness_score
            + weights["accuracy"] * accuracy_score
            + weights["completeness"] * completeness_score
            + weights["reliability"] * reliability_score
        )

        # Determine grade
        grade = self._determine_grade(overall_score)

        # Create metrics object
        metrics = QualityMetrics(
            freshness_score=freshness_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            reliability_score=reliability_score,
            overall_score=overall_score,
            grade=grade,
            anomalies=anomalies,
        )

        # Update reliability tracker
        self.reliability_tracker.update(source, overall_score, success=True)

        # Update history
        self._update_history(data)

        # Check for alerts
        if overall_score < self.alert_threshold:
            self._trigger_alerts(source, metrics)

        # Store as last known good if quality is acceptable
        if overall_score >= self.grade_thresholds[QualityGrade.ACCEPTABLE]:
            self.last_known_good[data.symbol] = data

        return metrics

    def _calculate_freshness_score(self, data: MarketData) -> float:
        """Calculate data freshness score based on timestamp"""
        if not data.timestamp:
            return 0.0

        age_seconds = (datetime.utcnow() - data.timestamp).total_seconds()

        if age_seconds <= self.freshness_threshold:
            return 1.0
        elif age_seconds <= self.freshness_threshold * 2:
            return 0.8
        elif age_seconds <= self.freshness_threshold * 5:
            return 0.6
        elif age_seconds <= self.freshness_threshold * 10:
            return 0.4
        else:
            return max(0.0, 1.0 - (age_seconds / (self.freshness_threshold * 20)))

    def _calculate_completeness_score(self, data: MarketData) -> float:
        """Calculate data completeness score based on required fields"""
        data_dict = data.to_dict()

        # Check required fields
        required_present = sum(
            1 for field in self.min_required_fields if field in data_dict and data_dict[field] is not None
        )
        required_score = required_present / len(self.min_required_fields)

        # Check optional but valuable fields
        optional_fields = ["bid", "ask", "high", "low", "open", "close", "bid_size", "ask_size", "vwap", "market_cap"]
        optional_present = sum(1 for field in optional_fields if field in data_dict and data_dict[field] is not None)
        optional_score = optional_present / len(optional_fields) if optional_fields else 0

        # Weighted combination (required fields are more important)
        return 0.7 * required_score + 0.3 * optional_score

    def _calculate_accuracy_score(
        self, data: MarketData, reference_data: Dict[str, MarketData]
    ) -> Tuple[float, List[str]]:
        """Calculate accuracy score by comparing with reference data"""
        anomalies = []
        scores = []

        # Compare with each reference source
        for source, ref_data in reference_data.items():
            if ref_data.symbol != data.symbol:
                continue

            # Price comparison
            if data.current_price and ref_data.current_price:
                price_diff = abs(data.current_price - ref_data.current_price)
                price_pct_diff = price_diff / ref_data.current_price

                if price_pct_diff <= self.price_deviation_threshold:
                    price_score = 1.0 - (price_pct_diff / self.price_deviation_threshold)
                else:
                    price_score = 0.0
                    anomalies.append(f"Price deviation from {source}: {price_pct_diff:.2%}")

                scores.append(price_score)

            # Volume comparison (less strict)
            if data.volume and ref_data.volume:
                volume_ratio = data.volume / ref_data.volume
                if 0.5 <= volume_ratio <= 2.0:
                    volume_score = 1.0
                else:
                    volume_score = max(0.0, 1.0 - abs(np.log(volume_ratio)))
                    if volume_score < 0.5:
                        anomalies.append(f"Volume mismatch with {source}: {volume_ratio:.2f}x")

                scores.append(volume_score * 0.5)  # Volume less important

        if not scores:
            return 0.5, anomalies  # No reference data available

        return np.mean(scores), anomalies

    def _estimate_accuracy_score(self, data: MarketData) -> float:
        """Estimate accuracy when no reference data is available"""
        score = 1.0

        # Basic sanity checks
        if data.current_price:
            if data.current_price <= 0:
                score *= 0.0
            elif data.current_price > 1000000:  # Unrealistic price
                score *= 0.5

        if data.volume and data.volume < 0:
            score *= 0.0

        # Check bid-ask spread if available
        if data.bid and data.ask:
            spread = (data.ask - data.bid) / data.current_price
            if spread > 0.1:  # More than 10% spread is suspicious
                score *= 0.7
            elif spread < 0:  # Inverted spread
                score *= 0.0

        return score

    def _detect_price_anomalies(self, data: MarketData) -> List[str]:
        """Detect price-related anomalies"""
        anomalies = []

        if not data.current_price or data.symbol not in self.price_history:
            return anomalies

        history = list(self.price_history[data.symbol])
        if len(history) < 5:
            return anomalies

        # Calculate statistics
        mean_price = np.mean(history)
        std_price = np.std(history)

        # Check for price spikes
        if std_price > 0:
            z_score = abs(data.current_price - mean_price) / std_price
            if z_score > 3:
                anomalies.append(f"Price spike detected: {z_score:.2f} standard deviations")

        # Check for sudden price changes
        if history:
            last_price = history[-1]
            price_change = abs(data.current_price - last_price) / last_price
            if price_change > 0.1:  # More than 10% change
                anomalies.append(f"Sudden price change: {price_change:.2%}")

        return anomalies

    def _detect_volume_anomalies(self, data: MarketData) -> List[str]:
        """Detect volume-related anomalies"""
        anomalies = []

        if not data.volume or data.symbol not in self.volume_history:
            return anomalies

        history = list(self.volume_history[data.symbol])
        if len(history) < 5:
            return anomalies

        # Calculate statistics
        median_volume = np.median(history)

        # Check for volume spikes
        if median_volume > 0:
            volume_ratio = data.volume / median_volume
            if volume_ratio > self.volume_spike_threshold:
                anomalies.append(f"Volume spike: {volume_ratio:.2f}x median")
            elif volume_ratio < 0.1:  # Very low volume
                anomalies.append(f"Unusually low volume: {volume_ratio:.2f}x median")

        return anomalies

    def _update_history(self, data: MarketData):
        """Update price and volume history"""
        if data.current_price:
            self.price_history[data.symbol].append(data.current_price)
        if data.volume:
            self.volume_history[data.symbol].append(data.volume)

    def _determine_grade(self, score: float) -> QualityGrade:
        """Determine quality grade based on score"""
        for grade, threshold in sorted(self.grade_thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return grade
        return QualityGrade.FAILED

    def _trigger_alerts(self, source: DataSourceType, metrics: QualityMetrics):
        """Trigger alerts for quality issues"""
        alert_msg = (
            f"Data quality alert for {source.value}: Grade={metrics.grade.value}, Score={metrics.overall_score:.2f}"
        )

        if metrics.anomalies:
            alert_msg += f", Anomalies: {', '.join(metrics.anomalies)}"

        logger.warning(alert_msg)

        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(source, metrics)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def register_alert_callback(self, callback):
        """Register a callback for quality alerts"""
        self.alert_callbacks.append(callback)

    def get_source_reliability_report(self) -> Dict[str, Any]:
        """Get reliability report for all sources"""
        report = {}

        for source in DataSourceType:
            reliability = self.reliability_tracker.get_reliability_score(source)
            total_requests = (
                self.reliability_tracker.success_counts[source] + self.reliability_tracker.error_counts[source]
            )

            report[source.value] = {
                "reliability_score": reliability,
                "total_requests": total_requests,
                "success_rate": (
                    self.reliability_tracker.success_counts[source] / total_requests if total_requests > 0 else 0.0
                ),
                "last_update": self.reliability_tracker.last_update[source].isoformat()
                if source in self.reliability_tracker.last_update
                else None,
                "grade": self._determine_grade(reliability).value,
            }

        return report

    def get_symbol_quality_trend(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get quality trend for a specific symbol"""
        # This would typically query a database of historical metrics
        # For now, return current state
        return {
            "symbol": symbol,
            "hours": hours,
            "current_price_history": list(self.price_history.get(symbol, [])),
            "current_volume_history": list(self.volume_history.get(symbol, [])),
            "last_known_good": (self.last_known_good[symbol].to_dict() if symbol in self.last_known_good else None),
        }
