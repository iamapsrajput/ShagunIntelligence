import pytest
from datetime import datetime, timedelta
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from backend.data_sources.market.models import MarketData
from backend.data_sources.data_quality_validator import DataSourceType
from backend.data_sources.data_quality_validator import (
    DataQualityValidator,
    QualityMetrics,
    QualityGrade,
    SourceReliabilityTracker
)


class TestQualityMetrics:
    def test_quality_metrics_initialization(self):
        metrics = QualityMetrics(
            freshness_score=0.9,
            accuracy_score=0.85,
            completeness_score=0.95,
            reliability_score=0.8,
            overall_score=0.875,
            grade=QualityGrade.GOOD,
            anomalies=["Test anomaly"]
        )

        assert metrics.freshness_score == 0.9
        assert metrics.accuracy_score == 0.85
        assert metrics.completeness_score == 0.95
        assert metrics.reliability_score == 0.8
        assert metrics.overall_score == 0.875
        assert metrics.grade == QualityGrade.GOOD
        assert len(metrics.anomalies) == 1
        assert metrics.anomalies[0] == "Test anomaly"

    def test_quality_metrics_to_dict(self):
        metrics = QualityMetrics(
            freshness_score=0.9,
            accuracy_score=0.85,
            completeness_score=0.95,
            reliability_score=0.8,
            overall_score=0.875,
            grade=QualityGrade.GOOD
        )

        result = metrics.to_dict()

        assert result["freshness_score"] == 0.9
        assert result["accuracy_score"] == 0.85
        assert result["completeness_score"] == 0.95
        assert result["reliability_score"] == 0.8
        assert result["overall_score"] == 0.875
        assert result["grade"] == "Good"
        assert "timestamp" in result


class TestSourceReliabilityTracker:
    def test_reliability_tracker_initialization(self):
        tracker = SourceReliabilityTracker(window_size=100)

        assert tracker.window_size == 100
        assert len(tracker.quality_history) == 0
        assert len(tracker.error_counts) == 0
        assert len(tracker.success_counts) == 0

    def test_update_tracking(self):
        tracker = SourceReliabilityTracker()
        source = DataSourceType.ZERODHA

        # Add some quality scores
        tracker.update(source, 0.9, success=True)
        tracker.update(source, 0.85, success=True)
        tracker.update(source, 0.7, success=False)

        assert len(tracker.quality_history[source]) == 3
        assert tracker.success_counts[source] == 2
        assert tracker.error_counts[source] == 1

    def test_get_reliability_score(self):
        tracker = SourceReliabilityTracker()
        source = DataSourceType.ZERODHA

        # No history should return default 0.5
        assert tracker.get_reliability_score(source) == 0.5

        # Add history
        tracker.update(source, 0.9, success=True)
        tracker.update(source, 0.8, success=True)
        tracker.update(source, 0.7, success=True)

        score = tracker.get_reliability_score(source)
        assert 0.7 <= score <= 0.9  # Should be weighted average

    def test_window_size_limit(self):
        tracker = SourceReliabilityTracker(window_size=5)
        source = DataSourceType.ZERODHA

        # Add more than window size
        for i in range(10):
            tracker.update(source, 0.5 + i * 0.05, success=True)

        # Should only keep last 5
        assert len(tracker.quality_history[source]) == 5


class TestDataQualityValidator:
    def setup_method(self):
        self.validator = DataQualityValidator(
            freshness_threshold_seconds=5,
            price_deviation_threshold=0.02,
            volume_spike_threshold=3.0
        )

    def test_validator_initialization(self):
        assert self.validator.freshness_threshold == 5
        assert self.validator.price_deviation_threshold == 0.02
        assert self.validator.volume_spike_threshold == 3.0
        assert len(self.validator.min_required_fields) == 4

    def test_freshness_score_calculation(self):
        # Fresh data
        fresh_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            timestamp=datetime.utcnow()
        )
        score = self.validator._calculate_freshness_score(fresh_data)
        assert score == 1.0

        # Slightly old data
        old_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            timestamp=datetime.utcnow() - timedelta(seconds=7)
        )
        score = self.validator._calculate_freshness_score(old_data)
        assert 0.7 < score < 0.9

        # Very old data
        very_old_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            timestamp=datetime.utcnow() - timedelta(minutes=5)
        )
        score = self.validator._calculate_freshness_score(very_old_data)
        assert score < 0.5

    def test_completeness_score_calculation(self):
        # Complete data
        complete_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            timestamp=datetime.utcnow(),
            bid=149.95,
            ask=150.05,
            high=152.0,
            low=148.0,
            open=149.0,
            close=150.5
        )
        score = self.validator._calculate_completeness_score(complete_data)
        assert score > 0.8

        # Minimal data
        minimal_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            timestamp=datetime.utcnow()
        )
        score = self.validator._calculate_completeness_score(minimal_data)
        assert 0.5 < score < 0.8

    def test_accuracy_score_with_reference(self):
        data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            timestamp=datetime.utcnow()
        )

        # Matching reference data
        reference_data = {
            "source1": MarketData(
                symbol="AAPL",
                current_price=150.5,
                volume=1100000,
                timestamp=datetime.utcnow()
            )
        }

        score, anomalies = self.validator._calculate_accuracy_score(data, reference_data)
        assert score > 0.7  # Small deviation should still score well
        assert len(anomalies) == 0

        # Large deviation
        reference_data["source2"] = MarketData(
            symbol="AAPL",
            current_price=160.0,
            volume=1000000,
            timestamp=datetime.utcnow()
        )

        score, anomalies = self.validator._calculate_accuracy_score(data, reference_data)
        assert score < 0.5
        assert len(anomalies) > 0

    def test_price_anomaly_detection(self):
        # Build price history
        for i in range(10):
            self.validator._update_history(
                MarketData(
                    symbol="AAPL",
                    current_price=150.0 + i * 0.1,
                    volume=1000000,
                    timestamp=datetime.utcnow()
                )
            )

        # Normal price
        normal_data = MarketData(
            symbol="AAPL",
            current_price=151.0,
            volume=1000000,
            timestamp=datetime.utcnow()
        )
        anomalies = self.validator._detect_price_anomalies(normal_data)
        assert len(anomalies) == 0

        # Price spike
        spike_data = MarketData(
            symbol="AAPL",
            current_price=200.0,
            volume=1000000,
            timestamp=datetime.utcnow()
        )
        anomalies = self.validator._detect_price_anomalies(spike_data)
        assert len(anomalies) > 0
        assert "spike" in anomalies[0].lower()

    def test_volume_anomaly_detection(self):
        # Build volume history
        for i in range(10):
            self.validator._update_history(
                MarketData(
                    symbol="AAPL",
                    current_price=150.0,
                    volume=1000000 + i * 10000,
                    timestamp=datetime.utcnow()
                )
            )

        # Normal volume
        normal_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            volume=1050000,
            timestamp=datetime.utcnow()
        )
        anomalies = self.validator._detect_volume_anomalies(normal_data)
        assert len(anomalies) == 0

        # Volume spike
        spike_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            volume=5000000,
            timestamp=datetime.utcnow()
        )
        anomalies = self.validator._detect_volume_anomalies(spike_data)
        assert len(anomalies) > 0
        assert "spike" in anomalies[0].lower()

    def test_grade_determination(self):
        assert self.validator._determine_grade(0.95) == QualityGrade.EXCELLENT
        assert self.validator._determine_grade(0.8) == QualityGrade.GOOD
        assert self.validator._determine_grade(0.65) == QualityGrade.ACCEPTABLE
        assert self.validator._determine_grade(0.45) == QualityGrade.POOR
        assert self.validator._determine_grade(0.2) == QualityGrade.FAILED

    def test_validate_data_complete_flow(self):
        data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            timestamp=datetime.utcnow(),
            bid=149.95,
            ask=150.05
        )

        metrics = self.validator.validate_data(
            data,
            DataSourceType.ZERODHA
        )

        assert isinstance(metrics, QualityMetrics)
        assert 0 <= metrics.freshness_score <= 1
        assert 0 <= metrics.accuracy_score <= 1
        assert 0 <= metrics.completeness_score <= 1
        assert 0 <= metrics.reliability_score <= 1
        assert 0 <= metrics.overall_score <= 1
        assert isinstance(metrics.grade, QualityGrade)

    def test_alert_callback_registration(self):
        callback_called = False

        def test_callback(source, metrics):
            nonlocal callback_called
            callback_called = True

        self.validator.register_alert_callback(test_callback)
        self.validator.alert_threshold = 0.9  # High threshold to trigger alert

        # Create low quality data
        data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            volume=1000000,
            timestamp=datetime.utcnow() - timedelta(minutes=10)  # Old data
        )

        metrics = self.validator.validate_data(
            data,
            DataSourceType.ZERODHA
        )

        assert callback_called

    def test_source_reliability_report(self):
        # Add some data
        for i in range(5):
            data = MarketData(
                symbol="AAPL",
                current_price=150.0 + i,
                volume=1000000,
                timestamp=datetime.utcnow()
            )
            self.validator.validate_data(data, DataSourceType.ZERODHA)

        report = self.validator.get_source_reliability_report()

        assert DataSourceType.ZERODHA.value in report
        zerodha_report = report[DataSourceType.ZERODHA.value]
        assert "reliability_score" in zerodha_report
        assert "total_requests" in zerodha_report
        assert "success_rate" in zerodha_report
        assert "grade" in zerodha_report


@pytest.mark.asyncio
class TestDataQualityIntegration:
    async def test_multi_source_integration(self):
        """Test integration with MultiSourceDataManager"""
        from backend.data_sources.multi_source_manager import MultiSourceDataManager

        manager = MultiSourceDataManager()

        # Verify quality validator is initialized
        assert hasattr(manager, '_quality_validator')
        assert isinstance(manager._quality_validator, DataQualityValidator)

        # Test quality callback registration
        callback_called = False

        async def quality_callback(source, metrics):
            nonlocal callback_called
            callback_called = True

        manager.add_quality_callback(quality_callback)
        assert quality_callback in manager._quality_callbacks