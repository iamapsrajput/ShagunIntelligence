import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiohttp

from backend.data_sources.sentiment.grok_api import (
    GrokSentimentSource,
    GrokSentimentResponse,
    GrokConfidenceLevel,
    GrokPromptBuilder,
    GrokCostManager,
    GrokBatchProcessor
)
from backend.data_sources.base import DataSourceConfig, DataSourceType, DataSourceStatus


class TestGrokSentimentResponse:
    def test_grok_response_creation(self):
        response = GrokSentimentResponse(
            symbol="AAPL",
            sentiment_score=0.75,
            sentiment_label="bullish",
            confidence=0.85,
            confidence_level=GrokConfidenceLevel.HIGH,
            reasoning="Strong earnings report and positive analyst coverage",
            key_factors=["earnings beat", "analyst upgrades"],
            trend_direction="rising",
            social_volume=15000,
            influential_posts=[{
                "content": "AAPL crushing earnings!",
                "influence_score": 0.9,
                "author_type": "analyst"
            }],
            timestamp=datetime.utcnow(),
            cost_tokens=500
        )
        
        assert response.symbol == "AAPL"
        assert response.sentiment_score == 0.75
        assert response.confidence == 0.85
        assert response.confidence_level == GrokConfidenceLevel.HIGH
        assert len(response.key_factors) == 2
        assert response.social_volume == 15000
    
    def test_grok_response_to_dict(self):
        timestamp = datetime.utcnow()
        response = GrokSentimentResponse(
            symbol="TSLA",
            sentiment_score=-0.3,
            sentiment_label="bearish",
            confidence=0.7,
            confidence_level=GrokConfidenceLevel.HIGH,
            reasoning="Concerns about production delays",
            key_factors=["production issues"],
            trend_direction="falling",
            social_volume=25000,
            influential_posts=[],
            timestamp=timestamp
        )
        
        result = response.to_dict()
        
        assert result["symbol"] == "TSLA"
        assert result["sentiment_score"] == -0.3
        assert result["confidence_level"] == "high"
        assert result["timestamp"] == timestamp.isoformat()


class TestGrokPromptBuilder:
    def test_single_symbol_prompt(self):
        prompt = GrokPromptBuilder.build_single_symbol_prompt("AAPL", hours=2)
        
        assert "$AAPL" in prompt
        assert "2 hour(s)" in prompt
        assert "real-time X platform data" in prompt
        assert "sentiment score" in prompt
        assert "JSON" in prompt
    
    def test_batch_prompt(self):
        symbols = ["AAPL", "TSLA", "GOOGL"]
        prompt = GrokPromptBuilder.build_batch_prompt(symbols, hours=1)
        
        assert "$AAPL" in prompt
        assert "$TSLA" in prompt
        assert "$GOOGL" in prompt
        assert "1 hour(s)" in prompt
    
    def test_trend_analysis_prompt(self):
        prompt = GrokPromptBuilder.build_trend_analysis_prompt("MSFT", hours=24)
        
        assert "$MSFT" in prompt
        assert "24 hours" in prompt
        assert "hourly sentiment data" in prompt
        assert "significant events" in prompt


class TestGrokCostManager:
    def test_cost_manager_initialization(self):
        manager = GrokCostManager(daily_budget=50.0, cost_per_1k_tokens=0.05)
        
        assert manager.daily_budget == 50.0
        assert manager.cost_per_1k_tokens == 0.05
    
    def test_can_make_request(self):
        manager = GrokCostManager(daily_budget=10.0, cost_per_1k_tokens=0.10)
        
        # Should allow first request
        assert manager.can_make_request(1000) is True
        
        # Record usage
        manager.record_usage(50000)  # $5 worth
        assert manager.can_make_request(1000) is True
        
        # Record more usage to exceed budget
        manager.record_usage(60000)  # Another $6, total $11
        assert manager.can_make_request(1000) is False
    
    def test_usage_stats(self):
        manager = GrokCostManager(daily_budget=100.0)
        
        # Record some usage
        manager.record_usage(5000, cost=0.5)
        manager.record_usage(3000, cost=0.3)
        
        stats = manager.get_usage_stats()
        
        assert stats["daily_budget"] == 100.0
        assert stats["daily_usage"] == 0.8
        assert stats["daily_tokens"] == 8000
        assert stats["budget_remaining"] == 99.2
        assert stats["requests_today"] == 2


class TestGrokBatchProcessor:
    @pytest.mark.asyncio
    async def test_batch_processor_initialization(self):
        processor = GrokBatchProcessor(max_batch_size=5, batch_delay=1.0)
        
        assert processor.max_batch_size == 5
        assert processor.batch_delay == 1.0
    
    @pytest.mark.asyncio
    async def test_add_request(self):
        processor = GrokBatchProcessor(max_batch_size=3, batch_delay=0.1)
        
        callback = AsyncMock()
        
        await processor.add_request("AAPL", callback, priority=5)
        
        # Check request was added
        assert len(processor.pending_requests[5]) == 1
        assert processor.pending_requests[5][0]["symbol"] == "AAPL"


class TestGrokSentimentSource:
    def setup_method(self):
        self.config = DataSourceConfig(
            name="grok_test",
            source_type=DataSourceType.SENTIMENT,
            enabled=True,
            credentials={
                "api_key": "test_key",
                "api_url": "https://api.test.com",
                "daily_budget": 50.0
            }
        )
        self.source = GrokSentimentSource(self.config)
    
    def test_initialization(self):
        assert self.source.api_key == "test_key"
        assert self.source.api_url == "https://api.test.com"
        assert isinstance(self.source.prompt_builder, GrokPromptBuilder)
        assert isinstance(self.source.cost_manager, GrokCostManager)
        assert isinstance(self.source.batch_processor, GrokBatchProcessor)
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        with patch.object(self.source, '_make_request', return_value='{"test": "response"}'):
            self.source.session = MagicMock()
            result = await self.source.connect()
            
            assert result is True
            assert self.source.health.status == DataSourceStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        with patch.object(self.source, '_make_request', return_value=None):
            result = await self.source.connect()
            
            assert result is False
            assert self.source.health.status == DataSourceStatus.UNHEALTHY
    
    def test_parse_grok_response(self):
        response_text = json.dumps({
            "sentiment_score": 0.65,
            "sentiment_label": "bullish",
            "confidence": 0.8,
            "reasoning": "Positive market sentiment",
            "key_factors": ["strong earnings", "product launch"],
            "trend_direction": "rising",
            "social_volume": 12000,
            "influential_posts": [{
                "content": "Great quarter!",
                "influence_score": 0.85,
                "author_type": "verified"
            }]
        })
        
        result = self.source._parse_grok_response(response_text, "AAPL")
        
        assert result is not None
        assert result.symbol == "AAPL"
        assert result.sentiment_score == 0.65
        assert result.confidence == 0.8
        assert result.confidence_level == GrokConfidenceLevel.HIGH
        assert len(result.key_factors) == 2
    
    def test_get_confidence_level(self):
        assert self.source._get_confidence_level(0.95) == GrokConfidenceLevel.VERY_HIGH
        assert self.source._get_confidence_level(0.8) == GrokConfidenceLevel.HIGH
        assert self.source._get_confidence_level(0.6) == GrokConfidenceLevel.MEDIUM
        assert self.source._get_confidence_level(0.4) == GrokConfidenceLevel.LOW
        assert self.source._get_confidence_level(0.2) == GrokConfidenceLevel.VERY_LOW
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_cached(self):
        # Add to cache
        cached_response = GrokSentimentResponse(
            symbol="AAPL",
            sentiment_score=0.7,
            sentiment_label="bullish",
            confidence=0.8,
            confidence_level=GrokConfidenceLevel.HIGH,
            reasoning="Cached response",
            key_factors=[],
            trend_direction="stable",
            social_volume=1000,
            influential_posts=[],
            timestamp=datetime.utcnow()
        )
        
        cache_key = f"AAPL_{int(datetime.utcnow().timestamp() / self.source.cache_ttl)}"
        self.source.sentiment_cache[cache_key] = cached_response
        
        # Should return cached response
        result = await self.source.analyze_sentiment("AAPL", use_cache=True)
        
        assert result == cached_response
        assert result.reasoning == "Cached response"
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_api_call(self):
        response_text = json.dumps({
            "sentiment_score": 0.5,
            "sentiment_label": "neutral",
            "confidence": 0.7,
            "reasoning": "Mixed signals",
            "key_factors": ["volatility"],
            "trend_direction": "stable",
            "social_volume": 5000,
            "influential_posts": []
        })
        
        with patch.object(self.source, '_make_request', return_value=response_text):
            result = await self.source.analyze_sentiment("TSLA", use_cache=False)
            
            assert result is not None
            assert result.symbol == "TSLA"
            assert result.sentiment_score == 0.5
            assert result.sentiment_label == "neutral"
    
    @pytest.mark.asyncio
    async def test_analyze_batch(self):
        batch_response = json.dumps({
            "symbols": {
                "AAPL": {
                    "sentiment_score": 0.7,
                    "sentiment_label": "bullish",
                    "confidence": 0.8,
                    "reasoning": "Strong performance",
                    "trend_direction": "rising",
                    "social_volume": 10000
                },
                "TSLA": {
                    "sentiment_score": -0.2,
                    "sentiment_label": "bearish",
                    "confidence": 0.6,
                    "reasoning": "Production concerns",
                    "trend_direction": "falling",
                    "social_volume": 15000
                }
            }
        })
        
        with patch.object(self.source, '_make_request', return_value=batch_response):
            results = await self.source.analyze_batch(["AAPL", "TSLA"])
            
            assert len(results) == 2
            assert "AAPL" in results
            assert "TSLA" in results
            assert results["AAPL"].sentiment_score == 0.7
            assert results["TSLA"].sentiment_score == -0.2
    
    @pytest.mark.asyncio
    async def test_get_sentiment_score(self):
        with patch.object(self.source, 'analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = GrokSentimentResponse(
                symbol="MSFT",
                sentiment_score=0.6,
                sentiment_label="bullish",
                confidence=0.75,
                confidence_level=GrokConfidenceLevel.HIGH,
                reasoning="Positive outlook",
                key_factors=["cloud growth"],
                trend_direction="rising",
                social_volume=8000,
                influential_posts=[],
                timestamp=datetime.utcnow()
            )
            
            result = await self.source.get_sentiment_score("MSFT")
            
            assert result["symbol"] == "MSFT"
            assert result["sentiment_score"] == 0.6
            assert result["confidence_level"] == "high"
            assert result["source"] == "grok"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        # Fill rate limit
        now = datetime.utcnow().timestamp()
        self.source.request_times = [now - i for i in range(30)]  # 30 requests
        
        with patch('asyncio.sleep') as mock_sleep:
            await self.source._check_rate_limit()
            
            # Should have called sleep
            mock_sleep.assert_called_once()
    
    def test_update_metrics(self):
        # Success
        self.source._update_metrics(success=True, response_time=1.5)
        assert self.source.success_count == 1
        assert self.source.avg_response_time == 1.5
        
        # Failure
        self.source._update_metrics(success=False)
        assert self.source.error_count == 1
        
        # Check health status update
        assert self.source.health.status == DataSourceStatus.HEALTHY  # 50% success rate
    
    def test_get_cost_stats(self):
        self.source.cost_manager.record_usage(5000, cost=0.5)
        
        stats = self.source.get_cost_stats()
        
        assert "daily_budget" in stats
        assert "daily_usage" in stats
        assert stats["daily_usage"] == 0.5
    
    def test_get_reliability_score(self):
        # No requests yet
        assert self.source.get_reliability_score() == 1.0
        
        # Some successes and failures
        self.source.success_count = 8
        self.source.error_count = 2
        
        assert self.source.get_reliability_score() == 0.8


@pytest.mark.asyncio
class TestGrokIntegration:
    async def test_grok_with_sentiment_fusion(self):
        """Test Grok integration with sentiment fusion"""
        from backend.data_sources.sentiment.sentiment_fusion import MultiSourceSentimentFusion
        
        config = DataSourceConfig(
            name="grok_test",
            source_type=DataSourceType.SENTIMENT,
            enabled=True,
            credentials={"api_key": "test_key"}
        )
        
        grok_source = GrokSentimentSource(config)
        fusion = MultiSourceSentimentFusion()
        
        # Add Grok to fusion with higher weight
        fusion.add_source("grok", grok_source, weight=1.5)
        
        # Mock Grok response
        with patch.object(grok_source, 'get_sentiment_score') as mock_get:
            mock_get.return_value = {
                "sentiment_score": 0.8,
                "confidence": 0.9,
                "confidence_level": "very_high"
            }
            
            result = await fusion.get_fused_sentiment("AAPL")
            
            assert result["symbol"] == "AAPL"
            assert result["fused_sentiment_score"] > 0
            assert "grok" in result["sources"]