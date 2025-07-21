import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json

from backend.data_sources.sentiment.twitter_api import (
    TwitterSentimentSource,
    TweetSentiment,
    SymbolSentiment,
    TwitterRateLimiter
)
from backend.data_sources.sentiment.sentiment_fusion import (
    MultiSourceSentimentFusion,
    SentimentSource
)
from backend.data_sources.base import DataSourceConfig, DataSourceType, DataSourceStatus


class TestTweetSentiment:
    def test_tweet_sentiment_creation(self):
        tweet = TweetSentiment(
            tweet_id="123456",
            text="$AAPL is looking bullish! 🚀",
            author_id="user123",
            created_at=datetime.utcnow(),
            symbols=["AAPL"],
            sentiment_score=0.8,
            sentiment_label="positive",
            confidence=0.9,
            user_influence=0.7,
            engagement_metrics={"likes": 100, "retweets": 50}
        )
        
        assert tweet.tweet_id == "123456"
        assert tweet.symbols == ["AAPL"]
        assert tweet.sentiment_score == 0.8
        assert tweet.sentiment_label == "positive"
        assert tweet.confidence == 0.9
        assert tweet.user_influence == 0.7
    
    def test_tweet_sentiment_to_dict(self):
        created_at = datetime.utcnow()
        tweet = TweetSentiment(
            tweet_id="123456",
            text="Test tweet",
            author_id="user123",
            created_at=created_at,
            symbols=["AAPL"],
            sentiment_score=0.5,
            sentiment_label="neutral",
            confidence=0.8,
            user_influence=0.6,
            engagement_metrics={"likes": 10}
        )
        
        result = tweet.to_dict()
        
        assert result["tweet_id"] == "123456"
        assert result["created_at"] == created_at.isoformat()
        assert result["sentiment_score"] == 0.5
        assert "engagement_metrics" in result


class TestSymbolSentiment:
    def test_symbol_sentiment_initialization(self):
        sentiment = SymbolSentiment("AAPL")
        
        assert sentiment.symbol == "AAPL"
        assert len(sentiment.tweets) == 0
        assert len(sentiment.hourly_sentiment) == 0
    
    def test_add_tweet(self):
        sentiment = SymbolSentiment("AAPL")
        
        tweet = TweetSentiment(
            tweet_id="123",
            text="Test",
            author_id="user1",
            created_at=datetime.utcnow(),
            symbols=["AAPL"],
            sentiment_score=0.7,
            sentiment_label="positive",
            confidence=0.8,
            user_influence=0.5,
            engagement_metrics={}
        )
        
        sentiment.add_tweet(tweet)
        
        assert len(sentiment.tweets) == 1
        assert sentiment.tweets[0] == tweet
    
    def test_aggregate_sentiment_empty(self):
        sentiment = SymbolSentiment("AAPL")
        result = sentiment.get_aggregate_sentiment(hours=1)
        
        assert result["symbol"] == "AAPL"
        assert result["sentiment_score"] == 0.0
        assert result["sentiment_label"] == "neutral"
        assert result["confidence"] == 0.0
        assert result["tweet_count"] == 0
    
    def test_aggregate_sentiment_with_tweets(self):
        sentiment = SymbolSentiment("AAPL")
        
        # Add multiple tweets
        for i in range(5):
            tweet = TweetSentiment(
                tweet_id=str(i),
                text="Test",
                author_id="user1",
                created_at=datetime.utcnow(),
                symbols=["AAPL"],
                sentiment_score=0.6 if i % 2 == 0 else 0.8,
                sentiment_label="positive",
                confidence=0.9,
                user_influence=0.7,
                engagement_metrics={}
            )
            sentiment.add_tweet(tweet)
        
        result = sentiment.get_aggregate_sentiment(hours=1)
        
        assert result["symbol"] == "AAPL"
        assert 0.6 <= result["sentiment_score"] <= 0.8
        assert result["sentiment_label"] == "positive"
        assert result["confidence"] > 0
        assert result["tweet_count"] == 5


class TestTwitterSentimentSource:
    def setup_method(self):
        self.config = DataSourceConfig(
            name="twitter_test",
            source_type=DataSourceType.SENTIMENT,
            enabled=True,
            credentials={
                "bearer_token": "test_token",
                "api_key": "test_key",
                "api_secret": "test_secret"
            }
        )
        self.source = TwitterSentimentSource(self.config)
    
    def test_initialization(self):
        assert self.source.bearer_token == "test_token"
        assert self.source.api_key == "test_key"
        assert self.source.api_secret == "test_secret"
        assert len(self.source.financial_keywords) > 0
        assert isinstance(self.source.rate_limiter, TwitterRateLimiter)
    
    def test_extract_symbols(self):
        text1 = "I'm bullish on $AAPL and $TSLA today!"
        symbols1 = self.source._extract_symbols(text1)
        assert set(symbols1) == {"AAPL", "TSLA"}
        
        text2 = "Check out #MSFT and #GOOGL performance"
        symbols2 = self.source._extract_symbols(text2)
        assert set(symbols2) == {"MSFT", "GOOGL"}
        
        text3 = "No symbols in this tweet"
        symbols3 = self.source._extract_symbols(text3)
        assert symbols3 == []
    
    def test_clean_tweet_text(self):
        text = "@user Check $AAPL at https://example.com #trading"
        cleaned = self.source._clean_tweet_text(text)
        
        assert "@user" not in cleaned
        assert "$AAPL" not in cleaned
        assert "https://example.com" not in cleaned
        assert "Check" in cleaned
    
    def test_calculate_keyword_boost(self):
        bullish_text = "very bullish on this stock, buying calls"
        bullish_boost = self.source._calculate_keyword_boost(bullish_text)
        assert bullish_boost > 0
        
        bearish_text = "bearish outlook, selling puts and shorting"
        bearish_boost = self.source._calculate_keyword_boost(bearish_text)
        assert bearish_boost < 0
        
        neutral_text = "just watching the market today"
        neutral_boost = self.source._calculate_keyword_boost(neutral_text)
        assert neutral_boost == 0
    
    def test_analyze_sentiment(self):
        positive_text = "This stock is amazing! Great earnings, strong buy!"
        pos_score, pos_label, pos_conf = self.source._analyze_sentiment(positive_text)
        assert pos_score > 0
        assert pos_label == "positive"
        assert 0 <= pos_conf <= 1
        
        negative_text = "Terrible performance, huge losses, avoid at all costs!"
        neg_score, neg_label, neg_conf = self.source._analyze_sentiment(negative_text)
        assert neg_score < 0
        assert neg_label == "negative"
        
        neutral_text = "The stock traded sideways today."
        neu_score, neu_label, neu_conf = self.source._analyze_sentiment(neutral_text)
        assert -0.1 <= neu_score <= 0.1
        assert neu_label == "neutral"
    
    @pytest.mark.asyncio
    async def test_calculate_user_influence(self):
        # Test with user data
        user_data = {
            "public_metrics": {
                "followers_count": 10000,
                "following_count": 500,
                "tweet_count": 5000
            },
            "verified": False
        }
        
        influence = await self.source._calculate_user_influence("user123", user_data)
        assert 0 <= influence <= 1
        assert influence > 0.5  # Should be above average for 10k followers
        
        # Test with verified user
        verified_user = user_data.copy()
        verified_user["verified"] = True
        
        verified_influence = await self.source._calculate_user_influence("user456", verified_user)
        assert verified_influence > influence  # Verified users get bonus
        
        # Test without user data
        default_influence = await self.source._calculate_user_influence("user789", None)
        assert default_influence == 0.5
    
    @pytest.mark.asyncio
    async def test_get_sentiment_score(self):
        # Test with no data
        result = await self.source.get_sentiment_score("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["sentiment_score"] == 0.0
        assert result["sentiment_label"] == "neutral"
        assert "error" in result
        
        # Add some sentiment data
        self.source.symbol_sentiments["AAPL"] = SymbolSentiment("AAPL")
        tweet = TweetSentiment(
            tweet_id="123",
            text="Bullish",
            author_id="user1",
            created_at=datetime.utcnow(),
            symbols=["AAPL"],
            sentiment_score=0.8,
            sentiment_label="positive",
            confidence=0.9,
            user_influence=0.7,
            engagement_metrics={}
        )
        self.source.symbol_sentiments["AAPL"].add_tweet(tweet)
        
        result2 = await self.source.get_sentiment_score("AAPL")
        assert result2["symbol"] == "AAPL"
        assert result2["sentiment_score"] > 0
        assert result2["source"] == "twitter"


class TestTwitterRateLimiter:
    def test_rate_limiter_initialization(self):
        limiter = TwitterRateLimiter()
        
        assert "stream" in limiter.limits
        assert "lookup" in limiter.limits
        assert limiter.limits["stream"]["calls"] == 50
        assert limiter.limits["stream"]["window"] == 900
    
    @pytest.mark.asyncio
    async def test_rate_limit_check(self):
        limiter = TwitterRateLimiter()
        
        # First call should pass
        await limiter.check_limit("stream")
        assert len(limiter.calls["stream"]) == 1
        
        # Multiple calls within limit should pass
        for _ in range(5):
            await limiter.check_limit("stream")
        
        assert len(limiter.calls["stream"]) == 6


class TestMultiSourceSentimentFusion:
    def setup_method(self):
        self.fusion = MultiSourceSentimentFusion()
    
    def test_initialization(self):
        assert len(self.fusion.sources) == 0
        assert self.fusion.cache_ttl == 300
    
    def test_add_remove_source(self):
        mock_source = Mock()
        self.fusion.add_source("test_source", mock_source, weight=0.8)
        
        assert "test_source" in self.fusion.sources
        assert self.fusion.sources["test_source"].weight == 0.8
        
        self.fusion.remove_source("test_source")
        assert "test_source" not in self.fusion.sources
    
    @pytest.mark.asyncio
    async def test_get_fused_sentiment_no_sources(self):
        result = await self.fusion.get_fused_sentiment("AAPL")
        
        assert result["symbol"] == "AAPL"
        assert result["fused_sentiment_score"] == 0.0
        assert result["fused_sentiment_label"] == "neutral"
        assert result["source_count"] == 0
    
    @pytest.mark.asyncio
    async def test_get_fused_sentiment_with_sources(self):
        # Create mock sources
        mock_source1 = AsyncMock()
        mock_source1.get_sentiment_score.return_value = {
            "sentiment_score": 0.7,
            "confidence": 0.8
        }
        
        mock_source2 = AsyncMock()
        mock_source2.get_sentiment_score.return_value = {
            "sentiment_score": 0.5,
            "confidence": 0.6
        }
        
        self.fusion.add_source("source1", mock_source1, weight=1.0)
        self.fusion.add_source("source2", mock_source2, weight=0.5)
        
        result = await self.fusion.get_fused_sentiment("AAPL")
        
        assert result["symbol"] == "AAPL"
        assert 0.5 <= result["fused_sentiment_score"] <= 0.7
        assert result["fused_sentiment_label"] == "positive"
        assert result["source_count"] == 2
        assert "source1" in result["sources"]
        assert "source2" in result["sources"]
    
    def test_calculate_source_agreement(self):
        # High agreement
        sentiments1 = [
            {"score": 0.7, "weight": 1.0, "reliability": 1.0, "confidence": 0.8},
            {"score": 0.8, "weight": 1.0, "reliability": 1.0, "confidence": 0.8},
            {"score": 0.75, "weight": 1.0, "reliability": 1.0, "confidence": 0.8}
        ]
        agreement1 = self.fusion._calculate_source_agreement(sentiments1)
        assert agreement1 > 0.8
        
        # Low agreement
        sentiments2 = [
            {"score": -0.8, "weight": 1.0, "reliability": 1.0, "confidence": 0.8},
            {"score": 0.8, "weight": 1.0, "reliability": 1.0, "confidence": 0.8}
        ]
        agreement2 = self.fusion._calculate_source_agreement(sentiments2)
        assert agreement2 < 0.2
    
    @pytest.mark.asyncio
    async def test_get_real_time_alerts(self):
        # Mock source with high sentiment
        mock_source = AsyncMock()
        mock_source.get_sentiment_score.return_value = {
            "sentiment_score": 0.8,
            "confidence": 0.9
        }
        
        self.fusion.add_source("test", mock_source, weight=1.0)
        
        alerts = await self.fusion.get_real_time_alerts("AAPL", threshold=0.5)
        
        assert len(alerts) == 1
        assert alerts[0]["symbol"] == "AAPL"
        assert alerts[0]["alert_type"] == "bullish"
        assert alerts[0]["sentiment_score"] > 0.5


@pytest.mark.asyncio
class TestIntegration:
    async def test_twitter_source_with_fusion(self):
        """Test Twitter source integrated with sentiment fusion"""
        config = DataSourceConfig(
            name="twitter_test",
            source_type=DataSourceType.SENTIMENT,
            enabled=True,
            credentials={"bearer_token": "test_token"}
        )
        
        twitter_source = TwitterSentimentSource(config)
        fusion = MultiSourceSentimentFusion()
        
        # Add Twitter to fusion
        fusion.add_source("twitter", twitter_source, weight=1.0)
        
        # Add test data to Twitter source
        twitter_source.symbol_sentiments["AAPL"] = SymbolSentiment("AAPL")
        for i in range(3):
            tweet = TweetSentiment(
                tweet_id=str(i),
                text="Test",
                author_id="user1",
                created_at=datetime.utcnow(),
                symbols=["AAPL"],
                sentiment_score=0.7,
                sentiment_label="positive",
                confidence=0.8,
                user_influence=0.6,
                engagement_metrics={}
            )
            twitter_source.symbol_sentiments["AAPL"].add_tweet(tweet)
        
        # Get fused sentiment
        result = await fusion.get_fused_sentiment("AAPL")
        
        assert result["symbol"] == "AAPL"
        assert result["fused_sentiment_score"] > 0
        assert result["source_count"] == 1
        assert "twitter" in result["sources"]