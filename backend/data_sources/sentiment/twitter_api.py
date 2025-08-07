import asyncio
import json
import re
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import numpy as np
from loguru import logger
from textblob import TextBlob

from backend.data_sources.base import (
    DataSourceConfig,
    DataSourceStatus,
    SentimentDataSource,
)


class TweetSentiment:
    """Represents sentiment analysis results for a tweet"""

    def __init__(
        self,
        tweet_id: str,
        text: str,
        author_id: str,
        created_at: datetime,
        symbols: list[str],
        sentiment_score: float,
        sentiment_label: str,
        confidence: float,
        user_influence: float,
        engagement_metrics: dict[str, int],
    ):
        self.tweet_id = tweet_id
        self.text = text
        self.author_id = author_id
        self.created_at = created_at
        self.symbols = symbols
        self.sentiment_score = sentiment_score
        self.sentiment_label = sentiment_label
        self.confidence = confidence
        self.user_influence = user_influence
        self.engagement_metrics = engagement_metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "tweet_id": self.tweet_id,
            "text": self.text,
            "author_id": self.author_id,
            "created_at": self.created_at.isoformat(),
            "symbols": self.symbols,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "confidence": self.confidence,
            "user_influence": self.user_influence,
            "engagement_metrics": self.engagement_metrics,
        }


class SymbolSentiment:
    """Aggregated sentiment for a specific symbol"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.tweets = deque(maxlen=1000)  # Keep last 1000 tweets
        self.hourly_sentiment = defaultdict(list)
        self.last_update = datetime.utcnow()

    def add_tweet(self, tweet: TweetSentiment):
        self.tweets.append(tweet)
        hour_key = tweet.created_at.replace(minute=0, second=0, microsecond=0)
        self.hourly_sentiment[hour_key].append(tweet)
        self.last_update = datetime.utcnow()

    def get_aggregate_sentiment(self, hours: int = 1) -> dict[str, Any]:
        """Calculate aggregate sentiment over specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_tweets = [t for t in self.tweets if t.created_at > cutoff_time]

        if not recent_tweets:
            return {
                "symbol": self.symbol,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "tweet_count": 0,
                "period_hours": hours,
            }

        # Calculate weighted sentiment based on influence and confidence
        total_weight = 0
        weighted_sentiment = 0

        for tweet in recent_tweets:
            weight = tweet.confidence * tweet.user_influence
            weighted_sentiment += tweet.sentiment_score * weight
            total_weight += weight

        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0

        # Determine label
        if avg_sentiment > 0.1:
            label = "positive"
        elif avg_sentiment < -0.1:
            label = "negative"
        else:
            label = "neutral"

        # Calculate confidence based on volume and consistency
        sentiment_scores = [t.sentiment_score for t in recent_tweets]
        sentiment_std = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 1.0
        volume_confidence = min(
            len(recent_tweets) / 50, 1.0
        )  # Max confidence at 50 tweets
        consistency_confidence = max(0, 1.0 - sentiment_std)

        overall_confidence = (volume_confidence + consistency_confidence) / 2

        return {
            "symbol": self.symbol,
            "sentiment_score": avg_sentiment,
            "sentiment_label": label,
            "confidence": overall_confidence,
            "tweet_count": len(recent_tweets),
            "period_hours": hours,
            "timestamp": datetime.utcnow().isoformat(),
        }


class TwitterSentimentSource(SentimentDataSource):
    """Twitter API v2 integration for financial sentiment analysis"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.bearer_token = config.credentials.get("bearer_token")
        self.api_key = config.credentials.get("api_key")
        self.api_secret = config.credentials.get("api_secret")

        # Streaming configuration
        self.stream_url = "https://api.twitter.com/2/tweets/search/stream"
        self.rules_url = "https://api.twitter.com/2/tweets/search/stream/rules"

        # Rate limiting
        self.rate_limiter = TwitterRateLimiter()

        # Sentiment analysis
        self.symbol_sentiments: dict[str, SymbolSentiment] = {}
        self.financial_keywords = {
            "bullish",
            "bearish",
            "buy",
            "sell",
            "long",
            "short",
            "calls",
            "puts",
            "moon",
            "dump",
            "rally",
            "crash",
            "breakout",
            "resistance",
            "support",
            "earnings",
            "upgrade",
            "downgrade",
            "target",
            "forecast",
        }

        # User influence cache
        self.user_cache: dict[str, dict[str, Any]] = {}
        self.user_cache_ttl = 3600  # 1 hour

        # Streaming state
        self.stream_session: aiohttp.ClientSession | None = None
        self.stream_task: asyncio.Task | None = None
        self.reconnect_delay = 1
        self.max_reconnect_delay = 300

        # Callbacks
        self.sentiment_callbacks: list[Callable] = []

    async def connect(self) -> bool:
        """Connect to Twitter API v2"""
        try:
            if not self.bearer_token:
                raise ValueError("Twitter bearer token not provided")

            # Create session
            self.stream_session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.bearer_token}"}
            )

            # Set up streaming rules
            await self._setup_stream_rules()

            # Start streaming
            self.stream_task = asyncio.create_task(self._stream_tweets())

            self.update_health_status(DataSourceStatus.HEALTHY)
            logger.info("Connected to Twitter API v2")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Twitter API: {e}")
            self.update_health_status(DataSourceStatus.UNHEALTHY, str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from Twitter API"""
        try:
            # Cancel streaming
            if self.stream_task:
                self.stream_task.cancel()
                try:
                    await self.stream_task
                except asyncio.CancelledError:
                    pass

            # Close session
            if self.stream_session:
                await self.stream_session.close()

            self.update_health_status(DataSourceStatus.DISCONNECTED)
            logger.info("Disconnected from Twitter API")

        except Exception as e:
            logger.error(f"Error disconnecting from Twitter API: {e}")

    async def _setup_stream_rules(self) -> None:
        """Set up streaming rules for financial content"""
        # Get existing rules
        async with self.stream_session.get(self.rules_url) as response:
            if response.status == 200:
                data = await response.json()
                existing_rules = data.get("data", [])

                # Delete existing rules
                if existing_rules:
                    rule_ids = [rule["id"] for rule in existing_rules]
                    await self._delete_stream_rules(rule_ids)

        # Add new rules for stock symbols and financial keywords
        rules = []

        # Add rules for tracked symbols
        if hasattr(self, "_tracked_symbols") and self._tracked_symbols:
            for symbol in self._tracked_symbols:
                rules.append(
                    {"value": f"${symbol} OR #{symbol}", "tag": f"symbol_{symbol}"}
                )

        # Add general financial keywords rule
        keywords = " OR ".join(self.financial_keywords)
        rules.append(
            {"value": f"({keywords}) lang:en -is:retweet", "tag": "financial_keywords"}
        )

        # Submit rules
        if rules:
            await self._add_stream_rules(rules)

    async def _add_stream_rules(self, rules: list[dict[str, str]]) -> None:
        """Add streaming rules"""
        payload = {"add": rules}

        async with self.stream_session.post(self.rules_url, json=payload) as response:
            if response.status != 201:
                text = await response.text()
                raise Exception(f"Failed to add stream rules: {text}")

            logger.info(f"Added {len(rules)} streaming rules")

    async def _delete_stream_rules(self, rule_ids: list[str]) -> None:
        """Delete streaming rules"""
        payload = {"delete": {"ids": rule_ids}}

        async with self.stream_session.post(self.rules_url, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                logger.error(f"Failed to delete stream rules: {text}")

    async def _stream_tweets(self) -> None:
        """Stream tweets in real-time"""
        while True:
            try:
                # Check rate limits
                await self.rate_limiter.check_limit("stream")

                # Stream parameters
                params = {
                    "tweet.fields": "created_at,author_id,public_metrics,context_annotations",
                    "user.fields": "public_metrics,verified",
                    "expansions": "author_id",
                }

                async with self.stream_session.get(
                    self.stream_url, params=params
                ) as response:
                    if response.status == 429:
                        # Rate limited
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, retrying after {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue

                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"Stream error: {response.status} - {text}")

                    # Reset reconnect delay on successful connection
                    self.reconnect_delay = 1

                    # Process stream
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                await self._process_tweet(data)
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                logger.error(f"Error processing tweet: {e}")

            except asyncio.CancelledError:
                logger.info("Tweet streaming cancelled")
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")

                # Exponential backoff
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(
                    self.reconnect_delay * 2, self.max_reconnect_delay
                )

    async def _process_tweet(self, data: dict[str, Any]) -> None:
        """Process incoming tweet data"""
        if "data" not in data:
            return

        tweet_data = data["data"]
        includes = data.get("includes", {})

        # Extract tweet info
        tweet_id = tweet_data["id"]
        text = tweet_data["text"]
        author_id = tweet_data["author_id"]
        created_at = datetime.fromisoformat(
            tweet_data["created_at"].replace("Z", "+00:00")
        )

        # Extract symbols from text
        symbols = self._extract_symbols(text)
        if not symbols:
            return

        # Get user data
        user_data = None
        if "users" in includes:
            for user in includes["users"]:
                if user["id"] == author_id:
                    user_data = user
                    break

        # Calculate user influence
        user_influence = await self._calculate_user_influence(author_id, user_data)

        # Analyze sentiment
        sentiment_score, sentiment_label, confidence = self._analyze_sentiment(text)

        # Get engagement metrics
        engagement_metrics = tweet_data.get("public_metrics", {})

        # Create tweet sentiment object
        tweet_sentiment = TweetSentiment(
            tweet_id=tweet_id,
            text=text,
            author_id=author_id,
            created_at=created_at,
            symbols=symbols,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            user_influence=user_influence,
            engagement_metrics=engagement_metrics,
        )

        # Update symbol sentiments
        for symbol in symbols:
            if symbol not in self.symbol_sentiments:
                self.symbol_sentiments[symbol] = SymbolSentiment(symbol)
            self.symbol_sentiments[symbol].add_tweet(tweet_sentiment)

        # Notify callbacks
        for callback in self.sentiment_callbacks:
            try:
                await callback(tweet_sentiment)
            except Exception as e:
                logger.error(f"Error in sentiment callback: {e}")

    def _extract_symbols(self, text: str) -> list[str]:
        """Extract stock symbols from tweet text"""
        # Match $SYMBOL or #SYMBOL patterns
        pattern = r"[$#]([A-Z]{1,5})\b"
        matches = re.findall(pattern, text.upper())

        # Filter valid symbols (you might want to validate against a known list)
        symbols = []
        for match in matches:
            if 1 <= len(match) <= 5 and match.isalpha():
                symbols.append(match)

        return list(set(symbols))

    def _analyze_sentiment(self, text: str) -> tuple[float, str, float]:
        """Analyze sentiment of tweet text"""
        try:
            # Clean text
            cleaned_text = self._clean_tweet_text(text)

            # Use TextBlob for sentiment analysis
            blob = TextBlob(cleaned_text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1

            # Adjust for financial keywords
            sentiment_boost = self._calculate_keyword_boost(cleaned_text.lower())
            adjusted_polarity = np.clip(polarity + sentiment_boost, -1, 1)

            # Determine label
            if adjusted_polarity > 0.1:
                label = "positive"
            elif adjusted_polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"

            # Calculate confidence based on subjectivity and keyword presence
            base_confidence = 1.0 - (
                subjectivity * 0.5
            )  # Less subjective = more confident
            keyword_confidence = min(
                abs(sentiment_boost) * 2, 0.3
            )  # Boost for financial keywords
            confidence = min(base_confidence + keyword_confidence, 1.0)

            return adjusted_polarity, label, confidence

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 0.0, "neutral", 0.5

    def _clean_tweet_text(self, text: str) -> str:
        """Clean tweet text for analysis"""
        # Remove URLs
        text = re.sub(r"http\S+|www.\S+", "", text)

        # Remove mentions but keep the text
        text = re.sub(r"@\w+", "", text)

        # Remove stock symbols for sentiment analysis
        text = re.sub(r"[$#][A-Z]{1,5}\b", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def _calculate_keyword_boost(self, text: str) -> float:
        """Calculate sentiment boost based on financial keywords"""
        boost = 0.0

        bullish_keywords = {
            "bullish",
            "buy",
            "long",
            "calls",
            "moon",
            "rally",
            "breakout",
            "upgrade",
        }
        bearish_keywords = {
            "bearish",
            "sell",
            "short",
            "puts",
            "dump",
            "crash",
            "resistance",
            "downgrade",
        }

        for word in bullish_keywords:
            if word in text:
                boost += 0.1

        for word in bearish_keywords:
            if word in text:
                boost -= 0.1

        return np.clip(boost, -0.3, 0.3)

    async def _calculate_user_influence(
        self, author_id: str, user_data: dict[str, Any] | None = None
    ) -> float:
        """Calculate user influence score"""
        # Check cache
        cache_key = (
            f"{author_id}_{int(datetime.utcnow().timestamp() / self.user_cache_ttl)}"
        )
        if cache_key in self.user_cache:
            return self.user_cache[cache_key]["influence"]

        influence = 0.5  # Default influence

        if user_data and "public_metrics" in user_data:
            metrics = user_data["public_metrics"]
            followers = metrics.get("followers_count", 0)
            following = metrics.get("following_count", 1)
            tweets = metrics.get("tweet_count", 0)

            # Calculate influence based on followers and engagement
            if followers > 0:
                # Log scale for followers
                follower_score = min(
                    np.log10(followers + 1) / 6, 1.0
                )  # Max at 1M followers

                # Follower/following ratio (quality indicator)
                ratio_score = min(followers / (following + 1), 10) / 10

                # Activity score
                activity_score = min(tweets / 10000, 1.0)

                # Verified bonus
                verified_bonus = 0.2 if user_data.get("verified", False) else 0

                # Weighted influence
                influence = (
                    follower_score * 0.5
                    + ratio_score * 0.3
                    + activity_score * 0.2
                    + verified_bonus
                )
                influence = min(influence, 1.0)

        # Cache result
        self.user_cache[cache_key] = {"influence": influence}

        return influence

    # SentimentDataSource interface methods

    async def get_sentiment_score(self, symbol: str) -> dict[str, Any]:
        """Get current sentiment score for a symbol"""
        if symbol not in self.symbol_sentiments:
            return {
                "symbol": symbol,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "source": "twitter",
                "error": "No data available",
            }

        sentiment = self.symbol_sentiments[symbol].get_aggregate_sentiment(hours=1)
        sentiment["source"] = "twitter"
        return sentiment

    async def get_news_sentiment(
        self,
        symbol: str,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get historical sentiment data"""
        if symbol not in self.symbol_sentiments:
            return []

        symbol_sentiment = self.symbol_sentiments[symbol]

        # Calculate hourly sentiments for the period
        results = []

        for hour, tweets in symbol_sentiment.hourly_sentiment.items():
            if from_date and hour < from_date:
                continue
            if to_date and hour > to_date:
                continue

            # Aggregate tweets for this hour
            if tweets:
                hourly_data = SymbolSentiment(symbol)
                for tweet in tweets:
                    hourly_data.add_tweet(tweet)

                sentiment = hourly_data.get_aggregate_sentiment(hours=1)
                sentiment["timestamp"] = hour.isoformat()
                sentiment["source"] = "twitter"
                results.append(sentiment)

        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    async def get_social_sentiment(
        self, symbol: str, platform: str = "twitter"
    ) -> dict[str, Any]:
        """Get social media sentiment"""
        return await self.get_sentiment_score(symbol)

    async def get_sentiment_trends(
        self, symbol: str, hours: int = 24
    ) -> dict[str, Any]:
        """Get sentiment trends over time"""
        if symbol not in self.symbol_sentiments:
            return {"symbol": symbol, "trends": [], "source": "twitter"}

        symbol_sentiment = self.symbol_sentiments[symbol]
        trends = []

        # Get hourly trends
        for i in range(hours):
            hour_time = datetime.utcnow() - timedelta(hours=i)
            hour_key = hour_time.replace(minute=0, second=0, microsecond=0)

            if hour_key in symbol_sentiment.hourly_sentiment:
                tweets = symbol_sentiment.hourly_sentiment[hour_key]
                if tweets:
                    # Calculate average sentiment for this hour
                    sentiments = [t.sentiment_score for t in tweets]
                    avg_sentiment = np.mean(sentiments) if sentiments else 0

                    trends.append(
                        {
                            "timestamp": hour_key.isoformat(),
                            "sentiment_score": avg_sentiment,
                            "tweet_count": len(tweets),
                            "confidence": min(len(tweets) / 10, 1.0),
                        }
                    )

        return {
            "symbol": symbol,
            "trends": sorted(trends, key=lambda x: x["timestamp"]),
            "source": "twitter",
            "period_hours": hours,
        }

    def add_sentiment_callback(self, callback: Callable) -> None:
        """Add callback for real-time sentiment updates"""
        self.sentiment_callbacks.append(callback)

    def track_symbols(self, symbols: list[str]) -> None:
        """Update tracked symbols"""
        self._tracked_symbols = set(symbols)

        # Update streaming rules
        if self.stream_session:
            asyncio.create_task(self._setup_stream_rules())


class TwitterRateLimiter:
    """Rate limiter for Twitter API v2"""

    def __init__(self):
        self.limits = {
            "stream": {"calls": 50, "window": 900},  # 50 per 15 min
            "lookup": {"calls": 300, "window": 900},  # 300 per 15 min
        }
        self.calls = defaultdict(list)

    async def check_limit(self, endpoint: str) -> None:
        """Check if rate limit allows the call"""
        if endpoint not in self.limits:
            return

        limit = self.limits[endpoint]
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=limit["window"])

        # Remove old calls
        self.calls[endpoint] = [
            call_time for call_time in self.calls[endpoint] if call_time > window_start
        ]

        # Check limit
        if len(self.calls[endpoint]) >= limit["calls"]:
            wait_time = (
                self.calls[endpoint][0] + timedelta(seconds=limit["window"]) - now
            ).total_seconds()
            if wait_time > 0:
                logger.warning(
                    f"Rate limit reached for {endpoint}, waiting {wait_time}s"
                )
                await asyncio.sleep(wait_time)

        # Record call
        self.calls[endpoint].append(now)
