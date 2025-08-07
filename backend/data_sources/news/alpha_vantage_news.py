from datetime import datetime, timedelta
from typing import Any

import aiohttp
import pytz
from loguru import logger

from backend.data_sources.base import DataSourceConfig, DataSourceStatus

from .base import NewsArticle, NewsCategory, NewsDataSource, NewsSentiment


class AlphaVantageNewsSource(NewsDataSource):
    """Alpha Vantage News & Sentiments API integration"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.api_key = config.credentials.get("api_key")
        self.base_url = "https://www.alphavantage.co/query"
        self.session: aiohttp.ClientSession | None = None

    async def connect(self) -> bool:
        """Connect to Alpha Vantage API"""
        try:
            if not self.api_key:
                raise ValueError("Alpha Vantage API key not provided")

            self.session = aiohttp.ClientSession()

            # Test connection with a simple request
            test_params = {
                "function": "NEWS_SENTIMENT",
                "tickers": "AAPL",
                "apikey": self.api_key,
                "limit": 1,
            }

            async with self.session.get(self.base_url, params=test_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "Error Message" in data or "Note" in data:
                        raise Exception(f"API error: {data}")

                    self.update_health_status(DataSourceStatus.HEALTHY)
                    logger.info("Connected to Alpha Vantage News API")
                    return True
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Failed to connect to Alpha Vantage: {e}")
            self.update_health_status(DataSourceStatus.UNHEALTHY, str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from Alpha Vantage API"""
        if self.session:
            await self.session.close()
        self.update_health_status(DataSourceStatus.DISCONNECTED)

    async def fetch_news(
        self,
        symbols: list[str] | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        categories: list[NewsCategory] | None = None,
    ) -> list[NewsArticle]:
        """Fetch news articles from Alpha Vantage"""
        try:
            articles = []

            # Alpha Vantage requires ticker symbols
            if not symbols:
                logger.warning("Alpha Vantage requires symbols for news queries")
                return articles

            # API parameters
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ",".join(symbols[:50]),  # Max 50 tickers
                "apikey": self.api_key,
                "limit": min(self.max_articles_per_request, 1000),
            }

            # Add time range if specified
            if from_date:
                params["time_from"] = from_date.strftime("%Y%m%dT%H%M")
            if to_date:
                params["time_to"] = to_date.strftime("%Y%m%dT%H%M")

            # Add topics filter if categories specified
            if categories:
                topics = self._map_categories_to_topics(categories)
                if topics:
                    params["topics"] = ",".join(topics)

            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Alpha Vantage API error: HTTP {response.status}")
                    return articles

                data = await response.json()

                if "Error Message" in data:
                    logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                    return articles

                if "Note" in data:
                    logger.warning(f"Alpha Vantage API note: {data['Note']}")
                    return articles

                # Parse feed items
                feed = data.get("feed", [])
                for item in feed:
                    article = self._parse_news_item(item, symbols)
                    if article:
                        articles.append(article)

            # Sort by published date
            articles.sort(key=lambda x: x.published_at, reverse=True)

            return articles

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            self.update_health_status(DataSourceStatus.DEGRADED, str(e))
            return []

    async def fetch_breaking_news(
        self, symbols: list[str] | None = None, minutes: int = 15
    ) -> list[NewsArticle]:
        """Fetch breaking/recent news from Alpha Vantage"""
        from_date = datetime.utcnow() - timedelta(minutes=minutes)
        return await self.fetch_news(symbols=symbols, from_date=from_date)

    def _parse_news_item(
        self, item: dict[str, Any], requested_symbols: list[str]
    ) -> NewsArticle | None:
        """Parse Alpha Vantage news item into NewsArticle"""
        try:
            # Extract basic info
            title = item.get("title", "")
            url = item.get("url", "")

            if not title or not url:
                return None

            # Parse timestamp
            time_published = item.get("time_published", "")
            published_at = self._parse_timestamp(time_published)

            # Extract symbols from ticker sentiment
            symbols = []
            ticker_sentiment = item.get("ticker_sentiment", [])
            for ticker_info in ticker_sentiment:
                symbol = ticker_info.get("ticker")
                if symbol:
                    symbols.append(symbol)

            # Create article
            article = NewsArticle(
                id=f"av_{item.get('url', '').split('/')[-1][:20]}",
                title=title,
                summary=item.get("summary", ""),
                content=item.get("summary", ""),  # AV doesn't provide full content
                url=url,
                source=item.get("source", "Unknown"),
                source_domain=item.get("source_domain", ""),
                published_at=published_at,
                symbols=symbols,
                author=None,  # Not provided by AV
                image_url=item.get("banner_image", None),
            )

            # Analyze sentiment from ticker sentiment data
            self._analyze_ticker_sentiment(article, ticker_sentiment)

            # Calculate relevance
            article.relevance_score = self.calculate_relevance_score(
                article, requested_symbols
            )

            # Detect categories
            article.categories = self.categorize_article(article)

            # Detect market impact
            article.market_impact = self.detect_market_impact(article)

            # Extract topics as tags
            topics = item.get("topics", [])
            article.tags = [
                topic.get("topic", "") for topic in topics if topic.get("topic")
            ]

            return article

        except Exception as e:
            logger.error(f"Error parsing Alpha Vantage news item: {e}")
            return None

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse Alpha Vantage timestamp format (YYYYMMDDThhmmss)"""
        try:
            # Format: 20240121T143000
            dt = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
            # Assume UTC
            return dt.replace(tzinfo=pytz.UTC)
        except:
            return datetime.utcnow().replace(tzinfo=pytz.UTC)

    def _analyze_ticker_sentiment(
        self, article: NewsArticle, ticker_sentiment: list[dict[str, Any]]
    ):
        """Analyze sentiment from Alpha Vantage ticker sentiment data"""
        if not ticker_sentiment:
            return

        # Average sentiment scores across all tickers
        sentiment_scores = []
        relevance_scores = []

        for ticker_info in ticker_sentiment:
            # Sentiment score (-1 to 1)
            sentiment_score = float(ticker_info.get("ticker_sentiment_score", 0))
            sentiment_scores.append(sentiment_score)

            # Relevance score (0 to 1)
            relevance = float(ticker_info.get("relevance_score", 0))
            relevance_scores.append(relevance)

        if sentiment_scores:
            # Average sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            article.sentiment_score = avg_sentiment

            # Map to sentiment enum
            if avg_sentiment >= 0.5:
                article.sentiment = NewsSentiment.VERY_BULLISH
            elif avg_sentiment >= 0.2:
                article.sentiment = NewsSentiment.BULLISH
            elif avg_sentiment <= -0.5:
                article.sentiment = NewsSentiment.VERY_BEARISH
            elif avg_sentiment <= -0.2:
                article.sentiment = NewsSentiment.BEARISH
            else:
                article.sentiment = NewsSentiment.NEUTRAL

            # Confidence based on consistency
            if len(sentiment_scores) > 1:
                # Check how consistent the sentiments are
                std_dev = self._calculate_std_dev(sentiment_scores)
                # Lower std dev = higher confidence
                article.sentiment_confidence = max(0, 1 - std_dev)
            else:
                article.sentiment_confidence = 0.7  # Default for single ticker

        if relevance_scores:
            # Use max relevance as article relevance
            article.relevance_score = max(relevance_scores)

    def _calculate_std_dev(self, values: list[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    def _map_categories_to_topics(self, categories: list[NewsCategory]) -> list[str]:
        """Map our categories to Alpha Vantage topics"""
        topic_mapping = {
            NewsCategory.EARNINGS: "earnings",
            NewsCategory.MERGER_ACQUISITION: "mergers_and_acquisitions",
            NewsCategory.REGULATORY: "finance",
            NewsCategory.PRODUCT_LAUNCH: "technology",
            NewsCategory.FINANCIAL_RESULTS: "earnings",
            NewsCategory.MARKET_ANALYSIS: "finance",
        }

        topics = []
        for category in categories:
            topic = topic_mapping.get(category)
            if topic and topic not in topics:
                topics.append(topic)

        return topics
