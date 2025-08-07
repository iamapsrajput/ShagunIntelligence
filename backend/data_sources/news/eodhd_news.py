from datetime import datetime, timedelta
from typing import Any

import aiohttp
import pytz
from loguru import logger

from backend.data_sources.base import DataSourceConfig, DataSourceStatus

from .base import MarketImpact, NewsArticle, NewsCategory, NewsDataSource, NewsSentiment


class EODHDNewsSource(NewsDataSource):
    """EODHD Financial News API integration"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.api_key = config.credentials.get("api_key")
        self.base_url = "https://eodhistoricaldata.com/api"
        self.session: aiohttp.ClientSession | None = None

    async def connect(self) -> bool:
        """Connect to EODHD API"""
        try:
            if not self.api_key:
                raise ValueError("EODHD API key not provided")

            self.session = aiohttp.ClientSession()

            # Test connection with a simple request
            test_url = f"{self.base_url}/news"
            test_params = {
                "api_token": self.api_key,
                "s": "AAPL.US",
                "limit": 1,
                "fmt": "json",
            }

            async with self.session.get(test_url, params=test_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict) and "error" in data:
                        raise Exception(f"API error: {data['error']}")

                    self.update_health_status(DataSourceStatus.HEALTHY)
                    logger.info("Connected to EODHD News API")
                    return True
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Failed to connect to EODHD: {e}")
            self.update_health_status(DataSourceStatus.UNHEALTHY, str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from EODHD API"""
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
        """Fetch news articles from EODHD"""
        try:
            articles = []

            # EODHD endpoints
            if symbols:
                # Fetch symbol-specific news
                for symbol in symbols[:10]:  # Limit to avoid too many requests
                    symbol_articles = await self._fetch_symbol_news(
                        symbol, from_date, to_date
                    )
                    articles.extend(symbol_articles)
            else:
                # Fetch general financial news
                general_articles = await self._fetch_general_news(from_date, to_date)
                articles.extend(general_articles)

            # Deduplicate by URL
            seen_urls = set()
            unique_articles = []
            for article in articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    unique_articles.append(article)

            # Filter by categories if specified
            if categories:
                unique_articles = [
                    article
                    for article in unique_articles
                    if any(cat in article.categories for cat in categories)
                ]

            # Sort by published date
            unique_articles.sort(key=lambda x: x.published_at, reverse=True)

            # Limit to max articles
            return unique_articles[: self.max_articles_per_request]

        except Exception as e:
            logger.error(f"Error fetching EODHD news: {e}")
            self.update_health_status(DataSourceStatus.DEGRADED, str(e))
            return []

    async def fetch_breaking_news(
        self, symbols: list[str] | None = None, minutes: int = 15
    ) -> list[NewsArticle]:
        """Fetch breaking/recent news from EODHD"""
        from_date = datetime.utcnow() - timedelta(minutes=minutes)
        articles = await self.fetch_news(symbols=symbols, from_date=from_date)

        # Filter for high-impact recent news
        breaking = [
            article
            for article in articles
            if article.market_impact in [MarketImpact.HIGH, MarketImpact.MEDIUM]
        ]

        return breaking

    async def _fetch_symbol_news(
        self,
        symbol: str,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[NewsArticle]:
        """Fetch news for specific symbol"""
        try:
            # EODHD uses .US suffix for US stocks
            if "." not in symbol:
                symbol = f"{symbol}.US"

            url = f"{self.base_url}/news"
            params = {
                "api_token": self.api_key,
                "s": symbol,
                "limit": 50,
                "fmt": "json",
            }

            if from_date:
                params["from"] = from_date.strftime("%Y-%m-%d")
            if to_date:
                params["to"] = to_date.strftime("%Y-%m-%d")

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"EODHD API error: HTTP {response.status}")
                    return []

                data = await response.json()

                if isinstance(data, dict) and "error" in data:
                    logger.error(f"EODHD API error: {data['error']}")
                    return []

                articles = []
                for item in data:
                    article = self._parse_news_item(item, [symbol.split(".")[0]])
                    if article:
                        articles.append(article)

                return articles

        except Exception as e:
            logger.error(f"Error fetching EODHD symbol news: {e}")
            return []

    async def _fetch_general_news(
        self, from_date: datetime | None = None, to_date: datetime | None = None
    ) -> list[NewsArticle]:
        """Fetch general financial news"""
        try:
            url = f"{self.base_url}/news"
            params = {"api_token": self.api_key, "limit": 100, "fmt": "json"}

            if from_date:
                params["from"] = from_date.strftime("%Y-%m-%d")
            if to_date:
                params["to"] = to_date.strftime("%Y-%m-%d")

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"EODHD API error: HTTP {response.status}")
                    return []

                data = await response.json()

                if isinstance(data, dict) and "error" in data:
                    logger.error(f"EODHD API error: {data['error']}")
                    return []

                articles = []
                for item in data:
                    article = self._parse_news_item(item, [])
                    if article:
                        articles.append(article)

                return articles

        except Exception as e:
            logger.error(f"Error fetching EODHD general news: {e}")
            return []

    def _parse_news_item(
        self, item: dict[str, Any], requested_symbols: list[str]
    ) -> NewsArticle | None:
        """Parse EODHD news item into NewsArticle"""
        try:
            # Extract basic info
            title = item.get("title", "")
            link = item.get("link", "")

            if not title or not link:
                return None

            # Parse timestamp
            date_str = item.get("date", "")
            published_at = self._parse_timestamp(date_str)

            # Extract symbols
            symbols = []
            related_symbols = item.get("symbols", [])
            if isinstance(related_symbols, list):
                symbols = [s.split(".")[0] for s in related_symbols]
            elif requested_symbols:
                symbols = requested_symbols

            # Create article
            article = NewsArticle(
                id=f"eodhd_{link.split('/')[-1][:20]}",
                title=title,
                summary=item.get("content", "")[:500],  # First 500 chars as summary
                content=item.get("content", ""),
                url=link,
                source="EODHD",
                source_domain="eodhistoricaldata.com",
                published_at=published_at,
                symbols=symbols,
                author=None,
                image_url=None,
            )

            # Analyze sentiment from content
            self._analyze_content_sentiment(article)

            # Calculate relevance
            article.relevance_score = self.calculate_relevance_score(
                article, requested_symbols
            )

            # Detect categories
            article.categories = self.categorize_article(article)

            # Detect market impact
            article.market_impact = self.detect_market_impact(article)

            # Extract tags
            tags = item.get("tags", [])
            if isinstance(tags, list):
                article.tags = tags

            return article

        except Exception as e:
            logger.error(f"Error parsing EODHD news item: {e}")
            return None

    def _parse_timestamp(self, date_str: str) -> datetime:
        """Parse EODHD timestamp format"""
        try:
            # Format: "2024-01-21 14:30:00"
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            # Assume UTC
            return dt.replace(tzinfo=pytz.UTC)
        except:
            try:
                # Alternative format: "2024-01-21"
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                return dt.replace(tzinfo=pytz.UTC)
            except:
                return datetime.utcnow().replace(tzinfo=pytz.UTC)

    def _analyze_content_sentiment(self, article: NewsArticle):
        """Analyze sentiment from article content"""
        content = f"{article.title} {article.summary}".lower()

        # Sentiment keywords
        very_bullish_keywords = [
            "soar",
            "surge",
            "skyrocket",
            "breakthrough",
            "record high",
            "exceptional",
            "outstanding",
            "massive growth",
        ]
        bullish_keywords = [
            "rise",
            "gain",
            "increase",
            "positive",
            "growth",
            "profit",
            "beat expectations",
            "upgrade",
            "strong",
            "improve",
        ]
        bearish_keywords = [
            "fall",
            "drop",
            "decline",
            "negative",
            "loss",
            "miss",
            "downgrade",
            "weak",
            "concern",
            "risk",
        ]
        very_bearish_keywords = [
            "crash",
            "plunge",
            "collapse",
            "bankruptcy",
            "crisis",
            "investigation",
            "fraud",
            "disaster",
        ]

        # Count sentiment indicators
        very_bullish_count = sum(
            1 for keyword in very_bullish_keywords if keyword in content
        )
        bullish_count = sum(1 for keyword in bullish_keywords if keyword in content)
        bearish_count = sum(1 for keyword in bearish_keywords if keyword in content)
        very_bearish_count = sum(
            1 for keyword in very_bearish_keywords if keyword in content
        )

        # Calculate sentiment score
        positive_score = very_bullish_count * 2 + bullish_count
        negative_score = very_bearish_count * 2 + bearish_count

        if positive_score + negative_score == 0:
            article.sentiment = NewsSentiment.NEUTRAL
            article.sentiment_score = 0.0
            article.sentiment_confidence = 0.5
        else:
            # Normalize to -1 to 1
            total_score = positive_score + negative_score
            sentiment_score = (positive_score - negative_score) / total_score
            article.sentiment_score = max(-1, min(1, sentiment_score))

            # Map to sentiment enum
            if sentiment_score >= 0.5:
                article.sentiment = NewsSentiment.VERY_BULLISH
            elif sentiment_score >= 0.2:
                article.sentiment = NewsSentiment.BULLISH
            elif sentiment_score <= -0.5:
                article.sentiment = NewsSentiment.VERY_BEARISH
            elif sentiment_score <= -0.2:
                article.sentiment = NewsSentiment.BEARISH
            else:
                article.sentiment = NewsSentiment.NEUTRAL

            # Confidence based on keyword count
            article.sentiment_confidence = min(1.0, total_score / 10)
