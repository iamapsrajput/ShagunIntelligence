from datetime import datetime, timedelta
from typing import Any

import aiohttp
import pytz
from loguru import logger

from backend.data_sources.base import DataSourceConfig, DataSourceStatus

from .base import MarketImpact, NewsArticle, NewsCategory, NewsDataSource, NewsSentiment


class MarketauxNewsSource(NewsDataSource):
    """Marketaux global financial news API integration"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.api_key = config.credentials.get("api_key")
        self.base_url = "https://api.marketaux.com/v1"
        self.session: aiohttp.ClientSession | None = None

    async def connect(self) -> bool:
        """Connect to Marketaux API"""
        try:
            if not self.api_key:
                raise ValueError("Marketaux API key not provided")

            self.session = aiohttp.ClientSession()

            # Test connection
            test_url = f"{self.base_url}/news/all"
            test_params = {"api_token": self.api_key, "limit": 1}

            async with self.session.get(test_url, params=test_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "error" in data:
                        raise Exception(f"API error: {data['error']}")

                    self.update_health_status(DataSourceStatus.HEALTHY)
                    logger.info("Connected to Marketaux News API")
                    return True
                else:
                    raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.error(f"Failed to connect to Marketaux: {e}")
            self.update_health_status(DataSourceStatus.UNHEALTHY, str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from Marketaux API"""
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
        """Fetch news articles from Marketaux"""
        try:
            url = f"{self.base_url}/news/all"

            # Build parameters
            params = {
                "api_token": self.api_key,
                "limit": min(self.max_articles_per_request, 100),
                "language": "en",
                "filter_entities": "true",
                "sort": "published_desc",
            }

            # Add symbols filter
            if symbols:
                params["symbols"] = ",".join(symbols[:20])  # Max 20 symbols

            # Add date range
            if from_date:
                params["published_after"] = from_date.isoformat()
            if to_date:
                params["published_before"] = to_date.isoformat()

            # Add industry/category filter
            if categories:
                industries = self._map_categories_to_industries(categories)
                if industries:
                    params["industries"] = ",".join(industries)

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Marketaux API error: HTTP {response.status}")
                    return []

                data = await response.json()

                if "error" in data:
                    logger.error(f"Marketaux API error: {data['error']}")
                    return []

                articles = []
                news_items = data.get("data", [])

                for item in news_items:
                    article = self._parse_news_item(item, symbols or [])
                    if article:
                        articles.append(article)

                return articles

        except Exception as e:
            logger.error(f"Error fetching Marketaux news: {e}")
            self.update_health_status(DataSourceStatus.DEGRADED, str(e))
            return []

    async def fetch_breaking_news(
        self, symbols: list[str] | None = None, minutes: int = 15
    ) -> list[NewsArticle]:
        """Fetch breaking/recent news from Marketaux"""
        from_date = datetime.utcnow() - timedelta(minutes=minutes)

        # Marketaux supports real-time filtering
        url = f"{self.base_url}/news/all"
        params = {
            "api_token": self.api_key,
            "limit": 50,
            "language": "en",
            "filter_entities": "true",
            "sort": "published_desc",
            "published_after": from_date.isoformat(),
        }

        if symbols:
            params["symbols"] = ",".join(symbols[:10])

        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                articles = []

                for item in data.get("data", []):
                    article = self._parse_news_item(item, symbols or [])
                    if article and article.market_impact in [
                        MarketImpact.HIGH,
                        MarketImpact.MEDIUM,
                    ]:
                        articles.append(article)

                return articles[:20]  # Limit breaking news

        except Exception as e:
            logger.error(f"Error fetching Marketaux breaking news: {e}")
            return []

    async def search_news(
        self,
        query: str,
        symbols: list[str] | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[NewsArticle]:
        """Search news articles with query"""
        try:
            url = f"{self.base_url}/news/all"

            params = {
                "api_token": self.api_key,
                "search": query,
                "limit": 50,
                "language": "en",
                "filter_entities": "true",
                "sort": "published_desc",
            }

            if symbols:
                params["symbols"] = ",".join(symbols[:10])

            if from_date:
                params["published_after"] = from_date.isoformat()
            if to_date:
                params["published_before"] = to_date.isoformat()

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                articles = []

                for item in data.get("data", []):
                    article = self._parse_news_item(item, symbols or [])
                    if article:
                        articles.append(article)

                return articles

        except Exception as e:
            logger.error(f"Error searching Marketaux news: {e}")
            return []

    def _parse_news_item(
        self, item: dict[str, Any], requested_symbols: list[str]
    ) -> NewsArticle | None:
        """Parse Marketaux news item into NewsArticle"""
        try:
            # Extract basic info
            title = item.get("title", "")
            url = item.get("url", "")

            if not title or not url:
                return None

            # Parse timestamp
            published_at = self._parse_timestamp(item.get("published_at", ""))

            # Extract symbols from entities
            symbols = []
            entities = item.get("entities", [])
            for entity in entities:
                if entity.get("type") == "equity" and entity.get("symbol"):
                    symbols.append(entity["symbol"])

            # If no symbols from entities, use requested symbols
            if not symbols and requested_symbols:
                symbols = requested_symbols

            # Create article
            article = NewsArticle(
                id=f"mktx_{item.get('uuid', '')[:20]}",
                title=title,
                summary=item.get("description", ""),
                content=item.get(
                    "description", ""
                ),  # Marketaux doesn't provide full content
                url=url,
                source=item.get("source", "Unknown"),
                source_domain=self._extract_domain(url),
                published_at=published_at,
                symbols=symbols,
                author=item.get("author", None),
                image_url=item.get("image_url", None),
            )

            # Analyze sentiment from Marketaux data
            self._analyze_marketaux_sentiment(article, item)

            # Calculate relevance
            article.relevance_score = self.calculate_relevance_score(
                article, requested_symbols
            )

            # Detect categories
            article.categories = self._extract_categories(item)

            # Detect market impact
            article.market_impact = self._analyze_market_impact(article, item)

            # Extract tags/keywords
            keywords = item.get("keywords", [])
            if isinstance(keywords, list):
                article.tags = keywords

            return article

        except Exception as e:
            logger.error(f"Error parsing Marketaux news item: {e}")
            return None

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse Marketaux ISO timestamp"""
        try:
            # ISO format with timezone
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt
        except:
            return datetime.utcnow().replace(tzinfo=pytz.UTC)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ""

    def _analyze_marketaux_sentiment(self, article: NewsArticle, item: dict[str, Any]):
        """Analyze sentiment from Marketaux data"""
        # Marketaux provides sentiment scores
        sentiment_data = item.get("sentiment", {})

        if sentiment_data:
            # Get sentiment scores
            positive = float(sentiment_data.get("positive", 0))
            negative = float(sentiment_data.get("negative", 0))
            neutral = float(sentiment_data.get("neutral", 0))

            # Calculate overall sentiment
            if positive + negative + neutral > 0:
                sentiment_score = (positive - negative) / (
                    positive + negative + neutral
                )
                article.sentiment_score = sentiment_score

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

                # Confidence based on sentiment strength
                article.sentiment_confidence = max(positive, negative, neutral)
            else:
                # Fallback to content analysis
                self._analyze_content_sentiment(article)
        else:
            # Fallback to content analysis
            self._analyze_content_sentiment(article)

    def _analyze_content_sentiment(self, article: NewsArticle):
        """Analyze sentiment from content (fallback)"""
        content = f"{article.title} {article.summary}".lower()

        # Use similar logic as EODHD
        positive_words = ["gain", "rise", "profit", "beat", "upgrade", "positive"]
        negative_words = ["loss", "fall", "drop", "miss", "downgrade", "negative"]

        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)

        if positive_count + negative_count > 0:
            sentiment_score = (positive_count - negative_count) / (
                positive_count + negative_count
            )
            article.sentiment_score = sentiment_score

            if sentiment_score > 0.3:
                article.sentiment = NewsSentiment.BULLISH
            elif sentiment_score < -0.3:
                article.sentiment = NewsSentiment.BEARISH
            else:
                article.sentiment = NewsSentiment.NEUTRAL

            article.sentiment_confidence = 0.6
        else:
            article.sentiment = NewsSentiment.NEUTRAL
            article.sentiment_score = 0.0
            article.sentiment_confidence = 0.5

    def _extract_categories(self, item: dict[str, Any]) -> list[NewsCategory]:
        """Extract categories from Marketaux data"""
        categories = []

        # Check highlights for category hints
        highlights = item.get("highlights", [])
        for highlight in highlights:
            highlight_type = highlight.get("highlight", "").lower()

            if "earnings" in highlight_type:
                categories.append(NewsCategory.EARNINGS)
            elif "merger" in highlight_type or "acquisition" in highlight_type:
                categories.append(NewsCategory.MERGER_ACQUISITION)
            elif "product" in highlight_type:
                categories.append(NewsCategory.PRODUCT_LAUNCH)

        # Check entities for regulatory
        entities = item.get("entities", [])
        for entity in entities:
            if entity.get("type") == "organization":
                org_name = entity.get("name", "").lower()
                if any(reg in org_name for reg in ["sec", "fda", "ftc", "doj"]):
                    categories.append(NewsCategory.REGULATORY)

        # Use base categorization as well
        base_categories = self.categorize_article(
            NewsArticle(
                id="temp",
                title=item.get("title", ""),
                summary=item.get("description", ""),
                content="",
                url="",
                source="",
                source_domain="",
                published_at=datetime.utcnow(),
                symbols=[],
            )
        )

        categories.extend(base_categories)

        # Deduplicate
        return list(set(categories))

    def _analyze_market_impact(
        self, article: NewsArticle, item: dict[str, Any]
    ) -> MarketImpact:
        """Analyze market impact using Marketaux data"""
        # Check highlights for impact indicators
        highlights = item.get("highlights", [])

        high_impact_highlights = [
            "earnings",
            "merger",
            "acquisition",
            "bankruptcy",
            "investigation",
        ]
        medium_impact_highlights = ["guidance", "analyst", "dividend", "product"]

        for highlight in highlights:
            highlight_text = highlight.get("highlight", "").lower()

            if any(keyword in highlight_text for keyword in high_impact_highlights):
                return MarketImpact.HIGH
            elif any(keyword in highlight_text for keyword in medium_impact_highlights):
                return MarketImpact.MEDIUM

        # Use base detection as fallback
        return self.detect_market_impact(article)

    def _map_categories_to_industries(
        self, categories: list[NewsCategory]
    ) -> list[str]:
        """Map our categories to Marketaux industries"""
        # Marketaux industry codes
        industry_mapping = {
            NewsCategory.EARNINGS: "Financial",
            NewsCategory.REGULATORY: "Government",
            NewsCategory.PRODUCT_LAUNCH: "Technology",
            NewsCategory.FINANCIAL_RESULTS: "Financial",
        }

        industries = []
        for category in categories:
            industry = industry_mapping.get(category)
            if industry and industry not in industries:
                industries.append(industry)

        return industries
