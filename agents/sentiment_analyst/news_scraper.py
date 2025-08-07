"""News scraper module for collecting financial news from various sources."""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import feedparser
from loguru import logger


class NewsScraper:
    """Scrape financial news from RSS feeds and news APIs."""

    def __init__(self):
        """Initialize the news scraper."""
        # RSS feeds for financial news
        self.rss_feeds = {
            "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
            "bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
            "cnbc_top": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories",
            "yahoo_finance": "https://finance.yahoo.com/rss/",
            "seeking_alpha": "https://seekingalpha.com/feed.xml",
            "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        }

        # News API configurations (can be extended with API keys)
        self.news_apis = {
            "newsapi": {
                "base_url": "https://newsapi.org/v2",
                "key": None,  # Set via environment variable
            },
            "alphavantage": {
                "base_url": "https://www.alphavantage.co/query",
                "key": None,  # Set via environment variable
            },
        }

        # Source reliability weights
        self.source_weights = {
            "reuters": 1.0,
            "bloomberg": 0.95,
            "wsj": 0.95,
            "cnbc": 0.85,
            "marketwatch": 0.8,
            "yahoo": 0.75,
            "seekingalpha": 0.7,
        }

        self.session = None
        logger.info("NewsScraper initialized")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def scrape_symbol_news(
        self, symbol: str, lookback_hours: int = 24
    ) -> list[dict[str, Any]]:
        """
        Scrape news for a specific stock symbol.

        Args:
            symbol: Stock symbol
            lookback_hours: Hours of historical news to fetch

        Returns:
            List of news articles with metadata
        """
        try:
            articles = []
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)

            # Create session if not exists
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Scrape RSS feeds
            rss_tasks = []
            for source_name, feed_url in self.rss_feeds.items():
                task = self._scrape_rss_feed(feed_url, symbol, cutoff_time, source_name)
                rss_tasks.append(task)

            rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)

            for result in rss_results:
                if isinstance(result, list):
                    articles.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"RSS scraping error: {result}")

            # Additional API calls if configured
            if self.news_apis["newsapi"]["key"]:
                api_articles = await self._fetch_newsapi(symbol, lookback_hours)
                articles.extend(api_articles)

            # Sort by timestamp and remove duplicates
            articles = self._deduplicate_articles(articles)
            articles.sort(key=lambda x: x["timestamp"], reverse=True)

            logger.info(f"Scraped {len(articles)} articles for {symbol}")
            return articles

        except Exception as e:
            logger.error(f"Error scraping news for {symbol}: {str(e)}")
            return []

    async def scrape_market_news(self) -> list[dict[str, Any]]:
        """
        Scrape general market news.

        Returns:
            List of market news articles
        """
        try:
            articles = []
            cutoff_time = datetime.now() - timedelta(hours=6)

            if not self.session:
                self.session = aiohttp.ClientSession()

            # Scrape all RSS feeds without symbol filter
            rss_tasks = []
            for source_name, feed_url in self.rss_feeds.items():
                task = self._scrape_rss_feed(feed_url, None, cutoff_time, source_name)
                rss_tasks.append(task)

            rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)

            for result in rss_results:
                if isinstance(result, list):
                    articles.extend(result)

            # Filter for market-related keywords
            market_keywords = [
                "market",
                "dow",
                "nasdaq",
                "s&p",
                "economy",
                "fed",
                "inflation",
            ]
            market_articles = []

            for article in articles:
                text = (article["title"] + " " + article.get("description", "")).lower()
                if any(keyword in text for keyword in market_keywords):
                    market_articles.append(article)

            # Sort and deduplicate
            market_articles = self._deduplicate_articles(market_articles)
            market_articles.sort(key=lambda x: x["timestamp"], reverse=True)

            return market_articles[:50]  # Return top 50 articles

        except Exception as e:
            logger.error(f"Error scraping market news: {str(e)}")
            return []

    async def scrape_sector_news(self, sector: str) -> list[dict[str, Any]]:
        """
        Scrape news for a specific sector.

        Args:
            sector: Sector name (e.g., 'technology', 'healthcare')

        Returns:
            List of sector news articles
        """
        try:
            # Define sector keywords
            sector_keywords = {
                "technology": [
                    "tech",
                    "software",
                    "hardware",
                    "semiconductor",
                    "ai",
                    "cloud",
                ],
                "healthcare": ["health", "pharma", "biotech", "medical", "drug", "fda"],
                "finance": ["bank", "financial", "insurance", "fintech", "payment"],
                "energy": ["oil", "gas", "renewable", "solar", "wind", "energy"],
                "consumer": [
                    "retail",
                    "consumer",
                    "e-commerce",
                    "restaurant",
                    "apparel",
                ],
                "industrial": [
                    "manufacturing",
                    "industrial",
                    "aerospace",
                    "defense",
                    "machinery",
                ],
            }

            keywords = sector_keywords.get(sector.lower(), [sector.lower()])

            # Get general market news
            all_articles = await self.scrape_market_news()

            # Filter for sector-specific articles
            sector_articles = []
            for article in all_articles:
                text = (article["title"] + " " + article.get("description", "")).lower()
                if any(keyword in text for keyword in keywords):
                    article["sector"] = sector
                    sector_articles.append(article)

            return sector_articles

        except Exception as e:
            logger.error(f"Error scraping sector news for {sector}: {str(e)}")
            return []

    async def _scrape_rss_feed(
        self,
        feed_url: str,
        symbol: str | None,
        cutoff_time: datetime,
        source_name: str,
    ) -> list[dict[str, Any]]:
        """Scrape articles from an RSS feed."""
        try:
            async with self.session.get(feed_url, timeout=10) as response:
                content = await response.text()

            feed = feedparser.parse(content)
            articles = []

            for entry in feed.entries:
                # Parse publication date
                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime.fromtimestamp(
                        datetime(*entry.published_parsed[:6]).timestamp()
                    )
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    pub_date = datetime.fromtimestamp(
                        datetime(*entry.updated_parsed[:6]).timestamp()
                    )

                # Skip old articles
                if pub_date and pub_date < cutoff_time:
                    continue

                # Get article text
                title = entry.get("title", "")
                description = entry.get("summary", "")
                content = title + " " + description

                # Filter by symbol if specified
                if symbol and symbol.upper() not in content.upper():
                    continue

                # Extract source reliability
                source_key = None
                for key in self.source_weights:
                    if key in source_name.lower():
                        source_key = key
                        break

                article = {
                    "title": title,
                    "description": self._clean_html(description),
                    "url": entry.get("link", ""),
                    "source": source_name,
                    "timestamp": pub_date or datetime.now(),
                    "reliability_weight": self.source_weights.get(source_key, 0.5),
                    "tags": self._extract_tags(content),
                }

                articles.append(article)

            return articles

        except Exception as e:
            logger.error(f"Error scraping RSS feed {feed_url}: {str(e)}")
            return []

    async def _fetch_newsapi(
        self, symbol: str, lookback_hours: int
    ) -> list[dict[str, Any]]:
        """Fetch news from News API."""
        try:
            if not self.news_apis["newsapi"]["key"]:
                return []

            from_date = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()

            params = {
                "q": symbol,
                "from": from_date,
                "sortBy": "relevancy",
                "apiKey": self.news_apis["newsapi"]["key"],
                "language": "en",
                "domains": "reuters.com,bloomberg.com,wsj.com,cnbc.com",
            }

            url = f"{self.news_apis['newsapi']['base_url']}/everything"

            async with self.session.get(url, params=params) as response:
                data = await response.json()

            if data.get("status") != "ok":
                logger.error(f"News API error: {data.get('message')}")
                return []

            articles = []
            for item in data.get("articles", []):
                article = {
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", {}).get("name", "NewsAPI"),
                    "timestamp": datetime.fromisoformat(
                        item.get("publishedAt", "").replace("Z", "+00:00")
                    ),
                    "reliability_weight": 0.8,
                    "tags": self._extract_tags(
                        item.get("title", "") + " " + item.get("description", "")
                    ),
                }
                articles.append(article)

            return articles

        except Exception as e:
            logger.error(f"Error fetching from News API: {str(e)}")
            return []

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Remove HTML tags
        clean_text = re.sub("<.*?>", "", text)
        # Remove extra whitespace
        clean_text = " ".join(clean_text.split())
        return clean_text

    def _extract_tags(self, text: str) -> list[str]:
        """Extract relevant tags from article text."""
        tags = []
        text_lower = text.lower()

        # Common financial terms to tag
        tag_patterns = {
            "earnings": r"\b(earnings|revenue|profit|loss)\b",
            "merger": r"\b(merger|acquisition|acquire|takeover)\b",
            "ipo": r"\b(ipo|initial public offering)\b",
            "regulation": r"\b(sec|regulation|regulatory|compliance)\b",
            "analyst": r"\b(upgrade|downgrade|analyst|rating)\b",
            "dividend": r"\b(dividend|distribution|yield)\b",
            "guidance": r"\b(guidance|forecast|outlook|projection)\b",
        }

        for tag, pattern in tag_patterns.items():
            if re.search(pattern, text_lower):
                tags.append(tag)

        return tags

    def _deduplicate_articles(
        self, articles: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate articles based on title similarity."""
        unique_articles = []
        seen_titles = set()

        for article in articles:
            # Create normalized title for comparison
            normalized_title = re.sub(r"[^\w\s]", "", article["title"].lower())
            normalized_title = " ".join(normalized_title.split())

            # Check for similar titles
            is_duplicate = False
            for seen in seen_titles:
                # Calculate simple similarity
                if self._calculate_similarity(normalized_title, seen) > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(normalized_title)

        return unique_articles

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def get_status(self) -> dict[str, Any]:
        """Get the status of the news scraper."""
        return {
            "status": "active",
            "rss_feeds": len(self.rss_feeds),
            "configured_apis": sum(1 for api in self.news_apis.values() if api["key"]),
            "session_active": self.session is not None and not self.session.closed,
        }
