import hashlib
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from backend.data_sources.base import BaseDataSource, DataSourceConfig, DataSourceType


class NewsSentiment(Enum):
    """News sentiment classification"""

    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class NewsCategory(Enum):
    """News category classification"""

    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY = "regulatory"
    MARKET_ANALYSIS = "market_analysis"
    EXECUTIVE_CHANGE = "executive_change"
    FINANCIAL_RESULTS = "financial_results"
    GUIDANCE = "guidance"
    PARTNERSHIP = "partnership"
    GENERAL = "general"


class MarketImpact(Enum):
    """Expected market impact level"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class NewsArticle:
    """Represents a financial news article"""

    id: str
    title: str
    summary: str
    content: str
    url: str
    source: str
    source_domain: str
    published_at: datetime
    symbols: list[str]

    # Sentiment analysis
    sentiment: NewsSentiment = NewsSentiment.NEUTRAL
    sentiment_score: float = 0.0  # -1 to 1
    sentiment_confidence: float = 0.5  # 0 to 1

    # Relevance and impact
    relevance_score: float = 0.5  # 0 to 1
    market_impact: MarketImpact = MarketImpact.LOW
    categories: list[NewsCategory] = field(default_factory=list)

    # Source metadata
    author: str | None = None
    image_url: str | None = None
    tags: list[str] = field(default_factory=list)

    # Tracking
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    hash: str | None = None

    def __post_init__(self):
        """Generate hash for deduplication"""
        if not self.hash:
            content = f"{self.title}_{self.source}_{self.published_at.isoformat()}"
            self.hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "source_domain": self.source_domain,
            "published_at": self.published_at.isoformat(),
            "symbols": self.symbols,
            "sentiment": self.sentiment.value,
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "relevance_score": self.relevance_score,
            "market_impact": self.market_impact.value,
            "categories": [cat.value for cat in self.categories],
            "author": self.author,
            "image_url": self.image_url,
            "tags": self.tags,
            "retrieved_at": self.retrieved_at.isoformat(),
            "hash": self.hash,
        }


class NewsDataSource(BaseDataSource):
    """Base class for news data sources"""

    def __init__(self, config: DataSourceConfig):
        # Override source type to NEWS
        config.source_type = DataSourceType.NEWS
        super().__init__(config)

        # News-specific configuration
        self.max_articles_per_request = config.extra_params.get("max_articles", 100)
        self.lookback_hours = config.extra_params.get("lookback_hours", 24)
        self.min_relevance_score = config.extra_params.get("min_relevance", 0.3)

        # Tracking
        self.articles_cache = {}
        self.last_fetch_time = {}

    @abstractmethod
    async def fetch_news(
        self,
        symbols: list[str] | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        categories: list[NewsCategory] | None = None,
    ) -> list[NewsArticle]:
        """Fetch news articles from the source"""
        pass

    @abstractmethod
    async def fetch_breaking_news(
        self, symbols: list[str] | None = None, minutes: int = 15
    ) -> list[NewsArticle]:
        """Fetch breaking/recent news"""
        pass

    async def search_news(
        self,
        query: str,
        symbols: list[str] | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[NewsArticle]:
        """Search news articles"""
        # Default implementation using fetch_news
        articles = await self.fetch_news(symbols, from_date, to_date)

        # Filter by query
        query_lower = query.lower()
        filtered = [
            article
            for article in articles
            if query_lower in article.title.lower()
            or query_lower in article.summary.lower()
            or (article.content and query_lower in article.content.lower())
        ]

        return filtered

    def calculate_relevance_score(
        self, article: NewsArticle, target_symbols: list[str]
    ) -> float:
        """Calculate relevance score for an article"""
        score = 0.0

        # Symbol match scoring
        if target_symbols:
            symbol_matches = sum(
                1 for symbol in target_symbols if symbol in article.symbols
            )
            symbol_score = symbol_matches / len(target_symbols)
            score += symbol_score * 0.4
        else:
            score += 0.2  # Base score if no specific symbols

        # Title relevance
        title_keywords = [
            "earnings",
            "revenue",
            "profit",
            "loss",
            "guidance",
            "merger",
            "acquisition",
            "fda",
            "sec",
            "lawsuit",
        ]
        title_matches = sum(
            1 for keyword in title_keywords if keyword in article.title.lower()
        )
        score += min(title_matches * 0.1, 0.3)

        # Category importance
        important_categories = [
            NewsCategory.EARNINGS,
            NewsCategory.MERGER_ACQUISITION,
            NewsCategory.REGULATORY,
            NewsCategory.GUIDANCE,
        ]
        category_matches = sum(
            1 for cat in article.categories if cat in important_categories
        )
        score += min(category_matches * 0.15, 0.3)

        return min(score, 1.0)

    def detect_market_impact(self, article: NewsArticle) -> MarketImpact:
        """Detect potential market impact of news"""
        # High impact keywords
        high_impact = [
            "bankruptcy",
            "fraud",
            "sec investigation",
            "fda approval",
            "fda rejection",
            "merger",
            "acquisition",
            "buyout",
            "earnings beat",
            "earnings miss",
            "guidance raised",
            "guidance lowered",
            "ceo resignation",
            "major partnership",
        ]

        # Medium impact keywords
        medium_impact = [
            "quarterly results",
            "product launch",
            "expansion",
            "restructuring",
            "dividend",
            "share buyback",
            "analyst upgrade",
            "analyst downgrade",
        ]

        content_lower = f"{article.title} {article.summary}".lower()

        # Check for high impact
        for keyword in high_impact:
            if keyword in content_lower:
                return MarketImpact.HIGH

        # Check for medium impact
        for keyword in medium_impact:
            if keyword in content_lower:
                return MarketImpact.MEDIUM

        # Check sentiment extremes
        if abs(article.sentiment_score) > 0.7:
            return MarketImpact.MEDIUM

        return MarketImpact.LOW

    def categorize_article(self, article: NewsArticle) -> list[NewsCategory]:
        """Categorize news article"""
        categories = []
        content_lower = f"{article.title} {article.summary}".lower()

        # Category keywords mapping
        category_keywords = {
            NewsCategory.EARNINGS: [
                "earnings",
                "revenue",
                "profit",
                "loss",
                "eps",
                "quarterly results",
            ],
            NewsCategory.MERGER_ACQUISITION: [
                "merger",
                "acquisition",
                "acquire",
                "buyout",
                "takeover",
            ],
            NewsCategory.PRODUCT_LAUNCH: [
                "launch",
                "unveil",
                "introduce",
                "new product",
                "release",
            ],
            NewsCategory.REGULATORY: [
                "sec",
                "fda",
                "regulatory",
                "investigation",
                "compliance",
                "lawsuit",
            ],
            NewsCategory.EXECUTIVE_CHANGE: [
                "ceo",
                "cfo",
                "executive",
                "resignation",
                "appointment",
                "hire",
            ],
            NewsCategory.FINANCIAL_RESULTS: [
                "financial results",
                "fiscal year",
                "annual report",
            ],
            NewsCategory.GUIDANCE: ["guidance", "forecast", "outlook", "projection"],
            NewsCategory.PARTNERSHIP: [
                "partnership",
                "collaboration",
                "joint venture",
                "agreement",
            ],
        }

        for category, keywords in category_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                categories.append(category)

        if not categories:
            categories.append(NewsCategory.GENERAL)

        return categories


class NewsSourceReliability:
    """Tracks and manages news source reliability"""

    def __init__(self):
        self.source_scores = {}
        self.source_article_count = {}
        self.source_accuracy_tracking = {}

        # Predefined reliability scores for known sources
        self.base_scores = {
            "reuters.com": 0.95,
            "bloomberg.com": 0.95,
            "wsj.com": 0.93,
            "ft.com": 0.92,
            "cnbc.com": 0.88,
            "marketwatch.com": 0.85,
            "seekingalpha.com": 0.75,
            "yahoo.com": 0.80,
            "benzinga.com": 0.70,
        }

    def get_reliability_score(self, source_domain: str) -> float:
        """Get reliability score for a news source"""
        # Check if we have a tracked score
        if source_domain in self.source_scores:
            return self.source_scores[source_domain]

        # Check base scores
        if source_domain in self.base_scores:
            return self.base_scores[source_domain]

        # Default score for unknown sources
        return 0.60

    def update_source_performance(
        self,
        source_domain: str,
        was_accurate: bool,
        impact_predicted: MarketImpact,
        impact_actual: MarketImpact | None = None,
    ):
        """Update source reliability based on performance"""
        if source_domain not in self.source_accuracy_tracking:
            self.source_accuracy_tracking[source_domain] = {
                "correct": 0,
                "total": 0,
                "impact_accuracy": [],
            }

        tracking = self.source_accuracy_tracking[source_domain]
        tracking["total"] += 1

        if was_accurate:
            tracking["correct"] += 1

        if impact_actual:
            tracking["impact_accuracy"].append(impact_predicted == impact_actual)

        # Update reliability score
        if tracking["total"] >= 10:  # Minimum articles before adjusting
            accuracy_rate = tracking["correct"] / tracking["total"]
            base_score = self.base_scores.get(source_domain, 0.60)

            # Blend base score with observed accuracy
            self.source_scores[source_domain] = base_score * 0.3 + accuracy_rate * 0.7

    def rank_sources(self, sources: list[str]) -> list[tuple[str, float]]:
        """Rank sources by reliability"""
        ranked = []
        for source in sources:
            score = self.get_reliability_score(source)
            ranked.append((source, score))

        return sorted(ranked, key=lambda x: x[1], reverse=True)
