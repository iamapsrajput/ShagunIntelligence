import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
from loguru import logger
import numpy as np
from textblob import TextBlob

from backend.data_sources.base import DataSourceConfig, DataSourceType
from .base import (
    NewsDataSource, NewsArticle, NewsSentiment, NewsCategory, 
    MarketImpact, NewsSourceReliability
)
from .alpha_vantage_news import AlphaVantageNewsSource
from .eodhd_news import EODHDNewsSource
from .marketaux_news import MarketauxNewsSource


class NewsAggregator:
    """Aggregates news from multiple sources with deduplication and analysis"""
    
    def __init__(self):
        self.sources: Dict[str, NewsDataSource] = {}
        self.reliability_tracker = NewsSourceReliability()
        self.article_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.alert_callbacks = []
        self.monitored_symbols = set()
        
        # Deduplication parameters
        self.similarity_threshold = 0.85
        self.time_window_hours = 24
        
        logger.info("Initialized NewsAggregator")
    
    def add_source(self, name: str, source: NewsDataSource):
        """Add a news source"""
        self.sources[name] = source
        logger.info(f"Added news source: {name}")
    
    def remove_source(self, name: str):
        """Remove a news source"""
        if name in self.sources:
            del self.sources[name]
            logger.info(f"Removed news source: {name}")
    
    async def initialize_sources(self):
        """Initialize all configured news sources"""
        # Initialize from environment variables
        import os
        
        # Alpha Vantage
        av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if av_key:
            config = DataSourceConfig(
                name="alpha_vantage_news",
                source_type=DataSourceType.NEWS,
                enabled=True,
                credentials={"api_key": av_key},
                priority=2
            )
            av_source = AlphaVantageNewsSource(config)
            if await av_source.connect():
                self.add_source("alpha_vantage", av_source)
        
        # EODHD
        eodhd_key = os.getenv("EODHD_API_KEY")
        if eodhd_key:
            config = DataSourceConfig(
                name="eodhd_news",
                source_type=DataSourceType.NEWS,
                enabled=True,
                credentials={"api_key": eodhd_key},
                priority=3
            )
            eodhd_source = EODHDNewsSource(config)
            if await eodhd_source.connect():
                self.add_source("eodhd", eodhd_source)
        
        # Marketaux
        marketaux_key = os.getenv("MARKETAUX_API_KEY")
        if marketaux_key:
            config = DataSourceConfig(
                name="marketaux_news",
                source_type=DataSourceType.NEWS,
                enabled=True,
                credentials={"api_key": marketaux_key},
                priority=1  # Highest priority for real-time
            )
            marketaux_source = MarketauxNewsSource(config)
            if await marketaux_source.connect():
                self.add_source("marketaux", marketaux_source)
        
        logger.info(f"Initialized {len(self.sources)} news sources")
    
    async def shutdown(self):
        """Shutdown all news sources"""
        for name, source in self.sources.items():
            try:
                await source.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {name}: {e}")
    
    async def fetch_aggregated_news(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        categories: Optional[List[NewsCategory]] = None,
        min_relevance: float = 0.3,
        deduplicate: bool = True
    ) -> List[NewsArticle]:
        """Fetch and aggregate news from all sources"""
        # Check cache
        cache_key = self._generate_cache_key(symbols, from_date, to_date, categories)
        if cache_key in self.article_cache:
            cached_time, cached_articles = self.article_cache[cache_key]
            if datetime.utcnow() - cached_time < timedelta(seconds=self.cache_ttl):
                return cached_articles
        
        # Fetch from all sources in parallel
        all_articles = []
        fetch_tasks = []
        
        for name, source in self.sources.items():
            if source.health.status.value in ["healthy", "degraded"]:
                task = self._fetch_from_source(
                    name, source, symbols, from_date, to_date, categories
                )
                fetch_tasks.append(task)
        
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error fetching news: {result}")
        
        # Process articles
        if deduplicate:
            all_articles = self._deduplicate_articles(all_articles)
        
        # Enhance sentiment analysis
        for article in all_articles:
            self.sentiment_analyzer.enhance_sentiment(article)
        
        # Apply reliability weighting
        all_articles = self._apply_reliability_weighting(all_articles)
        
        # Filter by relevance
        all_articles = [
            article for article in all_articles
            if article.relevance_score >= min_relevance
        ]
        
        # Sort by impact and recency
        all_articles.sort(
            key=lambda x: (
                self._get_impact_score(x.market_impact),
                x.relevance_score,
                x.published_at
            ),
            reverse=True
        )
        
        # Cache results
        self.article_cache[cache_key] = (datetime.utcnow(), all_articles)
        
        return all_articles
    
    async def fetch_breaking_news(
        self,
        symbols: Optional[List[str]] = None,
        minutes: int = 15
    ) -> List[NewsArticle]:
        """Fetch breaking news from all sources"""
        breaking_articles = []
        
        # Fetch from all sources in parallel
        tasks = []
        for name, source in self.sources.items():
            if source.health.status.value == "healthy":
                task = source.fetch_breaking_news(symbols, minutes)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                breaking_articles.extend(result)
        
        # Deduplicate
        breaking_articles = self._deduplicate_articles(breaking_articles)
        
        # Check for market-moving news
        market_moving = [
            article for article in breaking_articles
            if article.market_impact == MarketImpact.HIGH
        ]
        
        # Trigger alerts
        if market_moving:
            await self._trigger_breaking_news_alerts(market_moving)
        
        return breaking_articles
    
    async def monitor_symbols(self, symbols: List[str]):
        """Start monitoring symbols for news"""
        self.monitored_symbols.update(symbols)
        
        # Start monitoring task if not running
        if not hasattr(self, '_monitor_task') or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_news())
    
    async def _monitor_news(self):
        """Monitor news for tracked symbols"""
        while self.monitored_symbols:
            try:
                # Fetch recent news
                recent_news = await self.fetch_breaking_news(
                    list(self.monitored_symbols),
                    minutes=5
                )
                
                # Check for significant news
                for article in recent_news:
                    if article.market_impact in [MarketImpact.HIGH, MarketImpact.MEDIUM]:
                        await self._trigger_news_alert(article)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in news monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _fetch_from_source(
        self,
        name: str,
        source: NewsDataSource,
        symbols: Optional[List[str]],
        from_date: Optional[datetime],
        to_date: Optional[datetime],
        categories: Optional[List[NewsCategory]]
    ) -> List[NewsArticle]:
        """Fetch news from a single source with error handling"""
        try:
            articles = await source.fetch_news(symbols, from_date, to_date, categories)
            
            # Tag with source name
            for article in articles:
                article.source = f"{article.source} ({name})"
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from {name}: {e}")
            return []
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Deduplicate articles based on similarity"""
        if len(articles) <= 1:
            return articles
        
        # Group by time window
        time_groups = defaultdict(list)
        for article in articles:
            hour_key = article.published_at.replace(minute=0, second=0, microsecond=0)
            time_groups[hour_key].append(article)
        
        # Deduplicate within each time window
        unique_articles = []
        
        for hour, group in time_groups.items():
            # Sort by reliability (prefer better sources)
            group.sort(
                key=lambda x: self.reliability_tracker.get_reliability_score(x.source_domain),
                reverse=True
            )
            
            # Check for duplicates
            added_hashes = set()
            added_articles = []
            
            for article in group:
                # Check exact hash match
                if article.hash not in added_hashes:
                    # Check similarity with existing
                    is_duplicate = False
                    
                    for existing in added_articles:
                        similarity = self._calculate_similarity(article, existing)
                        if similarity >= self.similarity_threshold:
                            is_duplicate = True
                            # Merge information if needed
                            self._merge_article_info(existing, article)
                            break
                    
                    if not is_duplicate:
                        added_hashes.add(article.hash)
                        added_articles.append(article)
            
            unique_articles.extend(added_articles)
        
        return unique_articles
    
    def _calculate_similarity(self, article1: NewsArticle, article2: NewsArticle) -> float:
        """Calculate similarity between two articles"""
        # Title similarity (most important)
        title_sim = self._text_similarity(article1.title, article2.title)
        
        # Time similarity
        time_diff = abs((article1.published_at - article2.published_at).total_seconds())
        time_sim = max(0, 1 - time_diff / (3600 * 3))  # 3 hour window
        
        # Symbol overlap
        symbols1 = set(article1.symbols)
        symbols2 = set(article2.symbols)
        if symbols1 and symbols2:
            symbol_sim = len(symbols1 & symbols2) / len(symbols1 | symbols2)
        else:
            symbol_sim = 0.5
        
        # Weighted similarity
        similarity = (
            title_sim * 0.6 +
            time_sim * 0.2 +
            symbol_sim * 0.2
        )
        
        return similarity
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple word overlap"""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_article_info(self, primary: NewsArticle, secondary: NewsArticle):
        """Merge information from duplicate articles"""
        # Merge symbols
        primary.symbols = list(set(primary.symbols + secondary.symbols))
        
        # Update sentiment if secondary has higher confidence
        if secondary.sentiment_confidence > primary.sentiment_confidence:
            primary.sentiment = secondary.sentiment
            primary.sentiment_score = secondary.sentiment_score
            primary.sentiment_confidence = secondary.sentiment_confidence
        
        # Update categories
        primary.categories = list(set(primary.categories + secondary.categories))
        
        # Update tags
        primary.tags = list(set(primary.tags + secondary.tags))
    
    def _apply_reliability_weighting(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Apply source reliability weighting to articles"""
        for article in articles:
            reliability = self.reliability_tracker.get_reliability_score(article.source_domain)
            
            # Adjust relevance score based on source reliability
            article.relevance_score *= (0.7 + 0.3 * reliability)
            
            # Adjust sentiment confidence
            article.sentiment_confidence *= (0.8 + 0.2 * reliability)
        
        return articles
    
    def _get_impact_score(self, impact: MarketImpact) -> float:
        """Convert market impact to numeric score"""
        scores = {
            MarketImpact.HIGH: 1.0,
            MarketImpact.MEDIUM: 0.7,
            MarketImpact.LOW: 0.4,
            MarketImpact.MINIMAL: 0.1
        }
        return scores.get(impact, 0.0)
    
    def _generate_cache_key(
        self,
        symbols: Optional[List[str]],
        from_date: Optional[datetime],
        to_date: Optional[datetime],
        categories: Optional[List[NewsCategory]]
    ) -> str:
        """Generate cache key for request"""
        key_parts = []
        
        if symbols:
            key_parts.append("_".join(sorted(symbols)))
        if from_date:
            key_parts.append(from_date.isoformat())
        if to_date:
            key_parts.append(to_date.isoformat())
        if categories:
            key_parts.append("_".join(sorted([c.value for c in categories])))
        
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _trigger_breaking_news_alerts(self, articles: List[NewsArticle]):
        """Trigger alerts for breaking news"""
        for callback in self.alert_callbacks:
            try:
                await callback("breaking_news", articles)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    async def _trigger_news_alert(self, article: NewsArticle):
        """Trigger alert for significant news"""
        for callback in self.alert_callbacks:
            try:
                await callback("significant_news", [article])
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def register_alert_callback(self, callback):
        """Register callback for news alerts"""
        self.alert_callbacks.append(callback)
    
    def get_source_stats(self) -> Dict[str, Any]:
        """Get statistics about news sources"""
        stats = {
            "total_sources": len(self.sources),
            "active_sources": sum(
                1 for s in self.sources.values()
                if s.health.status.value == "healthy"
            ),
            "source_health": {
                name: {
                    "status": source.health.status.value,
                    "reliability": self.reliability_tracker.get_reliability_score(
                        source.config.name
                    )
                }
                for name, source in self.sources.items()
            }
        }
        return stats


class NewsSentimentAnalyzer:
    """Enhanced sentiment analysis for financial news"""
    
    def __init__(self):
        self.financial_lexicon = self._load_financial_lexicon()
    
    def enhance_sentiment(self, article: NewsArticle):
        """Enhance sentiment analysis with financial context"""
        # Skip if already has high confidence sentiment
        if article.sentiment_confidence > 0.8:
            return
        
        # Analyze with TextBlob
        try:
            text = f"{article.title} {article.summary}"
            blob = TextBlob(text)
            
            # Get polarity and subjectivity
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Adjust for financial keywords
            financial_adjustment = self._calculate_financial_adjustment(text.lower())
            
            # Combined sentiment
            adjusted_sentiment = np.clip(polarity + financial_adjustment, -1, 1)
            
            # Update article if better than existing
            new_confidence = (1.0 - subjectivity * 0.3) * 0.8  # TextBlob confidence
            
            if new_confidence > article.sentiment_confidence:
                article.sentiment_score = adjusted_sentiment
                article.sentiment_confidence = new_confidence
                
                # Update sentiment label
                if adjusted_sentiment >= 0.5:
                    article.sentiment = NewsSentiment.VERY_BULLISH
                elif adjusted_sentiment >= 0.2:
                    article.sentiment = NewsSentiment.BULLISH
                elif adjusted_sentiment <= -0.5:
                    article.sentiment = NewsSentiment.VERY_BEARISH
                elif adjusted_sentiment <= -0.2:
                    article.sentiment = NewsSentiment.BEARISH
                else:
                    article.sentiment = NewsSentiment.NEUTRAL
                    
        except Exception as e:
            logger.error(f"Error in sentiment enhancement: {e}")
    
    def _load_financial_lexicon(self) -> Dict[str, float]:
        """Load financial sentiment lexicon"""
        return {
            # Very positive
            "beat expectations": 0.3,
            "record profit": 0.3,
            "strong growth": 0.3,
            "upgrade": 0.25,
            "breakthrough": 0.25,
            
            # Positive
            "profit": 0.15,
            "gain": 0.15,
            "rise": 0.15,
            "positive": 0.15,
            "buy": 0.1,
            
            # Negative
            "loss": -0.15,
            "decline": -0.15,
            "fall": -0.15,
            "concern": -0.15,
            "sell": -0.1,
            
            # Very negative
            "miss expectations": -0.3,
            "bankruptcy": -0.4,
            "investigation": -0.3,
            "crash": -0.35,
            "plunge": -0.3
        }
    
    def _calculate_financial_adjustment(self, text: str) -> float:
        """Calculate sentiment adjustment based on financial keywords"""
        adjustment = 0.0
        
        for phrase, score in self.financial_lexicon.items():
            if phrase in text:
                adjustment += score
        
        # Cap adjustment
        return np.clip(adjustment, -0.5, 0.5)