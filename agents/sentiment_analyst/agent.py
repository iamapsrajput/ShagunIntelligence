"""Sentiment Analysis Agent for monitoring market sentiment from news and social media."""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from crewai import Agent
from loguru import logger

from ..base_quality_aware_agent import TradingMode
from .alert_manager import AlertManager
from .news_scraper import NewsScraper
from .report_generator import ReportGenerator
from .sentiment_scorer import SentimentScorer
from .social_media_monitor import SocialMediaMonitor


class SentimentAnalystAgent(Agent):
    """Agent responsible for analyzing market sentiment from various sources with quality awareness."""

    # Store API key as a class variable for model_post_init access
    _api_key_storage: str | None = None

    def __init__(self, openai_api_key: str | None = None):
        """Initialize the Sentiment Analyst Agent."""
        super().__init__(
            role="Multi-Source Sentiment Analyst",
            goal="Aggregate and analyze sentiment from multiple sources with quality-weighted confidence scoring",
            backstory="""You are an expert in multi-source sentiment analysis with the ability to aggregate
            sentiment from news, social media, and specialized AI sources (Twitter, Grok). You understand that
            sentiment data quality varies by source and time. You weight sentiment signals based on:
            - Source reliability and data quality
            - Recency and relevance of information
            - Volume and engagement metrics
            - Agreement between multiple sources
            Your analysis adapts based on data quality to provide reliable sentiment insights.""",
            verbose=True,
            allow_delegation=False,
        )

        # Store API key for model_post_init access
        SentimentAnalystAgent._api_key_storage = openai_api_key

    def model_post_init(self, __context):
        """Initialize components after Pydantic model creation."""
        from backend.data_sources.data_quality_validator import (
            DataQualityValidator,
            QualityGrade,
        )
        from backend.data_sources.integration import get_data_source_integration

        # Initialize quality components
        object.__setattr__(self, "data_integration", get_data_source_integration())
        object.__setattr__(self, "quality_validator", DataQualityValidator())

        # Initialize components
        object.__setattr__(self, "news_scraper", NewsScraper())
        object.__setattr__(self, "social_monitor", SocialMediaMonitor())
        object.__setattr__(
            self,
            "sentiment_scorer",
            SentimentScorer(SentimentAnalystAgent._api_key_storage),
        )
        object.__setattr__(self, "alert_manager", AlertManager())
        object.__setattr__(self, "report_generator", ReportGenerator())

        # Cache for sentiment scores
        object.__setattr__(self, "sentiment_cache", defaultdict(dict))
        object.__setattr__(self, "cache_duration", timedelta(minutes=5))

        # Sentiment-specific quality thresholds
        object.__setattr__(
            self,
            "quality_thresholds",
            {
                QualityGrade.EXCELLENT: 0.8,  # Lower threshold for sentiment
                QualityGrade.GOOD: 0.65,
                QualityGrade.ACCEPTABLE: 0.5,
            },
        )

        logger.info("Multi-Source Sentiment Analyst Agent initialized")

    async def analyze_symbol_sentiment(
        self, symbol: str, include_social: bool = True, lookback_hours: int = 24
    ) -> dict[str, Any]:
        """
        Analyze sentiment for a specific symbol using multiple sources with quality awareness.

        Args:
            symbol: Stock symbol to analyze
            include_social: Whether to include social media analysis
            lookback_hours: Hours of historical data to analyze

        Returns:
            Comprehensive multi-source sentiment analysis
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{lookback_hours}"
            if cache_key in self.sentiment_cache:
                cached = self.sentiment_cache[cache_key]
                if datetime.now() - cached["timestamp"] < self.cache_duration:
                    logger.debug(f"Using cached sentiment for {symbol}")
                    return cached["data"]

            # Get multi-source sentiment from integration layer
            multi_source_sentiment = await self._get_multi_source_sentiment(symbol)

            # Gather news articles
            news_articles = await self.news_scraper.scrape_symbol_news(
                symbol, lookback_hours
            )

            # Gather social media data if requested
            social_data = []
            if include_social:
                social_data = await self.social_monitor.get_symbol_mentions(
                    symbol, lookback_hours
                )

            # Analyze sentiment for each source
            news_sentiments = []
            for article in news_articles:
                sentiment = await self.sentiment_scorer.analyze_text(
                    article["title"] + " " + article.get("description", ""),
                    context=f"Stock: {symbol}",
                )
                news_sentiments.append({**article, "sentiment": sentiment})

            social_sentiments = []
            for post in social_data:
                sentiment = await self.sentiment_scorer.analyze_text(
                    post["text"], context=f"Stock: {symbol}, Source: {post['source']}"
                )
                social_sentiments.append({**post, "sentiment": sentiment})

            # Calculate aggregate scores with multi-source data
            aggregate_scores = self._calculate_multi_source_aggregate(
                news_sentiments, social_sentiments, multi_source_sentiment
            )

            # Check for significant changes
            sentiment_change = await self._detect_sentiment_change(
                symbol, aggregate_scores["overall_score"]
            )

            # Generate alerts if needed
            if abs(sentiment_change) > 0.3:
                await self.alert_manager.create_sentiment_alert(
                    symbol=symbol,
                    current_score=aggregate_scores["overall_score"],
                    change=sentiment_change,
                    top_stories=news_sentiments[:3],
                )

            # Compile results with quality awareness
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "lookback_hours": lookback_hours,
                "sentiment_scores": aggregate_scores,
                "sentiment_change": sentiment_change,
                "news_analysis": {
                    "total_articles": len(news_articles),
                    "articles": news_sentiments[:10],  # Top 10 articles
                    "average_sentiment": aggregate_scores["news_score"],
                },
                "social_analysis": (
                    {
                        "total_posts": len(social_data),
                        "posts": social_sentiments[:10],  # Top 10 posts
                        "average_sentiment": aggregate_scores["social_score"],
                    }
                    if include_social
                    else None
                ),
                "multi_source_data": multi_source_sentiment,
                "data_quality": aggregate_scores.get("data_quality", 1.0),
                "quality_level": aggregate_scores.get("quality_level", "EXCELLENT"),
                "market_impact": self._assess_quality_adjusted_impact(aggregate_scores),
                "recommendation": self._generate_quality_aware_recommendation(
                    aggregate_scores, sentiment_change
                ),
            }

            # Cache the results
            self.sentiment_cache[cache_key] = {
                "timestamp": datetime.now(),
                "data": analysis,
            }

            logger.info(
                f"Sentiment analysis completed for {symbol}: Score={aggregate_scores['overall_score']:.2f}"
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            raise

    async def analyze_market_sentiment(
        self,
        sectors: list[str] | None = None,
        top_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze overall market sentiment.

        Args:
            sectors: Specific sectors to analyze
            top_symbols: Top symbols to include in analysis

        Returns:
            Market-wide sentiment analysis
        """
        try:
            # Get general market news
            market_news = await self.news_scraper.scrape_market_news()

            # Get trending topics from social media
            trending_topics = await self.social_monitor.get_trending_topics()

            # Analyze general market sentiment
            market_sentiments = []
            for article in market_news:
                sentiment = await self.sentiment_scorer.analyze_text(
                    article["title"] + " " + article.get("description", ""),
                    context="General market",
                )
                market_sentiments.append({**article, "sentiment": sentiment})

            # Analyze sector sentiments if specified
            sector_sentiments = {}
            if sectors:
                for sector in sectors:
                    sector_news = await self.news_scraper.scrape_sector_news(sector)
                    sector_scores = []
                    for article in sector_news:
                        sentiment = await self.sentiment_scorer.analyze_text(
                            article["title"] + " " + article.get("description", ""),
                            context=f"Sector: {sector}",
                        )
                        sector_scores.append(sentiment["score"])

                    sector_sentiments[sector] = {
                        "average_sentiment": (
                            sum(sector_scores) / len(sector_scores)
                            if sector_scores
                            else 0
                        ),
                        "article_count": len(sector_news),
                    }

            # Analyze top symbols if specified
            symbol_sentiments = {}
            if top_symbols:
                tasks = [
                    self.analyze_symbol_sentiment(
                        symbol, include_social=False, lookback_hours=6
                    )
                    for symbol in top_symbols
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for symbol, result in zip(top_symbols, results, strict=False):
                    if not isinstance(result, Exception):
                        symbol_sentiments[symbol] = result["sentiment_scores"][
                            "overall_score"
                        ]
                    else:
                        logger.error(f"Error analyzing {symbol}: {result}")
                        symbol_sentiments[symbol] = 0.0

            # Calculate overall market sentiment
            all_scores = [s["sentiment"]["score"] for s in market_sentiments]
            market_score = sum(all_scores) / len(all_scores) if all_scores else 0

            return {
                "timestamp": datetime.now().isoformat(),
                "market_sentiment": {
                    "score": market_score,
                    "interpretation": self._interpret_score(market_score),
                    "article_count": len(market_news),
                },
                "sector_sentiments": sector_sentiments,
                "symbol_sentiments": symbol_sentiments,
                "trending_topics": trending_topics,
                "top_stories": market_sentiments[:5],
                "market_mood": self._determine_market_mood(
                    market_score, sector_sentiments
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {str(e)}")
            raise

    async def generate_daily_report(self, symbols: list[str]) -> dict[str, Any]:
        """
        Generate a comprehensive daily sentiment report.

        Args:
            symbols: List of symbols to include in report

        Returns:
            Daily sentiment report
        """
        try:
            # Analyze each symbol
            symbol_analyses = {}
            for symbol in symbols:
                try:
                    analysis = await self.analyze_symbol_sentiment(symbol)
                    symbol_analyses[symbol] = analysis
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} for report: {e}")
                    continue

            # Get market sentiment
            market_analysis = await self.analyze_market_sentiment(
                top_symbols=symbols[:10]
            )

            # Generate report
            report = self.report_generator.generate_daily_report(
                symbol_analyses=symbol_analyses, market_analysis=market_analysis
            )

            return report

        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            raise

    async def monitor_real_time(
        self,
        symbols: list[str],
        callback: callable = None,
        interval_seconds: int = 300,
    ):
        """
        Monitor sentiment in real-time.

        Args:
            symbols: Symbols to monitor
            callback: Function to call with updates
            interval_seconds: Update interval
        """
        logger.info(
            f"Starting real-time sentiment monitoring for {len(symbols)} symbols"
        )

        while True:
            try:
                for symbol in symbols:
                    # Quick sentiment check
                    analysis = await self.analyze_symbol_sentiment(
                        symbol, include_social=True, lookback_hours=1
                    )

                    # Check for significant changes
                    if abs(analysis["sentiment_change"]) > 0.2:
                        alert = {
                            "type": "sentiment_change",
                            "symbol": symbol,
                            "data": analysis,
                            "timestamp": datetime.now().isoformat(),
                        }

                        if callback:
                            await callback(alert)

                        logger.info(
                            f"Sentiment alert for {symbol}: Change={analysis['sentiment_change']:.2f}"
                        )

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in real-time monitoring: {str(e)}")
                await asyncio.sleep(interval_seconds)

    async def _get_multi_source_sentiment(self, symbol: str) -> dict[str, Any]:
        """Get sentiment from multiple integrated sources."""
        try:
            # Get sentiment from integration layer (includes Twitter, Grok, etc.)
            integrated_sentiment = await self.data_integration.get_sentiment_score(
                symbol
            )

            # Get Grok analysis if available
            grok_analysis = None
            try:
                grok_analysis = await self.data_integration.get_grok_analysis(symbol)
            except:
                logger.debug(f"Grok analysis not available for {symbol}")

            # Get sentiment trends
            sentiment_trends = await self.data_integration.get_sentiment_trends(
                symbol, hours=24
            )

            return {
                "integrated_sentiment": integrated_sentiment,
                "grok_analysis": grok_analysis,
                "sentiment_trends": sentiment_trends,
                "source_count": integrated_sentiment.get("source_count", 0),
            }
        except Exception as e:
            logger.error(f"Error getting multi-source sentiment: {e}")
            return {}

    def _calculate_multi_source_aggregate(
        self,
        news_sentiments: list[dict],
        social_sentiments: list[dict],
        multi_source_data: dict[str, Any],
    ) -> dict[str, float]:
        """Calculate aggregate sentiment scores from all sources with quality weighting."""
        # Traditional sentiment calculation
        traditional_scores = self._calculate_traditional_sentiment(
            news_sentiments, social_sentiments
        )

        # Multi-source sentiment data
        integrated_sentiment = multi_source_data.get("integrated_sentiment", {})
        grok_analysis = multi_source_data.get("grok_analysis", {})

        # Extract scores and quality metrics
        integrated_score = integrated_sentiment.get("sentiment", 0.0)
        integrated_confidence = integrated_sentiment.get("confidence", 0.0)
        agreement_score = integrated_sentiment.get("agreement_score", 0.5)

        # Grok sentiment (if available)
        grok_score = None
        if grok_analysis and not grok_analysis.get("error"):
            grok_score = grok_analysis.get("sentiment_score", 0.0)

        # Calculate data quality based on source availability and agreement
        data_quality = self._calculate_sentiment_data_quality(
            traditional_scores["confidence"],
            integrated_confidence,
            agreement_score,
            multi_source_data.get("source_count", 0),
        )

        # Determine quality level
        quality_level = self._determine_quality_level(data_quality)

        # Weight scores based on quality and availability
        weights = self._calculate_source_weights(
            has_news=len(news_sentiments) > 0,
            has_social=len(social_sentiments) > 0,
            has_integrated=integrated_confidence > 0,
            has_grok=grok_score is not None,
            quality_level=quality_level,
        )

        # Calculate weighted average
        weighted_sum = 0
        total_weight = 0

        if weights["news"] > 0:
            weighted_sum += traditional_scores["news_score"] * weights["news"]
            total_weight += weights["news"]

        if weights["social"] > 0:
            weighted_sum += traditional_scores["social_score"] * weights["social"]
            total_weight += weights["social"]

        if weights["integrated"] > 0:
            weighted_sum += integrated_score * weights["integrated"]
            total_weight += weights["integrated"]

        if weights["grok"] > 0 and grok_score is not None:
            weighted_sum += grok_score * weights["grok"]
            total_weight += weights["grok"]

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Adjust confidence based on data quality
        adjusted_confidence = traditional_scores["confidence"] * data_quality

        return {
            "overall_score": overall_score,
            "news_score": traditional_scores["news_score"],
            "social_score": traditional_scores["social_score"],
            "integrated_score": integrated_score,
            "grok_score": grok_score,
            "confidence": adjusted_confidence,
            "data_quality": data_quality,
            "quality_level": quality_level.value,
            "source_weights": weights,
            "agreement_score": agreement_score,
        }

    def _calculate_traditional_sentiment(
        self, news_sentiments: list[dict], social_sentiments: list[dict]
    ) -> dict[str, float]:
        """Calculate traditional sentiment scores from news and social."""
        # News sentiment (higher weight for more recent and reliable sources)
        news_scores = []
        for item in news_sentiments:
            score = item["sentiment"]["score"]
            weight = item.get("reliability_weight", 1.0)

            # Recency weight (newer articles have more impact)
            age_hours = (datetime.now() - item["timestamp"]).total_seconds() / 3600
            recency_weight = max(
                0.5, 1.0 - (age_hours / 48)
            )  # 50% weight after 48 hours

            news_scores.append(score * weight * recency_weight)

        news_avg = sum(news_scores) / len(news_scores) if news_scores else 0

        # Social sentiment (volume-weighted)
        social_scores = []
        for item in social_sentiments:
            score = item["sentiment"]["score"]
            engagement = item.get("engagement_score", 1)

            # Engagement weight (more engagement = more weight)
            engagement_weight = min(2.0, 1.0 + (engagement / 1000))

            social_scores.append(score * engagement_weight)

        social_avg = sum(social_scores) / len(social_scores) if social_scores else 0

        # Overall sentiment (70% news, 30% social)
        if news_scores and social_scores:
            overall = 0.7 * news_avg + 0.3 * social_avg
        elif news_scores:
            overall = news_avg
        else:
            overall = social_avg

        return {
            "overall_score": overall,
            "news_score": news_avg,
            "social_score": social_avg,
            "confidence": self._calculate_confidence(
                len(news_scores), len(social_scores)
            ),
        }

    def _calculate_sentiment_data_quality(
        self,
        traditional_confidence: float,
        integrated_confidence: float,
        agreement_score: float,
        source_count: int,
    ) -> float:
        """Calculate overall sentiment data quality."""
        # Base quality on confidence scores
        avg_confidence = (traditional_confidence + integrated_confidence) / 2

        # Bonus for multiple sources
        source_bonus = min(0.2, source_count * 0.05)

        # Factor in agreement between sources
        agreement_factor = 0.5 + (agreement_score * 0.5)

        # Calculate final quality
        quality = (avg_confidence * agreement_factor) + source_bonus

        return min(1.0, quality)

    def _calculate_source_weights(
        self,
        has_news: bool,
        has_social: bool,
        has_integrated: bool,
        has_grok: bool,
        quality_level,
    ) -> dict[str, float]:
        """Calculate weights for each sentiment source based on availability and quality."""
        from backend.data_sources.data_quality_validator import QualityGrade

        if quality_level == QualityGrade.EXCELLENT:
            # High quality: Use all sources with balanced weights
            weights = {
                "news": 0.35 if has_news else 0.0,
                "social": 0.15 if has_social else 0.0,
                "integrated": 0.30 if has_integrated else 0.0,
                "grok": 0.20 if has_grok else 0.0,
            }
        elif quality_level == QualityGrade.GOOD:
            # Medium quality: Prefer more reliable sources
            weights = {
                "news": 0.40 if has_news else 0.0,
                "social": 0.10 if has_social else 0.0,
                "integrated": 0.35 if has_integrated else 0.0,
                "grok": 0.15 if has_grok else 0.0,
            }
        else:
            # Low quality: Heavy weight on most reliable sources
            weights = {
                "news": 0.50 if has_news else 0.0,
                "social": 0.05 if has_social else 0.0,
                "integrated": 0.30 if has_integrated else 0.0,
                "grok": 0.15 if has_grok else 0.0,
            }

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    async def _detect_sentiment_change(
        self, symbol: str, current_score: float
    ) -> float:
        """Detect change in sentiment from previous analysis."""
        # Get previous score from cache or database
        previous_key = f"{symbol}_previous"
        previous_score = self.sentiment_cache.get(previous_key, {}).get(
            "score", current_score
        )

        # Update previous score
        self.sentiment_cache[previous_key] = {
            "score": current_score,
            "timestamp": datetime.now(),
        }

        return current_score - previous_score

    def _assess_quality_adjusted_impact(self, scores: dict[str, float]) -> str:
        """Assess market impact with data quality adjustment."""
        overall = scores["overall_score"]
        confidence = scores["confidence"]
        data_quality = scores.get("data_quality", 1.0)

        # Adjust thresholds based on data quality
        quality_multiplier = 1.0 if data_quality > 0.7 else 1.5

        if abs(overall) < 0.2 * quality_multiplier:
            return "minimal"
        elif abs(overall) < 0.5 * quality_multiplier:
            return "moderate"
        elif abs(overall) < 0.7 * quality_multiplier:
            return "significant" if confidence > 0.6 else "moderate"
        else:
            return "high" if confidence > 0.6 else "significant"

    def _generate_quality_aware_recommendation(
        self, scores: dict[str, float], sentiment_change: float
    ) -> dict[str, Any]:
        """Generate trading recommendation with data quality awareness."""
        overall = scores["overall_score"]
        confidence = scores["confidence"]
        data_quality = scores.get("data_quality", 1.0)
        from backend.data_sources.data_quality_validator import QualityGrade

        quality_level_str = scores.get("quality_level", "EXCELLENT")
        quality_level = (
            QualityGrade(quality_level_str)
            if isinstance(quality_level_str, str)
            else quality_level_str
        )

        # Get trading mode based on quality
        trading_mode = self.get_trading_mode(quality_level)

        if trading_mode == TradingMode.NORMAL:
            # High quality data - normal recommendations
            if overall > 0.6 and confidence > 0.7:
                action = "consider_long"
                reasoning = (
                    "Strong positive sentiment with high confidence and quality data"
                )
            elif overall < -0.6 and confidence > 0.7:
                action = "consider_short"
                reasoning = (
                    "Strong negative sentiment with high confidence and quality data"
                )
            elif sentiment_change > 0.4:
                action = "monitor_for_entry"
                reasoning = "Rapid improvement in sentiment"
            elif sentiment_change < -0.4:
                action = "monitor_for_exit"
                reasoning = "Rapid deterioration in sentiment"
            else:
                action = "hold"
                reasoning = "Neutral or mixed sentiment signals"

        elif trading_mode == TradingMode.CONSERVATIVE:
            # Medium quality - conservative recommendations
            if overall > 0.7 and confidence > 0.75:
                action = "consider_small_long"
                reasoning = (
                    f"Positive sentiment but medium data quality ({data_quality:.1%})"
                )
            elif overall < -0.7 and confidence > 0.75:
                action = "consider_small_short"
                reasoning = (
                    f"Negative sentiment but medium data quality ({data_quality:.1%})"
                )
            else:
                action = "hold"
                reasoning = "Conservative mode due to medium data quality"

        else:  # DEFENSIVE or EMERGENCY
            # Low quality - defensive only
            action = "hold_defensive"
            reasoning = f"Low data quality ({data_quality:.1%}) - defensive mode only"

        # Add position sizing based on quality
        position_multiplier = self.quality_position_multipliers.get(quality_level, 0.0)

        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "data_quality": data_quality,
            "quality_level": quality_level.value,
            "trading_mode": trading_mode.value,
            "position_size_multiplier": position_multiplier,
            "risk_level": self._assess_quality_adjusted_risk(
                overall, confidence, sentiment_change, data_quality
            ),
        }

    def _assess_quality_adjusted_risk(
        self, score: float, confidence: float, change: float, quality: float
    ) -> str:
        """Assess risk level with quality adjustment."""
        # Base risk assessment
        base_risk = self._assess_risk_level(score, confidence, change)

        # Increase risk level if data quality is low
        if quality < 0.5:
            risk_map = {
                "low": "medium",
                "medium": "medium_high",
                "medium_high": "high",
                "high": "extreme",
            }
            return risk_map.get(base_risk, "high")
        elif quality < 0.7:
            if base_risk == "low":
                return "medium"
            return base_risk
        else:
            return base_risk

    def _calculate_confidence(self, news_count: int, social_count: int) -> float:
        """Calculate confidence based on data availability."""
        # Base confidence on amount of data
        news_confidence = min(1.0, news_count / 10)  # Max confidence at 10+ articles
        social_confidence = min(1.0, social_count / 50)  # Max confidence at 50+ posts

        # Weight news more heavily
        if news_count > 0 and social_count > 0:
            return 0.7 * news_confidence + 0.3 * social_confidence
        elif news_count > 0:
            return news_confidence * 0.8  # Reduce confidence without social data
        elif social_count > 0:
            return social_confidence * 0.5  # Lower confidence for social-only
        else:
            return 0.0

    def _interpret_score(self, score: float) -> str:
        """Interpret sentiment score."""
        if score > 0.6:
            return "very_positive"
        elif score > 0.2:
            return "positive"
        elif score > -0.2:
            return "neutral"
        elif score > -0.6:
            return "negative"
        else:
            return "very_negative"

    def _determine_market_mood(
        self, market_score: float, sector_sentiments: dict[str, Any]
    ) -> str:
        """Determine overall market mood."""
        # Check sector consistency
        if sector_sentiments:
            sector_scores = [s["average_sentiment"] for s in sector_sentiments.values()]
            sector_avg = sum(sector_scores) / len(sector_scores)
            consistency = 1 - (abs(market_score - sector_avg) / 2)
        else:
            consistency = 0.5

        if market_score > 0.5 and consistency > 0.7:
            return "risk_on"
        elif market_score < -0.5 and consistency > 0.7:
            return "risk_off"
        elif abs(market_score) < 0.2:
            return "neutral"
        else:
            return "mixed"

    def _assess_risk_level(self, score: float, confidence: float, change: float) -> str:
        """Assess risk level based on sentiment metrics."""
        # High volatility in sentiment = higher risk
        if abs(change) > 0.5:
            return "high"
        # Low confidence = higher risk
        elif confidence < 0.5:
            return "medium_high"
        # Extreme sentiment = higher risk
        elif abs(score) > 0.8:
            return "medium_high"
        # Moderate conditions
        elif abs(score) > 0.4 and confidence > 0.7:
            return "medium"
        else:
            return "low"

    def get_status(self) -> dict[str, Any]:
        """Get the status of the sentiment analyst agent."""
        return {
            "agent": "SentimentAnalystAgent",
            "status": "active",
            "cache_size": len(self.sentiment_cache),
            "modules": {
                "news_scraper": self.news_scraper.get_status(),
                "social_monitor": self.social_monitor.get_status(),
                "sentiment_scorer": self.sentiment_scorer.get_status(),
                "alert_manager": self.alert_manager.get_status(),
            },
        }
