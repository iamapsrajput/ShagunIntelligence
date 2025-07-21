"""Sentiment Analysis Agent for monitoring market sentiment from news and social media."""

from crewai import Agent
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
import asyncio
from collections import defaultdict

from .news_scraper import NewsScraper
from .social_media_monitor import SocialMediaMonitor
from .sentiment_scorer import SentimentScorer
from .alert_manager import AlertManager
from .report_generator import ReportGenerator


class SentimentAnalystAgent(Agent):
    """Agent responsible for analyzing market sentiment from various sources."""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the Sentiment Analyst Agent."""
        super().__init__(
            role="Market Sentiment Analyst",
            goal="Monitor and analyze market sentiment from news and social media to provide trading insights",
            backstory="""You are an expert in market sentiment analysis with deep understanding of how news 
            and social media affect stock prices. You monitor multiple news sources, analyze social media trends,
            and use advanced NLP to gauge market sentiment. You can identify significant sentiment shifts,
            detect market-moving news early, and provide actionable insights based on collective market mood.
            Your analysis helps traders understand the psychological factors driving market movements.""",
            verbose=True,
            allow_delegation=False
        )
        
        self.news_scraper = NewsScraper()
        self.social_monitor = SocialMediaMonitor()
        self.sentiment_scorer = SentimentScorer(openai_api_key)
        self.alert_manager = AlertManager()
        self.report_generator = ReportGenerator()
        
        # Cache for sentiment scores
        self.sentiment_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.cache_duration = timedelta(minutes=5)
        
        logger.info("Sentiment Analyst Agent initialized")

    async def analyze_symbol_sentiment(
        self, 
        symbol: str,
        include_social: bool = True,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for a specific symbol.
        
        Args:
            symbol: Stock symbol to analyze
            include_social: Whether to include social media analysis
            lookback_hours: Hours of historical data to analyze
            
        Returns:
            Comprehensive sentiment analysis
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{lookback_hours}"
            if cache_key in self.sentiment_cache:
                cached = self.sentiment_cache[cache_key]
                if datetime.now() - cached['timestamp'] < self.cache_duration:
                    logger.debug(f"Using cached sentiment for {symbol}")
                    return cached['data']
            
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
                    article['title'] + ' ' + article.get('description', ''),
                    context=f"Stock: {symbol}"
                )
                news_sentiments.append({
                    **article,
                    'sentiment': sentiment
                })
            
            social_sentiments = []
            for post in social_data:
                sentiment = await self.sentiment_scorer.analyze_text(
                    post['text'],
                    context=f"Stock: {symbol}, Source: {post['source']}"
                )
                social_sentiments.append({
                    **post,
                    'sentiment': sentiment
                })
            
            # Calculate aggregate scores
            aggregate_scores = self._calculate_aggregate_sentiment(
                news_sentiments, social_sentiments
            )
            
            # Check for significant changes
            sentiment_change = await self._detect_sentiment_change(
                symbol, aggregate_scores['overall_score']
            )
            
            # Generate alerts if needed
            if abs(sentiment_change) > 0.3:
                await self.alert_manager.create_sentiment_alert(
                    symbol=symbol,
                    current_score=aggregate_scores['overall_score'],
                    change=sentiment_change,
                    top_stories=news_sentiments[:3]
                )
            
            # Compile results
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'lookback_hours': lookback_hours,
                'sentiment_scores': aggregate_scores,
                'sentiment_change': sentiment_change,
                'news_analysis': {
                    'total_articles': len(news_articles),
                    'articles': news_sentiments[:10],  # Top 10 articles
                    'average_sentiment': aggregate_scores['news_score']
                },
                'social_analysis': {
                    'total_posts': len(social_data),
                    'posts': social_sentiments[:10],  # Top 10 posts
                    'average_sentiment': aggregate_scores['social_score']
                } if include_social else None,
                'market_impact': self._assess_market_impact(aggregate_scores),
                'recommendation': self._generate_recommendation(aggregate_scores, sentiment_change)
            }
            
            # Cache the results
            self.sentiment_cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': analysis
            }
            
            logger.info(f"Sentiment analysis completed for {symbol}: "
                       f"Score={aggregate_scores['overall_score']:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            raise

    async def analyze_market_sentiment(
        self,
        sectors: Optional[List[str]] = None,
        top_symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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
                    article['title'] + ' ' + article.get('description', ''),
                    context="General market"
                )
                market_sentiments.append({
                    **article,
                    'sentiment': sentiment
                })
            
            # Analyze sector sentiments if specified
            sector_sentiments = {}
            if sectors:
                for sector in sectors:
                    sector_news = await self.news_scraper.scrape_sector_news(sector)
                    sector_scores = []
                    for article in sector_news:
                        sentiment = await self.sentiment_scorer.analyze_text(
                            article['title'] + ' ' + article.get('description', ''),
                            context=f"Sector: {sector}"
                        )
                        sector_scores.append(sentiment['score'])
                    
                    sector_sentiments[sector] = {
                        'average_sentiment': sum(sector_scores) / len(sector_scores) if sector_scores else 0,
                        'article_count': len(sector_news)
                    }
            
            # Analyze top symbols if specified
            symbol_sentiments = {}
            if top_symbols:
                tasks = [
                    self.analyze_symbol_sentiment(symbol, include_social=False, lookback_hours=6)
                    for symbol in top_symbols
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for symbol, result in zip(top_symbols, results):
                    if not isinstance(result, Exception):
                        symbol_sentiments[symbol] = result['sentiment_scores']['overall_score']
                    else:
                        logger.error(f"Error analyzing {symbol}: {result}")
                        symbol_sentiments[symbol] = 0.0
            
            # Calculate overall market sentiment
            all_scores = [s['sentiment']['score'] for s in market_sentiments]
            market_score = sum(all_scores) / len(all_scores) if all_scores else 0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'market_sentiment': {
                    'score': market_score,
                    'interpretation': self._interpret_score(market_score),
                    'article_count': len(market_news)
                },
                'sector_sentiments': sector_sentiments,
                'symbol_sentiments': symbol_sentiments,
                'trending_topics': trending_topics,
                'top_stories': market_sentiments[:5],
                'market_mood': self._determine_market_mood(market_score, sector_sentiments)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {str(e)}")
            raise

    async def generate_daily_report(self, symbols: List[str]) -> Dict[str, Any]:
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
                symbol_analyses=symbol_analyses,
                market_analysis=market_analysis
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            raise

    async def monitor_real_time(
        self,
        symbols: List[str],
        callback: Optional[callable] = None,
        interval_seconds: int = 300
    ):
        """
        Monitor sentiment in real-time.
        
        Args:
            symbols: Symbols to monitor
            callback: Function to call with updates
            interval_seconds: Update interval
        """
        logger.info(f"Starting real-time sentiment monitoring for {len(symbols)} symbols")
        
        while True:
            try:
                for symbol in symbols:
                    # Quick sentiment check
                    analysis = await self.analyze_symbol_sentiment(
                        symbol, 
                        include_social=True,
                        lookback_hours=1
                    )
                    
                    # Check for significant changes
                    if abs(analysis['sentiment_change']) > 0.2:
                        alert = {
                            'type': 'sentiment_change',
                            'symbol': symbol,
                            'data': analysis,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        if callback:
                            await callback(alert)
                        
                        logger.info(f"Sentiment alert for {symbol}: "
                                   f"Change={analysis['sentiment_change']:.2f}")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in real-time monitoring: {str(e)}")
                await asyncio.sleep(interval_seconds)

    def _calculate_aggregate_sentiment(
        self,
        news_sentiments: List[Dict],
        social_sentiments: List[Dict]
    ) -> Dict[str, float]:
        """Calculate aggregate sentiment scores."""
        # News sentiment (higher weight for more recent and reliable sources)
        news_scores = []
        for item in news_sentiments:
            score = item['sentiment']['score']
            weight = item.get('reliability_weight', 1.0)
            
            # Recency weight (newer articles have more impact)
            age_hours = (datetime.now() - item['timestamp']).total_seconds() / 3600
            recency_weight = max(0.5, 1.0 - (age_hours / 48))  # 50% weight after 48 hours
            
            news_scores.append(score * weight * recency_weight)
        
        news_avg = sum(news_scores) / len(news_scores) if news_scores else 0
        
        # Social sentiment (volume-weighted)
        social_scores = []
        for item in social_sentiments:
            score = item['sentiment']['score']
            engagement = item.get('engagement_score', 1)
            
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
            'overall_score': overall,
            'news_score': news_avg,
            'social_score': social_avg,
            'confidence': self._calculate_confidence(len(news_scores), len(social_scores))
        }

    async def _detect_sentiment_change(
        self,
        symbol: str,
        current_score: float
    ) -> float:
        """Detect change in sentiment from previous analysis."""
        # Get previous score from cache or database
        previous_key = f"{symbol}_previous"
        previous_score = self.sentiment_cache.get(previous_key, {}).get('score', current_score)
        
        # Update previous score
        self.sentiment_cache[previous_key] = {
            'score': current_score,
            'timestamp': datetime.now()
        }
        
        return current_score - previous_score

    def _assess_market_impact(self, scores: Dict[str, float]) -> str:
        """Assess potential market impact based on sentiment."""
        overall = scores['overall_score']
        confidence = scores['confidence']
        
        if abs(overall) < 0.2:
            return "minimal"
        elif abs(overall) < 0.5:
            return "moderate"
        elif abs(overall) < 0.7:
            return "significant" if confidence > 0.7 else "moderate"
        else:
            return "high" if confidence > 0.7 else "significant"

    def _generate_recommendation(
        self,
        scores: Dict[str, float],
        sentiment_change: float
    ) -> Dict[str, Any]:
        """Generate trading recommendation based on sentiment."""
        overall = scores['overall_score']
        confidence = scores['confidence']
        
        # Strong positive sentiment
        if overall > 0.6 and confidence > 0.7:
            action = "consider_long"
            reasoning = "Strong positive sentiment with high confidence"
        # Strong negative sentiment
        elif overall < -0.6 and confidence > 0.7:
            action = "consider_short"
            reasoning = "Strong negative sentiment with high confidence"
        # Rapid positive change
        elif sentiment_change > 0.4:
            action = "monitor_for_entry"
            reasoning = "Rapid improvement in sentiment"
        # Rapid negative change
        elif sentiment_change < -0.4:
            action = "monitor_for_exit"
            reasoning = "Rapid deterioration in sentiment"
        # Neutral
        else:
            action = "hold"
            reasoning = "Neutral or mixed sentiment signals"
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'risk_level': self._assess_risk_level(overall, confidence, sentiment_change)
        }

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
        self,
        market_score: float,
        sector_sentiments: Dict[str, Any]
    ) -> str:
        """Determine overall market mood."""
        # Check sector consistency
        if sector_sentiments:
            sector_scores = [s['average_sentiment'] for s in sector_sentiments.values()]
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

    def _assess_risk_level(
        self,
        score: float,
        confidence: float,
        change: float
    ) -> str:
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

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the sentiment analyst agent."""
        return {
            'agent': 'SentimentAnalystAgent',
            'status': 'active',
            'cache_size': len(self.sentiment_cache),
            'modules': {
                'news_scraper': self.news_scraper.get_status(),
                'social_monitor': self.social_monitor.get_status(),
                'sentiment_scorer': self.sentiment_scorer.get_status(),
                'alert_manager': self.alert_manager.get_status()
            }
        }