"""
News feed streaming handler for real-time financial news.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import json
import feedparser
import aiohttp
from loguru import logger
import hashlib

from ..realtime_pipeline import DataStreamHandler, StreamConfig, StreamMessage, StreamStatus


class NewsStreamHandler(DataStreamHandler):
    """Handler for streaming financial news from multiple RSS/API sources."""
    
    def __init__(self, config: StreamConfig, api_keys: Dict[str, str] = None):
        super().__init__(config)
        self.api_keys = api_keys or {}
        self.session = None
        self.polling_tasks = {}
        
        # News sources configuration
        self.news_sources = {
            'reuters': {
                'url': 'https://www.reuters.com/business/finance/rss',
                'type': 'rss',
                'interval': 60  # seconds
            },
            'bloomberg': {
                'url': 'https://feeds.bloomberg.com/markets/news.rss',
                'type': 'rss',
                'interval': 60
            },
            'cnbc': {
                'url': 'https://www.cnbc.com/id/10001147/device/rss/rss.html',
                'type': 'rss',
                'interval': 60
            },
            'marketwatch': {
                'url': 'https://feeds.marketwatch.com/marketwatch/topstories',
                'type': 'rss',
                'interval': 60
            },
            'newsapi': {
                'url': 'https://newsapi.org/v2/everything',
                'type': 'api',
                'interval': 300,  # 5 minutes due to rate limits
                'requires_key': True
            }
        }
        
        # Track seen articles to avoid duplicates
        self.seen_articles: Set[str] = set()
        self.article_cache_time = timedelta(hours=24)
        
        # Symbol detection patterns
        self.symbol_patterns = {}
        
        # News quality scoring
        self.trusted_sources = {
            'reuters.com': 0.9,
            'bloomberg.com': 0.9,
            'wsj.com': 0.9,
            'ft.com': 0.85,
            'cnbc.com': 0.8,
            'marketwatch.com': 0.75,
            'businessinsider.com': 0.7,
            'seekingalpha.com': 0.65
        }
        
    async def connect(self) -> bool:
        """Initialize HTTP session for news fetching."""
        try:
            self.status = StreamStatus.CONNECTING
            
            # Create session
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            self.status = StreamStatus.CONNECTED
            logger.info("Connected to news streaming sources")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize news streaming: {e}")
            self.status = StreamStatus.ERROR
            self.metrics.errors_count += 1
            return False
    
    async def disconnect(self):
        """Stop news streaming."""
        try:
            # Cancel all polling tasks
            for task in self.polling_tasks.values():
                task.cancel()
            self.polling_tasks.clear()
            
            # Close session
            if self.session:
                await self.session.close()
                
            self.status = StreamStatus.DISCONNECTED
            logger.info("Disconnected from news streaming")
            
        except Exception as e:
            logger.error(f"Error disconnecting news stream: {e}")
    
    async def subscribe(self, symbols: List[str]):
        """Start polling news for given symbols."""
        try:
            # Build symbol patterns for detection
            for symbol in symbols:
                patterns = self._build_symbol_patterns(symbol)
                self.symbol_patterns[symbol] = patterns
            
            # Start polling tasks for each news source
            for source_name, source_config in self.news_sources.items():
                if source_name not in self.polling_tasks:
                    # Check if API key is required
                    if source_config.get('requires_key') and source_name not in self.api_keys:
                        logger.warning(f"Skipping {source_name} - API key required")
                        continue
                    
                    task = asyncio.create_task(
                        self._poll_news_source(source_name, source_config)
                    )
                    self.polling_tasks[source_name] = task
                    logger.info(f"Started polling {source_name} for news")
                    
        except Exception as e:
            logger.error(f"Error subscribing to news: {e}")
            self.metrics.errors_count += 1
    
    async def process_message(self, message: Any) -> Optional[StreamMessage]:
        """News processing is handled by polling tasks."""
        return None
    
    async def send_heartbeat(self):
        """Check health of polling tasks."""
        self.last_heartbeat = datetime.now()
        
        # Restart dead tasks
        for source_name, task in list(self.polling_tasks.items()):
            if task.done():
                logger.warning(f"Restarting news polling for {source_name}")
                source_config = self.news_sources.get(source_name)
                if source_config:
                    new_task = asyncio.create_task(
                        self._poll_news_source(source_name, source_config)
                    )
                    self.polling_tasks[source_name] = new_task
    
    async def _poll_news_source(self, source_name: str, config: Dict[str, Any]):
        """Poll a specific news source."""
        consecutive_errors = 0
        
        while self.status == StreamStatus.CONNECTED:
            try:
                start_time = datetime.now()
                
                if config['type'] == 'rss':
                    articles = await self._fetch_rss_feed(config['url'])
                elif config['type'] == 'api':
                    articles = await self._fetch_api_news(source_name, config)
                else:
                    articles = []
                
                # Process articles
                new_articles = 0
                for article in articles:
                    if await self._process_article(article, source_name):
                        new_articles += 1
                
                if new_articles > 0:
                    logger.info(f"Processed {new_articles} new articles from {source_name}")
                
                # Update metrics
                self.metrics.messages_received += new_articles
                if new_articles > 0:
                    self.metrics.last_message_time = datetime.now()
                
                consecutive_errors = 0
                
            except Exception as e:
                logger.error(f"Error polling {source_name}: {e}")
                self.metrics.errors_count += 1
                consecutive_errors += 1
                
                if consecutive_errors > 5:
                    logger.error(f"Too many errors for {source_name}, stopping")
                    break
            
            # Wait before next poll
            await asyncio.sleep(config['interval'])
    
    async def _fetch_rss_feed(self, url: str) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    articles = []
                    for entry in feed.entries:
                        articles.append({
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published_parsed'),
                            'source': feed.feed.get('title', url)
                        })
                    
                    return articles
                else:
                    logger.error(f"RSS fetch error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching RSS feed: {e}")
            return []
    
    async def _fetch_api_news(self, source_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch news from API endpoints."""
        if source_name == 'newsapi' and 'newsapi' in self.api_keys:
            return await self._fetch_newsapi()
        
        return []
    
    async def _fetch_newsapi(self) -> List[Dict[str, Any]]:
        """Fetch from NewsAPI.org."""
        try:
            # Build query for financial news
            params = {
                'apiKey': self.api_keys['newsapi'],
                'q': 'stock market OR trading OR NYSE OR NASDAQ',
                'domains': 'reuters.com,bloomberg.com,wsj.com,cnbc.com',
                'sortBy': 'publishedAt',
                'pageSize': 50
            }
            
            async with self.session.get(
                'https://newsapi.org/v2/everything',
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    articles = []
                    for article in data.get('articles', []):
                        articles.append({
                            'title': article.get('title', ''),
                            'summary': article.get('description', ''),
                            'link': article.get('url', ''),
                            'published': article.get('publishedAt'),
                            'source': article.get('source', {}).get('name', 'NewsAPI'),
                            'author': article.get('author')
                        })
                    
                    return articles
                else:
                    logger.error(f"NewsAPI error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []
    
    async def _process_article(self, article: Dict[str, Any], source_name: str) -> bool:
        """Process a news article and create stream messages."""
        try:
            # Create unique ID for article
            article_id = hashlib.md5(
                f"{article['title']}:{article['link']}".encode()
            ).hexdigest()
            
            # Check if already seen
            if article_id in self.seen_articles:
                return False
            
            self.seen_articles.add(article_id)
            
            # Extract symbols mentioned in article
            mentioned_symbols = self._extract_symbols(article)
            
            if not mentioned_symbols:
                return False
            
            # Calculate article quality/relevance
            quality_score = self._calculate_article_quality(article, source_name)
            
            # Parse publication time
            if article.get('published'):
                if isinstance(article['published'], str):
                    pub_time = datetime.fromisoformat(article['published'].replace('Z', '+00:00'))
                else:
                    # Handle time.struct_time from feedparser
                    import time
                    pub_time = datetime.fromtimestamp(time.mktime(article['published']))
            else:
                pub_time = datetime.now()
            
            # Calculate "latency" (time since publication)
            latency_ms = (datetime.now() - pub_time).total_seconds() * 1000
            
            # Create messages for each mentioned symbol
            for symbol in mentioned_symbols:
                self.metrics.messages_processed += 1
                
                message = StreamMessage(
                    stream_name=self.config.name,
                    symbol=symbol,
                    data={
                        'type': 'news',
                        'title': article['title'],
                        'summary': article['summary'][:500],  # Limit summary length
                        'link': article['link'],
                        'source': article['source'],
                        'published': pub_time.isoformat(),
                        'relevance_score': self._calculate_relevance(article, symbol),
                        'sentiment': self._analyze_news_sentiment(article)
                    },
                    timestamp=datetime.now(),
                    latency_ms=latency_ms,
                    quality_score=quality_score
                )
                
                # This would be sent to the pipeline
                logger.debug(f"News for {symbol}: {article['title'][:100]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return False
    
    def _extract_symbols(self, article: Dict[str, Any]) -> Set[str]:
        """Extract stock symbols mentioned in article."""
        mentioned = set()
        
        # Combine title and summary for searching
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        
        # Check each tracked symbol
        for symbol, patterns in self.symbol_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text.lower():
                    mentioned.add(symbol)
                    break
        
        return mentioned
    
    def _build_symbol_patterns(self, symbol: str) -> List[str]:
        """Build search patterns for a symbol."""
        patterns = [
            symbol,
            f"${symbol}",  # Cashtag
            f"({symbol})",  # In parentheses
            f"{symbol} stock",
            f"{symbol} shares"
        ]
        
        # Add company names
        company_names = {
            'AAPL': ['Apple', 'Apple Inc'],
            'GOOGL': ['Google', 'Alphabet'],
            'MSFT': ['Microsoft'],
            'AMZN': ['Amazon'],
            'TSLA': ['Tesla'],
            'META': ['Meta', 'Facebook'],
            'NFLX': ['Netflix'],
            'RELIANCE': ['Reliance Industries', 'Reliance'],
            'TCS': ['Tata Consultancy', 'TCS'],
            'INFY': ['Infosys'],
            'HDFC': ['HDFC Bank', 'HDFC'],
            'ICICIBANK': ['ICICI Bank', 'ICICI']
        }
        
        if symbol in company_names:
            patterns.extend(company_names[symbol])
        
        return patterns
    
    def _calculate_article_quality(self, article: Dict[str, Any], source_name: str) -> float:
        """Calculate quality score for an article."""
        # Base score from source reputation
        source_url = article.get('link', '')
        base_score = 0.5
        
        for domain, score in self.trusted_sources.items():
            if domain in source_url:
                base_score = score
                break
        
        # Adjust for article completeness
        if article.get('title') and article.get('summary'):
            base_score += 0.1
        
        # Recency bonus
        if article.get('published'):
            age_hours = (datetime.now() - datetime.now()).total_seconds() / 3600
            if age_hours < 1:
                base_score += 0.1
            elif age_hours < 24:
                base_score += 0.05
        
        return min(1.0, base_score)
    
    def _calculate_relevance(self, article: Dict[str, Any], symbol: str) -> float:
        """Calculate relevance of article to specific symbol."""
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        text_lower = text.lower()
        symbol_lower = symbol.lower()
        
        # Count mentions
        mentions = text_lower.count(symbol_lower)
        
        # Check if symbol is in title (more relevant)
        title_mention = symbol_lower in article.get('title', '').lower()
        
        # Calculate score
        relevance = min(1.0, mentions * 0.2)
        if title_mention:
            relevance += 0.3
        
        return min(1.0, relevance)
    
    def _analyze_news_sentiment(self, article: Dict[str, Any]) -> float:
        """Simple sentiment analysis for news articles."""
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        text_lower = text.lower()
        
        # Positive indicators
        positive_terms = [
            'surge', 'gain', 'rise', 'jump', 'soar', 'rally', 'boom',
            'profit', 'beat', 'exceed', 'upgrade', 'buy', 'growth'
        ]
        
        # Negative indicators
        negative_terms = [
            'fall', 'drop', 'plunge', 'crash', 'loss', 'decline', 'slump',
            'miss', 'below', 'downgrade', 'sell', 'warning', 'cut'
        ]
        
        positive_count = sum(1 for term in positive_terms if term in text_lower)
        negative_count = sum(1 for term in negative_terms if term in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total
        return max(-1.0, min(1.0, sentiment))