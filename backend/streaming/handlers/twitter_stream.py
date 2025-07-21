"""
Twitter streaming handler for real-time social sentiment analysis.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import json
import re
from loguru import logger
import aiohttp

from ..realtime_pipeline import DataStreamHandler, StreamConfig, StreamMessage, StreamStatus


class TwitterStreamHandler(DataStreamHandler):
    """Handler for Twitter API v2 streaming for financial sentiment."""
    
    def __init__(self, config: StreamConfig, bearer_token: str):
        super().__init__(config)
        self.bearer_token = bearer_token
        self.session = None
        self.stream_response = None
        self.processing_task = None
        
        # Twitter API v2 endpoints
        self.stream_url = "https://api.twitter.com/2/tweets/search/stream"
        self.rules_url = "https://api.twitter.com/2/tweets/search/stream/rules"
        
        # Track symbols and their related keywords
        self.symbol_keywords: Dict[str, Set[str]] = {}
        self.cashtag_to_symbol: Dict[str, str] = {}  # $AAPL -> AAPL
        
        # Sentiment analysis patterns
        self.positive_words = {
            'bullish', 'buy', 'long', 'moon', 'rocket', 'up', 'green',
            'breakout', 'strong', 'gains', 'profit', 'winner', 'calls'
        }
        self.negative_words = {
            'bearish', 'sell', 'short', 'crash', 'down', 'red', 'dump',
            'breakdown', 'weak', 'loss', 'loser', 'puts', 'drop'
        }
        
    async def connect(self) -> bool:
        """Connect to Twitter streaming API."""
        try:
            self.status = StreamStatus.CONNECTING
            
            # Create session with auth headers
            headers = {
                "Authorization": f"Bearer {self.bearer_token}",
                "User-Agent": "Shagun IntelligenceStreamingBot/1.0"
            }
            
            timeout = aiohttp.ClientTimeout(total=0, sock_read=90)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
            
            # Clear existing rules
            await self._clear_rules()
            
            self.status = StreamStatus.CONNECTED
            logger.info("Connected to Twitter Streaming API")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Twitter: {e}")
            self.status = StreamStatus.ERROR
            self.metrics.errors_count += 1
            return False
    
    async def disconnect(self):
        """Disconnect from Twitter streaming."""
        try:
            if self.processing_task:
                self.processing_task.cancel()
                
            if self.stream_response:
                self.stream_response.close()
                
            if self.session:
                await self.session.close()
                
            self.status = StreamStatus.DISCONNECTED
            logger.info("Disconnected from Twitter Streaming API")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Twitter: {e}")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to Twitter streams for symbols."""
        try:
            # Build rules for symbols
            rules = []
            
            for symbol in symbols:
                # Create keywords for this symbol
                keywords = {
                    symbol,
                    f"${symbol}",  # Cashtag
                    f"#{symbol}",  # Hashtag
                }
                
                # Add company-specific keywords
                company_keywords = self._get_company_keywords(symbol)
                keywords.update(company_keywords)
                
                self.symbol_keywords[symbol] = keywords
                self.cashtag_to_symbol[f"${symbol}"] = symbol
                
                # Create Twitter rule
                # Look for cashtags and keywords with financial context
                rule_value = f'(${symbol} OR {symbol}) (stock OR trading OR market OR price)'
                
                rules.append({
                    "value": rule_value,
                    "tag": f"symbol:{symbol}"
                })
            
            # Add rules to Twitter
            if rules:
                await self._add_rules(rules)
                logger.info(f"Added {len(rules)} Twitter streaming rules")
                
                # Start streaming
                self.processing_task = asyncio.create_task(self._stream_tweets())
                
        except Exception as e:
            logger.error(f"Error subscribing to Twitter symbols: {e}")
            self.metrics.errors_count += 1
    
    async def process_message(self, message: Any) -> Optional[StreamMessage]:
        """Process tweets from the stream."""
        # Processing is handled in the streaming task
        return None
    
    async def send_heartbeat(self):
        """Check stream health."""
        self.last_heartbeat = datetime.now()
        
        # Check if stream is still active
        if self.processing_task and self.processing_task.done():
            logger.warning("Twitter stream task died, restarting...")
            self.processing_task = asyncio.create_task(self._stream_tweets())
    
    async def _stream_tweets(self):
        """Stream tweets from Twitter API."""
        backoff = 1
        
        while self.status == StreamStatus.CONNECTED:
            try:
                # Connect to stream
                async with self.session.get(
                    self.stream_url,
                    params={
                        "tweet.fields": "created_at,author_id,public_metrics,entities",
                        "expansions": "author_id"
                    }
                ) as response:
                    
                    if response.status == 429:
                        # Rate limited
                        reset_time = int(response.headers.get("x-rate-limit-reset", 0))
                        sleep_time = max(reset_time - int(datetime.now().timestamp()), 60)
                        logger.warning(f"Twitter rate limited, sleeping for {sleep_time}s")
                        await asyncio.sleep(sleep_time)
                        continue
                    
                    if response.status != 200:
                        raise Exception(f"Stream error: {response.status}")
                    
                    # Process stream
                    self.stream_response = response
                    backoff = 1  # Reset backoff on successful connection
                    
                    async for line in response.content:
                        if line:
                            await self._process_tweet(line)
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Twitter stream error: {e}")
                self.metrics.errors_count += 1
                
                # Exponential backoff
                await asyncio.sleep(min(backoff, 60))
                backoff *= 2
    
    async def _process_tweet(self, line: bytes):
        """Process a single tweet from the stream."""
        try:
            data = json.loads(line.decode('utf-8'))
            
            if 'data' not in data:
                return
            
            tweet = data['data']
            matching_rules = data.get('matching_rules', [])
            
            # Extract symbols from matching rules
            symbols = []
            for rule in matching_rules:
                tag = rule.get('tag', '')
                if tag.startswith('symbol:'):
                    symbols.append(tag.split(':')[1])
            
            if not symbols:
                return
            
            # Analyze sentiment
            text = tweet.get('text', '')
            sentiment_score = self._analyze_sentiment(text)
            
            # Extract metrics
            metrics = tweet.get('public_metrics', {})
            engagement_score = self._calculate_engagement_score(metrics)
            
            # Get author info
            author_id = tweet.get('author_id')
            author_influence = await self._get_author_influence(author_id)
            
            # Calculate quality based on engagement and author influence
            quality_score = min(1.0, (engagement_score * 0.5) + (author_influence * 0.5))
            
            # Create messages for each symbol
            for symbol in symbols:
                self.metrics.messages_received += 1
                self.metrics.messages_processed += 1
                self.metrics.last_message_time = datetime.now()
                
                message = StreamMessage(
                    stream_name=self.config.name,
                    symbol=symbol,
                    data={
                        'type': 'social_sentiment',
                        'source': 'twitter',
                        'text': text[:280],  # Truncate for storage
                        'sentiment_score': sentiment_score,
                        'engagement': metrics,
                        'engagement_score': engagement_score,
                        'author_influence': author_influence,
                        'created_at': tweet.get('created_at'),
                        'tweet_id': tweet.get('id')
                    },
                    timestamp=datetime.now(),
                    latency_ms=0,  # Social media doesn't have traditional latency
                    quality_score=quality_score
                )
                
                # This would be sent to the pipeline
                logger.debug(f"Twitter sentiment for {symbol}: {sentiment_score:.2f}")
                
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            self.metrics.errors_count += 1
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis for financial tweets."""
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Check for specific patterns
        if re.search(r'\b(buy|long)\s+calls?\b', text_lower):
            positive_count += 2
        if re.search(r'\b(buy|long)\s+puts?\b', text_lower):
            negative_count += 2
        
        # Calculate sentiment score (-1 to 1)
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_engagement_score(self, metrics: Dict[str, int]) -> float:
        """Calculate engagement score from tweet metrics."""
        retweets = metrics.get('retweet_count', 0)
        likes = metrics.get('like_count', 0)
        replies = metrics.get('reply_count', 0)
        quotes = metrics.get('quote_count', 0)
        
        # Weighted engagement score
        engagement = (retweets * 3) + (likes * 1) + (replies * 2) + (quotes * 2.5)
        
        # Normalize to 0-1 range (assuming 1000 engagement is very high)
        return min(1.0, engagement / 1000)
    
    async def _get_author_influence(self, author_id: str) -> float:
        """Get author influence score (simplified)."""
        # In production, you would fetch user details and calculate
        # based on followers, verification status, etc.
        # For now, return a default score
        return 0.5
    
    async def _add_rules(self, rules: List[Dict[str, str]]):
        """Add streaming rules to Twitter."""
        try:
            async with self.session.post(
                self.rules_url,
                json={"add": rules}
            ) as response:
                if response.status != 201:
                    text = await response.text()
                    raise Exception(f"Failed to add rules: {text}")
                    
                data = await response.json()
                logger.info(f"Added Twitter rules: {data}")
                
        except Exception as e:
            logger.error(f"Error adding Twitter rules: {e}")
            raise
    
    async def _clear_rules(self):
        """Clear existing streaming rules."""
        try:
            # Get existing rules
            async with self.session.get(self.rules_url) as response:
                if response.status != 200:
                    return
                    
                data = await response.json()
                rules = data.get('data', [])
                
                if not rules:
                    return
                
                # Delete all rules
                rule_ids = [rule['id'] for rule in rules]
                
                async with self.session.post(
                    self.rules_url,
                    json={"delete": {"ids": rule_ids}}
                ) as delete_response:
                    if delete_response.status == 200:
                        logger.info(f"Cleared {len(rule_ids)} Twitter rules")
                        
        except Exception as e:
            logger.error(f"Error clearing Twitter rules: {e}")
    
    def _get_company_keywords(self, symbol: str) -> Set[str]:
        """Get company-specific keywords for better matching."""
        # Company name mappings
        company_map = {
            'AAPL': {'apple', 'iphone', 'tim cook'},
            'TSLA': {'tesla', 'elon', 'musk'},
            'AMZN': {'amazon', 'bezos', 'aws'},
            'GOOGL': {'google', 'alphabet', 'android'},
            'MSFT': {'microsoft', 'windows', 'azure'},
            'META': {'meta', 'facebook', 'zuckerberg'},
            'NFLX': {'netflix'},
            'NVDA': {'nvidia', 'jensen'},
            'AMD': {'amd', 'ryzen'},
            'JPM': {'jpmorgan', 'chase', 'dimon'},
            'GS': {'goldman', 'sachs'},
            'RELIANCE': {'reliance', 'ambani', 'jio'},
            'TCS': {'tata', 'tcs'},
            'INFY': {'infosys', 'infosys'},
            'HDFC': {'hdfc'},
            'ICICIBANK': {'icici'},
        }
        
        return company_map.get(symbol, set())