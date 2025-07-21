import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import aiohttp
from loguru import logger
import numpy as np
from enum import Enum

from backend.data_sources.base import (
    SentimentDataSource,
    DataSourceConfig,
    DataSourceType,
    DataSourceStatus
)


class GrokConfidenceLevel(Enum):
    """Confidence levels for Grok responses"""
    VERY_HIGH = "very_high"  # 0.9-1.0
    HIGH = "high"            # 0.7-0.9
    MEDIUM = "medium"        # 0.5-0.7
    LOW = "low"              # 0.3-0.5
    VERY_LOW = "very_low"    # 0.0-0.3


class GrokSentimentResponse:
    """Structured response from Grok sentiment analysis"""
    
    def __init__(
        self,
        symbol: str,
        sentiment_score: float,
        sentiment_label: str,
        confidence: float,
        confidence_level: GrokConfidenceLevel,
        reasoning: str,
        key_factors: List[str],
        trend_direction: str,
        social_volume: int,
        influential_posts: List[Dict[str, Any]],
        timestamp: datetime,
        cost_tokens: int = 0
    ):
        self.symbol = symbol
        self.sentiment_score = sentiment_score
        self.sentiment_label = sentiment_label
        self.confidence = confidence
        self.confidence_level = confidence_level
        self.reasoning = reasoning
        self.key_factors = key_factors
        self.trend_direction = trend_direction
        self.social_volume = social_volume
        self.influential_posts = influential_posts
        self.timestamp = timestamp
        self.cost_tokens = cost_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "trend_direction": self.trend_direction,
            "social_volume": self.social_volume,
            "influential_posts": self.influential_posts,
            "timestamp": self.timestamp.isoformat(),
            "cost_tokens": self.cost_tokens
        }


class GrokPromptBuilder:
    """Builds optimized prompts for Grok sentiment analysis"""
    
    @staticmethod
    def build_single_symbol_prompt(symbol: str, hours: int = 1) -> str:
        """Build prompt for single symbol analysis"""
        return f"""Analyze the current sentiment and social media trends for ${symbol} stock over the past {hours} hour(s) using real-time X platform data.

Provide a comprehensive analysis including:
1. Overall sentiment score (-1.0 to 1.0 where -1 is extremely bearish, 0 is neutral, 1 is extremely bullish)
2. Sentiment label (bearish, neutral, bullish)
3. Confidence level in your analysis (0.0 to 1.0)
4. Key factors driving the sentiment
5. Current trend direction (rising, falling, stable)
6. Approximate social media volume
7. Most influential posts or discussions

Consider:
- Recent news and announcements
- Technical analysis discussions
- Options flow sentiment
- Retail vs institutional sentiment
- Sarcasm and context in posts
- Multi-language posts
- Spam and bot filtering

Format your response as JSON:
{{
    "sentiment_score": float,
    "sentiment_label": string,
    "confidence": float,
    "reasoning": string,
    "key_factors": [list of strings],
    "trend_direction": string,
    "social_volume": integer,
    "influential_posts": [
        {{
            "content": string,
            "influence_score": float,
            "author_type": string
        }}
    ]
}}"""
    
    @staticmethod
    def build_batch_prompt(symbols: List[str], hours: int = 1) -> str:
        """Build prompt for batch symbol analysis"""
        symbols_str = ", ".join([f"${s}" for s in symbols])
        return f"""Analyze the current sentiment for the following stocks: {symbols_str} over the past {hours} hour(s) using real-time X platform data.

For each symbol, provide:
1. Sentiment score (-1.0 to 1.0)
2. Sentiment label (bearish, neutral, bullish)
3. Confidence level (0.0 to 1.0)
4. Brief reasoning
5. Trend direction
6. Social volume indicator

Format your response as JSON:
{{
    "symbols": {{
        "SYMBOL1": {{
            "sentiment_score": float,
            "sentiment_label": string,
            "confidence": float,
            "reasoning": string,
            "trend_direction": string,
            "social_volume": integer
        }},
        ...
    }}
}}"""
    
    @staticmethod
    def build_trend_analysis_prompt(symbol: str, hours: int = 24) -> str:
        """Build prompt for trend analysis"""
        return f"""Analyze the sentiment trend for ${symbol} over the past {hours} hours using real-time X platform data.

Provide hourly sentiment data points showing how sentiment has evolved.
Include any significant events or catalysts that caused sentiment shifts.

Format your response as JSON:
{{
    "symbol": "{symbol}",
    "trend_data": [
        {{
            "hour": integer (hours ago),
            "sentiment_score": float,
            "volume": integer,
            "key_event": string or null
        }}
    ],
    "overall_trend": string,
    "significant_events": [list of strings]
}}"""


class GrokCostManager:
    """Manages API costs and usage optimization"""
    
    def __init__(self, daily_budget: float = 100.0, cost_per_1k_tokens: float = 0.10):
        self.daily_budget = daily_budget
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.daily_usage = defaultdict(float)
        self.token_usage = defaultdict(int)
        self.request_history = deque(maxlen=1000)
    
    def can_make_request(self, estimated_tokens: int = 1000) -> bool:
        """Check if request is within budget"""
        today = datetime.utcnow().date()
        estimated_cost = (estimated_tokens / 1000) * self.cost_per_1k_tokens
        
        return self.daily_usage[today] + estimated_cost <= self.daily_budget
    
    def record_usage(self, tokens: int, cost: float = None):
        """Record API usage"""
        today = datetime.utcnow().date()
        
        if cost is None:
            cost = (tokens / 1000) * self.cost_per_1k_tokens
        
        self.daily_usage[today] += cost
        self.token_usage[today] += tokens
        
        self.request_history.append({
            "timestamp": datetime.utcnow(),
            "tokens": tokens,
            "cost": cost
        })
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        today = datetime.utcnow().date()
        
        return {
            "daily_budget": self.daily_budget,
            "daily_usage": self.daily_usage[today],
            "daily_tokens": self.token_usage[today],
            "budget_remaining": self.daily_budget - self.daily_usage[today],
            "requests_today": len([r for r in self.request_history 
                                 if r["timestamp"].date() == today])
        }


class GrokBatchProcessor:
    """Handles intelligent batching of requests"""
    
    def __init__(self, max_batch_size: int = 10, batch_delay: float = 2.0):
        self.max_batch_size = max_batch_size
        self.batch_delay = batch_delay
        self.pending_requests = defaultdict(list)
        self.batch_lock = asyncio.Lock()
        self.processing_task = None
    
    async def add_request(
        self,
        symbol: str,
        callback: Any,
        priority: int = 5
    ) -> None:
        """Add request to batch queue"""
        async with self.batch_lock:
            self.pending_requests[priority].append({
                "symbol": symbol,
                "callback": callback,
                "timestamp": datetime.utcnow()
            })
            
            # Start processor if not running
            if not self.processing_task or self.processing_task.done():
                self.processing_task = asyncio.create_task(self._process_batches())
    
    async def _process_batches(self) -> None:
        """Process pending requests in batches"""
        await asyncio.sleep(self.batch_delay)  # Wait for batch to fill
        
        async with self.batch_lock:
            # Get requests by priority
            batch = []
            for priority in sorted(self.pending_requests.keys()):
                requests = self.pending_requests[priority]
                batch.extend(requests[:self.max_batch_size - len(batch)])
                self.pending_requests[priority] = requests[self.max_batch_size - len(batch):]
                
                if len(batch) >= self.max_batch_size:
                    break
            
            # Clear empty priorities
            self.pending_requests = defaultdict(list, 
                {k: v for k, v in self.pending_requests.items() if v})
        
        return batch


class GrokSentimentSource(SentimentDataSource):
    """xAI Grok API integration for advanced sentiment analysis"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.api_key = config.credentials.get("api_key")
        self.api_url = config.credentials.get("api_url", "https://api.x.ai/v1/chat/completions")
        
        # Components
        self.prompt_builder = GrokPromptBuilder()
        self.cost_manager = GrokCostManager(
            daily_budget=config.credentials.get("daily_budget", 100.0)
        )
        self.batch_processor = GrokBatchProcessor()
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Caching
        self.cache_ttl = 300  # 5 minutes
        self.sentiment_cache = {}
        
        # Reliability tracking
        self.success_count = 0
        self.error_count = 0
        self.avg_response_time = 0
        
        # Rate limiting
        self.requests_per_minute = 30
        self.request_times = deque(maxlen=self.requests_per_minute)
        
        logger.info("Initialized GrokSentimentSource")
    
    async def connect(self) -> bool:
        """Connect to Grok API"""
        try:
            if not self.api_key:
                raise ValueError("Grok API key not provided")
            
            # Create session with auth headers
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            # Test connection with a simple request
            test_response = await self._make_request(
                "Test connection", 
                max_tokens=10
            )
            
            if test_response:
                self.update_health_status(DataSourceStatus.HEALTHY)
                logger.info("Connected to Grok API successfully")
                return True
            else:
                raise Exception("Failed to validate Grok API connection")
                
        except Exception as e:
            logger.error(f"Failed to connect to Grok API: {e}")
            self.update_health_status(DataSourceStatus.UNHEALTHY, str(e))
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Grok API"""
        try:
            if self.session:
                await self.session.close()
            
            self.update_health_status(DataSourceStatus.DISCONNECTED)
            logger.info("Disconnected from Grok API")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Grok API: {e}")
    
    async def _make_request(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> Optional[str]:
        """Make request to Grok API with rate limiting and error handling"""
        # Rate limiting
        await self._check_rate_limit()
        
        # Check budget
        if not self.cost_manager.can_make_request(max_tokens):
            logger.warning("Daily budget exceeded for Grok API")
            return None
        
        start_time = time.time()
        
        try:
            payload = {
                "model": "grok-beta",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial sentiment analyst with access to real-time X platform data. Provide accurate, data-driven analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": {"type": "json_object"}
            }
            
            async with self.session.post(
                self.api_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract response
                    content = data["choices"][0]["message"]["content"]
                    tokens_used = data.get("usage", {}).get("total_tokens", max_tokens)
                    
                    # Record usage
                    self.cost_manager.record_usage(tokens_used)
                    
                    # Update metrics
                    response_time = time.time() - start_time
                    self._update_metrics(success=True, response_time=response_time)
                    
                    return content
                    
                elif response.status == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Grok API rate limited, retry after {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return None
                    
                else:
                    error_text = await response.text()
                    logger.error(f"Grok API error {response.status}: {error_text}")
                    self._update_metrics(success=False)
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("Grok API request timeout")
            self._update_metrics(success=False)
            return None
            
        except Exception as e:
            logger.error(f"Grok API request error: {e}")
            self._update_metrics(success=False)
            return None
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits"""
        now = time.time()
        
        # Remove old requests
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # Check if at limit
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Record request
        self.request_times.append(now)
    
    def _update_metrics(self, success: bool, response_time: float = 0):
        """Update reliability metrics"""
        if success:
            self.success_count += 1
            # Update average response time
            total_requests = self.success_count + self.error_count
            self.avg_response_time = (
                (self.avg_response_time * (total_requests - 1) + response_time) /
                total_requests
            )
        else:
            self.error_count += 1
        
        # Update health status
        success_rate = self.success_count / (self.success_count + self.error_count)
        if success_rate < 0.5:
            self.update_health_status(DataSourceStatus.UNHEALTHY)
        elif success_rate < 0.8:
            self.update_health_status(DataSourceStatus.DEGRADED)
        else:
            self.update_health_status(DataSourceStatus.HEALTHY)
    
    def _parse_grok_response(
        self,
        response_text: str,
        symbol: str
    ) -> Optional[GrokSentimentResponse]:
        """Parse Grok JSON response into structured format"""
        try:
            data = json.loads(response_text)
            
            # Determine confidence level
            confidence = data.get("confidence", 0.5)
            if confidence >= 0.9:
                confidence_level = GrokConfidenceLevel.VERY_HIGH
            elif confidence >= 0.7:
                confidence_level = GrokConfidenceLevel.HIGH
            elif confidence >= 0.5:
                confidence_level = GrokConfidenceLevel.MEDIUM
            elif confidence >= 0.3:
                confidence_level = GrokConfidenceLevel.LOW
            else:
                confidence_level = GrokConfidenceLevel.VERY_LOW
            
            return GrokSentimentResponse(
                symbol=symbol,
                sentiment_score=float(data.get("sentiment_score", 0)),
                sentiment_label=data.get("sentiment_label", "neutral"),
                confidence=confidence,
                confidence_level=confidence_level,
                reasoning=data.get("reasoning", ""),
                key_factors=data.get("key_factors", []),
                trend_direction=data.get("trend_direction", "stable"),
                social_volume=int(data.get("social_volume", 0)),
                influential_posts=data.get("influential_posts", []),
                timestamp=datetime.utcnow(),
                cost_tokens=data.get("tokens_used", 0)
            )
            
        except Exception as e:
            logger.error(f"Error parsing Grok response: {e}")
            logger.debug(f"Response text: {response_text}")
            return None
    
    async def analyze_sentiment(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[GrokSentimentResponse]:
        """Analyze sentiment for a single symbol"""
        # Check cache
        cache_key = f"{symbol}_{int(datetime.utcnow().timestamp() / self.cache_ttl)}"
        if use_cache and cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Build prompt
        prompt = self.prompt_builder.build_single_symbol_prompt(symbol)
        
        # Make request
        response_text = await self._make_request(prompt)
        if not response_text:
            return None
        
        # Parse response
        sentiment_response = self._parse_grok_response(response_text, symbol)
        if sentiment_response:
            # Cache result
            self.sentiment_cache[cache_key] = sentiment_response
        
        return sentiment_response
    
    async def analyze_batch(
        self,
        symbols: List[str],
        use_cache: bool = True
    ) -> Dict[str, GrokSentimentResponse]:
        """Analyze sentiment for multiple symbols in one request"""
        results = {}
        
        # Check cache for each symbol
        uncached_symbols = []
        for symbol in symbols:
            cache_key = f"{symbol}_{int(datetime.utcnow().timestamp() / self.cache_ttl)}"
            if use_cache and cache_key in self.sentiment_cache:
                results[symbol] = self.sentiment_cache[cache_key]
            else:
                uncached_symbols.append(symbol)
        
        if not uncached_symbols:
            return results
        
        # Batch uncached symbols
        batches = [uncached_symbols[i:i+10] for i in range(0, len(uncached_symbols), 10)]
        
        for batch in batches:
            prompt = self.prompt_builder.build_batch_prompt(batch)
            response_text = await self._make_request(prompt, max_tokens=2000)
            
            if response_text:
                try:
                    data = json.loads(response_text)
                    symbols_data = data.get("symbols", {})
                    
                    for symbol, symbol_data in symbols_data.items():
                        sentiment_response = GrokSentimentResponse(
                            symbol=symbol,
                            sentiment_score=float(symbol_data.get("sentiment_score", 0)),
                            sentiment_label=symbol_data.get("sentiment_label", "neutral"),
                            confidence=float(symbol_data.get("confidence", 0.5)),
                            confidence_level=self._get_confidence_level(
                                float(symbol_data.get("confidence", 0.5))
                            ),
                            reasoning=symbol_data.get("reasoning", ""),
                            key_factors=[],
                            trend_direction=symbol_data.get("trend_direction", "stable"),
                            social_volume=int(symbol_data.get("social_volume", 0)),
                            influential_posts=[],
                            timestamp=datetime.utcnow()
                        )
                        
                        results[symbol] = sentiment_response
                        
                        # Cache result
                        cache_key = f"{symbol}_{int(datetime.utcnow().timestamp() / self.cache_ttl)}"
                        self.sentiment_cache[cache_key] = sentiment_response
                        
                except Exception as e:
                    logger.error(f"Error parsing batch response: {e}")
        
        return results
    
    def _get_confidence_level(self, confidence: float) -> GrokConfidenceLevel:
        """Get confidence level enum from float value"""
        if confidence >= 0.9:
            return GrokConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return GrokConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return GrokConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return GrokConfidenceLevel.LOW
        else:
            return GrokConfidenceLevel.VERY_LOW
    
    # SentimentDataSource interface methods
    
    async def get_sentiment_score(self, symbol: str) -> Dict[str, Any]:
        """Get current sentiment score for a symbol"""
        try:
            response = await self.analyze_sentiment(symbol)
            
            if response:
                return {
                    "symbol": symbol,
                    "sentiment_score": response.sentiment_score,
                    "sentiment_label": response.sentiment_label,
                    "confidence": response.confidence,
                    "confidence_level": response.confidence_level.value,
                    "reasoning": response.reasoning,
                    "key_factors": response.key_factors,
                    "trend_direction": response.trend_direction,
                    "social_volume": response.social_volume,
                    "source": "grok",
                    "timestamp": response.timestamp.isoformat()
                }
            else:
                return {
                    "symbol": symbol,
                    "sentiment_score": 0.0,
                    "sentiment_label": "neutral",
                    "confidence": 0.0,
                    "source": "grok",
                    "error": "Failed to get Grok analysis"
                }
                
        except Exception as e:
            logger.error(f"Error getting Grok sentiment for {symbol}: {e}")
            return {
                "symbol": symbol,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "source": "grok",
                "error": str(e)
            }
    
    async def get_news_sentiment(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get historical sentiment data - Grok provides current analysis only"""
        # Grok focuses on real-time analysis
        current_sentiment = await self.get_sentiment_score(symbol)
        return [current_sentiment] if current_sentiment else []
    
    async def get_social_sentiment(
        self,
        symbol: str,
        platform: str = "x"
    ) -> Dict[str, Any]:
        """Get social media sentiment - Grok specializes in X platform data"""
        return await self.get_sentiment_score(symbol)
    
    async def get_sentiment_trends(
        self,
        symbol: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get sentiment trends over time"""
        try:
            prompt = self.prompt_builder.build_trend_analysis_prompt(symbol, hours)
            response_text = await self._make_request(prompt, max_tokens=1500)
            
            if response_text:
                data = json.loads(response_text)
                return {
                    "symbol": symbol,
                    "trends": data.get("trend_data", []),
                    "overall_trend": data.get("overall_trend", "stable"),
                    "significant_events": data.get("significant_events", []),
                    "source": "grok",
                    "period_hours": hours
                }
            else:
                return {
                    "symbol": symbol,
                    "trends": [],
                    "source": "grok",
                    "error": "Failed to get trend analysis"
                }
                
        except Exception as e:
            logger.error(f"Error getting Grok trends for {symbol}: {e}")
            return {
                "symbol": symbol,
                "trends": [],
                "source": "grok",
                "error": str(e)
            }
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """Get current API cost statistics"""
        return self.cost_manager.get_usage_stats()
    
    def get_reliability_score(self) -> float:
        """Get current reliability score"""
        total = self.success_count + self.error_count
        if total == 0:
            return 1.0
        return self.success_count / total