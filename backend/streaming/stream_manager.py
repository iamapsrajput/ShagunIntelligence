"""
Stream Manager for orchestrating all real-time data streams.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
from loguru import logger
from dataclasses import dataclass
import os

from .realtime_pipeline import RealTimeDataPipeline, StreamConfig, StreamMessage
from .handlers.kite_stream import KiteStreamHandler
from .handlers.alpha_vantage_stream import AlphaVantageStreamHandler
from .handlers.finnhub_stream import FinnhubStreamHandler
from .handlers.twitter_stream import TwitterStreamHandler
from .handlers.news_stream import NewsStreamHandler


@dataclass
class StreamManagerConfig:
    """Configuration for the stream manager."""
    redis_url: str = "redis://localhost:6379"
    enable_kite: bool = True
    enable_alpha_vantage: bool = True
    enable_finnhub: bool = True
    enable_twitter: bool = True
    enable_news: bool = True
    
    # API credentials (loaded from environment)
    kite_api_key: Optional[str] = None
    kite_access_token: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    newsapi_key: Optional[str] = None


class StreamManager:
    """
    High-level manager for all real-time data streams.
    
    Features:
    - Manages multiple streaming sources
    - Provides unified interface for agents
    - Handles stream prioritization and failover
    - Monitors overall system health
    """
    
    def __init__(self, config: StreamManagerConfig = None):
        self.config = config or self._load_default_config()
        self.pipeline = RealTimeDataPipeline(redis_url=self.config.redis_url)
        self.initialized = False
        self.subscribed_symbols: List[str] = []
        
        # Agent callbacks
        self.agent_callbacks: Dict[str, List[Callable]] = {}
        
        logger.info("StreamManager initialized")
    
    def _load_default_config(self) -> StreamManagerConfig:
        """Load configuration from environment variables."""
        return StreamManagerConfig(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            kite_api_key=os.getenv("KITE_API_KEY"),
            kite_access_token=os.getenv("KITE_ACCESS_TOKEN"),
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
            twitter_bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
            newsapi_key=os.getenv("NEWSAPI_KEY")
        )
    
    async def initialize(self):
        """Initialize the streaming system."""
        try:
            # Initialize pipeline
            await self.pipeline.initialize()
            
            # Add configured streams
            await self._setup_streams()
            
            self.initialized = True
            logger.info("StreamManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize StreamManager: {e}")
            raise
    
    async def _setup_streams(self):
        """Set up all configured streaming handlers."""
        
        # Kite Connect stream (highest priority for Indian markets)
        if self.config.enable_kite and self.config.kite_api_key:
            kite_config = StreamConfig(
                name="kite_connect",
                url="wss://ws.kite.trade",
                priority=10,  # Highest priority
                reconnect_interval=5.0,
                quality_threshold=0.85,
                latency_threshold_ms=50.0
            )
            
            kite_handler = KiteStreamHandler(
                config=kite_config,
                api_key=self.config.kite_api_key,
                access_token=self.config.kite_access_token
            )
            
            self.pipeline.add_stream(kite_handler)
            logger.info("Added Kite Connect stream")
        
        # Alpha Vantage stream
        if self.config.enable_alpha_vantage and self.config.alpha_vantage_api_key:
            av_config = StreamConfig(
                name="alpha_vantage",
                url="https://www.alphavantage.co",
                priority=7,
                reconnect_interval=10.0,
                quality_threshold=0.7,
                latency_threshold_ms=200.0
            )
            
            av_handler = AlphaVantageStreamHandler(
                config=av_config,
                api_key=self.config.alpha_vantage_api_key
            )
            
            self.pipeline.add_stream(av_handler)
            logger.info("Added Alpha Vantage stream")
        
        # Finnhub stream
        if self.config.enable_finnhub and self.config.finnhub_api_key:
            finnhub_config = StreamConfig(
                name="finnhub",
                url="wss://ws.finnhub.io",
                priority=8,
                reconnect_interval=5.0,
                quality_threshold=0.8,
                latency_threshold_ms=100.0
            )
            
            finnhub_handler = FinnhubStreamHandler(
                config=finnhub_config,
                api_key=self.config.finnhub_api_key
            )
            
            self.pipeline.add_stream(finnhub_handler)
            logger.info("Added Finnhub stream")
        
        # Twitter sentiment stream
        if self.config.enable_twitter and self.config.twitter_bearer_token:
            twitter_config = StreamConfig(
                name="twitter_sentiment",
                url="https://api.twitter.com/2/tweets/search/stream",
                priority=5,
                reconnect_interval=30.0,
                quality_threshold=0.6,
                latency_threshold_ms=1000.0  # Social media has higher latency tolerance
            )
            
            twitter_handler = TwitterStreamHandler(
                config=twitter_config,
                bearer_token=self.config.twitter_bearer_token
            )
            
            self.pipeline.add_stream(twitter_handler)
            logger.info("Added Twitter sentiment stream")
        
        # News stream
        if self.config.enable_news:
            news_config = StreamConfig(
                name="news_feed",
                url="multiple_sources",
                priority=6,
                reconnect_interval=60.0,
                quality_threshold=0.7,
                latency_threshold_ms=5000.0  # News can have higher latency
            )
            
            api_keys = {}
            if self.config.newsapi_key:
                api_keys['newsapi'] = self.config.newsapi_key
            
            news_handler = NewsStreamHandler(
                config=news_config,
                api_keys=api_keys
            )
            
            self.pipeline.add_stream(news_handler)
            logger.info("Added News feed stream")
    
    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to real-time data for given symbols."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Update subscribed symbols
            self.subscribed_symbols = list(set(self.subscribed_symbols + symbols))
            
            # Start streams for symbols
            for stream_name, handler in self.pipeline.streams.items():
                await self.pipeline.start_stream(stream_name, symbols)
                logger.info(f"Subscribed {len(symbols)} symbols to {stream_name}")
            
            # Set up primary streams for each symbol
            for symbol in symbols:
                primary_stream = await self.pipeline._find_best_stream_for_symbol(symbol)
                if primary_stream:
                    self.pipeline.primary_streams[symbol] = primary_stream
                    logger.info(f"Primary stream for {symbol}: {primary_stream}")
            
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {e}")
            raise
    
    def register_agent_callback(self, agent_id: str, callback: Callable):
        """Register an agent callback for real-time updates."""
        if agent_id not in self.agent_callbacks:
            self.agent_callbacks[agent_id] = []
        
        self.agent_callbacks[agent_id].append(callback)
        
        # Subscribe to all symbols for this agent
        for symbol in self.subscribed_symbols:
            self.pipeline.subscribe(symbol, callback)
        
        # Also subscribe to system events
        self.pipeline.subscribe('_quality_', callback)  # Quality alerts
        self.pipeline.subscribe('_status_', callback)   # Status updates
        
        logger.info(f"Registered callback for agent: {agent_id}")
    
    async def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest data for a symbol with quality info."""
        message = await self.pipeline.get_latest_data(symbol)
        
        if message:
            return {
                'symbol': message.symbol,
                'data': message.data,
                'timestamp': message.timestamp.isoformat(),
                'stream': message.stream_name,
                'quality_score': message.quality_score,
                'latency_ms': message.latency_ms,
                'primary_stream': self.pipeline.primary_streams.get(symbol)
            }
        
        return None
    
    async def get_multi_stream_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from all available streams for comparison."""
        multi_data = {
            'symbol': symbol,
            'streams': {},
            'best_quality_stream': None,
            'consensus': None
        }
        
        best_quality = 0.0
        prices = []
        
        # Collect data from each stream
        for stream_name, handler in self.pipeline.streams.items():
            if handler.status.value == "connected":
                # Try to get latest data from this stream's buffer
                stream_buffer = f"{symbol}:{stream_name}"
                if stream_buffer in self.pipeline.data_buffers:
                    buffer = self.pipeline.data_buffers[stream_buffer]
                    if buffer:
                        latest = buffer[-1]
                        stream_data = {
                            'price': latest.data.get('price') or latest.data.get('last_price'),
                            'timestamp': latest.timestamp.isoformat(),
                            'quality_score': latest.quality_score,
                            'latency_ms': latest.latency_ms
                        }
                        
                        multi_data['streams'][stream_name] = stream_data
                        
                        # Track best quality
                        if latest.quality_score > best_quality:
                            best_quality = latest.quality_score
                            multi_data['best_quality_stream'] = stream_name
                        
                        # Collect prices for consensus
                        if stream_data['price']:
                            prices.append(stream_data['price'])
        
        # Calculate consensus if multiple prices available
        if len(prices) > 1:
            import numpy as np
            multi_data['consensus'] = {
                'mean_price': np.mean(prices),
                'std_dev': np.std(prices),
                'min_price': min(prices),
                'max_price': max(prices),
                'spread_percent': (max(prices) - min(prices)) / np.mean(prices) * 100,
                'sources_count': len(prices)
            }
        
        return multi_data
    
    async def get_historical_buffer(
        self, 
        symbol: str, 
        seconds: int = 60,
        stream: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get historical data from buffer."""
        if stream:
            # Get from specific stream
            messages = await self.pipeline.get_historical_buffer(f"{symbol}:{stream}", seconds)
        else:
            # Get from primary stream
            messages = await self.pipeline.get_historical_buffer(symbol, seconds)
        
        return [
            {
                'data': msg.data,
                'timestamp': msg.timestamp.isoformat(),
                'quality_score': msg.quality_score,
                'latency_ms': msg.latency_ms,
                'stream': msg.stream_name
            }
            for msg in messages
        ]
    
    def get_stream_health(self) -> Dict[str, Any]:
        """Get comprehensive health report of all streams."""
        return self.pipeline.get_stream_quality_report()
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        metrics = self.pipeline.pipeline_metrics.copy()
        
        # Add derived metrics
        if metrics['total_messages'] > 0:
            uptime = sum(
                h.metrics.uptime_seconds 
                for h in self.pipeline.streams.values()
            ) / len(self.pipeline.streams)
            
            if uptime > 0:
                metrics['messages_per_minute'] = (
                    metrics['total_messages'] / (uptime / 60)
                )
        
        # Add stream-specific metrics
        metrics['stream_metrics'] = {}
        for name, handler in self.pipeline.streams.items():
            metrics['stream_metrics'][name] = {
                'status': handler.status.value,
                'messages': handler.metrics.messages_received,
                'errors': handler.metrics.errors_count,
                'latency_ms': handler.metrics.average_latency_ms,
                'quality': self.pipeline.stream_quality.get(name, 'unknown')
            }
        
        return metrics
    
    async def perform_quality_test(self, symbol: str) -> Dict[str, Any]:
        """Perform a quality test across all streams for a symbol."""
        logger.info(f"Performing quality test for {symbol}")
        
        test_results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'streams': {}
        }
        
        # Test each stream
        for stream_name, handler in self.pipeline.streams.items():
            if handler.status.value == "connected":
                try:
                    # Force a data fetch
                    start_time = datetime.now()
                    
                    # Wait for fresh data (max 5 seconds)
                    fresh_data = None
                    for _ in range(50):  # 50 * 0.1 = 5 seconds
                        data = await self.pipeline.get_latest_data(symbol)
                        if data and data.stream_name == stream_name:
                            if data.timestamp > start_time:
                                fresh_data = data
                                break
                        await asyncio.sleep(0.1)
                    
                    if fresh_data:
                        fetch_time = (datetime.now() - start_time).total_seconds()
                        
                        test_results['streams'][stream_name] = {
                            'success': True,
                            'fetch_time_seconds': fetch_time,
                            'quality_score': fresh_data.quality_score,
                            'latency_ms': fresh_data.latency_ms,
                            'data_received': bool(fresh_data.data)
                        }
                    else:
                        test_results['streams'][stream_name] = {
                            'success': False,
                            'error': 'No data received within timeout'
                        }
                        
                except Exception as e:
                    test_results['streams'][stream_name] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                test_results['streams'][stream_name] = {
                    'success': False,
                    'error': f'Stream not connected: {handler.status.value}'
                }
        
        # Determine best stream
        best_stream = None
        best_score = 0
        
        for stream, result in test_results['streams'].items():
            if result.get('success') and result.get('quality_score', 0) > best_score:
                best_score = result['quality_score']
                best_stream = stream
        
        test_results['recommendation'] = {
            'best_stream': best_stream,
            'best_quality_score': best_score
        }
        
        return test_results
    
    async def shutdown(self):
        """Gracefully shutdown the streaming system."""
        logger.info("Shutting down StreamManager")
        
        try:
            # Unsubscribe all callbacks
            for agent_id in self.agent_callbacks:
                for symbol in self.subscribed_symbols:
                    for callback in self.agent_callbacks[agent_id]:
                        self.pipeline.unsubscribe(symbol, callback)
            
            # Shutdown pipeline
            await self.pipeline.shutdown()
            
            logger.info("StreamManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Singleton instance
_stream_manager_instance = None

def get_stream_manager(config: StreamManagerConfig = None) -> StreamManager:
    """Get or create the global StreamManager instance."""
    global _stream_manager_instance
    
    if _stream_manager_instance is None:
        _stream_manager_instance = StreamManager(config)
    
    return _stream_manager_instance