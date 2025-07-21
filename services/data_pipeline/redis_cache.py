import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import pickle
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisCacheManager:
    """Optional Redis cache manager for data persistence and sharing"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pool: Optional[ConnectionPool] = None
        
        # Cache configuration
        self.cache_config = {
            "tick_ttl": 3600,        # 1 hour TTL for tick data
            "ohlc_ttl": 86400,       # 24 hours TTL for OHLC data
            "buffer_ttl": 300,       # 5 minutes TTL for buffer snapshots
            "max_ticks_per_key": 1000  # Max ticks to store per symbol
        }
        
        # Key prefixes
        self.key_prefixes = {
            "tick": "algohive:tick:",
            "ohlc": "algohive:ohlc:",
            "buffer": "algohive:buffer:",
            "stats": "algohive:stats:",
            "snapshot": "algohive:snapshot:"
        }
        
        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_writes": 0,
            "cache_errors": 0
        }
        
        logger.info(f"RedisCacheManager initialized with URL: {redis_url}")
    
    async def connect(self) -> None:
        """Connect to Redis"""
        try:
            # Create connection pool
            self.pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=50,
                decode_responses=False  # We'll handle encoding/decoding
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Connected to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    async def close(self) -> None:
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.pool:
            await self.pool.disconnect()
        
        logger.info("Redis connection closed")
    
    async def cache_tick(self, symbol: str, tick_data: Dict[str, Any]) -> None:
        """Cache tick data"""
        if not self.redis_client:
            return
        
        try:
            # Create key
            key = f"{self.key_prefixes['tick']}{symbol}"
            
            # Serialize tick data
            serialized = self._serialize(tick_data)
            
            # Add to list (capped)
            pipe = self.redis_client.pipeline()
            pipe.lpush(key, serialized)
            pipe.ltrim(key, 0, self.cache_config["max_ticks_per_key"] - 1)
            pipe.expire(key, self.cache_config["tick_ttl"])
            
            await pipe.execute()
            
            self.stats["cache_writes"] += 1
            
        except Exception as e:
            logger.error(f"Error caching tick: {str(e)}")
            self.stats["cache_errors"] += 1
    
    async def get_cached_ticks(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """Get cached tick data"""
        if not self.redis_client:
            return []
        
        try:
            key = f"{self.key_prefixes['tick']}{symbol}"
            
            # Get ticks from list
            serialized_ticks = await self.redis_client.lrange(key, 0, count - 1)
            
            if serialized_ticks:
                self.stats["cache_hits"] += 1
                return [self._deserialize(tick) for tick in serialized_ticks]
            else:
                self.stats["cache_misses"] += 1
                return []
                
        except Exception as e:
            logger.error(f"Error getting cached ticks: {str(e)}")
            self.stats["cache_errors"] += 1
            return []
    
    async def cache_ohlc(self, symbol: str, interval: str, 
                        ohlc_data: List[Dict[str, Any]]) -> None:
        """Cache OHLC data"""
        if not self.redis_client:
            return
        
        try:
            key = f"{self.key_prefixes['ohlc']}{symbol}:{interval}"
            
            # Serialize OHLC data
            serialized = self._serialize(ohlc_data)
            
            # Set with TTL
            await self.redis_client.setex(
                key,
                self.cache_config["ohlc_ttl"],
                serialized
            )
            
            self.stats["cache_writes"] += 1
            
        except Exception as e:
            logger.error(f"Error caching OHLC: {str(e)}")
            self.stats["cache_errors"] += 1
    
    async def get_cached_ohlc(self, symbol: str, interval: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached OHLC data"""
        if not self.redis_client:
            return None
        
        try:
            key = f"{self.key_prefixes['ohlc']}{symbol}:{interval}"
            
            serialized = await self.redis_client.get(key)
            
            if serialized:
                self.stats["cache_hits"] += 1
                return self._deserialize(serialized)
            else:
                self.stats["cache_misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached OHLC: {str(e)}")
            self.stats["cache_errors"] += 1
            return None
    
    async def save_buffer_snapshot(self, buffer_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Save buffer snapshot"""
        if not self.redis_client:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            key = f"{self.key_prefixes['snapshot']}{timestamp}"
            
            # Serialize buffer data
            serialized = self._serialize(buffer_data)
            
            # Save with TTL
            await self.redis_client.setex(
                key,
                self.cache_config["buffer_ttl"],
                serialized
            )
            
            # Also save to a latest snapshot key
            latest_key = f"{self.key_prefixes['snapshot']}latest"
            await self.redis_client.set(latest_key, serialized)
            
            logger.info(f"Saved buffer snapshot: {len(buffer_data)} symbols")
            
        except Exception as e:
            logger.error(f"Error saving buffer snapshot: {str(e)}")
            self.stats["cache_errors"] += 1
    
    async def get_latest_snapshot(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Get latest buffer snapshot"""
        if not self.redis_client:
            return None
        
        try:
            key = f"{self.key_prefixes['snapshot']}latest"
            serialized = await self.redis_client.get(key)
            
            if serialized:
                return self._deserialize(serialized)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest snapshot: {str(e)}")
            return None
    
    async def save_buffer(self, symbol: str, data: List[Dict[str, Any]]) -> None:
        """Save symbol-specific buffer data"""
        if not self.redis_client:
            return
        
        try:
            key = f"{self.key_prefixes['buffer']}{symbol}"
            
            # Serialize data
            serialized = self._serialize(data)
            
            # Save with TTL
            await self.redis_client.setex(
                key,
                self.cache_config["buffer_ttl"],
                serialized
            )
            
        except Exception as e:
            logger.error(f"Error saving buffer for {symbol}: {str(e)}")
            self.stats["cache_errors"] += 1
    
    async def get_buffer(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Get symbol-specific buffer data"""
        if not self.redis_client:
            return None
        
        try:
            key = f"{self.key_prefixes['buffer']}{symbol}"
            serialized = await self.redis_client.get(key)
            
            if serialized:
                return self._deserialize(serialized)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting buffer for {symbol}: {str(e)}")
            return None
    
    async def publish_tick(self, channel: str, tick_data: Dict[str, Any]) -> None:
        """Publish tick data to Redis pub/sub channel"""
        if not self.redis_client:
            return
        
        try:
            serialized = json.dumps(tick_data, default=str)
            await self.redis_client.publish(channel, serialized)
            
        except Exception as e:
            logger.error(f"Error publishing tick: {str(e)}")
    
    async def subscribe_to_channel(self, channel: str) -> redis.client.PubSub:
        """Subscribe to Redis pub/sub channel"""
        if not self.redis_client:
            raise ConnectionError("Redis not connected")
        
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        return pubsub
    
    async def save_stats(self, stats_type: str, stats_data: Dict[str, Any]) -> None:
        """Save statistics data"""
        if not self.redis_client:
            return
        
        try:
            key = f"{self.key_prefixes['stats']}{stats_type}"
            
            # Add timestamp
            stats_data["timestamp"] = datetime.now().isoformat()
            
            # Serialize and save
            serialized = self._serialize(stats_data)
            await self.redis_client.set(key, serialized)
            
        except Exception as e:
            logger.error(f"Error saving stats: {str(e)}")
    
    async def get_stats(self, stats_type: str) -> Optional[Dict[str, Any]]:
        """Get statistics data"""
        if not self.redis_client:
            return None
        
        try:
            key = f"{self.key_prefixes['stats']}{stats_type}"
            serialized = await self.redis_client.get(key)
            
            if serialized:
                return self._deserialize(serialized)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return None
    
    async def cleanup_old_data(self, days: int = 1) -> int:
        """Clean up old data from cache"""
        if not self.redis_client:
            return 0
        
        try:
            deleted_count = 0
            
            # Get all keys with pattern
            cursor = 0
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor,
                    match="algohive:*",
                    count=100
                )
                
                for key in keys:
                    # Check TTL
                    ttl = await self.redis_client.ttl(key)
                    
                    # Delete if expired or no TTL set
                    if ttl == -1:  # No TTL
                        await self.redis_client.delete(key)
                        deleted_count += 1
                
                if cursor == 0:
                    break
            
            logger.info(f"Cleaned up {deleted_count} keys from cache")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
            return 0
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage"""
        try:
            # Use pickle for complex objects
            return pickle.dumps(data)
        except Exception:
            # Fallback to JSON
            return json.dumps(data, default=str).encode('utf-8')
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage"""
        try:
            # Try pickle first
            return pickle.loads(data)
        except Exception:
            # Fallback to JSON
            return json.loads(data.decode('utf-8'))
    
    async def ping(self) -> bool:
        """Check if Redis is connected"""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = (self.stats["cache_hits"] + 
                         self.stats["cache_misses"])
        
        hit_rate = (
            (self.stats["cache_hits"] / total_requests * 100)
            if total_requests > 0 else 0
        )
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "connected": self.redis_client is not None
        }