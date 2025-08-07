import hashlib
import json
import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseCache:
    """Caches AI service responses to reduce costs and improve performance"""

    def __init__(
        self,
        cache_dir: str = "data/ai_cache",
        max_memory_items: int = 1000,
        default_ttl_hours: int = 24,
        enable_disk_cache: bool = True,
    ):
        self.cache_dir = Path(cache_dir) if enable_disk_cache else None
        self.max_memory_items = max_memory_items
        self.default_ttl_hours = default_ttl_hours
        self.enable_disk_cache = enable_disk_cache

        # In-memory cache using OrderedDict for LRU
        self.memory_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "disk_hits": 0,
            "evictions": 0,
            "total_saved_tokens": 0,
            "total_saved_cost": 0.0,
        }

        # TTL overrides for specific use cases
        self.ttl_overrides = {
            "market_sentiment": 1,  # 1 hour for rapidly changing sentiment
            "technical_analysis": 6,  # 6 hours for technical data
            "news_summary": 12,  # 12 hours for news
            "earnings_analysis": 168,  # 1 week for earnings data
            "company_info": 720,  # 30 days for static company info
        }

        # Initialize disk cache
        if self.enable_disk_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_old_cache_files()

        logger.info(
            f"ResponseCache initialized - Memory: {max_memory_items} items, "
            f"Disk: {'Enabled' if enable_disk_cache else 'Disabled'}"
        )

    def _generate_cache_key(
        self,
        provider: str,
        use_case: str,
        prompt: str,
        model: str | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> str:
        """Generate a unique cache key for the request"""
        # Create a dictionary of all parameters
        key_data = {
            "provider": provider,
            "use_case": use_case,
            "prompt": prompt,
            "model": model,
        }

        # Add additional parameters if provided
        if additional_params:
            key_data.update(additional_params)

        # Sort keys for consistent hashing
        key_string = json.dumps(key_data, sort_keys=True)

        # Generate SHA256 hash
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get(
        self, cache_key: str, use_case: str | None = None
    ) -> dict[str, Any] | None:
        """Retrieve cached response if available and not expired"""
        # Check memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]

            if self._is_entry_valid(entry, use_case):
                # Move to end (LRU)
                self.memory_cache.move_to_end(cache_key)

                self.stats["hits"] += 1
                self.stats["memory_hits"] += 1

                # Update saved statistics
                if "usage" in entry["response"]:
                    self.stats["total_saved_tokens"] += entry["response"]["usage"].get(
                        "total_tokens", 0
                    )
                if "cost" in entry["response"]:
                    self.stats["total_saved_cost"] += entry["response"]["cost"]

                logger.debug(f"Cache hit (memory): {cache_key[:8]}...")
                return entry["response"]
            else:
                # Remove expired entry
                del self.memory_cache[cache_key]

        # Check disk cache if enabled
        if self.enable_disk_cache:
            disk_entry = await self._load_from_disk(cache_key)
            if disk_entry and self._is_entry_valid(disk_entry, use_case):
                # Add to memory cache
                await self._add_to_memory_cache(cache_key, disk_entry)

                self.stats["hits"] += 1
                self.stats["disk_hits"] += 1

                # Update saved statistics
                if "usage" in disk_entry["response"]:
                    self.stats["total_saved_tokens"] += disk_entry["response"][
                        "usage"
                    ].get("total_tokens", 0)
                if "cost" in disk_entry["response"]:
                    self.stats["total_saved_cost"] += disk_entry["response"]["cost"]

                logger.debug(f"Cache hit (disk): {cache_key[:8]}...")
                return disk_entry["response"]

        self.stats["misses"] += 1
        return None

    async def set(
        self,
        cache_key: str,
        response: dict[str, Any],
        use_case: str | None = None,
        ttl_hours: int | None = None,
    ) -> None:
        """Store response in cache"""
        # Determine TTL
        if ttl_hours is None:
            ttl_hours = self.ttl_overrides.get(use_case, self.default_ttl_hours)

        expiry = datetime.now() + timedelta(hours=ttl_hours)

        # Create cache entry
        entry = {
            "response": response,
            "timestamp": datetime.now(),
            "expiry": expiry,
            "use_case": use_case,
            "ttl_hours": ttl_hours,
        }

        # Add to memory cache
        await self._add_to_memory_cache(cache_key, entry)

        # Save to disk if enabled
        if self.enable_disk_cache:
            await self._save_to_disk(cache_key, entry)

        logger.debug(f"Cached response: {cache_key[:8]}... (TTL: {ttl_hours}h)")

    async def _add_to_memory_cache(self, cache_key: str, entry: dict[str, Any]) -> None:
        """Add entry to memory cache with LRU eviction"""
        # Check if we need to evict
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest item (first in OrderedDict)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
            self.stats["evictions"] += 1

        # Add new entry
        self.memory_cache[cache_key] = entry

    def _is_entry_valid(
        self, entry: dict[str, Any], use_case: str | None = None
    ) -> bool:
        """Check if cache entry is still valid"""
        # Check expiry
        if datetime.now() > entry["expiry"]:
            return False

        # Additional validation based on use case
        if use_case and use_case == "market_sentiment":
            # For market sentiment, also check if it's during market hours
            now = datetime.now()
            if now.weekday() < 5 and 9 <= now.hour < 16:
                # During market hours, use shorter cache
                age = now - entry["timestamp"]
                if age > timedelta(minutes=30):
                    return False

        return True

    async def _save_to_disk(self, cache_key: str, entry: dict[str, Any]) -> None:
        """Save cache entry to disk"""
        if not self.cache_dir:
            return

        file_path = self.cache_dir / f"{cache_key}.json"

        try:
            data = {
                **entry,
                "timestamp": entry["timestamp"].isoformat(),
                "expiry": entry["expiry"].isoformat(),
            }
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data))
        except Exception as e:
            logger.error(f"Error saving to disk cache: {str(e)}")

    async def _load_from_disk(self, cache_key: str) -> dict[str, Any] | None:
        """Load cache entry from disk"""
        if not self.cache_dir:
            return None

        file_path = self.cache_dir / f"{cache_key}.json"

        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, "r") as f:
                data = await f.read()
                entry = json.loads(data)
                entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                entry["expiry"] = datetime.fromisoformat(entry["expiry"])
                return entry
        except Exception as e:
            logger.error(f"Error loading from disk cache: {str(e)}")
            # Remove corrupted file
            try:
                file_path.unlink()
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to remove corrupted cache file {file_path}: {cleanup_error}"
                )
            return None

    def _cleanup_old_cache_files(self) -> None:
        """Remove expired cache files from disk"""
        if not self.cache_dir:
            return

        removed_count = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                # Load and check if expired
                with open(cache_file) as f:
                    entry = json.load(f)
                    entry["expiry"] = datetime.fromisoformat(entry["expiry"])

                if datetime.now() > entry["expiry"]:
                    cache_file.unlink()
                    removed_count += 1
            except Exception as e:
                # Remove corrupted files
                logger.error(f"Error checking cache file {cache_file}: {e}")
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to remove corrupted cache file {cache_file}: {cleanup_error}"
                    )

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache files")

    async def clear_cache(self, use_case: str | None = None) -> int:
        """Clear cache entries, optionally filtered by use case"""
        cleared_count = 0

        # Clear memory cache
        if use_case:
            keys_to_remove = [
                key
                for key, entry in self.memory_cache.items()
                if entry.get("use_case") == use_case
            ]
            for key in keys_to_remove:
                del self.memory_cache[key]
                cleared_count += 1
        else:
            cleared_count = len(self.memory_cache)
            self.memory_cache.clear()

        # Clear disk cache
        if self.enable_disk_cache and self.cache_dir:
            if use_case:
                # Need to check each file
                for cache_file in self.cache_dir.glob("*.json"):
                    try:
                        with open(cache_file) as f:
                            entry = json.load(f)

                        if entry.get("use_case") == use_case:
                            cache_file.unlink()
                            cleared_count += 1
                    except Exception as e:
                        logger.error(f"Error removing cache file {cache_file}: {e}")
            else:
                # Clear all files
                for cache_file in self.cache_dir.glob("*.json"):
                    try:
                        cache_file.unlink()
                        cleared_count += 1
                    except Exception as e:
                        logger.error(f"Error removing cache file {cache_file}: {e}")

        logger.info(
            f"Cleared {cleared_count} cache entries"
            + (f" for use case: {use_case}" if use_case else "")
        )

        return cleared_count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]

        return {
            **self.stats,
            "memory_size": len(self.memory_cache),
            "hit_rate": (
                (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            ),
            "memory_hit_rate": (
                (self.stats["memory_hits"] / self.stats["hits"] * 100)
                if self.stats["hits"] > 0
                else 0
            ),
            "disk_hit_rate": (
                (self.stats["disk_hits"] / self.stats["hits"] * 100)
                if self.stats["hits"] > 0
                else 0
            ),
            "average_saved_tokens": (
                (self.stats["total_saved_tokens"] / self.stats["hits"])
                if self.stats["hits"] > 0
                else 0
            ),
            "average_saved_cost": (
                (self.stats["total_saved_cost"] / self.stats["hits"])
                if self.stats["hits"] > 0
                else 0
            ),
        }

    def get_memory_usage(self) -> dict[str, Any]:
        """Get detailed memory cache usage"""
        usage_by_use_case = {}
        oldest_entry = None
        newest_entry = None

        for _key, entry in self.memory_cache.items():
            use_case = entry.get("use_case", "unknown")

            if use_case not in usage_by_use_case:
                usage_by_use_case[use_case] = {"count": 0, "total_size": 0}

            usage_by_use_case[use_case]["count"] += 1

            # Track oldest and newest
            if oldest_entry is None or entry["timestamp"] < oldest_entry["timestamp"]:
                oldest_entry = entry
            if newest_entry is None or entry["timestamp"] > newest_entry["timestamp"]:
                newest_entry = entry

        return {
            "total_entries": len(self.memory_cache),
            "by_use_case": usage_by_use_case,
            "oldest_entry": {
                "timestamp": oldest_entry["timestamp"] if oldest_entry else None,
                "use_case": oldest_entry.get("use_case") if oldest_entry else None,
            },
            "newest_entry": {
                "timestamp": newest_entry["timestamp"] if newest_entry else None,
                "use_case": newest_entry.get("use_case") if newest_entry else None,
            },
        }

    async def warmup_cache(self, common_queries: list[dict[str, Any]]) -> None:
        """Pre-populate cache with common queries"""
        logger.info(f"Warming up cache with {len(common_queries)} queries")

        for query in common_queries:
            cache_key = self._generate_cache_key(
                provider=query["provider"],
                use_case=query["use_case"],
                prompt=query["prompt"],
                model=query.get("model"),
            )

            # Check if already cached
            existing = await self.get(cache_key, query["use_case"])
            if not existing and "response" in query:
                # Add to cache
                await self.set(
                    cache_key=cache_key,
                    response=query["response"],
                    use_case=query["use_case"],
                    ttl_hours=query.get("ttl_hours"),
                )
