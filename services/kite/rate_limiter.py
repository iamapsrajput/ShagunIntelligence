"""Rate limiter for API calls"""

import asyncio
import time
from typing import Dict, List
from collections import deque
from loguru import logger


class RateLimiter:
    """Rate limiter to control API request frequency"""
    
    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: deque = deque()
        self._lock = asyncio.Lock()
        
    async def acquire(self) -> None:
        """Acquire permission to make a request"""
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside the time window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            # If we're at the limit, wait
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.time_window - now
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)
                    
                    # Clean up old requests again after sleeping
                    now = time.time()
                    while self.requests and self.requests[0] <= now - self.time_window:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(now)
    
    def get_status(self) -> Dict[str, int]:
        """Get current rate limiter status"""
        now = time.time()
        
        # Clean up old requests
        while self.requests and self.requests[0] <= now - self.time_window:
            self.requests.popleft()
        
        return {
            "current_requests": len(self.requests),
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "remaining_requests": max(0, self.max_requests - len(self.requests))
        }