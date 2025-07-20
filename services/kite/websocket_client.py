"""WebSocket client for real-time market data streaming"""

import asyncio
import json
import struct
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime
import websockets
from loguru import logger

from kiteconnect import KiteTicker
from .exceptions import KiteWebSocketError, KiteAuthenticationError
from .auth import KiteAuthManager


class KiteWebSocketClient:
    """Handles real-time market data streaming via WebSocket"""
    
    def __init__(self, auth_manager: KiteAuthManager):
        self.auth_manager = auth_manager
        self.ticker: Optional[KiteTicker] = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        
        # Subscription management
        self.subscribed_tokens: Set[int] = set()
        self.subscription_modes: Dict[int, str] = {}
        
        # Callbacks
        self.on_tick_callback: Optional[Callable] = None
        self.on_order_update_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        self.on_connect_callback: Optional[Callable] = None
        self.on_disconnect_callback: Optional[Callable] = None
        
        # Data storage for latest ticks
        self.latest_ticks: Dict[int, Dict[str, Any]] = {}
        
    async def connect(self) -> bool:
        """Connect to Kite WebSocket"""
        try:
            if not self.auth_manager.is_authenticated():
                raise KiteAuthenticationError("Not authenticated. Please login first.")
            
            access_token = self.auth_manager.get_access_token()
            api_key = self.auth_manager.settings.KITE_API_KEY
            
            # Initialize KiteTicker
            self.ticker = KiteTicker(api_key, access_token)
            
            # Set up event handlers
            self.ticker.on_ticks = self._on_ticks
            self.ticker.on_connect = self._on_connect
            self.ticker.on_close = self._on_close
            self.ticker.on_error = self._on_error
            self.ticker.on_reconnect = self._on_reconnect
            self.ticker.on_noreconnect = self._on_noreconnect
            self.ticker.on_order_update = self._on_order_update
            
            # Connect in a separate thread
            await asyncio.to_thread(self.ticker.connect, threaded=True)
            
            # Wait for connection
            for _ in range(10):  # Wait up to 10 seconds
                if self.is_connected:
                    break
                await asyncio.sleep(1)
            
            if not self.is_connected:
                raise KiteWebSocketError("Failed to establish WebSocket connection")
            
            logger.info("WebSocket connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {str(e)}")
            raise KiteWebSocketError(f"Connection failed: {str(e)}")
    
    async def disconnect(self):
        """Disconnect WebSocket"""
        try:
            if self.ticker:
                await asyncio.to_thread(self.ticker.close)
                self.ticker = None
            
            self.is_connected = False
            self.subscribed_tokens.clear()
            self.subscription_modes.clear()
            
            logger.info("WebSocket disconnected successfully")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}")
    
    async def subscribe(self, tokens: List[int], mode: str = "full") -> bool:
        """Subscribe to market data for given tokens
        
        Args:
            tokens: List of instrument tokens
            mode: Subscription mode ("ltp", "quote", "full")
        """
        try:
            if not self.is_connected or not self.ticker:
                raise KiteWebSocketError("WebSocket not connected")
            
            # Validate mode
            valid_modes = ["ltp", "quote", "full"]
            if mode not in valid_modes:
                raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")
            
            # Convert mode to KiteTicker constants
            mode_map = {
                "ltp": self.ticker.MODE_LTP,
                "quote": self.ticker.MODE_QUOTE,
                "full": self.ticker.MODE_FULL
            }
            
            kite_mode = mode_map[mode]
            
            # Subscribe to tokens
            await asyncio.to_thread(self.ticker.subscribe, tokens)
            await asyncio.to_thread(self.ticker.set_mode, kite_mode, tokens)
            
            # Update subscription tracking
            for token in tokens:
                self.subscribed_tokens.add(token)
                self.subscription_modes[token] = mode
            
            logger.info(f"Subscribed to {len(tokens)} tokens in {mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to tokens: {str(e)}")
            return False
    
    async def unsubscribe(self, tokens: List[int]) -> bool:
        """Unsubscribe from market data for given tokens"""
        try:
            if not self.is_connected or not self.ticker:
                raise KiteWebSocketError("WebSocket not connected")
            
            await asyncio.to_thread(self.ticker.unsubscribe, tokens)
            
            # Update subscription tracking
            for token in tokens:
                self.subscribed_tokens.discard(token)
                self.subscription_modes.pop(token, None)
                self.latest_ticks.pop(token, None)
            
            logger.info(f"Unsubscribed from {len(tokens)} tokens")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from tokens: {str(e)}")
            return False
    
    async def change_mode(self, tokens: List[int], mode: str) -> bool:
        """Change subscription mode for given tokens"""
        try:
            if not self.is_connected or not self.ticker:
                raise KiteWebSocketError("WebSocket not connected")
            
            mode_map = {
                "ltp": self.ticker.MODE_LTP,
                "quote": self.ticker.MODE_QUOTE,
                "full": self.ticker.MODE_FULL
            }
            
            if mode not in mode_map:
                raise ValueError(f"Invalid mode: {mode}")
            
            kite_mode = mode_map[mode]
            await asyncio.to_thread(self.ticker.set_mode, kite_mode, tokens)
            
            # Update subscription tracking
            for token in tokens:
                if token in self.subscribed_tokens:
                    self.subscription_modes[token] = mode
            
            logger.info(f"Changed mode to {mode} for {len(tokens)} tokens")
            return True
            
        except Exception as e:
            logger.error(f"Failed to change mode: {str(e)}")
            return False
    
    def get_latest_tick(self, token: int) -> Optional[Dict[str, Any]]:
        """Get latest tick data for a token"""
        return self.latest_ticks.get(token)
    
    def get_all_latest_ticks(self) -> Dict[int, Dict[str, Any]]:
        """Get all latest tick data"""
        return self.latest_ticks.copy()
    
    def set_on_tick_callback(self, callback: Callable[[List[Dict]], None]):
        """Set callback for tick data"""
        self.on_tick_callback = callback
    
    def set_on_order_update_callback(self, callback: Callable[[Dict], None]):
        """Set callback for order updates"""
        self.on_order_update_callback = callback
    
    def set_on_error_callback(self, callback: Callable[[Exception], None]):
        """Set callback for errors"""
        self.on_error_callback = callback
    
    def set_on_connect_callback(self, callback: Callable[[], None]):
        """Set callback for connection events"""
        self.on_connect_callback = callback
    
    def set_on_disconnect_callback(self, callback: Callable[[], None]):
        """Set callback for disconnection events"""
        self.on_disconnect_callback = callback
    
    # Internal event handlers
    def _on_ticks(self, ws, ticks):
        """Handle incoming tick data"""
        try:
            # Process and store latest ticks
            for tick in ticks:
                token = tick.get('instrument_token')
                if token:
                    # Add timestamp to tick data
                    tick['timestamp'] = datetime.now()
                    self.latest_ticks[token] = tick
            
            # Call user callback
            if self.on_tick_callback:
                try:
                    self.on_tick_callback(ticks)
                except Exception as e:
                    logger.error(f"Error in tick callback: {str(e)}")
            
            logger.debug(f"Received {len(ticks)} ticks")
            
        except Exception as e:
            logger.error(f"Error processing ticks: {str(e)}")
    
    def _on_connect(self, ws, response):
        """Handle connection event"""
        self.is_connected = True
        self.reconnect_attempts = 0
        logger.info("WebSocket connected successfully")
        
        if self.on_connect_callback:
            try:
                self.on_connect_callback()
            except Exception as e:
                logger.error(f"Error in connect callback: {str(e)}")
    
    def _on_close(self, ws, code, reason):
        """Handle disconnection event"""
        self.is_connected = False
        logger.warning(f"WebSocket disconnected: {code} - {reason}")
        
        if self.on_disconnect_callback:
            try:
                self.on_disconnect_callback()
            except Exception as e:
                logger.error(f"Error in disconnect callback: {str(e)}")
    
    def _on_error(self, ws, code, reason):
        """Handle error event"""
        error = KiteWebSocketError(f"WebSocket error: {code} - {reason}")
        logger.error(str(error))
        
        if self.on_error_callback:
            try:
                self.on_error_callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {str(e)}")
    
    def _on_reconnect(self, ws, attempts_count):
        """Handle reconnection event"""
        self.reconnect_attempts = attempts_count
        logger.info(f"WebSocket reconnecting... attempt {attempts_count}")
    
    def _on_noreconnect(self, ws):
        """Handle no reconnection event"""
        logger.error("WebSocket reconnection failed, manual intervention required")
        self.is_connected = False
    
    def _on_order_update(self, ws, data):
        """Handle order update event"""
        try:
            logger.info(f"Order update received: {data}")
            
            if self.on_order_update_callback:
                try:
                    self.on_order_update_callback(data)
                except Exception as e:
                    logger.error(f"Error in order update callback: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing order update: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Get WebSocket connection health status"""
        return {
            "connected": self.is_connected,
            "subscribed_tokens_count": len(self.subscribed_tokens),
            "reconnect_attempts": self.reconnect_attempts,
            "latest_ticks_count": len(self.latest_ticks),
            "subscription_modes": dict(self.subscription_modes)
        }