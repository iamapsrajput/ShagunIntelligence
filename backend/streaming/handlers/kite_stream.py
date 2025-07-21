"""
Kite Connect WebSocket stream handler for real-time market data.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import time
from loguru import logger

from kiteconnect import KiteTicker
import websockets

from ..realtime_pipeline import DataStreamHandler, StreamConfig, StreamMessage, StreamStatus


class KiteStreamHandler(DataStreamHandler):
    """Handler for Kite Connect WebSocket streaming."""
    
    def __init__(self, config: StreamConfig, api_key: str, access_token: str):
        super().__init__(config)
        self.api_key = api_key
        self.access_token = access_token
        self.kite_ticker = None
        self.subscribed_tokens = {}  # symbol -> token mapping
        self.token_to_symbol = {}  # token -> symbol mapping
        self.last_tick_time = {}  # symbol -> last tick timestamp
        self.message_queue = asyncio.Queue()
        self.processing_task = None
        
    async def connect(self) -> bool:
        """Establish connection to Kite WebSocket."""
        try:
            self.status = StreamStatus.CONNECTING
            
            # Initialize KiteTicker
            self.kite_ticker = KiteTicker(self.api_key, self.access_token)
            
            # Set up callbacks
            self.kite_ticker.on_ticks = self._on_ticks
            self.kite_ticker.on_connect = self._on_connect
            self.kite_ticker.on_close = self._on_close
            self.kite_ticker.on_error = self._on_error
            self.kite_ticker.on_reconnect = self._on_reconnect
            
            # Connect in a separate thread (KiteTicker uses threading)
            await asyncio.get_event_loop().run_in_executor(
                None, self.kite_ticker.connect, True
            )
            
            # Start message processing
            self.processing_task = asyncio.create_task(self._process_message_queue())
            
            # Wait for connection
            await asyncio.sleep(2)
            
            if self.status == StreamStatus.CONNECTED:
                logger.info(f"Connected to Kite WebSocket")
                self.metrics.last_message_time = datetime.now()
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Kite WebSocket: {e}")
            self.status = StreamStatus.ERROR
            self.metrics.errors_count += 1
            return False
    
    async def disconnect(self):
        """Close the Kite WebSocket connection."""
        try:
            if self.processing_task:
                self.processing_task.cancel()
                
            if self.kite_ticker:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.kite_ticker.stop
                )
                
            self.status = StreamStatus.DISCONNECTED
            logger.info("Disconnected from Kite WebSocket")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Kite WebSocket: {e}")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols on Kite WebSocket."""
        try:
            # Map symbols to instrument tokens
            tokens_to_subscribe = []
            
            for symbol in symbols:
                # In production, you would fetch the instrument token from Kite API
                # For now, using mock token mapping
                token = self._get_instrument_token(symbol)
                if token:
                    self.subscribed_tokens[symbol] = token
                    self.token_to_symbol[token] = symbol
                    tokens_to_subscribe.append(token)
                    logger.info(f"Subscribing to {symbol} (token: {token})")
            
            if tokens_to_subscribe and self.kite_ticker:
                # Subscribe to tokens
                await asyncio.get_event_loop().run_in_executor(
                    None, self.kite_ticker.subscribe, tokens_to_subscribe
                )
                
                # Set mode to full quote for detailed data
                await asyncio.get_event_loop().run_in_executor(
                    None, self.kite_ticker.set_mode, 
                    self.kite_ticker.MODE_FULL, tokens_to_subscribe
                )
                
                logger.info(f"Subscribed to {len(tokens_to_subscribe)} instruments")
                
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {e}")
            self.metrics.errors_count += 1
    
    async def process_message(self, message: Any) -> Optional[StreamMessage]:
        """Process a tick from Kite WebSocket."""
        # Messages are processed through the queue
        try:
            # Get message from queue with timeout
            tick_data = await asyncio.wait_for(
                self.message_queue.get(), 
                timeout=1.0
            )
            
            return await self._process_tick(tick_data)
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error processing Kite message: {e}")
            self.metrics.errors_count += 1
            return None
    
    async def _process_tick(self, tick: Dict[str, Any]) -> Optional[StreamMessage]:
        """Process a single tick data."""
        try:
            # Get symbol from token
            token = tick.get('instrument_token')
            symbol = self.token_to_symbol.get(token)
            
            if not symbol:
                return None
            
            # Calculate latency
            receive_time = datetime.now()
            exchange_timestamp = tick.get('exchange_timestamp', receive_time)
            
            if isinstance(exchange_timestamp, datetime):
                latency_ms = (receive_time - exchange_timestamp).total_seconds() * 1000
            else:
                latency_ms = 0
            
            # Update metrics
            self.metrics.messages_processed += 1
            self.metrics.average_latency_ms = (
                (self.metrics.average_latency_ms * (self.metrics.messages_processed - 1) + latency_ms) /
                self.metrics.messages_processed
            )
            
            # Check for data gaps
            if symbol in self.last_tick_time:
                time_gap = (receive_time - self.last_tick_time[symbol]).total_seconds()
                if time_gap > 5:  # Gap of more than 5 seconds
                    self.metrics.data_gaps += 1
            
            self.last_tick_time[symbol] = receive_time
            
            # Calculate quality score
            has_gaps = self.metrics.data_gaps > 0
            quality_score = self.calculate_quality_score(latency_ms, has_gaps)
            
            # Create stream message
            message = StreamMessage(
                stream_name=self.config.name,
                symbol=symbol,
                data={
                    'last_price': tick.get('last_price'),
                    'last_traded_quantity': tick.get('last_traded_quantity'),
                    'average_traded_price': tick.get('average_traded_price'),
                    'volume_traded': tick.get('volume_traded'),
                    'total_buy_quantity': tick.get('total_buy_quantity'),
                    'total_sell_quantity': tick.get('total_sell_quantity'),
                    'ohlc': tick.get('ohlc', {}),
                    'change': tick.get('change'),
                    'last_trade_time': tick.get('last_trade_time'),
                    'oi': tick.get('oi'),  # Open Interest
                    'oi_day_high': tick.get('oi_day_high'),
                    'oi_day_low': tick.get('oi_day_low'),
                    'depth': tick.get('depth', {})  # Market depth
                },
                timestamp=receive_time,
                latency_ms=latency_ms,
                quality_score=quality_score
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            return None
    
    async def send_heartbeat(self):
        """Send heartbeat to keep connection alive."""
        # KiteTicker handles heartbeat automatically
        self.last_heartbeat = datetime.now()
    
    def _on_ticks(self, ws, ticks):
        """Callback for receiving ticks."""
        # Add ticks to async queue for processing
        for tick in ticks:
            try:
                asyncio.create_task(self.message_queue.put(tick))
            except Exception as e:
                logger.error(f"Error queuing tick: {e}")
    
    def _on_connect(self, ws, response):
        """Callback for successful connection."""
        self.status = StreamStatus.CONNECTED
        logger.info(f"Kite WebSocket connected: {response}")
    
    def _on_close(self, ws, code, reason):
        """Callback for connection close."""
        self.status = StreamStatus.DISCONNECTED
        logger.warning(f"Kite WebSocket closed: {code} - {reason}")
    
    def _on_error(self, ws, code, reason):
        """Callback for errors."""
        self.metrics.errors_count += 1
        logger.error(f"Kite WebSocket error: {code} - {reason}")
        
        if code in ["1006", "1011"]:  # Connection errors
            self.status = StreamStatus.ERROR
    
    def _on_reconnect(self, ws, attempts_count):
        """Callback for reconnection attempts."""
        self.metrics.reconnection_count = attempts_count
        logger.info(f"Kite WebSocket reconnecting... Attempt: {attempts_count}")
    
    async def _process_message_queue(self):
        """Process messages from the queue."""
        while self.status in [StreamStatus.CONNECTED, StreamStatus.CONNECTING]:
            try:
                # Process any queued messages
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in message queue processor: {e}")
    
    def _get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for a symbol."""
        # In production, this would fetch from Kite instruments API
        # Mock implementation for common symbols
        token_map = {
            'RELIANCE': 738561,
            'TCS': 2953217,
            'INFY': 408065,
            'HDFC': 340481,
            'ICICIBANK': 1270529,
            'SBIN': 779521,
            'BAJFINANCE': 81153,
            'MARUTI': 2815745,
            'HDFCBANK': 341249,
            'KOTAKBANK': 492033,
            'NIFTY': 256265,
            'BANKNIFTY': 260105,
            'SENSEX': 265
        }
        
        # Handle NSE/BSE suffix
        base_symbol = symbol.split('.')[0]
        return token_map.get(base_symbol)