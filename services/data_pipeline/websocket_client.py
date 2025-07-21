import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import websockets
from websockets import WebSocketClientProtocol
import struct
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Zerodha WebSocket message types"""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    MODE = "mode"
    HEARTBEAT = "heartbeat"


class TickMode(Enum):
    """Zerodha tick modes"""
    LTP = "ltp"  # Last traded price only
    QUOTE = "quote"  # LTP + OHLC + volume
    FULL = "full"  # Everything including market depth


class ZerodhaWebSocketClient:
    """Async WebSocket client for Zerodha Kite Connect live data"""
    
    def __init__(self, kite_client,
                 on_tick: Optional[Callable] = None,
                 on_connect: Optional[Callable] = None,
                 on_disconnect: Optional[Callable] = None,
                 on_error: Optional[Callable] = None):
        self.kite_client = kite_client
        
        # Callbacks
        self.on_tick = on_tick
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_error = on_error
        
        # Connection state
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.is_connected = False
        self.connection_task = None
        
        # Subscription management
        self.subscribed_tokens: Dict[int, TickMode] = {}
        self.pending_subscriptions: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.tick_count = 0
        self.last_tick_time = None
        self.connection_start_time = None
        
        # Configuration
        self.ws_url = "wss://ws.kite.trade"
        self.ping_interval = 30  # seconds
        self.ping_timeout = 10  # seconds
        self.max_reconnect_delay = 60  # seconds
        
    async def connect(self) -> None:
        """Connect to Zerodha WebSocket"""
        if self.is_connected:
            logger.warning("Already connected to WebSocket")
            return
        
        try:
            # Get WebSocket URL with access token
            ws_url = self._get_websocket_url()
            
            logger.info(f"Connecting to WebSocket: {ws_url}")
            self.connection_start_time = datetime.now()
            
            # Create connection with ping/pong
            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                max_size=10 * 1024 * 1024  # 10MB max message size
            )
            
            self.is_connected = True
            logger.info("WebSocket connected successfully")
            
            # Trigger connect callback
            if self.on_connect:
                await self._safe_callback(self.on_connect)
            
            # Process pending subscriptions
            await self._process_pending_subscriptions()
            
            # Start message handler
            self.connection_task = asyncio.create_task(self._handle_messages())
            
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            self.is_connected = False
            if self.on_error:
                await self._safe_callback(self.on_error, e)
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket"""
        logger.info("Disconnecting from WebSocket")
        
        self.is_connected = False
        
        # Cancel message handler
        if self.connection_task:
            self.connection_task.cancel()
            try:
                await self.connection_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        # Trigger disconnect callback
        if self.on_disconnect:
            await self._safe_callback(self.on_disconnect, 1000, "Client disconnect")
        
        logger.info("WebSocket disconnected")
    
    async def subscribe(self, tokens: List[int], mode: TickMode = TickMode.FULL) -> None:
        """Subscribe to instrument tokens"""
        if not tokens:
            return
        
        subscription = {
            "type": MessageType.SUBSCRIBE.value,
            "tokens": tokens,
            "mode": mode.value
        }
        
        if self.is_connected and self.websocket:
            await self._send_message(subscription)
            
            # Update subscribed tokens
            for token in tokens:
                self.subscribed_tokens[token] = mode
            
            logger.info(f"Subscribed to {len(tokens)} tokens in {mode.value} mode")
        else:
            # Queue for later
            self.pending_subscriptions.append(subscription)
            logger.info(f"Queued subscription for {len(tokens)} tokens")
    
    async def unsubscribe(self, tokens: List[int]) -> None:
        """Unsubscribe from instrument tokens"""
        if not tokens:
            return
        
        message = {
            "type": MessageType.UNSUBSCRIBE.value,
            "tokens": tokens
        }
        
        if self.is_connected and self.websocket:
            await self._send_message(message)
            
            # Remove from subscribed tokens
            for token in tokens:
                self.subscribed_tokens.pop(token, None)
            
            logger.info(f"Unsubscribed from {len(tokens)} tokens")
    
    async def set_mode(self, tokens: List[int], mode: TickMode) -> None:
        """Change subscription mode for tokens"""
        if not tokens:
            return
        
        message = {
            "type": MessageType.MODE.value,
            "tokens": tokens,
            "mode": mode.value
        }
        
        if self.is_connected and self.websocket:
            await self._send_message(message)
            
            # Update mode
            for token in tokens:
                if token in self.subscribed_tokens:
                    self.subscribed_tokens[token] = mode
            
            logger.info(f"Changed mode to {mode.value} for {len(tokens)} tokens")
    
    async def _handle_messages(self) -> None:
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    # Parse message based on type
                    if isinstance(message, bytes):
                        # Binary tick data
                        await self._handle_binary_message(message)
                    else:
                        # Text message (error or info)
                        await self._handle_text_message(message)
                        
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    if self.on_error:
                        await self._safe_callback(self.on_error, e)
                        
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e.code} - {e.reason}")
            self.is_connected = False
            if self.on_disconnect:
                await self._safe_callback(self.on_disconnect, e.code, e.reason)
                
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            self.is_connected = False
            if self.on_error:
                await self._safe_callback(self.on_error, e)
    
    async def _handle_binary_message(self, data: bytes) -> None:
        """Handle binary tick data"""
        # Parse binary data according to Zerodha format
        ticks = self._parse_binary_data(data)
        
        for tick in ticks:
            self.tick_count += 1
            self.last_tick_time = datetime.now()
            
            # Add metadata
            tick["timestamp"] = self.last_tick_time.isoformat()
            tick["sequence"] = self.tick_count
            
            # Trigger tick callback
            if self.on_tick:
                await self._safe_callback(self.on_tick, tick)
    
    async def _handle_text_message(self, message: str) -> None:
        """Handle text messages (errors, info)"""
        try:
            data = json.loads(message)
            
            if data.get("type") == "error":
                logger.error(f"Server error: {data.get('message')}")
                if self.on_error:
                    await self._safe_callback(
                        self.on_error,
                        Exception(data.get("message", "Unknown error"))
                    )
            else:
                logger.info(f"Server message: {message}")
                
        except json.JSONDecodeError:
            logger.warning(f"Received non-JSON text message: {message}")
    
    def _parse_binary_data(self, data: bytes) -> List[Dict[str, Any]]:
        """Parse binary tick data from Zerodha"""
        ticks = []
        
        # Zerodha binary format parsing
        # This is a simplified version - actual implementation would follow
        # Zerodha's binary protocol specification
        
        offset = 0
        packet_length = len(data)
        
        while offset < packet_length:
            try:
                # Read packet header (simplified)
                if offset + 44 > packet_length:  # Minimum tick size
                    break
                
                # Parse tick structure
                tick = {}
                
                # Instrument token (4 bytes)
                tick["instrument_token"] = struct.unpack(">I", data[offset:offset+4])[0]
                offset += 4
                
                # Mode detection based on packet size
                remaining = packet_length - offset
                
                if remaining >= 40:  # Full mode
                    # LTP (4 bytes)
                    tick["last_price"] = struct.unpack(">I", data[offset:offset+4])[0] / 100
                    offset += 4
                    
                    # Last quantity (4 bytes)
                    tick["last_quantity"] = struct.unpack(">I", data[offset:offset+4])[0]
                    offset += 4
                    
                    # Average price (4 bytes)
                    tick["average_price"] = struct.unpack(">I", data[offset:offset+4])[0] / 100
                    offset += 4
                    
                    # Volume (4 bytes)
                    tick["volume"] = struct.unpack(">I", data[offset:offset+4])[0]
                    offset += 4
                    
                    # Buy quantity (4 bytes)
                    tick["buy_quantity"] = struct.unpack(">I", data[offset:offset+4])[0]
                    offset += 4
                    
                    # Sell quantity (4 bytes)
                    tick["sell_quantity"] = struct.unpack(">I", data[offset:offset+4])[0]
                    offset += 4
                    
                    # OHLC
                    tick["ohlc"] = {
                        "open": struct.unpack(">I", data[offset:offset+4])[0] / 100,
                        "high": struct.unpack(">I", data[offset+4:offset+8])[0] / 100,
                        "low": struct.unpack(">I", data[offset+8:offset+12])[0] / 100,
                        "close": struct.unpack(">I", data[offset+12:offset+16])[0] / 100
                    }
                    offset += 16
                    
                    tick["mode"] = "full"
                    
                elif remaining >= 8:  # Quote mode
                    # LTP only
                    tick["last_price"] = struct.unpack(">I", data[offset:offset+4])[0] / 100
                    offset += 4
                    
                    # Last quantity
                    tick["last_quantity"] = struct.unpack(">I", data[offset:offset+4])[0]
                    offset += 4
                    
                    tick["mode"] = "quote"
                    
                else:  # LTP mode
                    # LTP only
                    tick["last_price"] = struct.unpack(">I", data[offset:offset+4])[0] / 100
                    offset += 4
                    
                    tick["mode"] = "ltp"
                
                ticks.append(tick)
                
            except struct.error as e:
                logger.error(f"Error parsing tick data: {str(e)}")
                break
        
        return ticks
    
    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send message to WebSocket server"""
        if not self.websocket:
            raise ConnectionError("WebSocket not connected")
        
        try:
            json_message = json.dumps(message)
            await self.websocket.send(json_message)
            logger.debug(f"Sent message: {json_message}")
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise
    
    async def _process_pending_subscriptions(self) -> None:
        """Process queued subscriptions"""
        if not self.pending_subscriptions:
            return
        
        logger.info(f"Processing {len(self.pending_subscriptions)} pending subscriptions")
        
        for subscription in self.pending_subscriptions:
            await self._send_message(subscription)
            
            # Update subscribed tokens
            tokens = subscription.get("tokens", [])
            mode = TickMode(subscription.get("mode", "full"))
            for token in tokens:
                self.subscribed_tokens[token] = mode
        
        self.pending_subscriptions.clear()
    
    def _get_websocket_url(self) -> str:
        """Get WebSocket URL with authentication"""
        # Get access token from Kite client
        access_token = self.kite_client.access_token
        api_key = self.kite_client.api_key
        
        return f"{self.ws_url}?api_key={api_key}&access_token={access_token}"
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs) -> None:
        """Safely execute callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback: {str(e)}")
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.is_connected and self.websocket is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket client statistics"""
        uptime = None
        if self.connection_start_time:
            uptime = (datetime.now() - self.connection_start_time).total_seconds()
        
        return {
            "connected": self.is_connected,
            "tick_count": self.tick_count,
            "last_tick_time": self.last_tick_time.isoformat() if self.last_tick_time else None,
            "subscribed_tokens": len(self.subscribed_tokens),
            "pending_subscriptions": len(self.pending_subscriptions),
            "uptime_seconds": uptime
        }