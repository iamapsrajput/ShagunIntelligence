import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
import json

from backend.data_sources.base import (
    MarketDataSource,
    DataSourceConfig,
    DataSourceStatus
)
from .models import (
    MarketData,
    MarketDepth,
    HistoricalBar,
    MarketDataQuality,
    DataCostTier,
    DataSourceCost
)


class GlobalDatafeedsSource(MarketDataSource):
    """Global Datafeeds - Authorized NSE/BSE real-time data provider"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.api_key = config.credentials.get("api_key")
        self.user_id = config.credentials.get("user_id")
        self.base_url = "https://api.globaldatafeeds.in/api/v1"
        self.ws_url = "wss://ws.globaldatafeeds.in"
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_session: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # Cost information
        self.cost_info = DataSourceCost(
            tier=DataCostTier.MEDIUM,
            monthly_cost=299.0,  # Professional plan
            per_request_cost=0.0,  # Unlimited in plan
            free_requests=0,
            requests_per_minute=300,
            requests_per_day=None,  # Unlimited
            includes_realtime=True,
            includes_historical=True,
            includes_options=True
        )
        
        # Exchange mappings
        self.exchange_map = {
            "NSE": "NSE_EQ",
            "BSE": "BSE_EQ",
            "NFO": "NSE_FO",
            "MCX": "MCX_FO"
        }
        
        logger.info("Initialized GlobalDatafeedsSource")
    
    async def connect(self) -> bool:
        """Connect to Global Datafeeds API"""
        try:
            if not self.api_key or not self.user_id:
                raise ValueError("Global Datafeeds credentials not provided")
            
            self.session = aiohttp.ClientSession()
            
            # Authenticate
            auth_url = f"{self.base_url}/authenticate"
            auth_data = {
                "api_key": self.api_key,
                "user_id": self.user_id
            }
            
            async with self.session.post(auth_url, json=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        self.update_health_status(DataSourceStatus.HEALTHY)
                        logger.info("Connected to Global Datafeeds")
                        return True
                    else:
                        raise Exception(f"Authentication failed: {data.get('message')}")
                else:
                    raise Exception(f"HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to connect to Global Datafeeds: {e}")
            self.update_health_status(DataSourceStatus.UNHEALTHY, str(e))
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Global Datafeeds"""
        if self.ws_session:
            await self.ws_session.close()
        if self.session:
            await self.session.close()
        self.update_health_status(DataSourceStatus.DISCONNECTED)
        logger.info("Disconnected from Global Datafeeds")
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        try:
            # Parse symbol format (e.g., "RELIANCE.NSE" -> "RELIANCE", "NSE_EQ")
            parts = symbol.split(".")
            ticker = parts[0]
            exchange = self.exchange_map.get(parts[1] if len(parts) > 1 else "NSE", "NSE_EQ")
            
            url = f"{self.base_url}/realtime/quote"
            params = {
                "api_key": self.api_key,
                "symbol": ticker,
                "exchange": exchange
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("success"):
                        quote_data = data.get("data", {})
                        
                        # Convert to standardized format
                        market_data = MarketData(
                            symbol=symbol,
                            exchange=parts[1] if len(parts) > 1 else "NSE",
                            timestamp=datetime.utcnow(),
                            source="global_datafeeds",
                            current_price=float(quote_data.get("ltp", 0)),
                            open=float(quote_data.get("open", 0)),
                            high=float(quote_data.get("high", 0)),
                            low=float(quote_data.get("low", 0)),
                            close=float(quote_data.get("close", 0)),
                            previous_close=float(quote_data.get("prev_close", 0)),
                            volume=int(quote_data.get("volume", 0)),
                            bid=float(quote_data.get("bid", 0)),
                            ask=float(quote_data.get("ask", 0)),
                            bid_size=int(quote_data.get("bid_qty", 0)),
                            ask_size=int(quote_data.get("ask_qty", 0)),
                            change=float(quote_data.get("change", 0)),
                            change_percent=float(quote_data.get("change_percent", 0)),
                            data_quality=MarketDataQuality.PROFESSIONAL,
                            latency_ms=50,  # Low latency for Indian markets
                            is_delayed=False
                        )
                        
                        return market_data.to_dict()
                    else:
                        raise Exception(f"API error: {data.get('message')}")
                else:
                    raise Exception(f"HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching quote from Global Datafeeds: {e}")
            raise
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols"""
        quotes = {}
        
        # Global Datafeeds supports batch requests
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            try:
                # Prepare batch request
                symbol_list = []
                for symbol in batch:
                    parts = symbol.split(".")
                    ticker = parts[0]
                    exchange = self.exchange_map.get(parts[1] if len(parts) > 1 else "NSE", "NSE_EQ")
                    symbol_list.append(f"{ticker}:{exchange}")
                
                url = f"{self.base_url}/realtime/quotes"
                params = {
                    "api_key": self.api_key,
                    "symbols": ",".join(symbol_list)
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("success"):
                            quotes_data = data.get("data", {})
                            
                            for symbol, quote_data in quotes_data.items():
                                # Convert back to original symbol format
                                ticker, exchange = symbol.split(":")
                                original_symbol = f"{ticker}.{self._reverse_exchange_map(exchange)}"
                                
                                market_data = MarketData(
                                    symbol=original_symbol,
                                    exchange=self._reverse_exchange_map(exchange),
                                    timestamp=datetime.utcnow(),
                                    source="global_datafeeds",
                                    current_price=float(quote_data.get("ltp", 0)),
                                    open=float(quote_data.get("open", 0)),
                                    high=float(quote_data.get("high", 0)),
                                    low=float(quote_data.get("low", 0)),
                                    close=float(quote_data.get("close", 0)),
                                    previous_close=float(quote_data.get("prev_close", 0)),
                                    volume=int(quote_data.get("volume", 0)),
                                    bid=float(quote_data.get("bid", 0)),
                                    ask=float(quote_data.get("ask", 0)),
                                    change=float(quote_data.get("change", 0)),
                                    change_percent=float(quote_data.get("change_percent", 0)),
                                    data_quality=MarketDataQuality.PROFESSIONAL,
                                    latency_ms=50,
                                    is_delayed=False
                                )
                                
                                quotes[original_symbol] = market_data.to_dict()
                                
            except Exception as e:
                logger.error(f"Error fetching batch quotes: {e}")
        
        return quotes
    
    async def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """Get market depth (order book) data"""
        try:
            parts = symbol.split(".")
            ticker = parts[0]
            exchange = self.exchange_map.get(parts[1] if len(parts) > 1 else "NSE", "NSE_EQ")
            
            url = f"{self.base_url}/realtime/depth"
            params = {
                "api_key": self.api_key,
                "symbol": ticker,
                "exchange": exchange
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("success"):
                        depth_data = data.get("data", {})
                        
                        # Convert to standardized format
                        market_depth = MarketDepth(
                            symbol=symbol,
                            timestamp=datetime.utcnow(),
                            source="global_datafeeds",
                            bids=[
                                {"price": float(bid["price"]), "size": int(bid["quantity"])}
                                for bid in depth_data.get("bids", [])[:5]
                            ],
                            asks=[
                                {"price": float(ask["price"]), "size": int(ask["quantity"])}
                                for ask in depth_data.get("asks", [])[:5]
                            ]
                        )
                        
                        # Calculate total volumes
                        market_depth.total_bid_volume = sum(
                            bid["size"] for bid in market_depth.bids
                        )
                        market_depth.total_ask_volume = sum(
                            ask["size"] for ask in market_depth.asks
                        )
                        
                        return market_depth.to_dict()
                    else:
                        raise Exception(f"API error: {data.get('message')}")
                else:
                    raise Exception(f"HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching market depth: {e}")
            return {"bids": [], "asks": []}
    
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        from_date: datetime,
        to_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get historical data for a symbol"""
        try:
            parts = symbol.split(".")
            ticker = parts[0]
            exchange = self.exchange_map.get(parts[1] if len(parts) > 1 else "NSE", "NSE_EQ")
            
            # Map interval to Global Datafeeds format
            interval_map = {
                "minute": "1minute",
                "5minute": "5minute",
                "15minute": "15minute",
                "30minute": "30minute",
                "60minute": "60minute",
                "day": "1day"
            }
            gdf_interval = interval_map.get(interval, "1day")
            
            url = f"{self.base_url}/historical/data"
            params = {
                "api_key": self.api_key,
                "symbol": ticker,
                "exchange": exchange,
                "interval": gdf_interval,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d")
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("success"):
                        candles = data.get("data", [])
                        
                        # Convert to standardized format
                        bars = []
                        for candle in candles:
                            bar = HistoricalBar(
                                timestamp=datetime.fromisoformat(candle["time"]),
                                open=float(candle["open"]),
                                high=float(candle["high"]),
                                low=float(candle["low"]),
                                close=float(candle["close"]),
                                volume=int(candle["volume"])
                            )
                            bars.append(bar.to_dict())
                        
                        return bars
                    else:
                        raise Exception(f"API error: {data.get('message')}")
                else:
                    raise Exception(f"HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    async def stream_quotes(self, symbols: List[str], callback: Any) -> None:
        """Stream real-time quotes via WebSocket"""
        try:
            # Connect to WebSocket
            ws_url = f"{self.ws_url}?api_key={self.api_key}"
            self.ws_session = await self.session.ws_connect(ws_url)
            
            # Subscribe to symbols
            subscribe_msg = {
                "type": "subscribe",
                "symbols": [self._format_ws_symbol(s) for s in symbols]
            }
            await self.ws_session.send_json(subscribe_msg)
            
            # Listen for updates
            async for msg in self.ws_session:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    if data.get("type") == "quote":
                        # Convert to standardized format
                        quote_data = data.get("data", {})
                        symbol = self._parse_ws_symbol(quote_data.get("symbol", ""))
                        
                        market_data = MarketData(
                            symbol=symbol,
                            exchange=symbol.split(".")[-1],
                            timestamp=datetime.utcnow(),
                            source="global_datafeeds",
                            current_price=float(quote_data.get("ltp", 0)),
                            bid=float(quote_data.get("bid", 0)),
                            ask=float(quote_data.get("ask", 0)),
                            volume=int(quote_data.get("volume", 0)),
                            data_quality=MarketDataQuality.INSTITUTIONAL,
                            latency_ms=10,  # Ultra-low latency
                            is_delayed=False
                        )
                        
                        await callback(symbol, market_data.to_dict())
                        
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in quote streaming: {e}")
            raise
    
    def _reverse_exchange_map(self, exchange: str) -> str:
        """Reverse map exchange codes"""
        reverse_map = {v: k for k, v in self.exchange_map.items()}
        return reverse_map.get(exchange, "NSE")
    
    def _format_ws_symbol(self, symbol: str) -> str:
        """Format symbol for WebSocket subscription"""
        parts = symbol.split(".")
        ticker = parts[0]
        exchange = self.exchange_map.get(parts[1] if len(parts) > 1 else "NSE", "NSE_EQ")
        return f"{ticker}:{exchange}"
    
    def _parse_ws_symbol(self, ws_symbol: str) -> str:
        """Parse WebSocket symbol back to standard format"""
        if ":" in ws_symbol:
            ticker, exchange = ws_symbol.split(":")
            return f"{ticker}.{self._reverse_exchange_map(exchange)}"
        return ws_symbol
    
    def get_cost_info(self) -> DataSourceCost:
        """Get cost information for this data source"""
        return self.cost_info
    
    def supports_market(self, market: str) -> bool:
        """Check if this source supports a market"""
        supported_markets = ["NSE", "BSE", "NFO", "MCX"]
        return market.upper() in supported_markets