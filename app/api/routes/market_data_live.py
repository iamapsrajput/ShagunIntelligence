"""
Live Market Data API Routes
Provides real-time and historical market data endpoints
"""

import json
from datetime import datetime
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from app.core.auth import get_current_user
from app.services.market_data_integration import (
    DataProvider,
    DataType,
    HistoricalDataService,
    MarketQuote,
    RealTimeDataFeed,
)

router = APIRouter(prefix="/market-data", tags=["market-data-live"])

# Global instances - in production, these would be dependency injected
real_time_feed = RealTimeDataFeed(DataProvider.ZERODHA_KITE)
historical_service = HistoricalDataService(DataProvider.YAHOO_FINANCE)


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.subscriptions: dict[WebSocket, list[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = []

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: str, symbol: str = None):
        disconnected = []
        for connection in self.active_connections:
            try:
                # Send to all connections or only those subscribed to the symbol
                if symbol is None or symbol in self.subscriptions.get(connection, []):
                    await connection.send_text(message)
            except:
                disconnected.append(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


class SubscriptionRequest(BaseModel):
    """WebSocket subscription request"""

    action: str = Field(..., description="subscribe or unsubscribe")
    symbols: list[str] = Field(..., description="List of symbols to subscribe to")
    data_types: list[str] = Field(
        default=["quote"], description="Types of data (quote, depth, trades)"
    )


class HistoricalDataRequest(BaseModel):
    """Historical data request"""

    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(
        default="1d", description="Timeframe (1min, 5min, 15min, 30min, 1h, 4h, 1d, 1w)"
    )
    start_date: datetime | None = Field(None, description="Start date (ISO format)")
    end_date: datetime | None = Field(None, description="End date (ISO format)")
    limit: int = Field(
        default=100, ge=1, le=1000, description="Number of periods to fetch"
    )


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time market data

    Supports:
    - Real-time quotes
    - Market depth
    - Trade data
    - Multiple symbol subscriptions
    """
    await manager.connect(websocket)

    try:
        while True:
            # Receive subscription requests
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("action") == "subscribe":
                symbols = message.get("symbols", [])
                data_types = message.get("data_types", ["quote"])

                # Add to client subscriptions
                manager.subscriptions[websocket].extend(symbols)

                # Subscribe to real-time feed
                data_type_enums = [
                    DataType(dt)
                    for dt in data_types
                    if dt in [e.value for e in DataType]
                ]

                async def quote_callback(quote: MarketQuote):
                    quote_data = {
                        "type": "quote",
                        "symbol": quote.symbol,
                        "data": {
                            "last_price": quote.last_price,
                            "change": quote.change,
                            "change_percent": quote.change_percent,
                            "volume": quote.volume,
                            "bid": quote.bid,
                            "ask": quote.ask,
                            "timestamp": quote.timestamp.isoformat(),
                        },
                    }
                    await manager.send_personal_message(
                        json.dumps(quote_data), websocket
                    )

                await real_time_feed.subscribe(symbols, data_type_enums, quote_callback)

                # Send confirmation
                confirmation = {
                    "type": "subscription_confirmed",
                    "symbols": symbols,
                    "data_types": data_types,
                    "client_id": client_id,
                }
                await websocket.send_text(json.dumps(confirmation))

            elif message.get("action") == "unsubscribe":
                symbols = message.get("symbols", [])
                data_types = message.get("data_types", ["quote"])

                # Remove from client subscriptions
                for symbol in symbols:
                    if symbol in manager.subscriptions[websocket]:
                        manager.subscriptions[websocket].remove(symbol)

                # Unsubscribe from real-time feed
                data_type_enums = [
                    DataType(dt)
                    for dt in data_types
                    if dt in [e.value for e in DataType]
                ]
                await real_time_feed.unsubscribe(symbols, data_type_enums)

                # Send confirmation
                confirmation = {
                    "type": "unsubscription_confirmed",
                    "symbols": symbols,
                    "client_id": client_id,
                }
                await websocket.send_text(json.dumps(confirmation))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        error_message = {"type": "error", "message": str(e), "client_id": client_id}
        await websocket.send_text(json.dumps(error_message))
        manager.disconnect(websocket)


@router.get("/quote/{symbol}")
async def get_real_time_quote(
    symbol: str, current_user: dict[str, Any] = Depends(get_current_user)
):
    """
    Get real-time quote for a symbol

    Returns current market data including:
    - Last traded price
    - Bid/Ask prices
    - Volume
    - Change and percentage change
    """
    try:
        # Try to get from cache first
        quote = await real_time_feed.get_cached_quote(symbol)

        if not quote:
            # If not in cache, this would typically trigger a subscription
            # For now, return an error
            raise HTTPException(
                status_code=404,
                detail=f"No real-time data available for {symbol}. Subscribe to WebSocket feed first.",
            )

        return {
            "success": True,
            "data": {
                "symbol": quote.symbol,
                "exchange": quote.exchange,
                "last_price": quote.last_price,
                "change": quote.change,
                "change_percent": quote.change_percent,
                "volume": quote.volume,
                "bid": quote.bid,
                "ask": quote.ask,
                "bid_quantity": quote.bid_quantity,
                "ask_quantity": quote.ask_quantity,
                "open": quote.open,
                "high": quote.high,
                "low": quote.low,
                "close": quote.close,
                "timestamp": quote.timestamp,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quote: {str(e)}")


@router.get("/depth/{symbol}")
async def get_market_depth(
    symbol: str, current_user: dict[str, Any] = Depends(get_current_user)
):
    """
    Get market depth (Level 2) data for a symbol

    Returns:
    - Top 5 bid and ask levels
    - Quantities at each level
    - Bid-ask spread
    """
    try:
        depth = await real_time_feed.get_cached_depth(symbol)

        if not depth:
            raise HTTPException(
                status_code=404, detail=f"No market depth data available for {symbol}"
            )

        return {
            "success": True,
            "data": {
                "symbol": depth.symbol,
                "exchange": depth.exchange,
                "bids": depth.bids,
                "asks": depth.asks,
                "spread": depth.spread,
                "timestamp": depth.timestamp,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get market depth: {str(e)}"
        )


@router.post("/historical")
async def get_historical_data(
    request: HistoricalDataRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Get historical OHLCV data for a symbol

    Supports multiple timeframes and date ranges
    """
    try:
        async with HistoricalDataService() as service:
            data = await service.get_historical_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date,
                limit=request.limit,
            )

        # Convert DataFrame to JSON-serializable format
        data_dict = data.reset_index().to_dict("records")

        return {
            "success": True,
            "data": {
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "records": len(data_dict),
                "data": data_dict,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get historical data: {str(e)}"
        )


@router.get("/historical/{symbol}")
async def get_historical_data_simple(
    symbol: str,
    timeframe: str = Query("1d", description="Timeframe"),
    limit: int = Query(100, ge=1, le=1000, description="Number of periods"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Simple endpoint to get historical data with query parameters
    """
    try:
        async with HistoricalDataService() as service:
            data = await service.get_historical_data(
                symbol=symbol, timeframe=timeframe, limit=limit
            )

        # Convert to JSON format
        data_dict = data.reset_index().to_dict("records")

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "records": len(data_dict),
                "data": data_dict,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get historical data: {str(e)}"
        )


@router.post("/historical/multiple")
async def get_multiple_historical_data(
    symbols: list[str] = Field(..., description="List of symbols"),
    timeframe: str = Field(default="1d", description="Timeframe"),
    limit: int = Field(default=100, ge=1, le=500, description="Number of periods"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Get historical data for multiple symbols
    """
    try:
        async with HistoricalDataService() as service:
            data = await service.get_multiple_symbols_data(
                symbols=symbols, timeframe=timeframe, limit=limit
            )

        # Convert all DataFrames to JSON format
        result = {}
        for symbol, df in data.items():
            if not df.empty:
                result[symbol] = df.reset_index().to_dict("records")
            else:
                result[symbol] = []

        return {
            "success": True,
            "data": {"symbols": symbols, "timeframe": timeframe, "results": result},
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get multiple historical data: {str(e)}"
        )


@router.get("/connection/status")
async def get_connection_status(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get real-time data feed connection status"""
    try:
        status = real_time_feed.get_connection_status()

        return {
            "success": True,
            "data": {
                "real_time_feed": status,
                "websocket_connections": len(manager.active_connections),
                "total_subscriptions": sum(
                    len(subs) for subs in manager.subscriptions.values()
                ),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get connection status: {str(e)}"
        )


@router.post("/connection/connect")
async def connect_data_feed(
    provider: str = Field(default="zerodha_kite", description="Data provider"),
    api_key: str | None = Field(None, description="API key"),
    access_token: str | None = Field(None, description="Access token"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Connect to real-time data feed"""
    try:
        # Validate provider
        try:
            provider_enum = DataProvider(provider)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")

        # Create new feed instance if provider changed
        global real_time_feed
        if real_time_feed.provider != provider_enum:
            await real_time_feed.disconnect()
            real_time_feed = RealTimeDataFeed(provider_enum)

        # Connect to the feed
        await real_time_feed.connect(api_key, access_token)

        return {
            "success": True,
            "message": f"Connected to {provider} data feed",
            "data": real_time_feed.get_connection_status(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to data feed: {str(e)}"
        )


@router.post("/connection/disconnect")
async def disconnect_data_feed(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Disconnect from real-time data feed"""
    try:
        await real_time_feed.disconnect()

        return {"success": True, "message": "Disconnected from data feed"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disconnect: {str(e)}")


@router.get("/providers")
async def get_supported_providers():
    """Get list of supported data providers"""
    return {
        "success": True,
        "data": {
            "providers": [
                {
                    "code": provider.value,
                    "name": provider.value.replace("_", " ").title(),
                    "description": f"{provider.value} market data provider",
                }
                for provider in DataProvider
            ],
            "data_types": [
                {
                    "code": data_type.value,
                    "name": data_type.value.title(),
                    "description": f"{data_type.value} market data",
                }
                for data_type in DataType
            ],
        },
    }
