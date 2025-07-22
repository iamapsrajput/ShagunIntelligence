from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from app.core.auth import get_current_user
from app.models.user import User
from app.schemas.market import MarketDepth, MarketQuote
from app.services.websocket_manager import websocket_broadcaster

router = APIRouter()


class WatchlistItem(BaseModel):
    symbol: str
    exchange: str = "NSE"
    added_at: datetime = Field(default_factory=datetime.utcnow)


class MarketStats(BaseModel):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    prev_close: float
    volume: int
    value: float
    vwap: float
    week_52_high: float
    week_52_low: float
    upper_circuit: Optional[float] = None
    lower_circuit: Optional[float] = None


@router.get("/quote/{symbol}", response_model=MarketQuote)
async def get_quote(
    symbol: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    exchange: str = Query("NSE", description="Exchange (NSE/BSE)"),
):
    """Get real-time quote for a symbol"""
    try:
        kite_client = request.app.state.kite_client
        quote_data = await kite_client.get_quote(f"{exchange}:{symbol}")

        # Broadcast to WebSocket subscribers
        await websocket_broadcaster.broadcast_market_update(symbol, quote_data)

        return MarketQuote(
            symbol=symbol,
            last_price=quote_data["last_price"],
            change=quote_data["change"],
            change_percent=quote_data["change_percent"],
            volume=quote_data["volume"],
            bid=quote_data.get("bid", 0),
            ask=quote_data.get("ask", 0),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quotes", response_model=List[MarketQuote])
async def get_multiple_quotes(
    request: Request,
    current_user: User = Depends(get_current_user),
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    exchange: str = Query("NSE", description="Exchange (NSE/BSE)"),
):
    """Get quotes for multiple symbols"""
    try:
        symbol_list = [s.strip() for s in symbols.split(",")]
        kite_client = request.app.state.kite_client

        quotes = []
        for symbol in symbol_list:
            try:
                quote_data = await kite_client.get_quote(f"{exchange}:{symbol}")
                quotes.append(
                    MarketQuote(
                        symbol=symbol,
                        last_price=quote_data["last_price"],
                        change=quote_data["change"],
                        change_percent=quote_data["change_percent"],
                        volume=quote_data["volume"],
                        bid=quote_data.get("bid", 0),
                        ask=quote_data.get("ask", 0),
                        timestamp=datetime.utcnow(),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to fetch quote for {symbol}: {str(e)}")

        return quotes
    except Exception as e:
        logger.error(f"Error fetching multiple quotes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    interval: str = Query("day", description="Interval: minute, 5minute, 15minute, hour, day"),
    days: int = Query(30, description="Number of days of data"),
):
    """Get historical data for a symbol with technical indicators"""
    try:
        kite_client = request.app.state.kite_client

        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        # Fetch historical data
        data = await kite_client.get_historical_data(
            symbol=f"NSE:{symbol}", from_date=from_date, to_date=to_date, interval=interval
        )

        # Calculate technical indicators
        if len(data) > 0:
            # Add technical indicators
            crew_manager = request.app.state.crew_manager
            indicators = await crew_manager.calculate_technical_indicators(data)

            # Merge indicators with historical data
            for i, record in enumerate(data):
                if i < len(indicators):
                    record["indicators"] = indicators[i]

        return data
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/depth/{symbol}", response_model=MarketDepth)
async def get_market_depth(
    symbol: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    exchange: str = Query("NSE", description="Exchange (NSE/BSE)"),
):
    """Get market depth (order book) for a symbol"""
    try:
        kite_client = request.app.state.kite_client
        depth_data = await kite_client.get_market_depth(f"{exchange}:{symbol}")

        return MarketDepth(
            symbol=symbol,
            buy_orders=depth_data.get("buy", []),
            sell_orders=depth_data.get("sell", []),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error fetching market depth for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{symbol}", response_model=MarketStats)
async def get_market_stats(
    symbol: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    exchange: str = Query("NSE", description="Exchange (NSE/BSE)"),
):
    """Get detailed market statistics for a symbol"""
    try:
        kite_client = request.app.state.kite_client
        ohlc_data = await kite_client.get_ohlc(f"{exchange}:{symbol}")

        return MarketStats(
            symbol=symbol,
            open=ohlc_data["open"],
            high=ohlc_data["high"],
            low=ohlc_data["low"],
            close=ohlc_data["close"],
            prev_close=ohlc_data["prev_close"],
            volume=ohlc_data["volume"],
            value=ohlc_data["value"],
            vwap=ohlc_data["vwap"],
            week_52_high=ohlc_data["week_52_high"],
            week_52_low=ohlc_data["week_52_low"],
            upper_circuit=ohlc_data.get("upper_circuit"),
            lower_circuit=ohlc_data.get("lower_circuit"),
        )
    except Exception as e:
        logger.error(f"Error fetching market stats for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/watchlist", response_model=List[WatchlistItem])
async def get_watchlist(request: Request, current_user: User = Depends(get_current_user)):
    """Get user's watchlist"""
    try:
        # In production, fetch from database
        # For now, return a default watchlist
        default_watchlist = [
            WatchlistItem(symbol="RELIANCE"),
            WatchlistItem(symbol="TCS"),
            WatchlistItem(symbol="INFY"),
            WatchlistItem(symbol="HDFC"),
            WatchlistItem(symbol="ICICIBANK"),
        ]
        return default_watchlist
    except Exception as e:
        logger.error(f"Error fetching watchlist: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watchlist/{symbol}")
async def add_to_watchlist(
    symbol: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    exchange: str = Query("NSE", description="Exchange (NSE/BSE)"),
):
    """Add symbol to watchlist"""
    try:
        # In production, save to database
        logger.info(f"User {current_user.username} added {symbol} to watchlist")

        return {"message": f"{symbol} added to watchlist", "symbol": symbol, "exchange": exchange}
    except Exception as e:
        logger.error(f"Error adding to watchlist: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str, request: Request, current_user: User = Depends(get_current_user)):
    """Remove symbol from watchlist"""
    try:
        # In production, remove from database
        logger.info(f"User {current_user.username} removed {symbol} from watchlist")

        return {"message": f"{symbol} removed from watchlist", "symbol": symbol}
    except Exception as e:
        logger.error(f"Error removing from watchlist: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instruments")
async def get_instruments(
    request: Request,
    current_user: User = Depends(get_current_user),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    segment: Optional[str] = Query(None, description="Filter by segment"),
):
    """Get list of tradeable instruments"""
    try:
        kite_client = request.app.state.kite_client
        instruments = await kite_client.get_instruments()

        # Apply filters
        if exchange:
            instruments = [i for i in instruments if i.get("exchange") == exchange]
        if segment:
            instruments = [i for i in instruments if i.get("segment") == segment]

        return instruments
    except Exception as e:
        logger.error(f"Error fetching instruments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subscribe/{symbol}")
async def subscribe_to_symbol(symbol: str, request: Request, current_user: User = Depends(get_current_user)):
    """Subscribe to real-time updates for a symbol"""
    try:
        kite_client = request.app.state.kite_client

        # Subscribe to ticker updates
        await kite_client.subscribe_ticker(f"NSE:{symbol}")

        logger.info(f"User {current_user.username} subscribed to {symbol}")

        return {"message": f"Subscribed to real-time updates for {symbol}", "symbol": symbol}
    except Exception as e:
        logger.error(f"Error subscribing to symbol: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
