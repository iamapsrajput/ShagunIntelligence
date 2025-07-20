from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date

router = APIRouter()

class MarketQuote(BaseModel):
    symbol: str
    last_price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime

class HistoricalData(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

@router.get("/quote/{symbol}", response_model=MarketQuote)
async def get_quote(symbol: str, request: Request):
    """Get real-time quote for a symbol"""
    try:
        kite_client = request.app.state.kite_client
        quote = await kite_client.get_quote(symbol)
        return MarketQuote(**quote)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quotes", response_model=List[MarketQuote])
async def get_multiple_quotes(symbols: str, request: Request):
    """Get quotes for multiple symbols (comma-separated)"""
    try:
        symbol_list = [s.strip() for s in symbols.split(",")]
        kite_client = request.app.state.kite_client
        quotes = await kite_client.get_quotes(symbol_list)
        return [MarketQuote(**quote) for quote in quotes]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical/{symbol}", response_model=List[HistoricalData])
async def get_historical_data(
    symbol: str, 
    from_date: date, 
    to_date: date,
    interval: str = "day",
    request: Request = None
):
    """Get historical data for a symbol"""
    try:
        kite_client = request.app.state.kite_client
        data = await kite_client.get_historical_data(symbol, from_date, to_date, interval)
        return [HistoricalData(**record) for record in data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/instruments")
async def get_instruments(exchange: Optional[str] = None, request: Request = None):
    """Get list of tradeable instruments"""
    try:
        kite_client = request.app.state.kite_client
        instruments = await kite_client.get_instruments(exchange)
        return instruments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))