from datetime import datetime

from pydantic import BaseModel


class MarketQuote(BaseModel):
    symbol: str
    price: float
    change: float
    changePercent: float
    volume: int
    bid: float
    ask: float
    timestamp: datetime

    class Config:
        from_attributes = True


class HistoricalData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    indicators: dict | None = None


class OrderBookEntry(BaseModel):
    price: float
    quantity: int
    orders: int


class MarketDepth(BaseModel):
    symbol: str
    buy_orders: list[OrderBookEntry]
    sell_orders: list[OrderBookEntry]
    timestamp: datetime


class TickData(BaseModel):
    symbol: str
    ltp: float  # Last traded price
    volume: int
    bid: float
    ask: float
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime
