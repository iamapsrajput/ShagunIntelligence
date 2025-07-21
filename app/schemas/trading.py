from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class OrderType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ProductType(str, Enum):
    MIS = "MIS"  # Intraday
    CNC = "CNC"  # Delivery
    NRML = "NRML"  # Normal


class PriceType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"  # Stop Loss
    SL_M = "SL-M"  # Stop Loss Market


class OrderRequest(BaseModel):
    symbol: str
    exchange: str = "NSE"
    order_type: OrderType
    quantity: int = Field(gt=0)
    product_type: ProductType = ProductType.MIS
    price_type: PriceType = PriceType.MARKET
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    validity: str = "DAY"
    
    # AI and risk management
    use_ai_analysis: bool = True
    auto_position_size: bool = False
    risk_percent: Optional[float] = Field(None, ge=0.1, le=5)
    stop_loss_percent: Optional[float] = Field(None, ge=0.5, le=10)
    position_size_percent: Optional[float] = Field(None, ge=1, le=20)
    
    @validator('price', always=True)
    def validate_price(cls, v, values):
        if values.get('price_type') in [PriceType.LIMIT, PriceType.SL] and v is None:
            raise ValueError('Price is required for LIMIT and SL orders')
        return v
    
    @validator('trigger_price', always=True)
    def validate_trigger_price(cls, v, values):
        if values.get('price_type') in [PriceType.SL, PriceType.SL_M] and v is None:
            raise ValueError('Trigger price is required for SL orders')
        return v


class OrderResponse(BaseModel):
    order_id: str
    status: str
    message: str
    trade_id: Optional[int] = None
    timestamp: datetime


class ModifyOrderRequest(BaseModel):
    quantity: Optional[int] = None
    price: Optional[float] = None
    trigger_price: Optional[float] = None


class PositionResponse(BaseModel):
    symbol: str
    quantity: int
    avgPrice: float
    currentPrice: float
    unrealizedPnL: float
    unrealizedPnLPercent: float
    
    class Config:
        from_attributes = True


class TradeHistoryResponse(BaseModel):
    id: int
    symbol: str
    action: str
    quantity: int
    price: float
    executed_price: Optional[float]
    status: str
    pnl: Optional[float]
    agent_confidence: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True