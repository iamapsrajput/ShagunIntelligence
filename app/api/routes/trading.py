from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter()

class TradeRequest(BaseModel):
    symbol: str
    quantity: int
    order_type: str  # "BUY" or "SELL"
    product_type: str = "MIS"  # MIS, CNC, NRML
    price: Optional[float] = None

class TradeResponse(BaseModel):
    order_id: str
    status: str
    message: str
    timestamp: datetime

class PositionResponse(BaseModel):
    symbol: str
    quantity: int
    average_price: float
    pnl: float
    product_type: str

@router.post("/order", response_model=TradeResponse)
async def place_order(trade_request: TradeRequest, request: Request):
    """Place a trading order through CrewAI analysis"""
    try:
        # Get Kite client from app state
        kite_client = request.app.state.kite_client
        crew_manager = request.app.state.crew_manager
        
        # Run CrewAI analysis before placing order
        analysis = await crew_manager.analyze_trade_opportunity(trade_request.symbol)
        
        if not analysis.get("recommended", False):
            raise HTTPException(
                status_code=400, 
                detail=f"Trade not recommended by AI analysis: {analysis.get('reason', 'Unknown')}"
            )
        
        # Place order through Kite
        order_response = await kite_client.place_order(trade_request)
        
        return TradeResponse(
            order_id=order_response["order_id"],
            status="success",
            message="Order placed successfully",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(request: Request):
    """Get current trading positions"""
    try:
        kite_client = request.app.state.kite_client
        positions = await kite_client.get_positions()
        return positions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orders")
async def get_orders(request: Request):
    """Get order history"""
    try:
        kite_client = request.app.state.kite_client
        orders = await kite_client.get_orders()
        return orders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/{symbol}")
async def analyze_symbol(symbol: str, request: Request):
    """Get AI analysis for a specific symbol"""
    try:
        crew_manager = request.app.state.crew_manager
        analysis = await crew_manager.analyze_trade_opportunity(symbol)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))