from fastapi import APIRouter, Depends, HTTPException, Request, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from loguru import logger
import uuid

from app.core.auth import get_current_user
from app.models.user import User
from app.models.trade import Trade, Position, TradeStatus, TradeAction
from app.db.session import get_db
from app.services.websocket_manager import websocket_broadcaster
from app.schemas.trading import (
    OrderRequest, OrderResponse, ModifyOrderRequest,
    PositionResponse, TradeHistoryResponse
)

router = APIRouter()


class AIAnalysisRequest(BaseModel):
    symbol: str
    quantity: int
    action: str = Field(..., regex="^(BUY|SELL)$")
    use_all_agents: bool = True


class RiskParameters(BaseModel):
    max_position_size: float = Field(10.0, ge=1, le=100)
    stop_loss_percent: float = Field(2.0, ge=0.5, le=10)
    take_profit_percent: float = Field(4.0, ge=1, le=20)
    use_trailing_stop: bool = True


@router.post("/orders", response_model=OrderResponse)
async def place_order(
    order: OrderRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Place a trading order with AI analysis"""
    try:
        kite_client = request.app.state.kite_client
        crew_manager = request.app.state.crew_manager
        
        # Run AI analysis if requested
        agent_decisions = {}
        if order.use_ai_analysis:
            logger.info(f"Running AI analysis for {order.symbol}")
            analysis = await crew_manager.analyze_trade_opportunity(
                symbol=order.symbol,
                action=order.order_type,
                quantity=order.quantity
            )
            
            if not analysis.get("recommended", False):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Trade not recommended by AI analysis",
                        "analysis": analysis,
                        "confidence": analysis.get("confidence", 0)
                    }
                )
            
            agent_decisions = analysis.get("agent_decisions", {})
        
        # Calculate position size based on risk parameters
        if order.auto_position_size:
            portfolio_value = await kite_client.get_portfolio_value()
            position_size = crew_manager.calculate_position_size(
                portfolio_value=portfolio_value,
                risk_percent=order.risk_percent or 2.0,
                stop_loss_percent=order.stop_loss_percent or 2.0
            )
            order.quantity = position_size
        
        # Create trade record
        trade = Trade(
            user_id=current_user.id,
            symbol=order.symbol,
            action=TradeAction[order.order_type],
            quantity=order.quantity,
            price=order.price or 0,
            status=TradeStatus.PENDING,
            agent_decisions=agent_decisions,
            rationale=agent_decisions.get("rationale", ""),
            confidence_score=agent_decisions.get("confidence", 0),
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            position_size_percent=order.position_size_percent
        )
        
        db.add(trade)
        db.commit()
        
        # Place order through Kite
        try:
            kite_order = {
                "tradingsymbol": order.symbol,
                "exchange": order.exchange,
                "transaction_type": order.order_type,
                "quantity": order.quantity,
                "product": order.product_type,
                "order_type": order.price_type,
                "price": order.price,
                "trigger_price": order.trigger_price,
                "validity": order.validity,
                "tag": f"algohive_{trade.id}"
            }
            
            order_response = await kite_client.place_order(**kite_order)
            
            # Update trade with order ID
            trade.order_id = order_response["order_id"]
            trade.exchange_order_id = order_response.get("exchange_order_id")
            trade.status = TradeStatus.EXECUTED
            db.commit()
            
            # Broadcast trade execution
            background_tasks.add_task(
                websocket_broadcaster.broadcast_trade_execution,
                {
                    "id": str(trade.id),
                    "symbol": trade.symbol,
                    "action": trade.action.value,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "status": trade.status.value,
                    "agentDecisions": agent_decisions,
                    "timestamp": trade.created_at.isoformat(),
                    "rationale": trade.rationale
                }
            )
            
            logger.info(f"Order placed successfully: {order_response['order_id']}")
            
            return OrderResponse(
                order_id=order_response["order_id"],
                status="success",
                message="Order placed successfully",
                trade_id=trade.id,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            # Update trade status on failure
            trade.status = TradeStatus.FAILED
            db.commit()
            raise
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error placing order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/orders/{order_id}")
async def modify_order(
    order_id: str,
    modify_request: ModifyOrderRequest,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Modify an existing order"""
    try:
        kite_client = request.app.state.kite_client
        
        result = await kite_client.modify_order(
            order_id=order_id,
            quantity=modify_request.quantity,
            price=modify_request.price,
            trigger_price=modify_request.trigger_price
        )
        
        logger.info(f"Order {order_id} modified by user {current_user.username}")
        
        return {
            "order_id": order_id,
            "status": "modified",
            "message": "Order modified successfully"
        }
    except Exception as e:
        logger.error(f"Error modifying order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/orders/{order_id}")
async def cancel_order(
    order_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Cancel an existing order"""
    try:
        kite_client = request.app.state.kite_client
        
        result = await kite_client.cancel_order(order_id)
        
        # Update trade status in database
        trade = db.query(Trade).filter(
            Trade.order_id == order_id,
            Trade.user_id == current_user.id
        ).first()
        
        if trade:
            trade.status = TradeStatus.CANCELLED
            db.commit()
        
        logger.info(f"Order {order_id} cancelled by user {current_user.username}")
        
        return {
            "order_id": order_id,
            "status": "cancelled",
            "message": "Order cancelled successfully"
        }
    except Exception as e:
        logger.error(f"Error cancelling order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get current trading positions"""
    try:
        kite_client = request.app.state.kite_client
        kite_positions = await kite_client.get_positions()
        
        # Sync with database
        positions = []
        for kite_pos in kite_positions.get("net", []):
            # Update or create position in database
            db_position = db.query(Position).filter(
                Position.user_id == current_user.id,
                Position.symbol == kite_pos["tradingsymbol"],
                Position.is_active == True
            ).first()
            
            if db_position:
                db_position.quantity = kite_pos["quantity"]
                db_position.avg_price = kite_pos["average_price"]
                db_position.current_price = kite_pos.get("last_price", 0)
                db_position.unrealized_pnl = kite_pos.get("pnl", 0)
                db_position.unrealized_pnl_percent = (
                    (kite_pos.get("pnl", 0) / (kite_pos["average_price"] * abs(kite_pos["quantity"]))) * 100
                    if kite_pos["quantity"] != 0 else 0
                )
            else:
                db_position = Position(
                    user_id=current_user.id,
                    symbol=kite_pos["tradingsymbol"],
                    quantity=kite_pos["quantity"],
                    avg_price=kite_pos["average_price"],
                    current_price=kite_pos.get("last_price", 0),
                    unrealized_pnl=kite_pos.get("pnl", 0),
                    unrealized_pnl_percent=(
                        (kite_pos.get("pnl", 0) / (kite_pos["average_price"] * abs(kite_pos["quantity"]))) * 100
                        if kite_pos["quantity"] != 0 else 0
                    )
                )
                db.add(db_position)
            
            positions.append(PositionResponse(
                symbol=db_position.symbol,
                quantity=db_position.quantity,
                avgPrice=db_position.avg_price,
                currentPrice=db_position.current_price,
                unrealizedPnL=db_position.unrealized_pnl,
                unrealizedPnLPercent=db_position.unrealized_pnl_percent
            ))
        
        db.commit()
        return positions
        
    except Exception as e:
        logger.error(f"Error fetching positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders")
async def get_orders(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, le=200),
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get order history"""
    try:
        kite_client = request.app.state.kite_client
        orders = await kite_client.get_orders()
        
        # Filter by status if provided
        if status:
            orders = [o for o in orders if o.get("status", "").upper() == status.upper()]
        
        # Limit results
        orders = orders[:limit]
        
        return orders
    except Exception as e:
        logger.error(f"Error fetching orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/history", response_model=List[TradeHistoryResponse])
async def get_trade_history(
    symbol: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get trade history with filters"""
    try:
        query = db.query(Trade).filter(Trade.user_id == current_user.id)
        
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        if start_date:
            query = query.filter(Trade.created_at >= start_date)
        if end_date:
            query = query.filter(Trade.created_at <= end_date)
        
        trades = query.order_by(Trade.created_at.desc()).limit(limit).all()
        
        return [
            TradeHistoryResponse(
                id=trade.id,
                symbol=trade.symbol,
                action=trade.action.value,
                quantity=trade.quantity,
                price=trade.price,
                executed_price=trade.executed_price,
                status=trade.status.value,
                pnl=trade.realized_pnl if trade.status == TradeStatus.EXECUTED else None,
                agent_confidence=trade.confidence_score,
                created_at=trade.created_at
            )
            for trade in trades
        ]
    except Exception as e:
        logger.error(f"Error fetching trade history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_trade(
    analysis_request: AIAnalysisRequest,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get AI analysis for a potential trade"""
    try:
        crew_manager = request.app.state.crew_manager
        
        # Run comprehensive analysis
        analysis = await crew_manager.analyze_trade_opportunity(
            symbol=analysis_request.symbol,
            action=analysis_request.action,
            quantity=analysis_request.quantity,
            use_all_agents=analysis_request.use_all_agents
        )
        
        # Log analysis request
        logger.info(f"User {current_user.username} requested analysis for {analysis_request.symbol}")
        
        return {
            "symbol": analysis_request.symbol,
            "action": analysis_request.action,
            "recommended": analysis.get("recommended", False),
            "confidence": analysis.get("confidence", 0),
            "analysis": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/paper-trade")
async def execute_paper_trade(
    order: OrderRequest,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Execute a paper trade (simulation)"""
    try:
        # Paper trading logic here
        trade_id = str(uuid.uuid4())
        
        logger.info(f"Paper trade executed: {order.symbol} {order.order_type} {order.quantity}")
        
        return {
            "trade_id": trade_id,
            "status": "paper_executed",
            "message": "Paper trade executed successfully",
            "details": {
                "symbol": order.symbol,
                "action": order.order_type,
                "quantity": order.quantity,
                "price": order.price or "market",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error executing paper trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))