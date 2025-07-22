from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from app.core.auth import get_current_user, get_current_active_superuser
from app.models.user import User
from app.services.websocket_manager import websocket_broadcaster

router = APIRouter()


class SystemSettings(BaseModel):
    trading_enabled: bool = True
    paper_trading_mode: bool = False
    max_daily_trades: int = Field(50, ge=1, le=500)
    max_position_value: float = Field(100000, ge=1000)
    market_hours_only: bool = True
    allowed_symbols: List[str] = []
    blocked_symbols: List[str] = []


class RiskParameters(BaseModel):
    max_position_size: float = Field(10.0, ge=1, le=100)
    max_portfolio_risk: float = Field(20.0, ge=5, le=50)
    max_daily_loss: float = Field(5.0, ge=1, le=20)
    stop_loss_percent: float = Field(2.0, ge=0.5, le=10)
    take_profit_percent: float = Field(4.0, ge=1, le=20)
    max_open_positions: int = Field(5, ge=1, le=20)
    allow_short_selling: bool = False
    use_trailing_stop: bool = True
    trailing_stop_percent: float = Field(1.5, ge=0.5, le=5)


class SystemStatus(BaseModel):
    is_active: bool
    trading_enabled: bool
    paper_trading_mode: bool
    active_agents: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH
    open_positions: int
    daily_trades: int
    daily_pnl: float
    system_health: str  # healthy, degraded, critical
    last_update: datetime


@router.get("/status", response_model=SystemStatus)
async def get_system_status(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get current system status"""
    try:
        crew_manager = request.app.state.crew_manager
        kite_client = request.app.state.kite_client

        # Get system components status
        agents_status = await crew_manager.get_all_agents_status()
        active_agents = [
            agent for agent, status in agents_status.items()
            if status.get("enabled", False)
        ]

        # Get trading statistics
        positions = await kite_client.get_positions()
        open_positions = len([p for p in positions.get("net", []) if p["quantity"] != 0])

        # Calculate daily P&L
        day_pnl = sum(pos.get("pnl", 0) for pos in positions.get("net", []))

        # Determine risk level
        risk_level = "LOW"
        if open_positions > 8:
            risk_level = "HIGH"
        elif open_positions > 5:
            risk_level = "MEDIUM"

        # System health check
        system_health = "healthy"
        if len(active_agents) < 3:
            system_health = "degraded"
        elif not kite_client.is_connected():
            system_health = "critical"

        status = SystemStatus(
            is_active=getattr(request.app.state, "trading_active", True),
            trading_enabled=getattr(request.app.state, "trading_enabled", True),
            paper_trading_mode=getattr(request.app.state, "paper_trading_mode", False),
            active_agents=active_agents,
            risk_level=risk_level,
            open_positions=open_positions,
            daily_trades=getattr(request.app.state, "daily_trades", 0),
            daily_pnl=day_pnl,
            system_health=system_health,
            last_update=datetime.utcnow()
        )

        # Broadcast status update
        await websocket_broadcaster.broadcast_system_status({
            "isActive": status.is_active,
            "activeAgents": status.active_agents,
            "riskLevel": status.risk_level,
            "lastUpdate": status.last_update.isoformat()
        })

        return status
    except Exception as e:
        logger.error(f"Error fetching system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/toggle")
async def toggle_system(
    active: bool,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Toggle system on/off"""
    try:
        request.app.state.trading_active = active

        # Start or stop background tasks
        if active:
            # Start trading scheduler
            trading_scheduler = request.app.state.get("trading_scheduler")
            if trading_scheduler and not trading_scheduler.running:
                trading_scheduler.start()

            message = "Trading system activated"
            logger.info(f"System activated by {current_user.username}")
        else:
            # Stop trading scheduler
            trading_scheduler = request.app.state.get("trading_scheduler")
            if trading_scheduler and trading_scheduler.running:
                trading_scheduler.pause()

            message = "Trading system deactivated"
            logger.warning(f"System deactivated by {current_user.username}")

        # Broadcast system status
        background_tasks.add_task(
            websocket_broadcaster.broadcast_system_status,
            {
                "isActive": active,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        return {
            "status": "success",
            "active": active,
            "message": message
        }
    except Exception as e:
        logger.error(f"Error toggling system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings", response_model=SystemSettings)
async def get_system_settings(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get system settings"""
    try:
        # In production, fetch from database
        settings = SystemSettings(
            trading_enabled=getattr(request.app.state, "trading_enabled", True),
            paper_trading_mode=getattr(request.app.state, "paper_trading_mode", False),
            max_daily_trades=getattr(request.app.state, "max_daily_trades", 50),
            max_position_value=getattr(request.app.state, "max_position_value", 100000),
            market_hours_only=getattr(request.app.state, "market_hours_only", True),
            allowed_symbols=getattr(request.app.state, "allowed_symbols", []),
            blocked_symbols=getattr(request.app.state, "blocked_symbols", [])
        )

        return settings
    except Exception as e:
        logger.error(f"Error fetching system settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/settings")
async def update_system_settings(
    settings: SystemSettings,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Update system settings"""
    try:
        # Update application state
        request.app.state.trading_enabled = settings.trading_enabled
        request.app.state.paper_trading_mode = settings.paper_trading_mode
        request.app.state.max_daily_trades = settings.max_daily_trades
        request.app.state.max_position_value = settings.max_position_value
        request.app.state.market_hours_only = settings.market_hours_only
        request.app.state.allowed_symbols = settings.allowed_symbols
        request.app.state.blocked_symbols = settings.blocked_symbols

        # In production, save to database

        logger.info(f"System settings updated by {current_user.username}")

        return {
            "status": "success",
            "message": "System settings updated successfully",
            "settings": settings.dict()
        }
    except Exception as e:
        logger.error(f"Error updating system settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-parameters", response_model=RiskParameters)
async def get_risk_parameters(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get risk management parameters"""
    try:
        # In production, fetch from database
        params = RiskParameters(
            max_position_size=getattr(request.app.state, "max_position_size", 10.0),
            max_portfolio_risk=getattr(request.app.state, "max_portfolio_risk", 20.0),
            max_daily_loss=getattr(request.app.state, "max_daily_loss", 5.0),
            stop_loss_percent=getattr(request.app.state, "stop_loss_percent", 2.0),
            take_profit_percent=getattr(request.app.state, "take_profit_percent", 4.0),
            max_open_positions=getattr(request.app.state, "max_open_positions", 5),
            allow_short_selling=getattr(request.app.state, "allow_short_selling", False),
            use_trailing_stop=getattr(request.app.state, "use_trailing_stop", True),
            trailing_stop_percent=getattr(request.app.state, "trailing_stop_percent", 1.5)
        )

        return params
    except Exception as e:
        logger.error(f"Error fetching risk parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/risk-parameters")
async def update_risk_parameters(
    params: RiskParameters,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Update risk management parameters"""
    try:
        # Update application state
        request.app.state.max_position_size = params.max_position_size
        request.app.state.max_portfolio_risk = params.max_portfolio_risk
        request.app.state.max_daily_loss = params.max_daily_loss
        request.app.state.stop_loss_percent = params.stop_loss_percent
        request.app.state.take_profit_percent = params.take_profit_percent
        request.app.state.max_open_positions = params.max_open_positions
        request.app.state.allow_short_selling = params.allow_short_selling
        request.app.state.use_trailing_stop = params.use_trailing_stop
        request.app.state.trailing_stop_percent = params.trailing_stop_percent

        # Update crew manager risk parameters
        crew_manager = request.app.state.crew_manager
        await crew_manager.update_risk_parameters(params.dict())

        # Broadcast update
        background_tasks.add_task(
            websocket_broadcaster.broadcast_system_status,
            {
                "type": "risk_update",
                "parameters": params.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        logger.info(f"Risk parameters updated by {current_user.username}")

        return {
            "status": "success",
            "message": "Risk parameters updated successfully",
            "parameters": params.dict()
        }
    except Exception as e:
        logger.error(f"Error updating risk parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emergency-stop")
async def emergency_stop(
    reason: str,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Emergency stop - close all positions and halt trading"""
    try:
        kite_client = request.app.state.kite_client

        # Set system to inactive
        request.app.state.trading_active = False
        request.app.state.trading_enabled = False

        # Get all open positions
        positions = await kite_client.get_positions()
        closed_positions = []

        # Close all positions
        for position in positions.get("net", []):
            if position["quantity"] != 0:
                try:
                    # Place opposite order to close position
                    order_type = "SELL" if position["quantity"] > 0 else "BUY"
                    order = await kite_client.place_order({
                        "tradingsymbol": position["tradingsymbol"],
                        "exchange": position["exchange"],
                        "transaction_type": order_type,
                        "quantity": abs(position["quantity"]),
                        "product": position["product"],
                        "order_type": "MARKET",
                        "tag": "emergency_stop"
                    })

                    closed_positions.append({
                        "symbol": position["tradingsymbol"],
                        "quantity": position["quantity"],
                        "order_id": order.get("order_id")
                    })
                except Exception as e:
                    logger.error(f"Failed to close position {position['tradingsymbol']}: {str(e)}")

        # Log emergency stop
        logger.critical(f"EMERGENCY STOP executed by {current_user.username}. Reason: {reason}")

        # Broadcast alert
        await websocket_broadcaster.broadcast_alert({
            "type": "emergency_stop",
            "severity": "critical",
            "message": f"Emergency stop executed: {reason}",
            "closed_positions": len(closed_positions),
            "timestamp": datetime.utcnow().isoformat()
        })

        return {
            "status": "emergency_stop_executed",
            "reason": reason,
            "closed_positions": closed_positions,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error executing emergency stop: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/recent")
async def get_recent_logs(
    current_user: User = Depends(get_current_active_superuser),
    level: Optional[str] = None,
    limit: int = 100
):
    """Get recent system logs (admin only)"""
    try:
        # In production, fetch from log storage
        logs = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "message": "Sample log entry",
                "module": "system"
            }
        ]

        return {
            "logs": logs,
            "count": len(logs),
            "filter": {"level": level} if level else None
        }
    except Exception as e:
        logger.error(f"Error fetching logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance-mode")
async def toggle_maintenance_mode(
    enabled: bool,
    request: Request,
    current_user: User = Depends(get_current_active_superuser),
    message: Optional[str] = None
):
    """Toggle maintenance mode (admin only)"""
    try:
        request.app.state.maintenance_mode = enabled
        request.app.state.maintenance_message = message or "System is under maintenance"

        if enabled:
            logger.warning(f"Maintenance mode enabled by {current_user.username}")
        else:
            logger.info(f"Maintenance mode disabled by {current_user.username}")

        return {
            "status": "success",
            "maintenance_mode": enabled,
            "message": request.app.state.maintenance_message
        }
    except Exception as e:
        logger.error(f"Error toggling maintenance mode: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))