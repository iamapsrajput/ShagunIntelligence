"""
Advanced Order Management API Routes
Provides endpoints for sophisticated order types and execution algorithms
"""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import get_current_user
from app.services.advanced_order_management import (
    AdvancedOrderManager,
    AdvancedOrderRequest,
    AdvancedOrderType,
    ExecutionStrategy,
)

router = APIRouter(prefix="/advanced-orders", tags=["advanced-orders"])

# Global instance - in production, this would be dependency injected
advanced_order_manager = AdvancedOrderManager()


class AdvancedOrderRequestModel(BaseModel):
    """API model for advanced order requests"""

    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(default="NSE", description="Exchange")
    transaction_type: str = Field(..., description="BUY or SELL")
    quantity: int = Field(..., gt=0, description="Order quantity")
    order_type: AdvancedOrderType = Field(..., description="Advanced order type")
    execution_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.BALANCED, description="Execution strategy"
    )

    # Price parameters
    price: float | None = Field(None, description="Limit price")
    trigger_price: float | None = Field(None, description="Trigger price")
    limit_price: float | None = Field(
        None, description="Limit price for stop orders"
    )

    # Advanced parameters
    max_participation_rate: float = Field(
        default=0.1, ge=0.01, le=0.5, description="Maximum participation rate (1-50%)"
    )
    max_slippage: float = Field(
        default=0.005,
        ge=0.001,
        le=0.02,
        description="Maximum acceptable slippage (0.1-2%)",
    )
    time_horizon: int = Field(
        default=300, ge=60, le=3600, description="Execution time horizon in seconds"
    )
    urgency_factor: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Urgency factor (0=patient, 1=urgent)"
    )

    # Iceberg parameters
    iceberg_visible_quantity: int | None = Field(
        None, description="Visible quantity for iceberg orders"
    )
    iceberg_variance: float = Field(
        default=0.1, ge=0.0, le=0.3, description="Variance in iceberg slice sizes"
    )

    # TWAP/VWAP parameters
    twap_intervals: int = Field(
        default=10, ge=2, le=50, description="Number of TWAP intervals"
    )
    vwap_lookback_periods: int = Field(
        default=20, ge=5, le=100, description="VWAP lookback periods"
    )

    # Risk controls
    max_order_value: float | None = Field(
        None, description="Maximum order value limit"
    )
    position_limit_check: bool = Field(
        default=True, description="Enable position limit checks"
    )

    # Metadata
    strategy_id: str | None = Field(None, description="Strategy identifier")
    tags: dict[str, Any] = Field(default_factory=dict, description="Order tags")


class OrderStatusResponse(BaseModel):
    """Response model for order status"""

    order_id: str
    status: str
    details: dict[str, Any]
    timestamp: datetime


class ExecutionMetricsResponse(BaseModel):
    """Response model for execution metrics"""

    total_orders: int
    successful_orders: int
    success_rate: float
    average_slippage: float
    average_execution_time: float
    market_impact_savings: float
    last_updated: datetime


@router.post("/place", response_model=dict[str, Any])
async def place_advanced_order(
    order_request: AdvancedOrderRequestModel,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Place an advanced order with sophisticated execution algorithms

    Supports:
    - TWAP (Time Weighted Average Price)
    - VWAP (Volume Weighted Average Price)
    - Iceberg orders
    - Smart order routing
    - Market impact analysis
    """
    try:
        # Convert API model to internal model
        advanced_request = AdvancedOrderRequest(
            symbol=order_request.symbol,
            exchange=order_request.exchange,
            transaction_type=order_request.transaction_type,
            quantity=order_request.quantity,
            order_type=order_request.order_type,
            execution_strategy=order_request.execution_strategy,
            price=order_request.price,
            trigger_price=order_request.trigger_price,
            limit_price=order_request.limit_price,
            max_participation_rate=order_request.max_participation_rate,
            max_slippage=order_request.max_slippage,
            time_horizon=order_request.time_horizon,
            urgency_factor=order_request.urgency_factor,
            iceberg_visible_quantity=order_request.iceberg_visible_quantity,
            iceberg_variance=order_request.iceberg_variance,
            twap_intervals=order_request.twap_intervals,
            vwap_lookback_periods=order_request.vwap_lookback_periods,
            max_order_value=order_request.max_order_value,
            position_limit_check=order_request.position_limit_check,
            strategy_id=order_request.strategy_id,
            tags={**order_request.tags, "user_id": current_user.get("user_id")},
        )

        # Execute the advanced order
        result = await advanced_order_manager.place_advanced_order(advanced_request)

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("reason"))

        return {
            "success": True,
            "message": "Advanced order placed successfully",
            "data": result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to place advanced order: {str(e)}"
        )


@router.get("/status/{order_id}", response_model=OrderStatusResponse)
async def get_order_status(
    order_id: str, current_user: dict[str, Any] = Depends(get_current_user)
):
    """Get detailed status of an advanced order"""
    try:
        status = advanced_order_manager.get_order_status(order_id)

        if status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Order not found")

        return OrderStatusResponse(**status)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get order status: {str(e)}"
        )


@router.delete("/cancel/{order_id}")
async def cancel_advanced_order(
    order_id: str, current_user: dict[str, Any] = Depends(get_current_user)
):
    """Cancel an advanced order and all its child orders"""
    try:
        result = await advanced_order_manager.cancel_advanced_order(order_id)

        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Order not found")

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error"))

        return {
            "success": True,
            "message": "Order cancelled successfully",
            "data": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel order: {str(e)}")


@router.get("/active")
async def get_active_orders(current_user: dict[str, Any] = Depends(get_current_user)):
    """Get all active advanced orders"""
    try:
        active_orders = advanced_order_manager.get_active_orders()
        return {"success": True, "data": active_orders}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get active orders: {str(e)}"
        )


@router.get("/metrics", response_model=ExecutionMetricsResponse)
async def get_execution_metrics(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get execution performance metrics"""
    try:
        metrics = advanced_order_manager.get_execution_metrics()
        return ExecutionMetricsResponse(**metrics)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get execution metrics: {str(e)}"
        )


@router.get("/algorithms")
async def get_available_algorithms():
    """Get list of available execution algorithms"""
    return {
        "success": True,
        "data": {
            "algorithms": [
                {
                    "type": "TWAP",
                    "name": "Time Weighted Average Price",
                    "description": "Executes orders evenly over time to minimize market impact",
                    "best_for": "Large orders, low urgency",
                },
                {
                    "type": "VWAP",
                    "name": "Volume Weighted Average Price",
                    "description": "Executes orders following historical volume patterns",
                    "best_for": "Orders seeking to match market volume distribution",
                },
                {
                    "type": "ICEBERG",
                    "name": "Iceberg Orders",
                    "description": "Hides large order size by showing only small portions",
                    "best_for": "Large orders requiring stealth execution",
                },
                {
                    "type": "MARKET",
                    "name": "Market Order",
                    "description": "Immediate execution at current market price",
                    "best_for": "High urgency, small orders",
                },
                {
                    "type": "LIMIT",
                    "name": "Limit Order",
                    "description": "Execution at specified price or better",
                    "best_for": "Price-sensitive orders",
                },
            ],
            "execution_strategies": [
                {"type": "AGGRESSIVE", "description": "Prioritizes speed over price"},
                {"type": "PASSIVE", "description": "Prioritizes price over speed"},
                {
                    "type": "BALANCED",
                    "description": "Balances speed and price optimization",
                },
                {
                    "type": "STEALTH",
                    "description": "Minimizes market impact and detection",
                },
            ],
        },
    }
