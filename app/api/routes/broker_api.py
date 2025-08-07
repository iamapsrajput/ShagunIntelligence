"""
Broker API Integration Routes
Unified interface for multiple broker APIs
"""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import get_current_user
from app.services.broker_integration import (
    BrokerManager,
    BrokerProvider,
    MockBrokerAPI,
    OrderRequest,
    OrderStatus,
    OrderType,
    TransactionType,
    ZerodhaKiteAPI,
)

router = APIRouter(prefix="/broker", tags=["broker-api"])

# Global broker manager - in production, this would be dependency injected
broker_manager = BrokerManager()

# Initialize with mock broker for development
mock_broker = MockBrokerAPI("mock_api_key", "mock_access_token")
broker_manager.add_broker("mock", mock_broker)


class BrokerConnectionRequest(BaseModel):
    """Request to connect a broker"""

    provider: str = Field(
        ..., description="Broker provider (zerodha_kite, angel_one, mock)"
    )
    api_key: str = Field(..., description="API key")
    access_token: str | None = Field(None, description="Access token")
    name: str | None = Field(
        None, description="Custom name for this broker connection"
    )


class OrderPlacementRequest(BaseModel):
    """Request to place an order"""

    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange (NSE, BSE)")
    transaction_type: str = Field(..., description="BUY or SELL")
    order_type: str = Field(..., description="MARKET, LIMIT, SL, SL-M")
    quantity: int = Field(..., gt=0, description="Order quantity")
    price: float | None = Field(None, description="Price for limit orders")
    trigger_price: float | None = Field(
        None, description="Trigger price for stop loss orders"
    )
    validity: str = Field(default="DAY", description="Order validity")
    disclosed_quantity: int = Field(default=0, description="Disclosed quantity")
    tag: str | None = Field(None, description="Order tag")
    broker_name: str | None = Field(None, description="Specific broker to use")


class OrderModificationRequest(BaseModel):
    """Request to modify an order"""

    order_id: str = Field(..., description="Order ID to modify")
    quantity: int | None = Field(None, description="New quantity")
    price: float | None = Field(None, description="New price")
    trigger_price: float | None = Field(None, description="New trigger price")
    broker_name: str | None = Field(None, description="Specific broker to use")


@router.post("/connect")
async def connect_broker(
    request: BrokerConnectionRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Connect to a broker API

    Supports multiple broker providers:
    - zerodha_kite: Zerodha Kite API
    - angel_one: Angel One API
    - mock: Mock broker for testing
    """
    try:
        # Validate provider
        try:
            provider = BrokerProvider(request.provider)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid provider: {request.provider}"
            )

        # Create broker API instance
        if provider == BrokerProvider.ZERODHA_KITE:
            broker_api = ZerodhaKiteAPI(request.api_key, request.access_token)
        elif provider == BrokerProvider.MOCK:
            broker_api = MockBrokerAPI(request.api_key, request.access_token)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Provider {request.provider} not yet implemented",
            )

        # Authenticate
        async with broker_api:
            authenticated = await broker_api.authenticate()

            if not authenticated:
                raise HTTPException(status_code=401, detail="Authentication failed")

        # Add to broker manager
        broker_name = (
            request.name or f"{request.provider}_{datetime.now().strftime('%H%M%S')}"
        )
        broker_manager.add_broker(broker_name, broker_api)

        return {
            "success": True,
            "message": f"Successfully connected to {request.provider}",
            "data": {
                "broker_name": broker_name,
                "provider": request.provider,
                "authenticated": True,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to connect broker: {str(e)}"
        )


@router.get("/status")
async def get_broker_status(current_user: dict[str, Any] = Depends(get_current_user)):
    """Get status of all connected brokers"""
    try:
        status = broker_manager.get_broker_status()

        # Get authentication status for all brokers
        auth_results = await broker_manager.authenticate_all()

        return {
            "success": True,
            "data": {**status, "authentication_status": auth_results},
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get broker status: {str(e)}"
        )


@router.post("/set-active")
async def set_active_broker(
    broker_name: str = Field(..., description="Name of broker to set as active"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Set the active broker for trading operations"""
    try:
        broker_manager.set_active_broker(broker_name)

        return {
            "success": True,
            "message": f"Active broker set to {broker_name}",
            "data": {"active_broker": broker_name},
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to set active broker: {str(e)}"
        )


@router.post("/orders/place")
async def place_order(
    request: OrderPlacementRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Place a trading order

    Supports all major order types:
    - MARKET: Execute immediately at market price
    - LIMIT: Execute at specified price or better
    - SL: Stop loss order
    - SL-M: Stop loss market order
    """
    try:
        # Validate enums
        try:
            transaction_type = TransactionType(request.transaction_type)
            order_type = OrderType(request.order_type)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid order parameter: {str(e)}"
            )

        # Create order request
        order = OrderRequest(
            symbol=request.symbol,
            exchange=request.exchange,
            transaction_type=transaction_type,
            order_type=order_type,
            quantity=request.quantity,
            price=request.price,
            trigger_price=request.trigger_price,
            validity=request.validity,
            disclosed_quantity=request.disclosed_quantity,
            tag=request.tag,
        )

        # Place order
        response = await broker_manager.place_order(order, request.broker_name)

        return {
            "success": response.status != OrderStatus.REJECTED,
            "data": {
                "order_id": response.order_id,
                "status": response.status.value,
                "message": response.message,
                "timestamp": response.timestamp,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to place order: {str(e)}")


@router.put("/orders/modify")
async def modify_order(
    request: OrderModificationRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Modify an existing order"""
    try:
        broker = (
            broker_manager.brokers.get(request.broker_name)
            if request.broker_name
            else broker_manager.get_active_broker()
        )

        if not broker:
            raise HTTPException(status_code=400, detail="No active broker available")

        # Prepare modification parameters
        modify_params = {}
        if request.quantity is not None:
            modify_params["quantity"] = request.quantity
        if request.price is not None:
            modify_params["price"] = request.price
        if request.trigger_price is not None:
            modify_params["trigger_price"] = request.trigger_price

        async with broker:
            response = await broker.modify_order(request.order_id, **modify_params)

        return {
            "success": response.status != OrderStatus.REJECTED,
            "data": {
                "order_id": response.order_id,
                "status": response.status.value,
                "message": response.message,
                "timestamp": response.timestamp,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify order: {str(e)}")


@router.delete("/orders/{order_id}")
async def cancel_order(
    order_id: str,
    broker_name: str | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Cancel an order"""
    try:
        broker = (
            broker_manager.brokers.get(broker_name)
            if broker_name
            else broker_manager.get_active_broker()
        )

        if not broker:
            raise HTTPException(status_code=400, detail="No active broker available")

        async with broker:
            response = await broker.cancel_order(order_id)

        return {
            "success": response.status != OrderStatus.REJECTED,
            "data": {
                "order_id": response.order_id,
                "status": response.status.value,
                "message": response.message,
                "timestamp": response.timestamp,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel order: {str(e)}")


@router.get("/orders/{order_id}")
async def get_order_status(
    order_id: str,
    broker_name: str | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get status of a specific order"""
    try:
        broker = (
            broker_manager.brokers.get(broker_name)
            if broker_name
            else broker_manager.get_active_broker()
        )

        if not broker:
            raise HTTPException(status_code=400, detail="No active broker available")

        async with broker:
            order_status = await broker.get_order_status(order_id)

        if "error" in order_status:
            raise HTTPException(status_code=404, detail=order_status["error"])

        return {"success": True, "data": order_status}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get order status: {str(e)}"
        )


@router.get("/orders")
async def get_all_orders(
    broker_name: str | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get all orders"""
    try:
        broker = (
            broker_manager.brokers.get(broker_name)
            if broker_name
            else broker_manager.get_active_broker()
        )

        if not broker:
            raise HTTPException(status_code=400, detail="No active broker available")

        async with broker:
            orders = await broker.get_orders()

        return {"success": True, "data": {"orders": orders, "count": len(orders)}}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get orders: {str(e)}")


@router.get("/positions")
async def get_positions(
    broker_name: str | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get current trading positions"""
    try:
        if broker_name:
            # Get positions from specific broker
            broker = broker_manager.brokers.get(broker_name)
            if not broker:
                raise HTTPException(
                    status_code=404, detail=f"Broker {broker_name} not found"
                )

            async with broker:
                positions = await broker.get_positions()

            return {
                "success": True,
                "data": {
                    "broker": broker_name,
                    "positions": [pos.__dict__ for pos in positions],
                    "count": len(positions),
                },
            }
        else:
            # Get consolidated positions from all brokers
            consolidated = await broker_manager.get_consolidated_positions()

            return {
                "success": True,
                "data": {
                    "consolidated_positions": {
                        broker: [pos.__dict__ for pos in positions]
                        for broker, positions in consolidated.items()
                    },
                    "total_brokers": len(consolidated),
                },
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get positions: {str(e)}"
        )


@router.get("/holdings")
async def get_holdings(
    broker_name: str | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get portfolio holdings"""
    try:
        if broker_name:
            # Get holdings from specific broker
            broker = broker_manager.brokers.get(broker_name)
            if not broker:
                raise HTTPException(
                    status_code=404, detail=f"Broker {broker_name} not found"
                )

            async with broker:
                holdings = await broker.get_holdings()

            return {
                "success": True,
                "data": {
                    "broker": broker_name,
                    "holdings": [holding.__dict__ for holding in holdings],
                    "count": len(holdings),
                },
            }
        else:
            # Get consolidated holdings from all brokers
            consolidated = await broker_manager.get_consolidated_holdings()

            return {
                "success": True,
                "data": {
                    "consolidated_holdings": {
                        broker: [holding.__dict__ for holding in holdings]
                        for broker, holdings in consolidated.items()
                    },
                    "total_brokers": len(consolidated),
                },
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get holdings: {str(e)}")


@router.get("/margins")
async def get_margins(
    broker_name: str | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get account margins"""
    try:
        broker = (
            broker_manager.brokers.get(broker_name)
            if broker_name
            else broker_manager.get_active_broker()
        )

        if not broker:
            raise HTTPException(status_code=400, detail="No active broker available")

        async with broker:
            margins = await broker.get_margins()

        return {"success": True, "data": margins}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get margins: {str(e)}")


@router.get("/providers")
async def get_supported_providers():
    """Get list of supported broker providers"""
    return {
        "success": True,
        "data": {
            "providers": [
                {
                    "code": provider.value,
                    "name": provider.value.replace("_", " ").title(),
                    "description": f"{provider.value} broker API",
                }
                for provider in BrokerProvider
            ],
            "order_types": [
                {
                    "code": order_type.value,
                    "name": order_type.value,
                    "description": f"{order_type.value} order type",
                }
                for order_type in OrderType
            ],
            "transaction_types": [
                {
                    "code": trans_type.value,
                    "name": trans_type.value,
                    "description": f"{trans_type.value} transaction",
                }
                for trans_type in TransactionType
            ],
        },
    }
