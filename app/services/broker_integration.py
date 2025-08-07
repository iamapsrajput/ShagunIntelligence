"""
Broker API Integration Service
Unified interface for multiple broker APIs (Zerodha Kite, Angel One, etc.)
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import aiohttp
from loguru import logger

from app.core.resilience import with_circuit_breaker, with_retry


class BrokerProvider(Enum):
    """Supported broker providers"""

    ZERODHA_KITE = "zerodha_kite"
    ANGEL_ONE = "angel_one"
    UPSTOX = "upstox"
    FYERS = "fyers"
    MOCK = "mock"


class OrderType(Enum):
    """Order types"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"  # Stop Loss
    SL_M = "SL-M"  # Stop Loss Market


class OrderStatus(Enum):
    """Order status"""

    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"


class TransactionType(Enum):
    """Transaction types"""

    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderRequest:
    """Unified order request"""

    symbol: str
    exchange: str
    transaction_type: TransactionType
    order_type: OrderType
    quantity: int
    price: float | None = None
    trigger_price: float | None = None
    validity: str = "DAY"
    disclosed_quantity: int = 0
    tag: str | None = None


@dataclass
class OrderResponse:
    """Unified order response"""

    order_id: str
    status: OrderStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Trading position"""

    symbol: str
    exchange: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    pnl_percent: float
    product: str
    instrument_token: str | None = None


@dataclass
class Holding:
    """Portfolio holding"""

    symbol: str
    exchange: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    pnl_percent: float
    collateral_quantity: int = 0
    collateral_type: str | None = None


class BaseBrokerAPI(ABC):
    """Abstract base class for broker APIs"""

    def __init__(self, api_key: str, access_token: str = None):
        self.api_key = api_key
        self.access_token = access_token
        self.session = None
        self.base_url = ""
        self.headers = {}

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the broker API"""
        pass

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a trading order"""
        pass

    @abstractmethod
    async def modify_order(self, order_id: str, **kwargs) -> OrderResponse:
        """Modify an existing order"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an order"""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get order status"""
        pass

    @abstractmethod
    async def get_orders(self) -> list[dict[str, Any]]:
        """Get all orders"""
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get current positions"""
        pass

    @abstractmethod
    async def get_holdings(self) -> list[Holding]:
        """Get portfolio holdings"""
        pass

    @abstractmethod
    async def get_margins(self) -> dict[str, Any]:
        """Get account margins"""
        pass


class ZerodhaKiteAPI(BaseBrokerAPI):
    """Zerodha Kite API implementation"""

    def __init__(self, api_key: str, access_token: str = None):
        super().__init__(api_key, access_token)
        self.base_url = "https://api.kite.trade"
        self.headers = {
            "X-Kite-Version": "3",
            "Authorization": f"token {api_key}:{access_token}" if access_token else "",
        }

    async def authenticate(self) -> bool:
        """Authenticate with Kite API"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(
                f"{self.base_url}/user/profile", headers=self.headers
            ) as response:
                if response.status == 200:
                    profile = await response.json()
                    logger.info(
                        f"Authenticated with Kite API: {profile['data']['user_name']}"
                    )
                    return True
                else:
                    logger.error(f"Kite authentication failed: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Kite authentication error: {str(e)}")
            return False

    @with_circuit_breaker("kite_api")
    @with_retry(max_retries=3, delay=1.0)
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place order via Kite API"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            order_data = {
                "tradingsymbol": order.symbol,
                "exchange": order.exchange,
                "transaction_type": order.transaction_type.value,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "validity": order.validity,
                "product": "MIS",  # Margin Intraday Square-off
                "disclosed_quantity": order.disclosed_quantity,
            }

            if order.price:
                order_data["price"] = order.price
            if order.trigger_price:
                order_data["trigger_price"] = order.trigger_price
            if order.tag:
                order_data["tag"] = order.tag

            async with self.session.post(
                f"{self.base_url}/orders/regular", headers=self.headers, data=order_data
            ) as response:
                result = await response.json()

                if response.status == 200 and result["status"] == "success":
                    return OrderResponse(
                        order_id=result["data"]["order_id"],
                        status=OrderStatus.PENDING,
                        message="Order placed successfully",
                    )
                else:
                    error_msg = result.get("message", "Unknown error")
                    return OrderResponse(
                        order_id="",
                        status=OrderStatus.REJECTED,
                        message=f"Order rejected: {error_msg}",
                    )

        except Exception as e:
            logger.error(f"Error placing Kite order: {str(e)}")
            return OrderResponse(
                order_id="", status=OrderStatus.REJECTED, message=f"Error: {str(e)}"
            )

    async def modify_order(self, order_id: str, **kwargs) -> OrderResponse:
        """Modify Kite order"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.put(
                f"{self.base_url}/orders/regular/{order_id}",
                headers=self.headers,
                data=kwargs,
            ) as response:
                result = await response.json()

                if response.status == 200 and result["status"] == "success":
                    return OrderResponse(
                        order_id=order_id,
                        status=OrderStatus.MODIFIED,
                        message="Order modified successfully",
                    )
                else:
                    error_msg = result.get("message", "Unknown error")
                    return OrderResponse(
                        order_id=order_id,
                        status=OrderStatus.REJECTED,
                        message=f"Modification failed: {error_msg}",
                    )

        except Exception as e:
            logger.error(f"Error modifying Kite order: {str(e)}")
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message=f"Error: {str(e)}",
            )

    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel Kite order"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.delete(
                f"{self.base_url}/orders/regular/{order_id}", headers=self.headers
            ) as response:
                result = await response.json()

                if response.status == 200 and result["status"] == "success":
                    return OrderResponse(
                        order_id=order_id,
                        status=OrderStatus.CANCELLED,
                        message="Order cancelled successfully",
                    )
                else:
                    error_msg = result.get("message", "Unknown error")
                    return OrderResponse(
                        order_id=order_id,
                        status=OrderStatus.REJECTED,
                        message=f"Cancellation failed: {error_msg}",
                    )

        except Exception as e:
            logger.error(f"Error cancelling Kite order: {str(e)}")
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message=f"Error: {str(e)}",
            )

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get Kite order status"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(
                f"{self.base_url}/orders", headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    orders = result.get("data", [])

                    for order in orders:
                        if order["order_id"] == order_id:
                            return {
                                "order_id": order["order_id"],
                                "status": order["status"],
                                "tradingsymbol": order["tradingsymbol"],
                                "transaction_type": order["transaction_type"],
                                "quantity": order["quantity"],
                                "filled_quantity": order["filled_quantity"],
                                "pending_quantity": order["pending_quantity"],
                                "price": order["price"],
                                "average_price": order["average_price"],
                                "order_timestamp": order["order_timestamp"],
                            }

                    return {"error": "Order not found"}
                else:
                    return {"error": f"API error: {response.status}"}

        except Exception as e:
            logger.error(f"Error getting Kite order status: {str(e)}")
            return {"error": str(e)}

    async def get_orders(self) -> list[dict[str, Any]]:
        """Get all Kite orders"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(
                f"{self.base_url}/orders", headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("data", [])
                else:
                    logger.error(f"Error getting orders: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error getting Kite orders: {str(e)}")
            return []

    async def get_positions(self) -> list[Position]:
        """Get Kite positions"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(
                f"{self.base_url}/portfolio/positions", headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    positions = []

                    for pos_data in result.get("data", {}).get("net", []):
                        if pos_data["quantity"] != 0:  # Only non-zero positions
                            position = Position(
                                symbol=pos_data["tradingsymbol"],
                                exchange=pos_data["exchange"],
                                quantity=pos_data["quantity"],
                                average_price=pos_data["average_price"],
                                last_price=pos_data["last_price"],
                                pnl=pos_data["pnl"],
                                pnl_percent=(
                                    (
                                        pos_data["pnl"]
                                        / (
                                            pos_data["average_price"]
                                            * abs(pos_data["quantity"])
                                        )
                                    )
                                    * 100
                                    if pos_data["average_price"] > 0
                                    else 0
                                ),
                                product=pos_data["product"],
                                instrument_token=str(pos_data["instrument_token"]),
                            )
                            positions.append(position)

                    return positions
                else:
                    logger.error(f"Error getting positions: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error getting Kite positions: {str(e)}")
            return []

    async def get_holdings(self) -> list[Holding]:
        """Get Kite holdings"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(
                f"{self.base_url}/portfolio/holdings", headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    holdings = []

                    for holding_data in result.get("data", []):
                        if holding_data["quantity"] > 0:  # Only positive holdings
                            holding = Holding(
                                symbol=holding_data["tradingsymbol"],
                                exchange=holding_data["exchange"],
                                quantity=holding_data["quantity"],
                                average_price=holding_data["average_price"],
                                last_price=holding_data["last_price"],
                                pnl=holding_data["pnl"],
                                pnl_percent=(
                                    (
                                        holding_data["pnl"]
                                        / (
                                            holding_data["average_price"]
                                            * holding_data["quantity"]
                                        )
                                    )
                                    * 100
                                    if holding_data["average_price"] > 0
                                    else 0
                                ),
                                collateral_quantity=holding_data.get(
                                    "collateral_quantity", 0
                                ),
                                collateral_type=holding_data.get("collateral_type"),
                            )
                            holdings.append(holding)

                    return holdings
                else:
                    logger.error(f"Error getting holdings: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error getting Kite holdings: {str(e)}")
            return []

    async def get_margins(self) -> dict[str, Any]:
        """Get Kite account margins"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(
                f"{self.base_url}/user/margins", headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("data", {})
                else:
                    logger.error(f"Error getting margins: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting Kite margins: {str(e)}")
            return {}


class MockBrokerAPI(BaseBrokerAPI):
    """Mock broker API for testing and development"""

    def __init__(self, api_key: str, access_token: str = None):
        super().__init__(api_key, access_token)
        self.orders = {}
        self.order_counter = 1000
        self.mock_positions = []
        self.mock_holdings = []

        # Initialize with some mock data
        self._initialize_mock_data()

    def _initialize_mock_data(self):
        """Initialize mock positions and holdings"""
        self.mock_positions = [
            Position(
                symbol="RELIANCE",
                exchange="NSE",
                quantity=100,
                average_price=2450.0,
                last_price=2500.0,
                pnl=5000.0,
                pnl_percent=2.04,
                product="MIS",
            ),
            Position(
                symbol="TCS",
                exchange="NSE",
                quantity=-50,  # Short position
                average_price=3600.0,
                last_price=3550.0,
                pnl=2500.0,
                pnl_percent=1.39,
                product="MIS",
            ),
        ]

        self.mock_holdings = [
            Holding(
                symbol="HDFCBANK",
                exchange="NSE",
                quantity=75,
                average_price=1550.0,
                last_price=1600.0,
                pnl=3750.0,
                pnl_percent=3.23,
            ),
            Holding(
                symbol="INFY",
                exchange="NSE",
                quantity=60,
                average_price=2400.0,
                last_price=2500.0,
                pnl=6000.0,
                pnl_percent=4.17,
            ),
        ]

    async def authenticate(self) -> bool:
        """Mock authentication (always succeeds)"""
        logger.info("Mock broker authentication successful")
        return True

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Mock order placement"""
        try:
            order_id = f"MOCK_{self.order_counter}"
            self.order_counter += 1

            # Simulate order validation
            if order.quantity <= 0:
                return OrderResponse(
                    order_id="", status=OrderStatus.REJECTED, message="Invalid quantity"
                )

            if order.order_type == OrderType.LIMIT and not order.price:
                return OrderResponse(
                    order_id="",
                    status=OrderStatus.REJECTED,
                    message="Price required for limit orders",
                )

            # Store order details
            self.orders[order_id] = {
                "order_id": order_id,
                "symbol": order.symbol,
                "exchange": order.exchange,
                "transaction_type": order.transaction_type.value,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "price": order.price,
                "trigger_price": order.trigger_price,
                "status": "OPEN",
                "filled_quantity": 0,
                "pending_quantity": order.quantity,
                "average_price": 0.0,
                "order_timestamp": datetime.now(),
                "exchange_timestamp": datetime.now(),
            }

            # Simulate immediate execution for market orders
            if order.order_type == OrderType.MARKET:
                await asyncio.sleep(0.1)  # Simulate processing delay
                self.orders[order_id]["status"] = "COMPLETE"
                self.orders[order_id]["filled_quantity"] = order.quantity
                self.orders[order_id]["pending_quantity"] = 0
                self.orders[order_id]["average_price"] = (
                    order.price or 2500.0
                )  # Mock price

            return OrderResponse(
                order_id=order_id,
                status=(
                    OrderStatus.OPEN
                    if order.order_type != OrderType.MARKET
                    else OrderStatus.COMPLETE
                ),
                message="Order placed successfully",
            )

        except Exception as e:
            logger.error(f"Error in mock order placement: {str(e)}")
            return OrderResponse(
                order_id="", status=OrderStatus.REJECTED, message=f"Error: {str(e)}"
            )

    async def modify_order(self, order_id: str, **kwargs) -> OrderResponse:
        """Mock order modification"""
        if order_id not in self.orders:
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message="Order not found",
            )

        # Update order details
        for key, value in kwargs.items():
            if key in self.orders[order_id]:
                self.orders[order_id][key] = value

        self.orders[order_id]["status"] = "MODIFIED"

        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.MODIFIED,
            message="Order modified successfully",
        )

    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Mock order cancellation"""
        if order_id not in self.orders:
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message="Order not found",
            )

        if self.orders[order_id]["status"] == "COMPLETE":
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message="Cannot cancel completed order",
            )

        self.orders[order_id]["status"] = "CANCELLED"

        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            message="Order cancelled successfully",
        )

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get mock order status"""
        if order_id in self.orders:
            return self.orders[order_id]
        else:
            return {"error": "Order not found"}

    async def get_orders(self) -> list[dict[str, Any]]:
        """Get all mock orders"""
        return list(self.orders.values())

    async def get_positions(self) -> list[Position]:
        """Get mock positions"""
        return self.mock_positions.copy()

    async def get_holdings(self) -> list[Holding]:
        """Get mock holdings"""
        return self.mock_holdings.copy()

    async def get_margins(self) -> dict[str, Any]:
        """Get mock margins"""
        return {
            "equity": {
                "enabled": True,
                "net": 100000.0,
                "available": {
                    "adhoc_margin": 0.0,
                    "cash": 50000.0,
                    "opening_balance": 100000.0,
                    "live_balance": 75000.0,
                    "collateral": 25000.0,
                    "intraday_payin": 0.0,
                },
                "utilised": {
                    "debits": 25000.0,
                    "exposure": 15000.0,
                    "m2m_realised": 0.0,
                    "m2m_unrealised": 2000.0,
                    "option_premium": 0.0,
                    "payout": 0.0,
                    "span": 8000.0,
                    "holding_sales": 0.0,
                    "turnover": 0.0,
                    "liquid_collateral": 0.0,
                    "stock_collateral": 25000.0,
                },
            }
        }


class BrokerManager:
    """Unified broker management interface"""

    def __init__(self):
        self.brokers: dict[str, BaseBrokerAPI] = {}
        self.active_broker: str | None = None

    def add_broker(self, name: str, broker_api: BaseBrokerAPI):
        """Add a broker API instance"""
        self.brokers[name] = broker_api
        if not self.active_broker:
            self.active_broker = name
        logger.info(f"Added broker: {name}")

    def set_active_broker(self, name: str):
        """Set the active broker"""
        if name in self.brokers:
            self.active_broker = name
            logger.info(f"Active broker set to: {name}")
        else:
            raise ValueError(f"Broker {name} not found")

    def get_active_broker(self) -> BaseBrokerAPI | None:
        """Get the active broker API"""
        if self.active_broker and self.active_broker in self.brokers:
            return self.brokers[self.active_broker]
        return None

    async def authenticate_all(self) -> dict[str, bool]:
        """Authenticate all brokers"""
        results = {}
        for name, broker in self.brokers.items():
            try:
                results[name] = await broker.authenticate()
            except Exception as e:
                logger.error(f"Authentication failed for {name}: {str(e)}")
                results[name] = False
        return results

    async def place_order(
        self, order: OrderRequest, broker_name: str | None = None
    ) -> OrderResponse:
        """Place order using specified or active broker"""
        broker = (
            self.brokers.get(broker_name) if broker_name else self.get_active_broker()
        )

        if not broker:
            return OrderResponse(
                order_id="",
                status=OrderStatus.REJECTED,
                message="No active broker available",
            )

        return await broker.place_order(order)

    async def get_consolidated_positions(self) -> dict[str, list[Position]]:
        """Get positions from all brokers"""
        consolidated = {}
        for name, broker in self.brokers.items():
            try:
                positions = await broker.get_positions()
                consolidated[name] = positions
            except Exception as e:
                logger.error(f"Error getting positions from {name}: {str(e)}")
                consolidated[name] = []
        return consolidated

    async def get_consolidated_holdings(self) -> dict[str, list[Holding]]:
        """Get holdings from all brokers"""
        consolidated = {}
        for name, broker in self.brokers.items():
            try:
                holdings = await broker.get_holdings()
                consolidated[name] = holdings
            except Exception as e:
                logger.error(f"Error getting holdings from {name}: {str(e)}")
                consolidated[name] = []
        return consolidated

    def get_broker_status(self) -> dict[str, Any]:
        """Get status of all brokers"""
        return {
            "active_broker": self.active_broker,
            "available_brokers": list(self.brokers.keys()),
            "total_brokers": len(self.brokers),
        }
