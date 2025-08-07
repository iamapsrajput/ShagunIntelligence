"""Advanced order management system for trading operations"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from .auth import KiteAuthManager
from .exceptions import KiteOrderError, KiteValidationError
from .rate_limiter import RateLimiter


class OrderType(Enum):
    """Order types supported by the system"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "SL"
    STOP_LOSS_MARKET = "SL-M"
    BRACKET_ORDER = "BO"
    COVER_ORDER = "CO"


class TransactionType(Enum):
    """Transaction types"""

    BUY = "BUY"
    SELL = "SELL"


class ProductType(Enum):
    """Product types"""

    CNC = "CNC"  # Cash and Carry (delivery)
    MIS = "MIS"  # Margin Intraday Square-off
    NRML = "NRML"  # Normal (carry forward)


class OrderStatus(Enum):
    """Order status"""

    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


class Validity(Enum):
    """Order validity"""

    DAY = "DAY"
    IOC = "IOC"  # Immediate or Cancel
    TTL = "TTL"  # Till cancelled


@dataclass
class OrderRequest:
    """Order request structure"""

    symbol: str
    exchange: str
    transaction_type: TransactionType
    quantity: int
    order_type: OrderType
    product_type: ProductType = ProductType.MIS
    price: float | None = None
    trigger_price: float | None = None
    validity: Validity = Validity.DAY
    disclosed_quantity: int | None = None
    squareoff: float | None = None  # For bracket orders
    stoploss: float | None = None  # For bracket orders
    trailing_stoploss: float | None = None
    tag: str | None = None

    def __post_init__(self):
        # Validate order parameters
        if (
            self.order_type in [OrderType.LIMIT, OrderType.STOP_LOSS]
            and self.price is None
        ):
            raise KiteValidationError("Price is required for LIMIT and SL orders")

        if (
            self.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_MARKET]
            and self.trigger_price is None
        ):
            raise KiteValidationError("Trigger price is required for stop loss orders")


@dataclass
class Order:
    """Order data structure"""

    order_id: str
    request: OrderRequest
    status: OrderStatus
    average_price: float | None = None
    filled_quantity: int = 0
    pending_quantity: int = 0
    cancelled_quantity: int = 0
    placed_at: datetime | None = None
    updated_at: datetime | None = None
    rejection_reason: str | None = None
    parent_order_id: str | None = None  # For bracket/cover orders
    child_orders: list[str] = None  # For bracket orders

    def __post_init__(self):
        if self.child_orders is None:
            self.child_orders = []


class OrderManager:
    """Advanced order management system"""

    def __init__(self, auth_manager: KiteAuthManager):
        self.auth_manager = auth_manager
        self.rate_limiter = RateLimiter(
            max_requests=20, time_window=60
        )  # 20 orders per minute
        self.orders: dict[str, Order] = {}
        self._order_update_callbacks: list[callable] = []

    async def place_order(self, order_request: OrderRequest) -> str:
        """Place a new order

        Returns:
            Order ID
        """
        try:
            # Wait for rate limiter
            await self.rate_limiter.acquire()

            # Generate internal order ID
            str(uuid.uuid4())

            # Validate order
            await self._validate_order(order_request)

            # Prepare order parameters for Kite API
            order_params = await self._prepare_order_params(order_request)

            # Place order via Kite API
            kite_order_id = await asyncio.to_thread(
                self.auth_manager.kite.place_order, **order_params
            )

            # Create order record
            order = Order(
                order_id=kite_order_id,
                request=order_request,
                status=OrderStatus.PENDING,
                placed_at=datetime.now(),
                updated_at=datetime.now(),
            )

            # Store order
            self.orders[kite_order_id] = order

            logger.info(
                f"Order placed successfully: {kite_order_id} for {order_request.symbol}"
            )

            # Notify callbacks
            await self._notify_order_update(order)

            return kite_order_id

        except Exception as e:
            logger.error(f"Failed to place order: {str(e)}")
            raise KiteOrderError(f"Order placement failed: {str(e)}")

    async def place_market_order(
        self,
        symbol: str,
        transaction_type: TransactionType,
        quantity: int,
        exchange: str = "NSE",
        product_type: ProductType = ProductType.MIS,
        tag: str = None,
    ) -> str:
        """Place a market order"""
        order_request = OrderRequest(
            symbol=symbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=OrderType.MARKET,
            product_type=product_type,
            tag=tag,
        )
        return await self.place_order(order_request)

    async def place_limit_order(
        self,
        symbol: str,
        transaction_type: TransactionType,
        quantity: int,
        price: float,
        exchange: str = "NSE",
        product_type: ProductType = ProductType.MIS,
        validity: Validity = Validity.DAY,
        tag: str = None,
    ) -> str:
        """Place a limit order"""
        order_request = OrderRequest(
            symbol=symbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            product_type=product_type,
            price=price,
            validity=validity,
            tag=tag,
        )
        return await self.place_order(order_request)

    async def place_stop_loss_order(
        self,
        symbol: str,
        transaction_type: TransactionType,
        quantity: int,
        trigger_price: float,
        price: float | None = None,
        exchange: str = "NSE",
        product_type: ProductType = ProductType.MIS,
        tag: str = None,
    ) -> str:
        """Place a stop loss order"""
        order_type = (
            OrderType.STOP_LOSS_MARKET if price is None else OrderType.STOP_LOSS
        )

        order_request = OrderRequest(
            symbol=symbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product_type=product_type,
            price=price,
            trigger_price=trigger_price,
            tag=tag,
        )
        return await self.place_order(order_request)

    async def place_bracket_order(
        self,
        symbol: str,
        transaction_type: TransactionType,
        quantity: int,
        price: float,
        squareoff: float,
        stoploss: float,
        exchange: str = "NSE",
        trailing_stoploss: float | None = None,
        tag: str = None,
    ) -> str:
        """Place a bracket order"""
        order_request = OrderRequest(
            symbol=symbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=OrderType.BRACKET_ORDER,
            product_type=ProductType.MIS,  # BO only supports MIS
            price=price,
            squareoff=squareoff,
            stoploss=stoploss,
            trailing_stoploss=trailing_stoploss,
            tag=tag,
        )
        return await self.place_order(order_request)

    async def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
        trigger_price: float | None = None,
        order_type: OrderType | None = None,
        validity: Validity | None = None,
    ) -> bool:
        """Modify an existing order"""
        try:
            await self.rate_limiter.acquire()

            # Prepare modification parameters
            modify_params = {}
            if quantity is not None:
                modify_params["quantity"] = quantity
            if price is not None:
                modify_params["price"] = price
            if trigger_price is not None:
                modify_params["trigger_price"] = trigger_price
            if order_type is not None:
                modify_params["order_type"] = self._get_kite_order_type(order_type)
            if validity is not None:
                modify_params["validity"] = validity.value

            # Modify order via Kite API
            await asyncio.to_thread(
                self.auth_manager.kite.modify_order, order_id, **modify_params
            )

            # Update local order record
            if order_id in self.orders:
                order = self.orders[order_id]
                order.updated_at = datetime.now()
                await self._notify_order_update(order)

            logger.info(f"Order modified successfully: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {str(e)}")
            raise KiteOrderError(f"Order modification failed: {str(e)}")

    async def cancel_order(self, order_id: str, variety: str = "regular") -> bool:
        """Cancel an order"""
        try:
            await self.rate_limiter.acquire()

            # Cancel order via Kite API
            await asyncio.to_thread(
                self.auth_manager.kite.cancel_order, variety, order_id
            )

            # Update local order record
            if order_id in self.orders:
                order = self.orders[order_id]
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                await self._notify_order_update(order)

            logger.info(f"Order cancelled successfully: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            raise KiteOrderError(f"Order cancellation failed: {str(e)}")

    async def get_order_status(self, order_id: str) -> Order | None:
        """Get order status"""
        try:
            # Fetch from Kite API
            order_data = await asyncio.to_thread(
                self.auth_manager.kite.order_history, order_id
            )

            if order_data:
                # Update local order record
                latest_order = order_data[-1]  # Latest status
                await self._update_order_from_kite_data(order_id, latest_order)
                return self.orders.get(order_id)

            return None

        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            return self.orders.get(order_id)

    async def get_all_orders(self) -> list[Order]:
        """Get all orders for the day"""
        try:
            # Fetch from Kite API
            orders_data = await asyncio.to_thread(self.auth_manager.kite.orders)

            # Update local records
            for order_data in orders_data:
                order_id = order_data.get("order_id")
                if order_id:
                    await self._update_order_from_kite_data(order_id, order_data)

            return list(self.orders.values())

        except Exception as e:
            logger.error(f"Failed to get all orders: {str(e)}")
            return list(self.orders.values())

    async def get_order_history(self, order_id: str) -> list[dict[str, Any]]:
        """Get complete order history"""
        try:
            history = await asyncio.to_thread(
                self.auth_manager.kite.order_history, order_id
            )
            return history or []

        except Exception as e:
            logger.error(f"Failed to get order history for {order_id}: {str(e)}")
            return []

    async def cancel_all_orders(self) -> dict[str, bool]:
        """Cancel all pending orders"""
        try:
            orders = await self.get_all_orders()
            results = {}

            for order in orders:
                if order.status in [OrderStatus.OPEN, OrderStatus.PENDING]:
                    try:
                        success = await self.cancel_order(order.order_id)
                        results[order.order_id] = success
                    except Exception as e:
                        logger.error(
                            f"Failed to cancel order {order.order_id}: {str(e)}"
                        )
                        results[order.order_id] = False

            return results

        except Exception as e:
            logger.error(f"Failed to cancel all orders: {str(e)}")
            return {}

    def add_order_update_callback(self, callback: callable):
        """Add callback for order updates"""
        self._order_update_callbacks.append(callback)

    def remove_order_update_callback(self, callback: callable):
        """Remove order update callback"""
        if callback in self._order_update_callbacks:
            self._order_update_callbacks.remove(callback)

    async def _validate_order(self, order_request: OrderRequest):
        """Validate order parameters"""
        # Basic validations
        if order_request.quantity <= 0:
            raise KiteValidationError("Quantity must be positive")

        if order_request.price is not None and order_request.price <= 0:
            raise KiteValidationError("Price must be positive")

        if order_request.trigger_price is not None and order_request.trigger_price <= 0:
            raise KiteValidationError("Trigger price must be positive")

        # Specific validations for different order types
        if order_request.order_type == OrderType.BRACKET_ORDER:
            if order_request.squareoff is None or order_request.stoploss is None:
                raise KiteValidationError(
                    "Squareoff and stoploss are required for bracket orders"
                )

        # Add more validations as needed
        logger.debug(f"Order validation passed for {order_request.symbol}")

    async def _prepare_order_params(
        self, order_request: OrderRequest
    ) -> dict[str, Any]:
        """Prepare order parameters for Kite API"""
        params = {
            "variety": self._get_variety(order_request.order_type),
            "exchange": order_request.exchange,
            "tradingsymbol": order_request.symbol,
            "transaction_type": order_request.transaction_type.value,
            "quantity": order_request.quantity,
            "product": order_request.product_type.value,
            "order_type": self._get_kite_order_type(order_request.order_type),
            "validity": order_request.validity.value,
        }

        # Add optional parameters
        if order_request.price is not None:
            params["price"] = order_request.price

        if order_request.trigger_price is not None:
            params["trigger_price"] = order_request.trigger_price

        if order_request.disclosed_quantity is not None:
            params["disclosed_quantity"] = order_request.disclosed_quantity

        if order_request.squareoff is not None:
            params["squareoff"] = order_request.squareoff

        if order_request.stoploss is not None:
            params["stoploss"] = order_request.stoploss

        if order_request.trailing_stoploss is not None:
            params["trailing_stoploss"] = order_request.trailing_stoploss

        if order_request.tag is not None:
            params["tag"] = order_request.tag

        return params

    def _get_variety(self, order_type: OrderType) -> str:
        """Get order variety for Kite API"""
        variety_map = {
            OrderType.MARKET: "regular",
            OrderType.LIMIT: "regular",
            OrderType.STOP_LOSS: "regular",
            OrderType.STOP_LOSS_MARKET: "regular",
            OrderType.BRACKET_ORDER: "bo",
            OrderType.COVER_ORDER: "co",
        }
        return variety_map.get(order_type, "regular")

    def _get_kite_order_type(self, order_type: OrderType) -> str:
        """Convert OrderType to Kite API order type"""
        type_map = {
            OrderType.MARKET: self.auth_manager.kite.ORDER_TYPE_MARKET,
            OrderType.LIMIT: self.auth_manager.kite.ORDER_TYPE_LIMIT,
            OrderType.STOP_LOSS: self.auth_manager.kite.ORDER_TYPE_SL,
            OrderType.STOP_LOSS_MARKET: self.auth_manager.kite.ORDER_TYPE_SLM,
            OrderType.BRACKET_ORDER: self.auth_manager.kite.ORDER_TYPE_LIMIT,
            OrderType.COVER_ORDER: self.auth_manager.kite.ORDER_TYPE_MARKET,
        }
        return type_map.get(order_type, self.auth_manager.kite.ORDER_TYPE_MARKET)

    async def _update_order_from_kite_data(
        self, order_id: str, kite_data: dict[str, Any]
    ):
        """Update local order record from Kite API data"""
        try:
            if order_id not in self.orders:
                # Create order record if it doesn't exist
                self.orders[order_id] = Order(
                    order_id=order_id,
                    request=None,  # We don't have the original request
                    status=OrderStatus.PENDING,
                )

            order = self.orders[order_id]

            # Update status
            status_map = {
                "OPEN": OrderStatus.OPEN,
                "COMPLETE": OrderStatus.COMPLETE,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
            }

            kite_status = kite_data.get("status", "").upper()
            order.status = status_map.get(kite_status, OrderStatus.PENDING)

            # Update other fields
            order.average_price = kite_data.get("average_price", 0) or None
            order.filled_quantity = kite_data.get("filled_quantity", 0)
            order.pending_quantity = kite_data.get("pending_quantity", 0)
            order.cancelled_quantity = kite_data.get("cancelled_quantity", 0)
            order.rejection_reason = kite_data.get("status_message")
            order.updated_at = datetime.now()

            # Parse timestamps
            if kite_data.get("order_timestamp"):
                try:
                    order.placed_at = datetime.strptime(
                        kite_data["order_timestamp"], "%Y-%m-%d %H:%M:%S"
                    )
                except:
                    pass

            await self._notify_order_update(order)

        except Exception as e:
            logger.error(f"Failed to update order from Kite data: {str(e)}")

    async def _notify_order_update(self, order: Order):
        """Notify callbacks about order updates"""
        for callback in self._order_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.error(f"Error in order update callback: {str(e)}")
