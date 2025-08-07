"""Comprehensive Kite Connect service module with all features integrated"""

import asyncio
from datetime import date, datetime
from typing import Any

from loguru import logger

from app.core.config import get_settings

from .auth import KiteAuthManager
from .error_handler import global_error_handler
from .exceptions import KiteAuthenticationError, KiteException
from .historical_data import HistoricalDataService, Interval
from .monitoring import monitoring_service
from .order_management import (
    OrderManager,
    OrderRequest,
    OrderType,
    ProductType,
    TransactionType,
)
from .portfolio_manager import PortfolioManager
from .validators import DataValidator
from .websocket_client import KiteWebSocketClient


class KiteConnectService:
    """Comprehensive Kite Connect service with all features"""

    def __init__(self):
        self.settings = get_settings()

        # Initialize core components
        self.auth_manager = KiteAuthManager()
        self.websocket_client = None
        self.historical_data_service = None
        self.order_manager = None
        self.portfolio_manager = None

        # Initialize monitoring
        monitoring_service.start_monitoring()

        # Connection status
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the Kite service with authentication"""
        try:
            logger.info("Initializing Kite Connect service...")

            # Initialize authentication
            auth_success = await self.auth_manager.initialize()
            if not auth_success:
                logger.error("Authentication failed. Please login manually.")
                return False

            # Initialize other services
            self.websocket_client = KiteWebSocketClient(self.auth_manager)
            self.historical_data_service = HistoricalDataService(self.auth_manager)
            self.order_manager = OrderManager(self.auth_manager)
            self.portfolio_manager = PortfolioManager(self.auth_manager)

            self.is_initialized = True
            logger.info("Kite Connect service initialized successfully")

            # Record initialization metric
            monitoring_service.metrics_collector.record_metric(
                "kite_service_initialized", 1.0, {"status": "success"}
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Kite service: {str(e)}")
            monitoring_service.metrics_collector.record_metric(
                "kite_service_initialized", 0.0, {"status": "error"}
            )
            return False

    def get_login_url(self) -> str:
        """Get login URL for manual authentication"""
        return self.auth_manager.get_login_url()

    @global_error_handler.with_error_handling("authentication")
    async def generate_session(self, request_token: str) -> dict[str, Any]:
        """Generate session using request token"""
        monitoring_service.logger.log_api_call(
            "generate_session", {"request_token": "***"}
        )

        result = await self.auth_manager.generate_session(request_token)

        monitoring_service.logger.log_api_response(
            "generate_session", len(str(result)), 0, True
        )
        return result

    # Market Data Methods
    @global_error_handler.with_error_handling("market_data")
    @monitoring_service.performance_monitor.monitor_operation("get_quote")
    async def get_quote(self, symbol: str, exchange: str = "NSE") -> dict[str, Any]:
        """Get real-time quote for a symbol"""
        self._ensure_initialized()

        # Validate symbol
        validation = DataValidator.validate_symbol(symbol, exchange)
        if not validation.is_valid:
            raise KiteException(f"Invalid symbol: {', '.join(validation.errors)}")

        monitoring_service.logger.log_api_call(
            "get_quote", {"symbol": symbol, "exchange": exchange}
        )

        try:
            instrument_key = f"{exchange}:{validation.value}"
            quote = await asyncio.to_thread(
                self.auth_manager.kite.quote, instrument_key
            )

            if instrument_key in quote:
                data = quote[instrument_key]
                result = {
                    "symbol": symbol,
                    "last_price": data["last_price"],
                    "change": data.get("net_change", 0),
                    "change_percent": (
                        (
                            (data["last_price"] - data["ohlc"]["close"])
                            / data["ohlc"]["close"]
                        )
                        * 100
                        if data["ohlc"]["close"]
                        else 0
                    ),
                    "volume": data.get("volume", 0),
                    "timestamp": datetime.now(),
                    "ohlc": data.get("ohlc", {}),
                    "exchange": exchange,
                }

                monitoring_service.logger.log_api_response(
                    "get_quote", len(str(result)), 0, True
                )
                return result
            else:
                raise ValueError(f"No data found for symbol: {symbol}")

        except Exception as e:
            monitoring_service.logger.log_error("get_quote", e, {"symbol": symbol})
            raise

    @global_error_handler.with_error_handling("market_data")
    @monitoring_service.performance_monitor.monitor_operation("get_quotes")
    async def get_quotes(
        self, symbols: list[str], exchange: str = "NSE"
    ) -> list[dict[str, Any]]:
        """Get quotes for multiple symbols"""
        self._ensure_initialized()

        monitoring_service.logger.log_api_call(
            "get_quotes", {"symbols_count": len(symbols), "exchange": exchange}
        )

        try:
            # Validate all symbols
            validated_symbols = []
            for symbol in symbols:
                validation = DataValidator.validate_symbol(symbol, exchange)
                if validation.is_valid:
                    validated_symbols.append(validation.value)
                else:
                    logger.warning(
                        f"Skipping invalid symbol {symbol}: {validation.errors}"
                    )

            if not validated_symbols:
                return []

            instrument_keys = [f"{exchange}:{symbol}" for symbol in validated_symbols]
            quotes = await asyncio.to_thread(
                self.auth_manager.kite.quote, instrument_keys
            )

            result = []
            for symbol in validated_symbols:
                key = f"{exchange}:{symbol}"
                if key in quotes:
                    data = quotes[key]
                    result.append(
                        {
                            "symbol": symbol,
                            "last_price": data["last_price"],
                            "change": data.get("net_change", 0),
                            "change_percent": (
                                (
                                    (data["last_price"] - data["ohlc"]["close"])
                                    / data["ohlc"]["close"]
                                )
                                * 100
                                if data["ohlc"]["close"]
                                else 0
                            ),
                            "volume": data.get("volume", 0),
                            "timestamp": datetime.now(),
                            "exchange": exchange,
                        }
                    )

            monitoring_service.logger.log_api_response(
                "get_quotes", len(str(result)), 0, True
            )
            return result

        except Exception as e:
            monitoring_service.logger.log_error(
                "get_quotes", e, {"symbols_count": len(symbols)}
            )
            raise

    @global_error_handler.with_error_handling("market_data")
    async def get_historical_data(
        self,
        symbol: str,
        from_date: date | datetime | str,
        to_date: date | datetime | str,
        interval: str = "day",
        exchange: str = "NSE",
    ) -> list[dict[str, Any]]:
        """Get historical OHLC data"""
        self._ensure_initialized()

        # Validate request
        validation = DataValidator.validate_historical_data_request(
            symbol, from_date, to_date, interval, exchange
        )
        if not validation.is_valid:
            raise KiteException(f"Invalid request: {', '.join(validation.errors)}")

        try:
            interval_enum = (
                Interval(interval.upper())
                if hasattr(Interval, interval.upper())
                else Interval.DAY
            )

            ohlc_data = await self.historical_data_service.get_ohlc_data(
                validation.value["symbol"],
                validation.value["from_date"],
                validation.value["to_date"],
                interval_enum,
                validation.value["exchange"],
            )

            return [
                {
                    "date": (
                        ohlc.timestamp.date()
                        if hasattr(ohlc.timestamp, "date")
                        else ohlc.timestamp
                    ),
                    "open": ohlc.open,
                    "high": ohlc.high,
                    "low": ohlc.low,
                    "close": ohlc.close,
                    "volume": ohlc.volume,
                }
                for ohlc in ohlc_data
            ]

        except Exception as e:
            monitoring_service.logger.log_error(
                "get_historical_data", e, {"symbol": symbol}
            )
            raise

    # Order Management Methods
    @global_error_handler.with_error_handling("orders")
    @monitoring_service.performance_monitor.monitor_operation("place_order")
    async def place_order(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """Place a trading order with comprehensive validation"""
        self._ensure_initialized()

        # Validate order data
        validation = DataValidator.validate_order_data(order_data)
        if not validation.is_valid:
            raise KiteException(f"Invalid order data: {', '.join(validation.errors)}")

        monitoring_service.logger.log_order_event(
            "place_order_request", validation.value
        )

        try:
            # Create order request
            order_request = OrderRequest(
                symbol=validation.value["symbol"],
                exchange=validation.value["exchange"],
                transaction_type=TransactionType(validation.value["transaction_type"]),
                quantity=validation.value["quantity"],
                order_type=OrderType(validation.value["order_type"].replace("-", "_")),
                product_type=ProductType(validation.value["product"]),
                price=validation.value.get("price"),
                trigger_price=validation.value.get("trigger_price"),
                tag=validation.value.get("tag"),
            )

            order_id = await self.order_manager.place_order(order_request)

            result = {
                "order_id": order_id,
                "status": "success",
                "message": "Order placed successfully",
                "timestamp": datetime.now(),
            }

            monitoring_service.logger.log_order_event("order_placed", result)
            return result

        except Exception as e:
            monitoring_service.logger.log_error("place_order", e, validation.value)
            raise

    @global_error_handler.with_error_handling("orders")
    async def place_market_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        exchange: str = "NSE",
        product_type: str = "MIS",
    ) -> str:
        """Place a market order"""
        self._ensure_initialized()

        return await self.order_manager.place_market_order(
            symbol=symbol,
            transaction_type=TransactionType(transaction_type.upper()),
            quantity=quantity,
            exchange=exchange,
            product_type=ProductType(product_type.upper()),
        )

    @global_error_handler.with_error_handling("orders")
    async def place_limit_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        price: float,
        exchange: str = "NSE",
        product_type: str = "MIS",
    ) -> str:
        """Place a limit order"""
        self._ensure_initialized()

        return await self.order_manager.place_limit_order(
            symbol=symbol,
            transaction_type=TransactionType(transaction_type.upper()),
            quantity=quantity,
            price=price,
            exchange=exchange,
            product_type=ProductType(product_type.upper()),
        )

    @global_error_handler.with_error_handling("orders")
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        self._ensure_initialized()

        monitoring_service.logger.log_order_event(
            "cancel_order_request", {"order_id": order_id}
        )

        result = await self.order_manager.cancel_order(order_id)

        monitoring_service.logger.log_order_event(
            "order_cancelled", {"order_id": order_id, "success": result}
        )
        return result

    @global_error_handler.with_error_handling("orders")
    async def get_orders(self) -> list[dict[str, Any]]:
        """Get all orders"""
        self._ensure_initialized()

        orders = await self.order_manager.get_all_orders()
        return [
            {
                "order_id": order.order_id,
                "symbol": order.request.symbol if order.request else "",
                "transaction_type": (
                    order.request.transaction_type.value if order.request else ""
                ),
                "quantity": order.request.quantity if order.request else 0,
                "order_type": order.request.order_type.value if order.request else "",
                "status": order.status.value,
                "average_price": order.average_price,
                "filled_quantity": order.filled_quantity,
                "placed_at": order.placed_at,
                "updated_at": order.updated_at,
            }
            for order in orders
        ]

    # Portfolio Management Methods
    @global_error_handler.with_error_handling("portfolio")
    async def get_positions(self) -> dict[str, list[dict[str, Any]]]:
        """Get current positions"""
        self._ensure_initialized()

        positions = await self.portfolio_manager.get_positions()

        result = {}
        for position_type, position_list in positions.items():
            result[position_type] = [
                {
                    "symbol": pos.symbol,
                    "exchange": pos.exchange,
                    "quantity": pos.quantity,
                    "average_price": pos.average_price,
                    "last_price": pos.last_price,
                    "pnl": pos.pnl,
                    "product": pos.product,
                    "value": pos.value,
                }
                for pos in position_list
            ]

        return result

    @global_error_handler.with_error_handling("portfolio")
    async def get_holdings(self) -> list[dict[str, Any]]:
        """Get current holdings"""
        self._ensure_initialized()

        holdings = await self.portfolio_manager.get_holdings()

        return [
            {
                "symbol": holding.symbol,
                "exchange": holding.exchange,
                "quantity": holding.quantity,
                "average_price": holding.average_price,
                "last_price": holding.last_price,
                "pnl": holding.pnl,
                "day_change": holding.day_change,
                "day_change_percentage": holding.day_change_percentage,
                "value": holding.quantity * holding.last_price,
            }
            for holding in holdings
        ]

    @global_error_handler.with_error_handling("portfolio")
    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary"""
        self._ensure_initialized()

        summary = await self.portfolio_manager.get_portfolio_summary()

        return {
            "total_portfolio_value": summary.total_portfolio_value,
            "total_pnl": summary.total_pnl,
            "total_day_change": summary.total_day_change,
            "total_day_change_percentage": summary.total_day_change_percentage,
            "equity": summary.equity,
            "commodity": summary.commodity,
            "margin_available": summary.margin_available,
            "margin_utilised": summary.margin_utilised,
            "timestamp": summary.timestamp,
        }

    # WebSocket Methods
    async def connect_websocket(self) -> bool:
        """Connect to WebSocket for real-time data"""
        self._ensure_initialized()

        if not self.websocket_client:
            raise KiteException("WebSocket client not initialized")

        monitoring_service.logger.log_websocket_event("connect_request", {})

        try:
            success = await self.websocket_client.connect()
            monitoring_service.logger.log_websocket_event(
                "connect_response", {"success": success}
            )
            return success
        except Exception as e:
            monitoring_service.logger.log_error("websocket_connect", e)
            raise

    async def subscribe_symbols(self, symbols: list[str], mode: str = "full") -> bool:
        """Subscribe to symbols for real-time data"""
        self._ensure_initialized()

        if not self.websocket_client or not self.websocket_client.is_connected:
            raise KiteException("WebSocket not connected")

        # Get instrument tokens
        instrument_tokens = []
        for symbol in symbols:
            try:
                token = await self.historical_data_service._get_instrument_token(symbol)
                instrument_tokens.append(token)
            except Exception as e:
                logger.warning(f"Could not get instrument token for {symbol}: {str(e)}")

        if not instrument_tokens:
            return False

        return await self.websocket_client.subscribe(instrument_tokens, mode)

    def set_tick_callback(self, callback):
        """Set callback for tick data"""
        if self.websocket_client:
            self.websocket_client.set_on_tick_callback(callback)

    # Utility Methods
    async def get_instruments(
        self, exchange: str | None = None
    ) -> list[dict[str, Any]]:
        """Get tradeable instruments"""
        self._ensure_initialized()

        return await self.historical_data_service.get_available_instruments(exchange)

    def get_health_status(self) -> dict[str, Any]:
        """Get service health status"""
        return monitoring_service.get_health_status()

    def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self.is_initialized:
            raise KiteAuthenticationError(
                "Kite service not initialized. Please call initialize() first."
            )

    async def shutdown(self):
        """Shutdown the service gracefully"""
        logger.info("Shutting down Kite Connect service...")

        try:
            if self.websocket_client and self.websocket_client.is_connected:
                await self.websocket_client.disconnect()

            if self.auth_manager:
                await self.auth_manager.logout()

            monitoring_service.stop_monitoring()

            logger.info("Kite Connect service shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")


# Maintain backward compatibility
KiteClient = KiteConnectService
