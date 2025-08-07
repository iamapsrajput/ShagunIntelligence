import random
import uuid
from datetime import datetime, timedelta
from typing import Any


class MockKiteClient:
    """Mock Kite client for testing without actual API calls"""

    def __init__(self):
        self.connected = True
        self.orders = {}
        self.positions = []
        self.portfolio_value = 1000000  # 10 lakh starting capital
        self.market_data = self._generate_market_data()

    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self.connected

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get mock quote for a symbol"""
        if symbol not in self.market_data:
            self.market_data[symbol] = self._generate_quote(symbol)

        quote = self.market_data[symbol]
        # Add some randomness to simulate price movement
        quote["last_price"] *= 1 + random.uniform(-0.002, 0.002)
        quote["timestamp"] = datetime.now()

        return quote

    async def get_quotes(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Get quotes for multiple symbols"""
        quotes = []
        for symbol in symbols:
            quote = await self.get_quote(symbol)
            quotes.append(quote)
        return quotes

    async def get_historical_data(
        self, symbol: str, from_date: datetime, to_date: datetime, interval: str = "day"
    ) -> list[dict[str, Any]]:
        """Generate mock historical data"""
        data = []
        current_date = from_date
        base_price = 2500.0

        while current_date <= to_date:
            # Generate OHLCV data with some randomness
            open_price = base_price * (1 + random.uniform(-0.02, 0.02))
            close_price = open_price * (1 + random.uniform(-0.03, 0.03))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.02))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.02))

            data.append(
                {
                    "date": current_date,
                    "timestamp": current_date.isoformat(),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": random.randint(1000000, 5000000),
                }
            )

            base_price = close_price

            if interval == "minute":
                current_date += timedelta(minutes=1)
            elif interval == "5minute":
                current_date += timedelta(minutes=5)
            elif interval == "hour":
                current_date += timedelta(hours=1)
            else:  # day
                current_date += timedelta(days=1)

        return data

    async def place_order(self, **kwargs) -> dict[str, Any]:
        """Place a mock order"""
        order_id = str(uuid.uuid4())

        order = {
            "order_id": order_id,
            "exchange_order_id": f"NSE_{order_id[:8]}",
            "status": "COMPLETE",
            "status_message": "Order placed successfully",
            "order_timestamp": datetime.now(),
            **kwargs,
        }

        self.orders[order_id] = order

        # Update positions if order is complete
        if order["status"] == "COMPLETE":
            self._update_positions(order)

        return {"order_id": order_id, "status": "success"}

    async def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
        trigger_price: float | None = None,
    ) -> dict[str, Any]:
        """Modify a mock order"""
        if order_id not in self.orders:
            raise Exception("Order not found")

        order = self.orders[order_id]

        if quantity:
            order["quantity"] = quantity
        if price:
            order["price"] = price
        if trigger_price:
            order["trigger_price"] = trigger_price

        order["modified_timestamp"] = datetime.now()

        return {"status": "success", "order_id": order_id}

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a mock order"""
        if order_id not in self.orders:
            raise Exception("Order not found")

        self.orders[order_id]["status"] = "CANCELLED"
        self.orders[order_id]["cancelled_timestamp"] = datetime.now()

        return {"status": "success", "order_id": order_id}

    async def get_orders(self) -> list[dict[str, Any]]:
        """Get all orders"""
        return list(self.orders.values())

    async def get_positions(self) -> dict[str, list[dict[str, Any]]]:
        """Get current positions"""
        net_positions = []
        day_positions = []

        for position in self.positions:
            # Calculate P&L
            current_price = self.market_data.get(
                position["tradingsymbol"], {"last_price": position["average_price"]}
            )["last_price"]

            pnl = (current_price - position["average_price"]) * position["quantity"]
            pnl_percent = (
                pnl / (position["average_price"] * abs(position["quantity"]))
            ) * 100

            position_data = {
                **position,
                "last_price": current_price,
                "pnl": round(pnl, 2),
                "pnl_percent": round(pnl_percent, 2),
                "value": round(current_price * abs(position["quantity"]), 2),
            }

            net_positions.append(position_data)
            if position.get("product") == "MIS":
                day_positions.append(position_data)

        return {"net": net_positions, "day": day_positions}

    async def get_portfolio(self) -> dict[str, Any]:
        """Get portfolio summary"""
        positions = await self.get_positions()

        # Calculate total position value
        position_value = sum(pos["value"] for pos in positions["net"])
        total_pnl = sum(pos["pnl"] for pos in positions["net"])

        # Calculate available cash
        available_cash = self.portfolio_value - position_value

        return {
            "equity": self.portfolio_value + total_pnl,
            "available_cash": available_cash,
            "used_margin": position_value,
            "total_pnl": total_pnl,
        }

    async def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        portfolio = await self.get_portfolio()
        return portfolio["equity"]

    async def get_market_depth(self, symbol: str) -> dict[str, Any]:
        """Get mock market depth"""
        base_price = self.market_data.get(symbol, {"last_price": 2500})["last_price"]

        buy_orders = []
        sell_orders = []

        # Generate 5 levels of depth
        for i in range(5):
            buy_price = base_price - (i + 1) * 0.25
            sell_price = base_price + (i + 1) * 0.25

            buy_orders.append(
                {
                    "price": round(buy_price, 2),
                    "quantity": random.randint(100, 1000),
                    "orders": random.randint(1, 10),
                }
            )

            sell_orders.append(
                {
                    "price": round(sell_price, 2),
                    "quantity": random.randint(100, 1000),
                    "orders": random.randint(1, 10),
                }
            )

        return {"buy": buy_orders, "sell": sell_orders}

    async def get_ohlc(self, symbol: str) -> dict[str, Any]:
        """Get OHLC data for a symbol"""
        quote = await self.get_quote(symbol)

        return {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "prev_close": quote["open"] * 0.99,
            "volume": quote["volume"],
            "value": quote["volume"] * quote["last_price"],
            "vwap": quote["last_price"] * 0.995,
            "week_52_high": quote["high"] * 1.2,
            "week_52_low": quote["low"] * 0.8,
            "upper_circuit": quote["last_price"] * 1.1,
            "lower_circuit": quote["last_price"] * 0.9,
        }

    async def get_instruments(
        self, exchange: str | None = None
    ) -> list[dict[str, Any]]:
        """Get list of instruments"""
        instruments = [
            {"symbol": "RELIANCE", "exchange": "NSE", "segment": "EQ"},
            {"symbol": "TCS", "exchange": "NSE", "segment": "EQ"},
            {"symbol": "INFY", "exchange": "NSE", "segment": "EQ"},
            {"symbol": "HDFC", "exchange": "NSE", "segment": "EQ"},
            {"symbol": "ICICIBANK", "exchange": "NSE", "segment": "EQ"},
            {"symbol": "SBIN", "exchange": "NSE", "segment": "EQ"},
            {"symbol": "BHARTIARTL", "exchange": "NSE", "segment": "EQ"},
            {"symbol": "ITC", "exchange": "NSE", "segment": "EQ"},
            {"symbol": "KOTAKBANK", "exchange": "NSE", "segment": "EQ"},
            {"symbol": "LT", "exchange": "NSE", "segment": "EQ"},
        ]

        if exchange:
            instruments = [i for i in instruments if i["exchange"] == exchange]

        return instruments

    async def subscribe_ticker(self, symbol: str):
        """Mock ticker subscription"""
        pass

    async def unsubscribe_all_tickers(self):
        """Mock unsubscribe all tickers"""
        pass

    def _generate_market_data(self) -> dict[str, dict[str, Any]]:
        """Generate initial market data"""
        symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"]
        data = {}

        for symbol in symbols:
            data[symbol] = self._generate_quote(symbol)

        return data

    def _generate_quote(self, symbol: str) -> dict[str, Any]:
        """Generate a mock quote"""
        base_prices = {
            "RELIANCE": 2500.0,
            "TCS": 3500.0,
            "INFY": 1500.0,
            "HDFC": 1600.0,
            "ICICIBANK": 900.0,
        }

        base_price = base_prices.get(symbol, 1000.0)

        return {
            "symbol": symbol,
            "last_price": base_price,
            "change": round(random.uniform(-50, 50), 2),
            "change_percent": round(random.uniform(-2, 2), 2),
            "volume": random.randint(1000000, 5000000),
            "bid": base_price - 0.25,
            "ask": base_price + 0.25,
            "open": base_price * 0.99,
            "high": base_price * 1.02,
            "low": base_price * 0.98,
            "close": base_price,
            "timestamp": datetime.now(),
        }

    def _update_positions(self, order: dict[str, Any]):
        """Update positions based on executed order"""
        symbol = order["tradingsymbol"]
        quantity = order["quantity"]
        price = order.get(
            "price", self.market_data.get(symbol, {"last_price": 2500})["last_price"]
        )

        # Find existing position
        existing_position = None
        for pos in self.positions:
            if pos["tradingsymbol"] == symbol:
                existing_position = pos
                break

        if order["transaction_type"] == "BUY":
            if existing_position:
                # Average the position
                total_qty = existing_position["quantity"] + quantity
                total_value = (
                    existing_position["average_price"] * existing_position["quantity"]
                ) + (price * quantity)
                existing_position["quantity"] = total_qty
                existing_position["average_price"] = round(total_value / total_qty, 2)
            else:
                # Create new position
                self.positions.append(
                    {
                        "tradingsymbol": symbol,
                        "exchange": order.get("exchange", "NSE"),
                        "quantity": quantity,
                        "average_price": price,
                        "product": order.get("product", "MIS"),
                        "overnight_quantity": 0,
                        "multiplier": 1,
                        "buy_quantity": quantity,
                        "sell_quantity": 0,
                    }
                )

        elif order["transaction_type"] == "SELL":
            if existing_position:
                existing_position["quantity"] -= quantity
                if existing_position["quantity"] == 0:
                    self.positions.remove(existing_position)
