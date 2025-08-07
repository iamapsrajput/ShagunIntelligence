import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

from crewai import Agent
from loguru import logger as loguru_logger

from ..base_quality_aware_agent import BaseQualityAwareAgent, DataQualityLevel
from .order_manager import OrderManager
from .order_timing_optimizer import OrderTimingOptimizer
from .paper_trading_manager import PaperTradingManager
from .position_monitor import PositionMonitor
from .trade_logger import TradeLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Trade signal with all necessary information including data quality"""

    symbol: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str  # MARKET, LIMIT, SL, SL-M
    price: float | None = None
    trigger_price: float | None = None
    stop_loss: float | None = None
    target: float | None = None
    confidence: float = 0.0
    strategy_name: str = ""
    metadata: dict[str, Any] = None
    data_quality_score: float = 1.0  # Data quality at signal generation
    quality_level: str = "high"  # Quality level


class TradeExecutorAgent(BaseQualityAwareAgent, Agent):
    """Quality-aware agent for executing trades with pre-execution data quality checks"""

    def __init__(self, kite_client=None, paper_trading=False):
        BaseQualityAwareAgent.__init__(self)
        Agent.__init__(
            self,
            name="Quality-Aware Trade Executor",
            role="Execute trades with pre-execution data quality validation",
            goal="Execute trades only when data quality meets minimum thresholds",
            backstory="""You are an expert trade execution specialist who understands that
            execution quality depends on data reliability. You perform pre-execution checks:
            - Verify current quote quality before placing orders
            - Validate price levels against multiple sources
            - Adjust order types based on data confidence
            - Block trades when data quality is below threshold
            High quality (>85%): Execute all order types normally
            Medium quality (65-85%): Limit orders only, no market orders
            Low quality (<65%): Block all new trades, manage existing only""",
            verbose=True,
            allow_delegation=False,
        )

        self.kite_client = kite_client
        self.paper_trading = paper_trading

        # Initialize components
        self.order_manager = OrderManager(kite_client, paper_trading)
        self.position_monitor = PositionMonitor(kite_client, paper_trading)
        self.paper_trading_manager = PaperTradingManager() if paper_trading else None
        self.trade_logger = TradeLogger()
        self.timing_optimizer = OrderTimingOptimizer()

        # Trading session times (NSE)
        self.market_open = time(9, 15)
        self.market_close = time(15, 30)
        self.intraday_square_off = time(15, 15)  # Square off 15 mins before close

        # Quality thresholds for execution
        self.execution_quality_thresholds = {
            "market_order": 0.85,  # High quality required for market orders
            "limit_order": 0.65,  # Medium quality acceptable for limit orders
            "stop_loss": 0.50,  # Lower threshold for protective orders
            "any_order": 0.40,  # Minimum for any execution
        }

    async def execute_trade_with_quality_check(
        self, signal: TradeSignal
    ) -> dict[str, Any]:
        """Execute trade with pre-execution data quality validation"""
        try:
            # Log the incoming signal
            self.trade_logger.log_signal(signal)

            # Check if market is open
            if not self._is_market_open():
                logger.warning(
                    f"Market is closed. Cannot execute trade for {signal.symbol}"
                )
                return {"status": "rejected", "reason": "market_closed"}

            # Pre-execution quality check
            quality_check = await self._pre_execution_quality_check(signal)

            if not quality_check["approved"]:
                logger.warning(
                    f"Trade blocked for {signal.symbol}: {quality_check['reason']}"
                )
                return {
                    "status": "rejected",
                    "reason": quality_check["reason"],
                    "data_quality": quality_check["data_quality"],
                    "quality_level": quality_check["quality_level"],
                }

            # Adjust order based on quality
            adjusted_signal = self._adjust_signal_for_quality(
                signal, quality_check["quality_level"]
            )

            # Check paper trading mode
            if self.paper_trading:
                result = self.paper_trading_manager.execute_trade(adjusted_signal)
                result["data_quality"] = quality_check["data_quality"]
                self.trade_logger.log_trade(result)
                return result

            # Validate prices against current market
            price_validation = await self._validate_prices(adjusted_signal)
            if not price_validation["valid"]:
                return {"status": "rejected", "reason": price_validation["reason"]}

            # Optimize order timing
            timing_params = self.timing_optimizer.get_optimal_timing(
                adjusted_signal.symbol, adjusted_signal.action, adjusted_signal.quantity
            )

            # Execute based on order type
            if adjusted_signal.order_type == "BRACKET":
                result = self._execute_bracket_order(adjusted_signal, timing_params)
            elif adjusted_signal.order_type == "LIMIT":
                result = self._execute_limit_order(adjusted_signal, timing_params)
            else:  # MARKET
                result = self._execute_market_order(adjusted_signal, timing_params)

            # Add quality metadata
            result["execution_quality"] = quality_check

            # Log the execution result
            self.trade_logger.log_trade(result)

            # Start monitoring the position if order was successful
            if result.get("status") == "success":
                self.position_monitor.add_position(result["order_id"], adjusted_signal)

            return result

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _execute_market_order(self, signal: TradeSignal, timing_params: dict) -> dict:
        """Execute a market order"""
        return self.order_manager.place_market_order(
            symbol=signal.symbol,
            action=signal.action,
            quantity=signal.quantity,
            **timing_params,
        )

    def _execute_limit_order(self, signal: TradeSignal, timing_params: dict) -> dict:
        """Execute a limit order with smart pricing"""
        # Get optimal limit price
        limit_price = signal.price
        if not limit_price:
            limit_price = self._calculate_limit_price(signal.symbol, signal.action)

        return self.order_manager.place_limit_order(
            symbol=signal.symbol,
            action=signal.action,
            quantity=signal.quantity,
            price=limit_price,
            **timing_params,
        )

    def _execute_bracket_order(self, signal: TradeSignal, timing_params: dict) -> dict:
        """Execute a bracket order with stop loss and target"""
        return self.order_manager.place_bracket_order(
            symbol=signal.symbol,
            action=signal.action,
            quantity=signal.quantity,
            price=signal.price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            **timing_params,
        )

    def monitor_positions(self) -> list[dict]:
        """Monitor all open positions and handle exits"""
        positions = self.position_monitor.get_open_positions()

        for position in positions:
            # Check exit conditions
            exit_signal = self._check_exit_conditions(position)

            if exit_signal:
                # Execute exit trade
                exit_result = self.execute_trade(exit_signal)
                logger.info(
                    f"Exit trade executed for {position['symbol']}: {exit_result}"
                )

        return positions

    def _check_exit_conditions(self, position: dict) -> TradeSignal | None:
        """Check if position should be exited"""
        current_time = datetime.now().time()

        # Force square off near market close for intraday
        if current_time >= self.intraday_square_off:
            return TradeSignal(
                symbol=position["symbol"],
                action="SELL" if position["side"] == "BUY" else "BUY",
                quantity=position["quantity"],
                order_type="MARKET",
                strategy_name="intraday_square_off",
            )

        # Check stop loss hit
        if self._is_stop_loss_hit(position):
            return TradeSignal(
                symbol=position["symbol"],
                action="SELL" if position["side"] == "BUY" else "BUY",
                quantity=position["quantity"],
                order_type="MARKET",
                strategy_name="stop_loss_hit",
            )

        # Check target hit
        if self._is_target_hit(position):
            return TradeSignal(
                symbol=position["symbol"],
                action="SELL" if position["side"] == "BUY" else "BUY",
                quantity=position["quantity"],
                order_type="MARKET",
                strategy_name="target_hit",
            )

        return None

    def _is_stop_loss_hit(self, position: dict) -> bool:
        """Check if stop loss is hit for a position"""
        if not position.get("stop_loss"):
            return False

        current_price = self._get_current_price(position["symbol"])

        if position["side"] == "BUY":
            return current_price <= position["stop_loss"]
        else:
            return current_price >= position["stop_loss"]

    def _is_target_hit(self, position: dict) -> bool:
        """Check if target is hit for a position"""
        if not position.get("target"):
            return False

        current_price = self._get_current_price(position["symbol"])

        if position["side"] == "BUY":
            return current_price >= position["target"]
        else:
            return current_price <= position["target"]

    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        if self.paper_trading:
            return self.paper_trading_manager.get_price(symbol)

        try:
            quote = self.kite_client.quote([symbol])
            return quote[symbol]["last_price"]
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return 0.0

    def _calculate_limit_price(self, symbol: str, action: str) -> float:
        """Calculate optimal limit price based on order book"""
        try:
            quote = self.kite_client.quote([symbol])
            bid = quote[symbol]["depth"]["buy"][0]["price"]
            ask = quote[symbol]["depth"]["sell"][0]["price"]

            # Place limit order slightly better than best bid/ask
            if action == "BUY":
                return bid + 0.05  # 5 paise above best bid
            else:
                return ask - 0.05  # 5 paise below best ask

        except Exception as e:
            logger.error(f"Error calculating limit price: {str(e)}")
            return 0.0

    async def _pre_execution_quality_check(self, signal: TradeSignal) -> dict[str, Any]:
        """Perform pre-execution data quality validation."""
        try:
            # Get current quote with quality assessment
            (
                quote_data,
                data_quality,
                quality_level,
            ) = await self.get_quality_weighted_data(signal.symbol, "quote")

            if not quote_data:
                return {
                    "approved": False,
                    "reason": "Unable to fetch current market data",
                    "data_quality": 0.0,
                    "quality_level": DataQualityLevel.CRITICAL.value,
                }

            # Check minimum quality threshold
            min_threshold = self.execution_quality_thresholds.get(
                signal.order_type.lower(),
                self.execution_quality_thresholds["any_order"],
            )

            if data_quality < min_threshold:
                return {
                    "approved": False,
                    "reason": f"Data quality {data_quality:.1%} below threshold {min_threshold:.1%}",
                    "data_quality": data_quality,
                    "quality_level": quality_level.value,
                    "required_quality": min_threshold,
                }

            # Additional check for market orders
            if signal.order_type == "MARKET" and quality_level != DataQualityLevel.HIGH:
                return {
                    "approved": False,
                    "reason": "Market orders require high data quality",
                    "data_quality": data_quality,
                    "quality_level": quality_level.value,
                    "suggested_order_type": "LIMIT",
                }

            # Get multi-source validation for large orders
            if (
                signal.quantity * quote_data.get("current_price", 0) > 100000
            ):  # Large order
                (
                    consensus_data,
                    consensus_confidence,
                ) = await self.get_multi_source_consensus(signal.symbol)

                if consensus_confidence < 0.7:
                    return {
                        "approved": False,
                        "reason": "Large order requires high consensus confidence",
                        "data_quality": data_quality,
                        "quality_level": quality_level.value,
                        "consensus_confidence": consensus_confidence,
                    }

            return {
                "approved": True,
                "data_quality": data_quality,
                "quality_level": quality_level,
                "current_price": quote_data.get("current_price"),
                "bid": quote_data.get("bid"),
                "ask": quote_data.get("ask"),
                "data_source": quote_data.get("data_source"),
            }

        except Exception as e:
            loguru_logger.error(f"Error in pre-execution quality check: {e}")
            return {
                "approved": False,
                "reason": f"Quality check error: {str(e)}",
                "data_quality": 0.0,
                "quality_level": DataQualityLevel.CRITICAL.value,
            }

    def _adjust_signal_for_quality(
        self, signal: TradeSignal, quality_level: DataQualityLevel
    ) -> TradeSignal:
        """Adjust trade signal based on data quality."""
        adjusted_signal = TradeSignal(
            symbol=signal.symbol,
            action=signal.action,
            quantity=signal.quantity,
            order_type=signal.order_type,
            price=signal.price,
            trigger_price=signal.trigger_price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            confidence=signal.confidence,
            strategy_name=signal.strategy_name,
            metadata=signal.metadata,
            data_quality_score=signal.data_quality_score,
            quality_level=quality_level.value,
        )

        # Adjust based on quality
        if quality_level == DataQualityLevel.MEDIUM:
            # Convert market orders to limit orders
            if adjusted_signal.order_type == "MARKET":
                adjusted_signal.order_type = "LIMIT"
                # Price will be set during validation

            # Reduce quantity for medium quality
            adjusted_signal.quantity = int(adjusted_signal.quantity * 0.7)

            # Widen stop loss for uncertainty
            if adjusted_signal.stop_loss:
                if adjusted_signal.action == "BUY":
                    adjusted_signal.stop_loss *= 0.98  # 2% wider stop
                else:
                    adjusted_signal.stop_loss *= 1.02

        elif quality_level == DataQualityLevel.LOW:
            # Very conservative adjustments
            adjusted_signal.order_type = "LIMIT"
            adjusted_signal.quantity = int(adjusted_signal.quantity * 0.3)

            # Very wide stop loss
            if adjusted_signal.stop_loss:
                if adjusted_signal.action == "BUY":
                    adjusted_signal.stop_loss *= 0.95  # 5% wider stop
                else:
                    adjusted_signal.stop_loss *= 1.05

        return adjusted_signal

    async def _validate_prices(self, signal: TradeSignal) -> dict[str, Any]:
        """Validate signal prices against current market."""
        try:
            # Get current quote
            quote_data, _, _ = await self.get_quality_weighted_data(
                signal.symbol, "quote"
            )

            if not quote_data:
                return {"valid": False, "reason": "Unable to fetch current prices"}

            current_price = quote_data.get("current_price", 0)
            bid = quote_data.get("bid", current_price)
            ask = quote_data.get("ask", current_price)

            # Set limit price if not provided
            if signal.order_type == "LIMIT" and not signal.price:
                if signal.action == "BUY":
                    signal.price = bid + 0.05  # Slightly above bid
                else:
                    signal.price = ask - 0.05  # Slightly below ask

            # Validate limit price
            if signal.order_type == "LIMIT":
                if signal.action == "BUY" and signal.price > ask * 1.01:
                    return {
                        "valid": False,
                        "reason": f"Buy limit price {signal.price} too far from ask {ask}",
                    }
                elif signal.action == "SELL" and signal.price < bid * 0.99:
                    return {
                        "valid": False,
                        "reason": f"Sell limit price {signal.price} too far from bid {bid}",
                    }

            # Validate stop loss
            if signal.stop_loss:
                if signal.action == "BUY" and signal.stop_loss > current_price:
                    return {
                        "valid": False,
                        "reason": "Buy stop loss above current price",
                    }
                elif signal.action == "SELL" and signal.stop_loss < current_price:
                    return {
                        "valid": False,
                        "reason": "Sell stop loss below current price",
                    }

            return {
                "valid": True,
                "current_price": current_price,
                "bid": bid,
                "ask": ask,
            }

        except Exception as e:
            loguru_logger.error(f"Error validating prices: {e}")
            return {"valid": False, "reason": str(e)}

    def execute_trade(self, signal: TradeSignal) -> dict[str, Any]:
        """Legacy synchronous execute - wraps async version."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.execute_trade_with_quality_check(signal)
            )
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.execute_trade_with_quality_check(signal)
            )

    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        current_time = datetime.now().time()

        # Check if it's a weekday
        if datetime.now().weekday() >= 5:  # Saturday or Sunday
            return False

        # Check market hours
        return self.market_open <= current_time <= self.market_close

    def get_execution_summary(self) -> dict[str, Any]:
        """Get summary of all executions for the day"""
        return {
            "total_trades": self.trade_logger.get_trade_count(),
            "successful_trades": self.trade_logger.get_successful_trades(),
            "failed_trades": self.trade_logger.get_failed_trades(),
            "open_positions": len(self.position_monitor.get_open_positions()),
            "pnl": self.position_monitor.get_total_pnl(),
            "execution_stats": self.order_manager.get_execution_stats(),
        }

    def close_all_positions(self) -> list[dict]:
        """Close all open positions (emergency or end of day)"""
        positions = self.position_monitor.get_open_positions()
        results = []

        for position in positions:
            exit_signal = TradeSignal(
                symbol=position["symbol"],
                action="SELL" if position["side"] == "BUY" else "BUY",
                quantity=position["quantity"],
                order_type="MARKET",
                strategy_name="close_all_positions",
            )

            result = self.execute_trade(exit_signal)
            results.append(result)

        return results
